"""MatildaModel inference-chain tests (torch-gated; fake Maia-3 wrapper).

The keystone test: with a zero-delta (freshly generated) checkpoint the whole
chain — featurization, candidate/vocab mapping, TXTC forward, and the map back
to real-board UCIs — must reproduce the Maia-3 prior exactly. That property
holds end-to-end only if every step is faithful.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import chess  # noqa: E402

from matilda_uci.matilda.inference import MatildaModel  # noqa: E402
from matilda_uci.matilda.maia3_wrapper import Maia3Result  # noqa: E402
from matilda_uci.matilda.move_vocab import VOCAB_SIZE, legal_entries  # noqa: E402
from matilda_uci.matilda.search import (  # noqa: E402
    CP_UNSCORED,
    MoveScore,
    scores_to_arrays,
)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from generate_checkpoint import generate_base, generate_posthoc, generate_style  # noqa: E402


class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeMaia3Wrapper:
    """Deterministic Maia-3 stand-in: consistent logits / probs / top_moves."""

    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self.calls: list[dict] = []

    def infer(self, board, *, elo_self, elo_oppo, board_history=None):
        self.calls.append(dict(elo_self=elo_self, elo_oppo=elo_oppo,
                               board_history=board_history))
        rng = np.random.RandomState(board.ply())  # deterministic per position
        logits = rng.randn(VOCAB_SIZE).astype(np.float32) * 2.0
        entries = legal_entries(board)
        sub = np.array([logits[i] for _, i in entries], dtype=np.float64)
        p = np.exp(sub - sub.max())
        p /= p.sum()
        move_probs = {uci: float(pi) for (uci, _), pi in zip(entries, p)}
        top = tuple(sorted(move_probs.items(), key=lambda kv: kv[1], reverse=True)[:16])
        return Maia3Result(
            move_probs=move_probs,
            logits=logits,
            win_prob=0.55,
            top_moves=top,
            hidden=rng.randn(512).astype(np.float32),
            importance=rng.rand(8, 8).astype(np.float32),
        )

    def close(self):
        pass


@pytest.fixture()
def zero_ckpt(tmp_path) -> str:
    sd = generate_base(_Args(seed=0, n_players=1, dropout=0.1, random_head=False))
    path = tmp_path / "zero_base.pt"
    torch.save(sd, path)
    return str(path)


def make_model(ckpt: str) -> tuple[MatildaModel, FakeMaia3Wrapper]:
    wrapper = FakeMaia3Wrapper()
    return MatildaModel(ckpt, wrapper=wrapper), wrapper


def test_zero_delta_model_reproduces_maia3_prior(zero_ckpt) -> None:
    model, wrapper = make_model(zero_ckpt)
    board = chess.Board()
    for mv in ["e2e4", "e7e5", "g1f3"]:
        board.push_uci(mv)
    pred = model.predict(board, board_history=["x"], elo_self=1500, elo_oppo=1500)
    prior = wrapper.infer(board, elo_self=1500, elo_oppo=1500).move_probs
    assert set(pred.move_probs) == set(prior)
    for uci, p in prior.items():
        # fp16 feature quantization perturbs the logits slightly by design
        # (training saw the same quantization), hence the loose-ish tolerance.
        assert pred.move_probs[uci] == pytest.approx(p, abs=5e-3), uci
    assert abs(sum(pred.move_probs.values()) - 1.0) < 1e-6
    assert pred.win_prob == pytest.approx(0.55)
    assert not pred.engine_used


def test_black_to_move_mirroring(zero_ckpt) -> None:
    model, wrapper = make_model(zero_ckpt)
    board = chess.Board()
    board.push_uci("e2e4")  # black to move: vocab is mirrored
    pred = model.predict(board)
    prior = wrapper.infer(board, elo_self=1500, elo_oppo=1500).move_probs
    legal = {m.uci() for m in board.legal_moves}
    assert set(pred.move_probs) == legal
    for uci in legal:
        assert pred.move_probs[uci] == pytest.approx(prior[uci], abs=5e-3), uci


def test_mate_position_returns_none(zero_ckpt) -> None:
    model, _ = make_model(zero_ckpt)
    assert model.predict(chess.Board("7k/5KQ1/8/8/8/8/8/8 b - - 0 1")) is None


def test_controller_scores_flow_through(zero_ckpt) -> None:
    class OneMoveController:
        def __init__(self):
            self.seen: list[list[str]] = []

        def score(self, board, candidates):
            self.seen.append(list(candidates))
            return [MoveScore(candidates[0], cp=120, depth=8)]

        def close(self):
            pass

    model, _ = make_model(zero_ckpt)
    ctrl = OneMoveController()
    pred = model.predict(chess.Board(), controller=ctrl)
    assert pred.engine_used
    assert ctrl.seen and ctrl.seen[0] == list(pred.candidates)
    # zero-delta model: engine features cannot change the output
    base = make_model(zero_ckpt)[0].predict(chess.Board())
    for uci, p in base.move_probs.items():
        assert pred.move_probs[uci] == pytest.approx(p, abs=1e-9)


def test_controller_failure_degrades_gracefully(zero_ckpt) -> None:
    class Boom:
        def score(self, board, candidates):
            raise RuntimeError("engine died")

        def close(self):
            pass

    model, _ = make_model(zero_ckpt)
    pred = model.predict(chess.Board(), controller=Boom())
    assert pred is not None and not pred.engine_used


def test_style_overlay_and_posthoc_rows(zero_ckpt, tmp_path) -> None:
    style = generate_style(_Args(seed=1, n_players=10, base=zero_ckpt))
    ph = generate_posthoc(_Args(seed=2, n_new=5, n_old=11,
                                manifest="m.json", k=0))
    style_path, ph_path = tmp_path / "style.pt", tmp_path / "ph.pt"
    torch.save(style, style_path)
    torch.save(ph, ph_path)

    model, _ = make_model(zero_ckpt)
    rows = model.load_style(str(style_path), str(ph_path))
    assert rows == 16  # 11 style rows + 5 post-hoc
    assert torch.equal(
        model._model.spe.weight[11:], ph["spe_new"]
    )
    pred = model.predict(chess.Board(), pid=12)  # a post-hoc player
    assert abs(sum(pred.move_probs.values()) - 1.0) < 1e-6


def test_pid_without_style_overlay_is_ignored(zero_ckpt) -> None:
    model, _ = make_model(zero_ckpt)
    pred = model.predict(chess.Board(), pid=3)
    assert pred is not None  # warns and plays style-free


def test_hidden_none_degrades_gracefully(zero_ckpt) -> None:
    """A wrapper whose hidden hook failed returns hidden=None; predict must
    zero-fill (like importance) instead of crashing every move."""

    class NoHiddenWrapper(FakeMaia3Wrapper):
        def infer(self, board, **kwargs):
            r = super().infer(board, **kwargs)
            return Maia3Result(
                move_probs=r.move_probs, logits=r.logits, win_prob=r.win_prob,
                top_moves=r.top_moves, hidden=None, importance=None,
            )

    model = MatildaModel(zero_ckpt, wrapper=NoHiddenWrapper())
    pred = model.predict(chess.Board())
    assert pred is not None
    assert abs(sum(pred.move_probs.values()) - 1.0) < 1e-6


def test_checkpoint_style_rows_are_derived_not_hardcoded(tmp_path) -> None:
    """A base checkpoint with a non-default style table must still strict-load."""
    sd = generate_base(_Args(seed=0, n_players=6998, dropout=0.1, random_head=False))
    path = tmp_path / "wide_base.pt"
    torch.save(sd, path)
    model = MatildaModel(str(path), wrapper=FakeMaia3Wrapper())
    assert model.predict(chess.Board()) is not None  # loads without size mismatch


def test_wrong_variant_feature_shapes_fail_loudly(zero_ckpt) -> None:
    """A Maia-3 variant with different dims must raise an actionable error,
    not a cryptic matmul failure deep in the transformer."""
    from matilda_uci.matilda.inference import Maia3FeatureError

    class WrongShapeWrapper(FakeMaia3Wrapper):
        def infer(self, board, **kwargs):
            r = super().infer(board, **kwargs)
            return Maia3Result(
                move_probs=r.move_probs,
                logits=r.logits,
                win_prob=r.win_prob,
                top_moves=r.top_moves,
                hidden=np.zeros(1024, np.float32),  # a bigger variant's width
                importance=r.importance,
            )

    model = MatildaModel(zero_ckpt, wrapper=WrongShapeWrapper())
    with pytest.raises(Maia3FeatureError, match="hidden state: 1024"):
        model.predict(chess.Board())


def test_feature_check_passes_once_for_correct_shapes(zero_ckpt) -> None:
    model, _ = make_model(zero_ckpt)
    assert model.predict(chess.Board()) is not None
    assert model._features_checked


def test_non_23m_variant_warns(zero_ckpt, caplog) -> None:
    import logging

    with caplog.at_level(logging.WARNING, logger="matilda_uci.matilda.inference"):
        MatildaModel(zero_ckpt, maia3_model="79m", wrapper=FakeMaia3Wrapper())
    assert any("79m" in rec.message for rec in caplog.records)


def test_close_leaves_injected_wrapper_alone(zero_ckpt) -> None:
    class ClosableWrapper(FakeMaia3Wrapper):
        def __init__(self):
            super().__init__()
            self.closed = False

        def close(self):
            self.closed = True

    wrapper = ClosableWrapper()
    model = MatildaModel(zero_ckpt, wrapper=wrapper)
    model.predict(chess.Board())
    model.close()
    assert not wrapper.closed  # caller-owned: close() must not touch it
    # and a reused model keeps using the injected wrapper, not a real Maia-3
    assert model.predict(chess.Board()) is not None
    assert len(wrapper.calls) == 2


# --- scores_to_arrays unit tests ---------------------------------------------------

def test_scores_to_arrays_rank_derivation() -> None:
    cands = ["e2e4", "d2d4", "g1f3"] + [""] * 13
    scores = [MoveScore("d2d4", cp=30), MoveScore("e2e4", cp=50)]
    sf_cp, sf_rank, valid = scores_to_arrays(cands, scores)
    assert valid == 1
    assert sf_cp[0] == 50 and sf_rank[0] == 1  # e2e4: higher cp -> rank 1
    assert sf_cp[1] == 30 and sf_rank[1] == 2
    assert sf_cp[2] == CP_UNSCORED and sf_rank[2] == 0  # unscored candidate


def test_scores_to_arrays_mixed_ranks_rederived() -> None:
    """Partial ranks can't be honored: rank 0 means 'unscored' to the model,
    so scored-but-unranked moves must get a real 1-based rank."""
    cands = ["e2e4", "d2d4", "g1f3"] + [""] * 13
    scores = [
        MoveScore("d2d4", cp=30, rank=1),  # controller ranked only its top pick
        MoveScore("e2e4", cp=50),
        MoveScore("g1f3", cp=10),
    ]
    sf_cp, sf_rank, valid = scores_to_arrays(cands, scores)
    assert valid == 1
    assert sf_rank[0] == 1 and sf_rank[1] == 2 and sf_rank[2] == 3  # by cp desc
    assert (sf_rank[:3] > 0).all()  # no scored move carries the unscored rank


def test_scores_to_arrays_clamps_and_respects_given_ranks() -> None:
    cands = ["e2e4", "d2d4"] + [""] * 14
    scores = [
        MoveScore("e2e4", cp=999_999, rank=1),
        MoveScore("d2d4", cp=-999_999, rank=2),
    ]
    sf_cp, sf_rank, valid = scores_to_arrays(cands, scores)
    assert sf_cp[0] == 32000 and sf_cp[1] == -32000
    assert sf_rank[0] == 1 and sf_rank[1] == 2


def test_scores_to_arrays_empty() -> None:
    sf_cp, sf_rank, valid = scores_to_arrays([""] * 16, [])
    assert valid == 0
    assert (sf_cp == CP_UNSCORED).all() and (sf_rank == 0).all()
