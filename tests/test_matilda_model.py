"""Tests for the Matilda inference core (TXTC) and the checkpoint generator.

The model/featurizer are faithful ports of the paper pipeline
(github.com/GarryChess/matilda1-paper); bitwise parity against the paper code
itself is asserted by ``scripts/verify_checkpoint.py``. These tests cover the
port's invariants without needing the paper repo.

They need torch, which is a runtime dependency but not required by the
UCI-engine test-suite; they skip cleanly when torch is absent. Tests that
compare against the *real* shipped checkpoint additionally skip when
``checkpoints/base_3k.pt`` is not staged (weights are gitignored).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from matilda_uci.matilda.features import tensors_sf  # noqa: E402
from matilda_uci.matilda.model import (  # noqa: E402
    N_CANDIDATES,
    VOCAB_SIZE,
    TXTC,
    maia3_reference_logprob,
)

REPO = Path(__file__).resolve().parent.parent
BASE_CKPT = REPO / "checkpoints" / "base_3k.pt"

# The generator lives in scripts/, not the package.
sys.path.insert(0, str(REPO / "scripts"))
from generate_checkpoint import generate_base, generate_posthoc, generate_style  # noqa: E402


class _Args:
    def __init__(self, **kw: object) -> None:
        self.__dict__.update(kw)


def _fake_inputs(n: int = 2, seed: int = 0):
    """Random tensors matching the TXTC feature contract."""
    g = torch.Generator().manual_seed(seed)
    ml = torch.randn(n, VOCAB_SIZE, generator=g)
    hid = torch.randn(n, 512, generator=g)
    imp = torch.randn(n, 64, generator=g)
    ci = torch.stack(
        [torch.randperm(VOCAB_SIZE, generator=g)[:N_CANDIDATES] for _ in range(n)]
    )
    lm = torch.zeros(n, VOCAB_SIZE)
    for r in range(n):
        lm[r, ci[r]] = 1.0
        lm[r, torch.randperm(VOCAB_SIZE, generator=g)[:20]] = 1.0
    tok = torch.randn(n, N_CANDIDATES, 9, generator=g)
    val = torch.ones(n, N_CANDIDATES)
    tcf = torch.randn(n, 2, generator=g)
    return ml, hid, imp, lm, ci, tok, val, tcf


def test_param_count_matches_checkpoint_family() -> None:
    assert sum(p.numel() for p in TXTC().parameters()) == 1_672_449


def test_zero_delta_equals_maia3() -> None:
    """The paper's invariant: untrained (zero-init dl) == Maia-3 exactly."""
    model = TXTC()  # dl is zero-initialised in __init__, as in training
    model.eval()
    ml, hid, imp, lm, ci, tok, val, tcf = _fake_inputs()
    with torch.no_grad():
        out = model(ml, hid, imp, lm, ci, tok, val, tcf, pid=None, style=False)
        ref = maia3_reference_logprob(ml, lm)
    assert torch.equal(out, ref)


def test_nonzero_delta_moves_only_candidates() -> None:
    """A trained head may only shift probability via the 16 candidate logits."""
    model = TXTC()
    torch.nn.init.normal_(model.dl.weight, std=0.5)
    torch.nn.init.normal_(model.dl.bias, std=0.5)
    model.eval()
    ml, hid, imp, lm, ci, tok, val, tcf = _fake_inputs()
    with torch.no_grad():
        out = model(ml, hid, imp, lm, ci, tok, val, tcf, pid=None, style=False)
        ref = maia3_reference_logprob(ml, lm)
    # Log-prob *gaps* between two legal non-candidate moves only share the
    # normalizer, so their pairwise differences must be intact.
    for r in range(ml.shape[0]):
        noncand = [
            i for i in torch.nonzero(lm[r]).flatten().tolist()
            if i not in set(ci[r].tolist())
        ]
        a, b = noncand[0], noncand[1]
        assert torch.allclose(
            out[r, a] - out[r, b], ref[r, a] - ref[r, b], atol=1e-5
        )


def test_invalid_candidates_are_inert() -> None:
    """Padded candidate slots (val=0) must not change the output."""
    model = TXTC()
    torch.nn.init.normal_(model.dl.weight, std=0.5)
    model.eval()
    ml, hid, imp, lm, ci, tok, val, tcf = _fake_inputs()
    val2 = val.clone()
    val2[:, 10:] = 0.0  # drop the last 6 candidates
    tok_junk = tok.clone()
    tok_junk[:, 10:] = 999.0  # junk features in the dropped slots
    with torch.no_grad():
        a = model(ml, hid, imp, lm, ci, tok, val2, tcf, pid=None, style=False)
        b = model(ml, hid, imp, lm, ci, tok_junk, val2, tcf, pid=None, style=False)
    assert torch.allclose(a, b, atol=1e-6)


def test_style_token_changes_output_only_when_enabled() -> None:
    model = TXTC(n_players=4)
    torch.nn.init.normal_(model.dl.weight, std=0.5)
    # spp is zero-initialised (trained separately, base frozen); give it real
    # values so the style path actually does something.
    torch.nn.init.normal_(model.spp.weight, std=0.5)
    model.eval()
    ml, hid, imp, lm, ci, tok, val, tcf = _fake_inputs()
    pid = torch.tensor([1, 2])
    with torch.no_grad():
        plain = model(ml, hid, imp, lm, ci, tok, val, tcf, pid=None, style=False)
        off = model(ml, hid, imp, lm, ci, tok, val, tcf, pid=pid, style=False)
        on = model(ml, hid, imp, lm, ci, tok, val, tcf, pid=pid, style=True)
    assert torch.equal(off, plain)
    assert not torch.allclose(on, plain)


def test_featurizer_sf_block_zeroes_gracefully() -> None:
    """Without sf_* inputs (or with use_sf=False) the engine block is all-zero."""
    rng = np.random.RandomState(0)
    n = 8
    d = {
        "maia_logits": rng.randn(n, VOCAB_SIZE).astype(np.float32),
        "maia_hidden": rng.randn(n, 512).astype(np.float32),
        "maia_importance": rng.randn(n, 8, 8).astype(np.float32),
        "legal_mask": (rng.rand(n, VOCAB_SIZE) < 0.01).astype(np.int8),
        "cand_idx": rng.randint(-1, VOCAB_SIZE, (n, N_CANDIDATES)).astype(np.int64),
        "cand_logit": (-rng.rand(n, N_CANDIDATES) * 8).astype(np.float32),
        "cand_valid": np.ones((n, N_CANDIDATES), np.float32),
        "tc_base": np.full(n, 180, np.int32),
        "tc_inc": np.zeros(n, np.int32),
    }
    T = tensors_sf(d, use_sf=True)  # no sf_* keys at all
    assert T["tok"].shape == (n, N_CANDIDATES, 9)
    assert torch.equal(T["tok"][..., 4:], torch.zeros(n, N_CANDIDATES, 5))
    assert "tg" not in T  # no target at inference time
    # top-1 flag marks exactly one candidate per row
    assert torch.equal(T["tok"][..., 3].sum(1), torch.ones(n))


def test_generated_formats_have_shipped_layout() -> None:
    base = generate_base(_Args(seed=0, n_players=1, dropout=0.1, random_head=False))
    TXTC().load_state_dict(base, strict=True)  # raises on any drift

    style = generate_style(_Args(seed=0, n_players=6998, base="checkpoints/base_3k.pt"))
    assert set(style) == {"state_dict", "sdim", "n_players", "base"}
    assert style["state_dict"]["spe.weight"].shape == (6999, 32)

    ph = generate_posthoc(
        _Args(seed=0, n_new=4745, n_old=6999, manifest="lostyle_manifest.json", k=0)
    )
    assert set(ph) == {"spe_new", "n_old", "manifest", "k"}
    assert ph["spe_new"].shape == (4745, 32)


@pytest.mark.skipif(not BASE_CKPT.exists(), reason="real checkpoint not staged")
def test_real_base_3k_loads_strict() -> None:
    sd = torch.load(BASE_CKPT, map_location="cpu")
    model = TXTC(n_players=sd["spe.weight"].shape[0] - 1)
    model.load_state_dict(sd, strict=True)  # exact key/shape match or it raises
