"""MatildaPolicy + engine go-parsing tests. Torch-free: a fake model is injected.

The real inference chain (Maia-3 -> featurizer -> TXTC) is covered by the
torch-gated tests in ``test_matilda_inference.py``; here we test the UCI-facing
behaviour: option plumbing, the time-control latch, history reconstruction,
temperature sampling, and the engine's ``go`` parameter parsing.
"""

from __future__ import annotations

import chess

from matilda_uci.engine import UciEngine, _parse_go
from matilda_uci.matilda_policy import MatildaPolicy, board_history_fens
from matilda_uci.policy import MovePolicy


class FakePrediction:
    def __init__(self, move_probs, win_prob=0.5, engine_used=False):
        self.move_probs = move_probs
        self.win_prob = win_prob
        self.engine_used = engine_used
        self.candidates = tuple(move_probs)
        self.prior_probs = dict(move_probs)


class FakeModel:
    """Records predict() kwargs; favours the alphabetically-first legal move."""

    def __init__(self):
        self.calls: list[dict] = []
        self.style_loads: list[tuple] = []
        self.closed = False

    def predict(self, board, **kwargs):
        self.calls.append(kwargs)
        legal = sorted(m.uci() for m in board.legal_moves)
        if not legal:
            return None
        probs = {u: (0.7 if i == 0 else 0.3 / (len(legal) - 1 or 1))
                 for i, u in enumerate(legal)}
        return FakePrediction(probs)

    def load_style(self, style, posthoc=None):
        self.style_loads.append((style, posthoc))
        return 7000

    def close(self):
        self.closed = True


def make_policy(**kwargs) -> tuple[MatildaPolicy, FakeModel]:
    model = FakeModel()
    policy = MatildaPolicy(model=model, **kwargs)
    return policy, model


def test_conforms_to_protocol() -> None:
    policy, _ = make_policy()
    assert isinstance(policy, MovePolicy)


def test_select_normalizes_and_picks_best() -> None:
    policy, model = make_policy()
    result = policy.select(chess.Board())
    assert result.best_uci == sorted(m.uci() for m in chess.Board().legal_moves)[0]
    assert abs(sum(p for _, p in result.ranked) - 1.0) < 1e-9
    assert result.score_cp == 0  # win_prob 0.5 -> even
    assert "matilda" in result.info


def test_no_legal_moves_returns_none_move() -> None:
    policy, _ = make_policy()
    board = chess.Board("7k/5KQ1/8/8/8/8/8/8 b - - 0 1")  # checkmated
    assert policy.select(board).best_uci is None


def test_history_passed_to_model() -> None:
    policy, model = make_policy()
    board = chess.Board()
    for mv in ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"]:
        board.push_uci(mv)
    policy.select(board)
    hist = model.calls[-1]["board_history"]
    assert len(hist) == 5  # all five prior positions (fewer than the 8 cap)
    assert hist[0] == chess.Board().fen()  # oldest first


def test_board_history_caps_at_eight() -> None:
    board = chess.Board()
    for mv in ["g1f3", "g8f6", "f3g1", "f6g8"] * 3:  # 12 plies
        board.push_uci(mv)
    fens = board_history_fens(board)
    assert len(fens) == 8
    # newest of the history = position just before current
    prev = board.copy()
    prev.pop()
    assert fens[-1] == prev.fen()


def test_tc_latch_from_first_go() -> None:
    policy, model = make_policy(tc_base=180, tc_inc=0)
    board = chess.Board()  # white to move, ply 0
    policy.observe_go({"wtime": 300_000, "btime": 300_000, "winc": 2000, "binc": 2000}, board)
    assert policy.tc_base == 300.0 and policy.tc_inc == 2.0
    policy.select(board)
    assert model.calls[-1]["tc_base"] == 300.0
    # mid-game go must not re-latch the base
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    policy.observe_go({"wtime": 100_000, "winc": 2000}, board)
    assert policy.tc_base == 300.0


def test_tc_latch_black_side_and_reset() -> None:
    policy, _ = make_policy()
    board = chess.Board()
    board.push_uci("e2e4")  # black to move, ply 1 (black's first go)
    policy.observe_go({"wtime": 60_000, "btime": 61_000, "winc": 1000, "binc": 1000}, board)
    assert policy.tc_base == 61.0  # black's own clock
    policy.new_game()
    assert policy.tc_base == 180.0 and not policy._tc_latched


def test_tc_option_fallback_when_no_clock() -> None:
    policy, model = make_policy()
    policy.set_option("TimeControlBase", "600")
    policy.set_option("TimeControlInc", "5")
    policy.observe_go({"infinite": True}, chess.Board())  # analysis: no clocks
    policy.select(chess.Board())
    assert model.calls[-1]["tc_base"] == 600.0
    assert model.calls[-1]["tc_inc"] == 5.0


def test_elo_options_and_limit_strength() -> None:
    policy, model = make_policy()
    policy.set_option("UCI_Elo", "2400")
    policy.set_option("OpponentElo", "2200")
    policy.select(chess.Board())
    assert model.calls[-1]["elo_self"] == 2400
    assert model.calls[-1]["elo_oppo"] == 2200
    policy.set_option("UCI_LimitStrength", "false")
    policy.select(chess.Board())
    assert model.calls[-1]["elo_self"] == policy.elo_max


def test_style_loaded_once_and_pid_passed() -> None:
    policy, model = make_policy(style_checkpoint="style.pt", style_player_id=42)
    policy.select(chess.Board())
    policy.select(chess.Board())
    assert model.style_loads == [("style.pt", None)]
    assert model.calls[-1]["pid"] == 42
    policy.set_option("StylePlayerId", "-1")
    policy.select(chess.Board())
    assert model.calls[-1]["pid"] is None


def test_temperature_sampling_is_seeded() -> None:
    a, _ = make_policy(temperature=3.0, seed=7)
    b, _ = make_policy(temperature=3.0, seed=7)
    board = chess.Board()
    picks_a = [a.select(board).best_uci for _ in range(10)]
    picks_b = [b.select(board).best_uci for _ in range(10)]
    assert picks_a == picks_b
    assert len(set(picks_a)) > 1  # actually samples


def test_option_declarations_include_matilda_specific() -> None:
    policy, _ = make_policy()
    names = {o.name for o in policy.uci_options()}
    assert {"UCI_Elo", "OpponentElo", "TimeControlBase", "TimeControlInc",
            "AutoLatchTC", "Checkpoint", "StyleCheckpoint", "StylePlayerId",
            "EngineCmd", "EngineDepth", "EngineNodes"} <= names


def test_empty_placeholder_option_value() -> None:
    policy, _ = make_policy(engine_cmd="stockfish")
    policy.set_option("EngineCmd", "<empty>")
    assert policy.engine_cmd == ""


# --- engine go-parsing integration -----------------------------------------------

def test_parse_go() -> None:
    assert _parse_go("wtime 300000 btime 299000 winc 2000 binc 2000".split()) == {
        "wtime": 300_000, "btime": 299_000, "winc": 2000, "binc": 2000,
    }
    assert _parse_go(["infinite"]) == {"infinite": True}
    assert _parse_go("depth 10 movetime 500".split()) == {"depth": 10, "movetime": 500}


def test_engine_forwards_go_params_to_policy() -> None:
    import io

    policy, _ = make_policy()
    out = io.StringIO()
    engine = UciEngine(policy, out=out)
    engine.handle_command("position startpos")
    engine.handle_command("go wtime 180000 btime 180000 winc 1000 binc 1000")
    engine.join_search(timeout=5)
    assert policy.tc_base == 180.0 and policy.tc_inc == 1.0
    assert "bestmove" in out.getvalue()
