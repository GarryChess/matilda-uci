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

    def load_style_vector(self, style, vector):
        self.style_loads.append((style, vector))
        return 1

    def close(self):
        self.closed = True


def make_policy(**kwargs) -> tuple[MatildaPolicy, FakeModel]:
    model = FakeModel()
    # No engine by default: tests must not depend on (or spawn) a real
    # stockfish; engine-required behaviour is tested explicitly.
    kwargs.setdefault("engine_cmd", None)
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


def test_history_never_leaks_across_games() -> None:
    """A new game (or a bare-FEN position) must start with an empty history —
    stale FENs from the previous game would corrupt Maia-3's conditioning."""
    import io

    policy, model = make_policy()
    engine = UciEngine(policy, out=io.StringIO())
    engine.handle_command("position startpos moves e2e4 e7e5 g1f3")
    engine.handle_command("go")
    engine.join_search(timeout=5)
    assert len(model.calls[-1]["board_history"]) == 3

    engine.handle_command("ucinewgame")
    engine.handle_command("position startpos")
    engine.handle_command("go")
    engine.join_search(timeout=5)
    assert model.calls[-1]["board_history"] == []  # fresh game, no leak

    # analysis from a bare FEN: no history either
    engine.handle_command(
        "position fen r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
    )
    engine.handle_command("go")
    engine.join_search(timeout=5)
    assert model.calls[-1]["board_history"] == []


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


def test_style_vector_loaded_once_and_pid_passed() -> None:
    policy, model = make_policy(style_checkpoint="style.pt", style_vector="tal.pt")
    policy.select(chess.Board())
    policy.select(chess.Board())
    assert model.style_loads == [("style.pt", "tal.pt")]
    assert model.calls[-1]["pid"] == 1  # the loaded vector's row


def test_style_checkpoint_without_vector_plays_style_free() -> None:
    policy, model = make_policy(style_checkpoint="style.pt")
    policy.select(chess.Board())
    assert model.style_loads == []  # nothing to load without a vector
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
            "AutoLatchTC", "Checkpoint", "StyleCheckpoint", "StyleVector",
            "EngineCmd", "EngineDepth", "EngineNodes", "EngineMovetime",
            "Threads", "CacheSize"} <= names


def test_empty_placeholder_option_value() -> None:
    policy, _ = make_policy(engine_cmd="stockfish")
    policy.set_option("EngineCmd", "<empty>")
    assert policy.engine_cmd == ""


def test_engine_movetime_option_milliseconds() -> None:
    policy, _ = make_policy()
    policy.set_option("EngineMovetime", "1500")
    assert policy.engine_movetime == 1.5
    policy.set_option("EngineMovetime", "0")
    assert policy.engine_movetime == 0.0


def test_explicit_no_engine_means_no_controller() -> None:
    policy, _ = make_policy()  # engine_cmd=None via the helper
    assert policy.engine_cmd == ""
    assert policy._ensure_controller() is None


def test_engine_auto_resolves_or_raises(monkeypatch) -> None:
    import matilda_uci.matilda_policy as mp

    monkeypatch.setattr(mp.shutil, "which", lambda name: "/fake/bin/stockfish")
    policy, _ = make_policy(engine_cmd="auto")
    assert policy.engine_cmd == "/fake/bin/stockfish"

    monkeypatch.setattr(mp.shutil, "which", lambda name: None)
    import pytest

    with pytest.raises(FileNotFoundError):
        make_policy(engine_cmd="auto")


def test_engine_budget_from_tc_clock_and_options() -> None:
    policy, _ = make_policy(engine_cmd="stockfish", tc_base=300, tc_inc=2)
    board = chess.Board()
    # 300+2 is blitz: cap 15s, but only remaining/30 of the clock may be spent.
    policy.observe_go(
        {"wtime": 300_000, "btime": 300_000, "winc": 2000, "binc": 2000}, board
    )
    assert abs(policy._engine_budget() - 10.0) < 1e-9  # min(15, 300/30)
    # low on the clock: the guard shrinks the budget
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    policy.observe_go({"wtime": 30_000, "btime": 200_000, "winc": 2000}, board)
    assert abs(policy._engine_budget() - 1.0) < 1e-9  # 30/30
    # an explicit EngineMovetime takes the min with the TC cap
    policy.new_game()
    policy.set_option("EngineMovetime", "500")
    assert abs(policy._engine_budget() - 0.5) < 1e-9
    policy.set_option("EngineMovetime", "60000")
    assert abs(policy._engine_budget() - 15.0) < 1e-9  # blitz cap wins the min


def test_engine_budget_by_speed_and_gui_movetime() -> None:
    policy, _ = make_policy(engine_cmd="stockfish", tc_base=60, tc_inc=0)
    assert abs(policy._engine_budget() - 2.0) < 1e-9  # bullet
    policy2, _ = make_policy(engine_cmd="stockfish", tc_base=900, tc_inc=10)
    assert abs(policy2._engine_budget() - 30.0) < 1e-9  # rapid
    # go movetime from the GUI caps the search as well
    policy2.observe_go({"movetime": 500}, chess.Board())
    assert abs(policy2._engine_budget() - 0.5) < 1e-9
    policy2.new_game()
    assert abs(policy2._engine_budget() - 30.0) < 1e-9


def test_fresh_seed_when_unset_reproducible_when_set() -> None:
    a, _ = make_policy()
    b, _ = make_policy()
    assert a.seed != b.seed  # fresh entropy each construction
    c, _ = make_policy(seed=7)
    assert c.seed == 7


# --- engine go-parsing integration -----------------------------------------------

def test_parse_go() -> None:
    assert _parse_go("wtime 300000 btime 299000 winc 2000 binc 2000".split()) == {
        "wtime": 300_000, "btime": 299_000, "winc": 2000, "binc": 2000,
    }
    assert _parse_go(["infinite"]) == {"infinite": True}
    assert _parse_go("depth 10 movetime 500".split()) == {"depth": 10, "movetime": 500}


def test_parse_go_searchmoves_does_not_swallow_later_params() -> None:
    # UCI allows parameters after the searchmoves list; the move tokens end at
    # the first non-move token.
    parsed = _parse_go("searchmoves e2e4 d2d4 infinite".split())
    assert parsed == {"searchmoves": ["e2e4", "d2d4"], "infinite": True}
    parsed = _parse_go("searchmoves e7e8q wtime 60000 winc 1000".split())
    assert parsed == {"searchmoves": ["e7e8q"], "wtime": 60_000, "winc": 1000}
    # ...and the conventional trailing position still works
    parsed = _parse_go("infinite searchmoves e2e4".split())
    assert parsed == {"infinite": True, "searchmoves": ["e2e4"]}


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


def test_setoption_aborts_running_search() -> None:
    """setoption must serialize with the search thread: it may tear down the
    model/controller the search is using (checkpoint/engine swaps)."""
    import io
    import threading

    release = threading.Event()

    class SlowPolicy:
        def __init__(self):
            self.options_set = []

        def select(self, board):
            release.wait(timeout=5)
            legal = sorted(m.uci() for m in board.legal_moves)
            from matilda_uci.policy import PolicyResult

            return PolicyResult(best_uci=legal[0])

        def uci_options(self):
            return []

        def set_option(self, name, value):
            self.options_set.append((name, value))

        def observe_go(self, params, board):
            pass

        def new_game(self):
            pass

        def close(self):
            release.set()

    policy = SlowPolicy()
    engine = UciEngine(policy, out=io.StringIO())
    engine.handle_command("position startpos")
    engine.handle_command("go infinite")
    assert engine._search_thread is not None and engine._search_thread.is_alive()
    release.set()  # let the slow select finish so the abort join succeeds
    engine.handle_command("setoption name EngineCmd value lc0")
    # the search thread was aborted (joined) before the option was applied
    assert engine._search_thread is None
    assert policy.options_set == [("EngineCmd", "lc0")]


def test_search_failure_reports_error_info_string() -> None:
    import io

    class BoomPolicy:
        def select(self, board):
            raise RuntimeError("checkpoint not found")

        def uci_options(self):
            return []

        def set_option(self, name, value):
            pass

        def observe_go(self, params, board):
            pass

        def new_game(self):
            pass

        def close(self):
            pass

    out = io.StringIO()
    engine = UciEngine(BoomPolicy(), out=out)
    engine.handle_command("position startpos")
    engine.handle_command("go")
    engine.join_search(timeout=5)
    text = out.getvalue()
    assert "info string error: RuntimeError: checkpoint not found" in text
    assert "bestmove 0000" in text
