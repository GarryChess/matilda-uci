from __future__ import annotations

import io

import chess

from matilda_uci.engine import UciEngine
from matilda_uci.policy import PolicyResult, UciOption


class FakePolicy:
    """Deterministic policy: always plays the lexicographically-first legal move."""

    def __init__(self):
        self.options_set = []
        self.new_games = 0
        self.closed = False

    def select(self, board: chess.Board) -> PolicyResult:
        legal = sorted(move.uci() for move in board.legal_moves)
        if not legal:
            return PolicyResult(best_uci=None)
        ranked = tuple((uci, 1.0 / len(legal)) for uci in legal)
        return PolicyResult(best_uci=legal[0], ranked=ranked, score_cp=12, info="fake")

    def uci_options(self):
        return [UciOption("UCI_Elo", "spin", default="1500", min=1100, max=2000)]

    def set_option(self, name, value):
        self.options_set.append((name, value))

    def new_game(self):
        self.new_games += 1

    def close(self):
        self.closed = True


def make_engine():
    out = io.StringIO()
    return UciEngine(FakePolicy(), out=out), out


def lines(out: io.StringIO) -> list[str]:
    return out.getvalue().splitlines()


def test_uci_handshake_advertises_id_and_options():
    engine, out = make_engine()
    engine.handle_command("uci")
    produced = lines(out)
    assert "id name Matilda" in produced
    assert any(line.startswith("id author") for line in produced)
    assert any(line.startswith("option name UCI_Elo type spin") for line in produced)
    assert produced[-1] == "uciok"


def test_isready_returns_readyok():
    engine, out = make_engine()
    engine.handle_command("isready")
    assert lines(out) == ["readyok"]


def test_position_startpos_with_moves_updates_board():
    engine, _ = make_engine()
    engine.handle_command("position startpos moves e2e4 e7e5")
    expected = chess.Board()
    expected.push_uci("e2e4")
    expected.push_uci("e7e5")
    assert engine.board.fen() == expected.fen()


def test_position_fen_with_moves():
    engine, _ = make_engine()
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    engine.handle_command(f"position fen {fen} moves e7e5")
    board = chess.Board(fen)
    board.push_uci("e7e5")
    assert engine.board.fen() == board.fen()


def test_ucinewgame_resets_board_and_notifies_policy():
    engine, _ = make_engine()
    engine.handle_command("position startpos moves e2e4")
    engine.handle_command("ucinewgame")
    assert engine.board.fen() == chess.Board().fen()
    assert engine.policy.new_games == 1


def test_setoption_is_forwarded_to_policy():
    engine, _ = make_engine()
    engine.handle_command("setoption name UCI_Elo value 1800")
    assert ("UCI_Elo", "1800") in engine.policy.options_set


def test_go_emits_single_legal_bestmove_with_info():
    engine, out = make_engine()
    engine.handle_command("position startpos")
    engine.handle_command("go depth 1")
    engine.join_search(timeout=5.0)
    produced = lines(out)
    best = [line for line in produced if line.startswith("bestmove ")]
    assert len(best) == 1
    move = best[0].split()[1]
    assert chess.Move.from_uci(move) in chess.Board().legal_moves
    assert any(line.startswith("info depth 1") and "score cp 12" in line for line in produced)


def test_go_with_no_legal_moves_emits_null_move():
    engine, out = make_engine()
    engine.handle_command(
        "position fen rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    )
    engine.handle_command("go depth 1")
    engine.join_search(timeout=5.0)
    assert "bestmove 0000" in lines(out)


def test_go_infinite_holds_bestmove_until_stop():
    engine, out = make_engine()
    engine.handle_command("position startpos")
    engine.handle_command("go infinite")
    # The held search must not have produced a bestmove yet...
    engine.handle_command("stop")  # ...stop releases exactly one.
    produced = lines(out)
    assert sum(line.startswith("bestmove ") for line in produced) == 1


def test_quit_stops_loop_and_closes_policy():
    engine, _ = make_engine()
    script = io.StringIO("uci\nposition startpos\ngo depth 1\nquit\n")
    engine.run(in_stream=script)
    assert engine.policy.closed is True


def test_full_run_script_produces_handshake_and_move():
    engine, out = make_engine()
    script = io.StringIO("uci\nisready\nposition startpos\ngo depth 1\nquit\n")
    engine.run(in_stream=script)
    produced = lines(out)
    assert "uciok" in produced
    assert "readyok" in produced
    assert any(line.startswith("bestmove ") for line in produced)
