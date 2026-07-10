from __future__ import annotations

import chess
import pytest

from matilda_uci.policy import MaiaPolicy, winprob_to_cp


class FakeMaiaResult:
    def __init__(self, move_probs, win_prob):
        self.move_probs = move_probs
        self.win_prob = win_prob
        self.top_moves = ()
        self.embedding = None


class FakeWrapper:
    """Stand-in for MaiaWrapper so policy tests never import the maia2 runtime."""

    def __init__(self, move_probs, win_prob=0.6):
        self._move_probs = move_probs
        self._win_prob = win_prob
        self.calls = []
        self.closed = False

    def infer(self, board, *, elo_self, elo_oppo):
        self.calls.append((board.fen(), elo_self, elo_oppo))
        return FakeMaiaResult(self._move_probs, self._win_prob)

    def close(self):
        self.closed = True


def test_selects_highest_prob_legal_move_ignoring_illegal_mass():
    # 'a2a3' is illegal mass that must be ignored; 'e2e4' is the best legal move.
    wrapper = FakeWrapper({"zzz9z9": 0.9, "e2e4": 0.5, "d2d4": 0.2})
    policy = MaiaPolicy(wrapper=wrapper)
    result = policy.select(chess.Board())
    assert result.best_uci == "e2e4"
    # Ranked probabilities are renormalized over legal moves only.
    assert pytest.approx(sum(p for _, p in result.ranked), abs=1e-9) == 1.0
    assert "zzz9z9" not in dict(result.ranked)


def test_uniform_fallback_when_no_legal_mass():
    wrapper = FakeWrapper({"e7e5": 0.9})  # none of these are legal at startpos
    policy = MaiaPolicy(wrapper=wrapper)
    result = policy.select(chess.Board())
    probs = [p for _, p in result.ranked]
    assert result.best_uci is not None
    assert pytest.approx(min(probs)) == max(probs)  # uniform


def test_no_legal_moves_returns_none_without_calling_maia():
    wrapper = FakeWrapper({"e2e4": 1.0})
    policy = MaiaPolicy(wrapper=wrapper)
    mate = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert mate.is_checkmate()
    result = policy.select(mate)
    assert result.best_uci is None
    assert wrapper.calls == []


def test_limit_strength_clamps_elo_passed_to_maia():
    wrapper = FakeWrapper({"e2e4": 1.0})
    policy = MaiaPolicy(wrapper=wrapper, elo_self=5000, elo_max=2000)
    policy.select(chess.Board())
    assert wrapper.calls[0][1] == 2000  # clamped self elo


def test_limit_strength_false_uses_max_elo():
    wrapper = FakeWrapper({"e2e4": 1.0})
    policy = MaiaPolicy(wrapper=wrapper, elo_self=1300, elo_max=2000, limit_strength=False)
    policy.select(chess.Board())
    assert wrapper.calls[0][1] == 2000


def test_set_option_updates_elos_and_opponent():
    wrapper = FakeWrapper({"e2e4": 1.0})
    policy = MaiaPolicy(wrapper=wrapper)
    policy.set_option("UCI_Elo", "1800")
    policy.set_option("OpponentElo", "1700")
    policy.select(chess.Board())
    _, elo_self, elo_oppo = wrapper.calls[0]
    assert (elo_self, elo_oppo) == (1800, 1700)


def test_changing_maia_type_rebuilds_wrapper():
    built = [FakeWrapper({"e2e4": 1.0}), FakeWrapper({"d2d4": 1.0})]
    order = iter(built)
    policy = MaiaPolicy(wrapper_factory=lambda: next(order))

    first = policy.select(chess.Board())
    assert first.best_uci == "e2e4"

    policy.set_option("MaiaType", "blitz")
    assert built[0].closed is True  # old wrapper released

    second = policy.select(chess.Board())
    assert second.best_uci == "d2d4"  # rebuilt from the factory


def test_temperature_zero_is_deterministic_argmax():
    wrapper = FakeWrapper({"e2e4": 0.4, "d2d4": 0.3, "g1f3": 0.3})
    policy = MaiaPolicy(wrapper=wrapper, temperature=0.0)
    picks = {policy.select(chess.Board()).best_uci for _ in range(5)}
    assert picks == {"e2e4"}


def test_winprob_to_cp_is_signed_and_monotonic():
    assert winprob_to_cp(0.5) == 0
    assert winprob_to_cp(0.75) > 0
    assert winprob_to_cp(0.25) < 0
    assert winprob_to_cp(0.9) > winprob_to_cp(0.6)
    # Extreme inputs stay finite (clamped).
    assert abs(winprob_to_cp(1.0)) <= 10_000
    assert abs(winprob_to_cp(0.0)) <= 10_000
