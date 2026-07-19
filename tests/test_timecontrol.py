"""Time-control classification and per-move engine budget math."""

from __future__ import annotations

from matilda_uci.timecontrol import MoveTimeConfig, classify_tc


def test_classify_tc_lichess_convention() -> None:
    # estimated duration = base + 40 * inc
    assert classify_tc(60, 0) == "bullet"
    assert classify_tc(120, 1) == "bullet"  # 160
    assert classify_tc(180, 0) == "blitz"  # boundary goes up
    assert classify_tc(300, 2) == "blitz"  # 380
    assert classify_tc(300, 5) == "rapid"  # 500
    assert classify_tc(900, 10) == "rapid"  # 1300
    assert classify_tc(1500, 0) == "classical"
    assert classify_tc(1800, 20) == "classical"


def test_budget_speed_caps() -> None:
    cfg = MoveTimeConfig()
    assert cfg.budget(60, 0) == 2.0
    assert cfg.budget(300, 2) == 15.0
    assert cfg.budget(900, 10) == 30.0
    assert cfg.budget(3600, 30) == 60.0


def test_budget_min_with_explicit_movetime() -> None:
    cfg = MoveTimeConfig()
    assert cfg.budget(300, 2, explicit_s=0.5) == 0.5  # explicit below the cap
    assert cfg.budget(300, 2, explicit_s=60.0) == 15.0  # cap below the explicit
    assert cfg.budget(300, 2, explicit_s=0.0) == 15.0  # 0 = unset


def test_budget_remaining_clock_guard() -> None:
    cfg = MoveTimeConfig()
    assert cfg.budget(300, 2, remaining_s=300.0) == 10.0  # 300/30 < 15
    assert cfg.budget(300, 2, remaining_s=30.0) == 1.0
    # the floor stops the budget collapsing to nothing on a dead clock
    assert cfg.budget(300, 2, remaining_s=0.3) == cfg.floor_s


def test_budget_gui_movetime_and_overrides() -> None:
    cfg = MoveTimeConfig()
    assert cfg.budget(300, 2, gui_movetime_s=0.25) == 0.25
    # everything at once: the smallest limit wins
    assert (
        cfg.budget(300, 2, explicit_s=5.0, remaining_s=60.0, gui_movetime_s=1.0)
        == 1.0
    )
    custom = MoveTimeConfig(blitz_s=4.0)
    assert custom.budget(300, 2) == 4.0
