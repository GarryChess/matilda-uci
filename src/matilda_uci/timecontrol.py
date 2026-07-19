"""Time-control classification and per-move engine search budgets.

The game's time control decides how long the search controller (Stockfish/Lc0)
may think per move: a bullet game cannot afford a 15-second engine call, while
a classical game can. Classification follows the lichess convention — estimated
game duration = base + 40 x increment — and each speed gets a per-move cap.

The final budget is the minimum of every limit that applies:

* the speed cap for the game's time control,
* an explicit ``EngineMovetime`` / ``--engine-movetime`` setting,
* the remaining clock (never more than ``remaining x clock_fraction``),
* a GUI-imposed ``go movetime``.
"""

from __future__ import annotations

from dataclasses import dataclass

# Lichess speed thresholds on estimated duration (base + 40*inc), in seconds.
_BULLET_MAX = 180.0
_BLITZ_MAX = 480.0
_RAPID_MAX = 1500.0


def classify_tc(base_s: float, inc_s: float) -> str:
    """Name the speed of a ``base+inc`` time control: bullet/blitz/rapid/classical."""
    estimated = float(base_s) + 40.0 * float(inc_s)
    if estimated < _BULLET_MAX:
        return "bullet"
    if estimated < _BLITZ_MAX:
        return "blitz"
    if estimated < _RAPID_MAX:
        return "rapid"
    return "classical"


@dataclass
class MoveTimeConfig:
    """Per-move engine time caps by game speed, plus the in-game clock guard.

    ``clock_fraction`` bounds any single move by a share of the remaining
    clock, so a long cap can never flag us in a winning position.
    ``floor_s`` keeps the budget from collapsing to a value the engine
    cannot do anything useful with.
    """

    bullet_s: float = 2.0
    blitz_s: float = 15.0
    rapid_s: float = 30.0
    classical_s: float = 60.0
    clock_fraction: float = 1.0 / 30.0
    floor_s: float = 0.05

    def cap_for_tc(self, base_s: float, inc_s: float) -> float:
        return {
            "bullet": self.bullet_s,
            "blitz": self.blitz_s,
            "rapid": self.rapid_s,
            "classical": self.classical_s,
        }[classify_tc(base_s, inc_s)]

    def budget(
        self,
        base_s: float,
        inc_s: float,
        *,
        explicit_s: float = 0.0,
        remaining_s: float | None = None,
        gui_movetime_s: float | None = None,
    ) -> float:
        """Seconds the search controller may spend on this move."""
        cap = self.cap_for_tc(base_s, inc_s)
        if explicit_s and explicit_s > 0.0:
            cap = min(cap, explicit_s)
        if remaining_s is not None and remaining_s > 0.0:
            cap = min(cap, remaining_s * self.clock_fraction)
        if gui_movetime_s is not None and gui_movetime_s > 0.0:
            cap = min(cap, gui_movetime_s)
        return max(cap, self.floor_s)
