from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import chess

logger = logging.getLogger(__name__)

# Logistic constant from the lichess win% model:
#   win_prob = 1 / (1 + exp(-K * cp))   ->   cp = ln(p / (1 - p)) / K
_WINPROB_K = 0.00368208
_CP_CLAMP = 10_000


def winprob_to_cp(win_prob: float) -> int:
    """Convert a side-to-move win probability to a UCI ``score cp`` value.

    Uses the inverse of the lichess logistic win% model so GUIs can render a
    sensible evaluation bar. Purely cosmetic: Maia's strength is set by Elo, not
    by this number.
    """
    p = min(max(float(win_prob), 1e-4), 1.0 - 1e-4)
    cp = math.log(p / (1.0 - p)) / _WINPROB_K
    return int(round(max(-_CP_CLAMP, min(_CP_CLAMP, cp))))


def cp_to_winprob(cp: float) -> float:
    """Inverse of :func:`winprob_to_cp`: centipawns -> side-to-move win probability.

    The forward direction of the lichess logistic win% model. Used by the
    Elo-conditioned eval bar to mix Stockfish scores in bounded expected-score
    space instead of raw centipawns.
    """
    return 1.0 / (1.0 + math.exp(-_WINPROB_K * float(cp)))


@dataclass(frozen=True)
class UciOption:
    """A single ``option name ...`` line advertised during the UCI handshake."""

    name: str
    type: str  # "check" | "spin" | "combo" | "string" | "button"
    default: str | None = None
    min: int | None = None
    max: int | None = None
    var: tuple[str, ...] = ()

    def declaration(self) -> str:
        parts = [f"option name {self.name} type {self.type}"]
        if self.default is not None:
            parts.append(f"default {self.default}")
        if self.min is not None:
            parts.append(f"min {self.min}")
        if self.max is not None:
            parts.append(f"max {self.max}")
        for value in self.var:
            parts.append(f"var {value}")
        return " ".join(parts)


@dataclass(frozen=True)
class PolicyResult:
    """One move decision for a board.

    ``best_uci`` is ``None`` only when the position has no legal move (mate or
    stalemate); the engine emits ``bestmove 0000`` in that case.
    """

    best_uci: str | None
    ranked: tuple[tuple[str, float], ...] = ()  # (uci, prob) desc, legal-only
    score_cp: int | None = None
    info: str | None = None  # optional detail surfaced as a UCI 'info string'


@runtime_checkable
class MovePolicy(Protocol):
    """Anything that can pick a move for a board and be driven over UCI.

    Keeping this minimal is the whole point: a future RL-trained or
    embedding-conditioned style policy implements the same surface and slots
    into :class:`~garrychessscience.uci.engine.UciEngine` unchanged.
    """

    def select(self, board: chess.Board) -> PolicyResult: ...

    def uci_options(self) -> list[UciOption]: ...

    def set_option(self, name: str, value: str) -> None: ...

    def observe_go(self, params: dict, board: chess.Board) -> None:
        """Called with the parsed ``go`` parameters before each search starts.

        Policies that condition on the clock (e.g. Matilda's time-control
        latch) read ``wtime``/``winc`` here; others do nothing.
        """
        ...

    def new_game(self) -> None: ...

    def close(self) -> None: ...


class MaiaPolicy:
    """Style policy backed by real Maia-2 human-move prediction.

    Maia is rating-conditioned, so ``UCI_Elo`` maps directly onto playing
    strength: the engine plays like a human of that rating. The heavy ``maia2``
    dependency is only touched when a real move is requested, so importing this
    module (and the test-suite) stays cheap. A ``wrapper`` or ``wrapper_factory``
    can be injected to play without the real model (used by the tests).
    """

    def __init__(
        self,
        *,
        wrapper: object | None = None,
        wrapper_factory: Callable[[], object] | None = None,
        elo_self: int = 1500,
        elo_oppo: int = 1500,
        maia_type: str = "rapid",
        device: str = "cpu",
        temperature: float = 0.0,
        limit_strength: bool = True,
        elo_min: int = 1100,
        elo_max: int = 2000,
        seed: int | None = None,
    ) -> None:
        self.elo_self = int(elo_self)
        self.elo_oppo = int(elo_oppo)
        self.maia_type = maia_type
        self.device = device
        self.temperature = float(temperature)
        self.limit_strength = bool(limit_strength)
        self.elo_min = int(elo_min)
        self.elo_max = int(elo_max)
        self._wrapper = wrapper
        self._wrapper_factory = wrapper_factory
        if seed is None:
            seed = random.SystemRandom().randrange(2**32)
            logger.info("sampling seed (fresh): %d — pass seed= to reproduce", seed)
        self.seed = int(seed)
        self._rng = random.Random(self.seed)

    # --- MovePolicy surface ----------------------------------------------------
    def select(self, board: chess.Board) -> PolicyResult:
        legal = [move.uci() for move in board.legal_moves]
        if not legal:
            return PolicyResult(best_uci=None)

        wrapper = self._ensure_wrapper()
        elo_self = self._effective_elo()
        result = wrapper.infer(board, elo_self=elo_self, elo_oppo=self.elo_oppo)

        probs = {uci: max(float(result.move_probs.get(uci, 0.0)), 0.0) for uci in legal}
        total = sum(probs.values())
        if total <= 0.0:  # Maia gave no mass to any legal move; fall back to uniform.
            probs = {uci: 1.0 / len(legal) for uci in legal}
            total = 1.0
        probs = {uci: value / total for uci, value in probs.items()}
        ranked = tuple(sorted(probs.items(), key=lambda kv: kv[1], reverse=True))

        best = self._choose(ranked)
        info = (
            f"maia type={self.maia_type} elo_self={elo_self} elo_oppo={self.elo_oppo} "
            f"p_best={ranked[0][1]:.3f} win={result.win_prob:.3f}"
        )
        return PolicyResult(
            best_uci=best,
            ranked=ranked,
            score_cp=winprob_to_cp(result.win_prob),
            info=info,
        )

    def uci_options(self) -> list[UciOption]:
        return [
            UciOption("UCI_LimitStrength", "check", default="true"),
            UciOption(
                "UCI_Elo", "spin", default=str(self.elo_self),
                min=self.elo_min, max=self.elo_max,
            ),
            UciOption(
                "OpponentElo", "spin", default=str(self.elo_oppo),
                min=self.elo_min, max=self.elo_max,
            ),
            UciOption("MaiaType", "combo", default=self.maia_type, var=("rapid", "blitz")),
            UciOption("Device", "combo", default=self.device, var=("cpu", "gpu")),
            UciOption("Temperature", "string", default=f"{self.temperature:.2f}"),
        ]

    def set_option(self, name: str, value: str) -> None:
        key = name.strip().lower()
        value = value.strip()
        if key == "uci_elo":
            self.elo_self = _safe_int(value, self.elo_self)
        elif key == "opponentelo":
            self.elo_oppo = _safe_int(value, self.elo_oppo)
        elif key == "uci_limitstrength":
            self.limit_strength = value.lower() in ("true", "1", "yes", "on")
        elif key == "temperature":
            self.temperature = _safe_float(value, self.temperature)
        elif key == "maiatype" and value in ("rapid", "blitz") and value != self.maia_type:
            self.maia_type = value
            self._reset_wrapper()
        elif key == "device" and value in ("cpu", "gpu") and value != self.device:
            self.device = value
            self._reset_wrapper()
        else:
            logger.debug("MaiaPolicy: ignoring option %s=%s", name, value)

    def observe_go(self, params: dict, board: chess.Board) -> None:
        # Maia-2 does not condition on the clock; nothing to observe.
        pass

    def new_game(self) -> None:
        # Maia decides each position independently; nothing to reset between games.
        pass

    def close(self) -> None:
        self._reset_wrapper()

    # --- internals -------------------------------------------------------------
    def _ensure_wrapper(self) -> object:
        if self._wrapper is None:
            factory = self._wrapper_factory or self._build_wrapper
            self._wrapper = factory()
        return self._wrapper

    def _reset_wrapper(self) -> None:
        if self._wrapper is not None:
            close = getattr(self._wrapper, "close", None)
            if callable(close):
                close()
            self._wrapper = None

    def _build_wrapper(self) -> object:
        from .maia_wrapper import MaiaWrapper

        return MaiaWrapper(
            maia_type=self.maia_type,
            device=self.device,
            top_k=5,
            capture_embedding=False,
        )

    def _effective_elo(self) -> int:
        return effective_elo(self.elo_self, self.elo_min, self.elo_max, self.limit_strength)

    def _choose(self, ranked: tuple[tuple[str, float], ...]) -> str:
        return choose_with_temperature(ranked, self.temperature, self._rng)


def effective_elo(elo: int, elo_min: int, elo_max: int, limit_strength: bool) -> int:
    """The Elo actually fed to the model: clamped, or ``elo_max`` when unlimited."""
    if not limit_strength:
        return elo_max
    return max(elo_min, min(elo_max, elo))


def choose_with_temperature(
    ranked: tuple[tuple[str, float], ...], temperature: float, rng: random.Random
) -> str:
    """Pick from a ranked ``(uci, prob)`` tuple; 0 = argmax, >0 = temperature sample."""
    if temperature <= 0.0 or len(ranked) == 1:
        return ranked[0][0]
    inv_t = 1.0 / temperature
    weights = [max(prob, 1e-12) ** inv_t for _, prob in ranked]
    total = sum(weights)
    draw = rng.random() * total
    upto = 0.0
    for (uci, _prob), weight in zip(ranked, weights):
        upto += weight
        if draw <= upto:
            return uci
    return ranked[0][0]


def _safe_int(value: str, fallback: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return fallback


def _safe_float(value: str, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback
