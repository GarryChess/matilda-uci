"""MatildaPolicy: the Matilda re-ranker as a UCI :class:`MovePolicy`.

Drives :class:`matilda_uci.matilda.MatildaModel` (frozen Maia-3 + the trained
set-transformer from ``base_3k.pt``) over the UCI protocol. Everything heavy
(torch, Maia-3) is imported lazily on the first real move request, so this
module — and the engine handshake — stays instant and torch-free; tests inject
a fake model.

UCI-protocol impedance matching:

* **Move history**: Maia-3 conditions on the trailing 8 positions. UCI gives
  them for free — ``position startpos moves ...`` builds a board whose
  ``move_stack`` we rewind. A bare-FEN analysis position has no history, which
  matches early-game training rows (graceful, in-distribution).
* **Time control**: the model takes (base, inc) directly, but UCI never
  transmits base explicitly. At the first ``go`` of a game the side-to-move
  clock ~= base and the increment is exact, so we latch them then
  (``AutoLatchTC``); ``TimeControlBase``/``TimeControlInc`` options are the
  fallback, defaulting to 180+0 blitz (Maia-3 is blitz-trained). The same TC
  also sets the search controller's per-move time budget (see
  :mod:`matilda_uci.timecontrol`), bounded further by the remaining clock and
  any explicit ``EngineMovetime`` / ``go movetime``.
* **Elos**: both players' Elos are inputs UCI does not provide; standard
  ``UCI_Elo`` plus an ``OpponentElo`` option cover them.
"""

from __future__ import annotations

import logging
import random
import shutil
from typing import Callable

import chess

from .policy import (
    PolicyResult,
    UciOption,
    _safe_float,
    _safe_int,
    choose_with_temperature,
    effective_elo,
    winprob_to_cp,
)
from .timecontrol import MoveTimeConfig

logger = logging.getLogger(__name__)

_HISTORY_PLIES = 8

STOCKFISH_INSTALL_HINT = (
    "stockfish not found on PATH — install it "
    "(https://stockfishchess.org/download/; macOS: brew install stockfish; "
    "Debian/Ubuntu: apt install stockfish), or pass an explicit engine "
    "command, or opt out of engine assistance explicitly."
)


def resolve_engine_cmd(engine_cmd: str | None) -> str:
    """Turn the ``engine_cmd`` setting into a runnable command.

    ``"auto"`` (the default) resolves stockfish from PATH and raises
    :class:`FileNotFoundError` when it is missing — high-Elo play without a
    search engine is silently much weaker, so absence must be loud. ``None``
    or ``""`` means the caller explicitly wants no engine.
    """
    if engine_cmd is None:
        return ""
    cmd = engine_cmd.strip()
    if cmd != "auto":
        return cmd
    found = shutil.which("stockfish")
    if found is None:
        raise FileNotFoundError(STOCKFISH_INSTALL_HINT)
    return found


def _resolve_if_released(spec: str) -> str:
    """Resolve released checkpoint names (download on first use); anything
    else — an existing path, or a caller-supplied identifier for an injected
    model — passes through untouched."""
    from .assets import resolve_if_released

    return resolve_if_released(spec)


def board_history_fens(board: chess.Board, plies: int = _HISTORY_PLIES) -> list[str]:
    """The FENs preceding ``board`` (oldest first), rewound from its move stack."""
    b = board.copy()
    fens: list[str] = []
    while b.move_stack and len(fens) < plies:
        b.pop()
        fens.append(b.fen())
    fens.reverse()
    return fens


class MatildaPolicy:
    """Style policy backed by Matilda (Maia-3 + trained re-ranker).

    ``model`` / ``model_factory`` inject a stand-in for tests; the real
    :class:`~matilda_uci.matilda.MatildaModel` is built lazily otherwise.
    """

    def __init__(
        self,
        *,
        model: object | None = None,
        model_factory: Callable[[], object] | None = None,
        checkpoint: str = "checkpoints/base_3k.pt",
        device: str = "cpu",
        maia3_model: str = "23m",
        elo_self: int = 1500,
        elo_oppo: int = 1500,
        tc_base: float = 180.0,
        tc_inc: float = 0.0,
        auto_latch_tc: bool = True,
        temperature: float = 0.0,
        limit_strength: bool = True,
        elo_min: int = 1000,
        elo_max: int = 3200,
        style_checkpoint: str = "",
        style_vector: str = "",
        engine_cmd: str | None = "auto",
        engine_depth: int = 22,
        engine_nodes: int = 0,
        engine_movetime: float = 0.0,
        movetime_config: MoveTimeConfig | None = None,
        threads: int = 0,
        cache_size: int = 4096,
        seed: int | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.maia3_model = maia3_model
        self.elo_self = int(elo_self)
        self.elo_oppo = int(elo_oppo)
        self.cfg_tc_base = float(tc_base)
        self.cfg_tc_inc = float(tc_inc)
        self.tc_base = float(tc_base)
        self.tc_inc = float(tc_inc)
        self.auto_latch_tc = bool(auto_latch_tc)
        self._tc_latched = False
        self.temperature = float(temperature)
        self.limit_strength = bool(limit_strength)
        self.elo_min = int(elo_min)
        self.elo_max = int(elo_max)
        self.style_checkpoint = style_checkpoint
        self.style_vector = style_vector
        self._style_pid: int | None = None
        self.engine_cmd = resolve_engine_cmd(engine_cmd)
        self.engine_depth = int(engine_depth)
        self.engine_nodes = int(engine_nodes)
        self.engine_movetime = float(engine_movetime)
        self.movetime_config = movetime_config or MoveTimeConfig()
        self._clock_remaining_s: float | None = None
        self._gui_movetime_s: float | None = None
        self.threads = int(threads)
        self.cache_size = int(cache_size)
        self._model = model
        self._model_factory = model_factory
        self._style_applied = False
        self._controller: object | None = None
        if seed is None:
            seed = random.SystemRandom().randrange(2**32)
            logger.info("sampling seed (fresh): %d — pass seed= to reproduce", seed)
        self.seed = int(seed)
        self._rng = random.Random(self.seed)

    # --- MovePolicy surface ------------------------------------------------------
    def select(self, board: chess.Board) -> PolicyResult:
        legal = [move.uci() for move in board.legal_moves]
        if not legal:
            return PolicyResult(best_uci=None)

        model = self._ensure_model()
        pred = model.predict(
            board,
            board_history=board_history_fens(board),
            elo_self=self._effective_elo(),
            elo_oppo=self.elo_oppo,
            tc_base=self.tc_base,
            tc_inc=self.tc_inc,
            controller=self._ensure_controller(),
            pid=self._style_pid,
        )
        if pred is None:
            return PolicyResult(best_uci=None)

        probs = {u: max(float(pred.move_probs.get(u, 0.0)), 0.0) for u in legal}
        total = sum(probs.values())
        if total <= 0.0:
            probs = {u: 1.0 / len(legal) for u in legal}
            total = 1.0
        probs = {u: p / total for u, p in probs.items()}
        ranked = tuple(sorted(probs.items(), key=lambda kv: kv[1], reverse=True))

        best = self._choose(ranked)
        info = (
            f"matilda elo={self._effective_elo()} opp={self.elo_oppo} "
            f"tc={self.tc_base:.0f}+{self.tc_inc:.0f}"
            f"{' latched' if self._tc_latched else ''} "
            f"engine={'on' if pred.engine_used else 'off'} "
            f"p_best={ranked[0][1]:.3f} win={pred.win_prob:.3f}"
        )
        return PolicyResult(
            best_uci=best,
            ranked=ranked,
            score_cp=winprob_to_cp(pred.win_prob),
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
            UciOption(
                "TimeControlBase", "spin", default=str(int(self.cfg_tc_base)),
                min=0, max=10800,
            ),
            UciOption(
                "TimeControlInc", "spin", default=str(int(self.cfg_tc_inc)),
                min=0, max=180,
            ),
            UciOption("AutoLatchTC", "check",
                      default="true" if self.auto_latch_tc else "false"),
            UciOption("Temperature", "string", default=f"{self.temperature:.2f}"),
            UciOption("Checkpoint", "string", default=self.checkpoint),
            UciOption("StyleCheckpoint", "string", default=self.style_checkpoint or "<empty>"),
            UciOption("StyleVector", "string", default=self.style_vector or "<empty>"),
            UciOption("EngineCmd", "string", default=self.engine_cmd or "<empty>"),
            UciOption("EngineDepth", "spin", default=str(self.engine_depth), min=1, max=40),
            UciOption("EngineNodes", "spin", default=str(self.engine_nodes),
                      min=0, max=10_000_000),
            UciOption("EngineMovetime", "spin",
                      default=str(int(self.engine_movetime * 1000)),
                      min=0, max=600_000),  # ms; 0 = the TC-derived budget alone
            UciOption("Device", "combo", default=self.device, var=("cpu", "mps", "cuda")),
            UciOption("Threads", "spin", default=str(self.threads), min=0, max=64),
            UciOption("CacheSize", "spin", default=str(self.cache_size),
                      min=0, max=1_000_000),  # cached predictions; 0 = off
        ]

    def set_option(self, name: str, value: str) -> None:
        key = name.strip().lower()
        value = value.strip()
        if value == "<empty>":  # some GUIs echo our empty-string placeholder back
            value = ""
        if key == "uci_elo":
            self.elo_self = _safe_int(value, self.elo_self)
        elif key == "opponentelo":
            self.elo_oppo = _safe_int(value, self.elo_oppo)
        elif key == "uci_limitstrength":
            self.limit_strength = value.lower() in ("true", "1", "yes", "on")
        elif key == "temperature":
            self.temperature = _safe_float(value, self.temperature)
        elif key == "timecontrolbase":
            self.cfg_tc_base = _safe_float(value, self.cfg_tc_base)
            if not self._tc_latched:
                self.tc_base = self.cfg_tc_base
        elif key == "timecontrolinc":
            self.cfg_tc_inc = _safe_float(value, self.cfg_tc_inc)
            if not self._tc_latched:
                self.tc_inc = self.cfg_tc_inc
        elif key == "autolatchtc":
            self.auto_latch_tc = value.lower() in ("true", "1", "yes", "on")
        elif key == "checkpoint" and value and value != self.checkpoint:
            self.checkpoint = value
            self._reset_model()
        elif key == "stylecheckpoint" and value != self.style_checkpoint:
            self.style_checkpoint = value
            self._reset_model()
        elif key == "stylevector" and value != self.style_vector:
            self.style_vector = value
            self._reset_model()
        elif key == "enginecmd":
            try:
                resolved = resolve_engine_cmd(value)
            except FileNotFoundError as exc:
                logger.warning("EngineCmd %r ignored: %s", value, exc)
            else:
                if resolved != self.engine_cmd:
                    self.engine_cmd = resolved
                    self._reset_controller()
        elif key == "enginedepth":
            self.engine_depth = _safe_int(value, self.engine_depth)
            self._update_controller_limits()
        elif key == "enginenodes":
            self.engine_nodes = _safe_int(value, self.engine_nodes)
            self._update_controller_limits()
        elif key == "enginemovetime":  # milliseconds over UCI
            self.engine_movetime = _safe_int(value, int(self.engine_movetime * 1000)) / 1000.0
            self._update_controller_limits()
        elif key == "device" and value in ("cpu", "mps", "cuda") and value != self.device:
            self.device = value
            self._reset_model()
        elif key == "threads":
            self.threads = _safe_int(value, self.threads)
            if self._model is not None:
                set_threads = getattr(self._model, "set_threads", None)
                if callable(set_threads):
                    set_threads(self.threads)
        elif key == "cachesize":
            self.cache_size = _safe_int(value, self.cache_size)
            if self._model is not None:
                set_cache = getattr(self._model, "set_cache_size", None)
                if callable(set_cache):
                    set_cache(self.cache_size)
        else:
            logger.debug("MatildaPolicy: ignoring option %s=%s", name, value)

    def observe_go(self, params: dict, board: chess.Board) -> None:
        """Read the clock state from each ``go``: latch the TC, track the rest.

        At move one the side-to-move clock ~= base and the increment is exact;
        later in the game the clock has drifted, so only the increment is
        trusted then. The remaining clock and any ``go movetime`` feed the
        search controller's per-move budget on every call.
        """
        our_time = params.get("wtime" if board.turn == chess.WHITE else "btime")
        our_inc = params.get("winc" if board.turn == chess.WHITE else "binc")
        self._clock_remaining_s = (
            float(our_time) / 1000.0 if our_time is not None else None
        )
        movetime = params.get("movetime")
        self._gui_movetime_s = (
            float(movetime) / 1000.0 if movetime is not None else None
        )
        if not self.auto_latch_tc or self._tc_latched or our_time is None:
            return
        if board.ply() <= 1:
            self.tc_base = float(our_time) / 1000.0
            self.tc_inc = float(our_inc or 0) / 1000.0
            self._tc_latched = True
            logger.info("latched TC from clock: %.0f+%.0f", self.tc_base, self.tc_inc)
        elif our_inc is not None:
            self.tc_inc = float(our_inc) / 1000.0

    def new_game(self) -> None:
        self._tc_latched = False
        self.tc_base = self.cfg_tc_base
        self.tc_inc = self.cfg_tc_inc
        self._clock_remaining_s = None
        self._gui_movetime_s = None

    def close(self) -> None:
        self._reset_controller()
        self._reset_model()

    # --- internals -----------------------------------------------------------------
    def _ensure_model(self) -> object:
        if self._model is None:
            factory = self._model_factory or self._build_model
            self._model = factory()
            self._style_applied = False
            self._style_pid = None
        if self.style_checkpoint and self.style_vector and not self._style_applied:
            load_vec = getattr(self._model, "load_style_vector", None)
            if callable(load_vec):
                self._style_pid = load_vec(
                    _resolve_if_released(self.style_checkpoint), self.style_vector
                )
                logger.info(
                    "style vector %s loaded (transformation: %s)",
                    self.style_vector, self.style_checkpoint,
                )
            self._style_applied = True
        elif self.style_checkpoint and not self.style_vector and not self._style_applied:
            logger.warning(
                "StyleCheckpoint set without StyleVector; playing style-free "
                "(supply a 32-d embedding file — see demos/fit_style_vector.py)"
            )
            self._style_applied = True
        return self._model

    def _build_model(self) -> object:
        from .matilda import MatildaModel

        return MatildaModel(
            _resolve_if_released(self.checkpoint), device=self.device,
            maia3_model=self.maia3_model,
            threads=self.threads, cache_size=self.cache_size,
        )

    def _reset_model(self) -> None:
        if self._model is not None:
            close = getattr(self._model, "close", None)
            if callable(close):
                close()
            self._model = None
        self._style_applied = False

    def _ensure_controller(self) -> object | None:
        if not self.engine_cmd:
            return None
        if self._controller is None:
            from .matilda.search import UciSearchController

            self._controller = UciSearchController(
                self.engine_cmd, depth=self.engine_depth, nodes=self.engine_nodes,
                timeout_s=self._engine_budget(),
            )
        else:
            self._update_controller_limits()  # keep the per-move budget current
        return self._controller

    def _reset_controller(self) -> None:
        if self._controller is not None:
            close = getattr(self._controller, "close", None)
            if callable(close):
                close()
            self._controller = None

    def _engine_budget(self) -> float:
        """Seconds the search controller may spend on the next move: the TC's
        speed cap, bounded by the remaining clock, any explicit
        ``EngineMovetime``, and any GUI ``go movetime``."""
        return self.movetime_config.budget(
            self.tc_base, self.tc_inc,
            explicit_s=self.engine_movetime,
            remaining_s=self._clock_remaining_s,
            gui_movetime_s=self._gui_movetime_s,
        )

    def _update_controller_limits(self) -> None:
        """Apply depth/nodes/movetime to a live controller in place — they only
        feed the per-call search limit, so no reason to respawn the engine."""
        if self._controller is not None:
            self._controller.depth = self.engine_depth
            self._controller.nodes = self.engine_nodes
            self._controller.timeout_s = self._engine_budget()

    def _effective_elo(self) -> int:
        return effective_elo(self.elo_self, self.elo_min, self.elo_max, self.limit_strength)

    def _choose(self, ranked: tuple[tuple[str, float], ...]) -> str:
        return choose_with_temperature(ranked, self.temperature, self._rng)
