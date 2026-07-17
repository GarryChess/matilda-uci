"""MatildaPolicy: the Matilda re-ranker as a UCI :class:`MovePolicy`.

Drives :class:`matilda_uci.matilda.MatildaModel` (frozen Maia-3 + the trained
set-transformer from ``base_3k.pt``) over the UCI protocol. Everything heavy
(torch, Maia-3) is imported lazily on the first real move request, so this
module — and the engine handshake — stays instant and torch-free; tests inject
a fake model.

UCI-protocol impedance matching (see currentTask notes):

* **Move history**: Maia-3 conditions on the trailing 8 positions. UCI gives
  them for free — ``position startpos moves ...`` builds a board whose
  ``move_stack`` we rewind. A bare-FEN analysis position has no history, which
  matches early-game training rows (graceful, in-distribution).
* **Time control**: the model takes (base, inc) directly, but UCI never
  transmits base explicitly. At the first ``go`` of a game the side-to-move
  clock ~= base and the increment is exact, so we latch them then
  (``AutoLatchTC``); ``TimeControlBase``/``TimeControlInc`` options are the
  fallback, defaulting to 180+0 blitz (Maia-3 is blitz-trained).
* **Elos**: both players' Elos are inputs UCI does not provide; standard
  ``UCI_Elo`` plus an ``OpponentElo`` option cover them.
"""

from __future__ import annotations

import logging
import random
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

logger = logging.getLogger(__name__)

_HISTORY_PLIES = 8


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
        style_posthoc: str = "",
        style_player_id: int = -1,
        engine_cmd: str = "",
        engine_depth: int = 12,
        engine_nodes: int = 0,
        engine_movetime: float = 0.0,
        threads: int = 0,
        cache_size: int = 4096,
        seed: int = 0,
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
        self.style_posthoc = style_posthoc
        self.style_player_id = int(style_player_id)
        self.engine_cmd = engine_cmd
        self.engine_depth = int(engine_depth)
        self.engine_nodes = int(engine_nodes)
        self.engine_movetime = float(engine_movetime)
        self.threads = int(threads)
        self.cache_size = int(cache_size)
        self._model = model
        self._model_factory = model_factory
        self._style_applied = False
        self._controller: object | None = None
        self._rng = random.Random(seed)

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
            pid=self.style_player_id if self.style_player_id >= 0 else None,
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
            UciOption("StylePosthoc", "string", default=self.style_posthoc or "<empty>"),
            UciOption("StylePlayerId", "spin", default=str(self.style_player_id),
                      min=-1, max=1_000_000),
            UciOption("EngineCmd", "string", default=self.engine_cmd or "<empty>"),
            UciOption("EngineDepth", "spin", default=str(self.engine_depth), min=1, max=40),
            UciOption("EngineNodes", "spin", default=str(self.engine_nodes),
                      min=0, max=10_000_000),
            UciOption("EngineMovetime", "spin",
                      default=str(int(self.engine_movetime * 1000)),
                      min=0, max=600_000),  # milliseconds; 0 = uncapped
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
        elif key == "styleposthoc" and value != self.style_posthoc:
            self.style_posthoc = value
            self._reset_model()
        elif key == "styleplayerid":
            self.style_player_id = _safe_int(value, self.style_player_id)
        elif key == "enginecmd" and value != self.engine_cmd:
            self.engine_cmd = value
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
        """Latch the time control from the first ``go`` of the game.

        At move one the side-to-move clock ~= base and the increment is exact;
        later in the game the clock has drifted, so only the increment is
        trusted then.
        """
        if not self.auto_latch_tc or self._tc_latched:
            return
        our_time = params.get("wtime" if board.turn == chess.WHITE else "btime")
        our_inc = params.get("winc" if board.turn == chess.WHITE else "binc")
        if our_time is None:
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

    def close(self) -> None:
        self._reset_controller()
        self._reset_model()

    # --- internals -----------------------------------------------------------------
    def _ensure_model(self) -> object:
        if self._model is None:
            factory = self._model_factory or self._build_model
            self._model = factory()
            self._style_applied = False
        if self.style_checkpoint and not self._style_applied:
            load_style = getattr(self._model, "load_style", None)
            if callable(load_style):
                rows = load_style(self.style_checkpoint, self.style_posthoc or None)
                logger.info("style overlay loaded: %d player rows", rows)
            self._style_applied = True
        return self._model

    def _build_model(self) -> object:
        from .matilda import MatildaModel

        return MatildaModel(
            self.checkpoint, device=self.device, maia3_model=self.maia3_model,
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
                timeout_s=self.engine_movetime or None,
            )
        return self._controller

    def _reset_controller(self) -> None:
        if self._controller is not None:
            close = getattr(self._controller, "close", None)
            if callable(close):
                close()
            self._controller = None

    def _update_controller_limits(self) -> None:
        """Apply depth/nodes/movetime to a live controller in place — they only
        feed the per-call search limit, so no reason to respawn the engine."""
        if self._controller is not None:
            self._controller.depth = self.engine_depth
            self._controller.nodes = self.engine_nodes
            self._controller.timeout_s = self.engine_movetime or None

    def _effective_elo(self) -> int:
        return effective_elo(self.elo_self, self.elo_min, self.elo_max, self.limit_strength)

    def _choose(self, ranked: tuple[tuple[str, float], ...]) -> str:
        return choose_with_temperature(ranked, self.temperature, self._rng)
