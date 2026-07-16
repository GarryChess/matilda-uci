"""Pluggable search controllers: the engine-feature block of the Matilda model.

Matilda's set-transformer accepts an optional per-candidate *engine block* of 5
features (scored flag, centipawns, cp-loss-vs-best, rank, engine-top-1). During
training those came from Stockfish (depth 21) and, for one twin, Lc0 (800
nodes) — but the model only ever sees ``(cp, rank, scored)`` per candidate, so
**any** search backend that can score a set of moves plugs in here: Stockfish,
Lc0, or a custom controller that spends its budget however it likes across the
up-to-16 candidate moves. Without a controller the block zeroes out gracefully
(``sf_valid=0`` is in-distribution; the model stays exactly usable).

This is the package's deliberate extension point for human-feature-focused
search (Botvinnik's programme): Maia proposes the humanly plausible candidates;
your controller decides which of them deserve engine time.

Score semantics (must match the training annotator,
``pipeline/annotate/sf_annotate_worker.py`` in the paper repo):

* ``cp`` is from the **side-to-move POV**, mate-clamped to ``[-32000, 32000]``;
* unscored candidates use the sentinel ``-32001`` and rank ``0``;
* ``rank`` is 1-based among scored candidates, best first (multipv order for a
  UCI engine; derived from cp descending otherwise).
"""

from __future__ import annotations

import logging
import random
import shlex
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from .features import SF_CP_UNSCORED
from .model import N_CANDIDATES

logger = logging.getLogger(__name__)

MATE_SCORE = 32000
# Sentinel for "candidate not scored" — defined once in features.py (the
# consumer side of the contract) and re-exported here for producers.
CP_UNSCORED = SF_CP_UNSCORED
assert CP_UNSCORED == -MATE_SCORE - 1  # the two constants must stay coupled


@dataclass(frozen=True)
class MoveScore:
    """One scored candidate move.

    ``cp`` is centipawns from the side-to-move's point of view, clamped to
    ``[-32000, 32000]`` (mates map to the bound). ``rank`` may be omitted; it is
    then derived from cp ordering. ``depth``/``time_s`` are informational only —
    the model does not consume them (a controller may vary depth per move
    freely; that is the whole point of the API).
    """

    uci: str
    cp: int
    rank: int | None = None
    depth: int | None = None
    time_s: float | None = None


@runtime_checkable
class SearchController(Protocol):
    """Anything that can score candidate moves for a position.

    ``score`` receives the position and the candidate UCIs (Maia-3's top-16
    humanly plausible moves, best-first) and returns scores for the subset it
    chose to evaluate. Returning fewer moves — or an empty list — is fine; the
    unscored ones get the sentinel and the model degrades gracefully.
    """

    def score(self, board: "object", candidates: Sequence[str]) -> Sequence[MoveScore]: ...

    def close(self) -> None: ...


def scores_to_arrays(
    candidates: Sequence[str], scores: Sequence[MoveScore]
) -> tuple[np.ndarray, np.ndarray, int]:
    """Map controller output onto the model's ``(sf_cp, sf_rank, sf_valid)``.

    ``candidates`` is the padded 16-slot candidate list (empty string = unused
    slot). Ranks are used as given only when EVERY score carries one; if any is
    missing, all ranks are (re)derived 1-based by cp descending — the same
    ordering a UCI engine's multipv listing gives. A partial ranking cannot be
    honored consistently, and rank 0 means "unscored" to the model, so it must
    never leak onto a scored move.
    """
    by_uci: dict[str, MoveScore] = {}
    for s in scores:
        clamped = int(max(-MATE_SCORE, min(MATE_SCORE, s.cp)))
        by_uci[s.uci] = MoveScore(s.uci, clamped, s.rank, s.depth, s.time_s)

    if by_uci and any(s.rank is None for s in by_uci.values()):
        ordered = sorted(by_uci.values(), key=lambda s: s.cp, reverse=True)
        by_uci = {
            s.uci: MoveScore(s.uci, s.cp, i + 1, s.depth, s.time_s)
            for i, s in enumerate(ordered)
        }

    sf_cp = np.full(N_CANDIDATES, CP_UNSCORED, np.int32)
    sf_rank = np.zeros(N_CANDIDATES, np.int32)
    scored_any = 0
    for j, uci in enumerate(candidates[:N_CANDIDATES]):
        s = by_uci.get(uci)
        if s is None:
            continue
        sf_cp[j] = s.cp
        sf_rank[j] = int(s.rank or 0)
        scored_any = 1
    return sf_cp, sf_rank, scored_any


class UciSearchController:
    """Score candidates with any UCI engine binary (Stockfish, Lc0, ...).

    One ``analyse`` call per position with ``root_moves=<candidates>`` and
    ``multipv=len(candidates)`` ranks every candidate in a shared search tree —
    exactly how the training annotations were produced. ``nodes`` switches to a
    fixed-node budget (how the Lc0 twin was annotated: 800 nodes); otherwise
    ``depth`` (+ ``timeout_s`` cap) applies.

    The engine process is created lazily and *shared* across calls; pass an
    existing ``chess.engine.SimpleEngine`` as ``engine`` to share one you
    already run elsewhere (it will not be closed by :meth:`close`).
    """

    def __init__(
        self,
        cmd: str = "stockfish",
        *,
        depth: int = 12,
        nodes: int = 0,
        timeout_s: float | None = None,
        options: dict[str, object] | None = None,
        engine: "object | None" = None,
    ) -> None:
        self.cmd = cmd
        self.depth = int(depth)
        self.nodes = int(nodes)
        self.timeout_s = timeout_s
        self.options = dict(options or {})
        self._engine = engine
        # Ownership follows creation: an injected engine is caller-owned; any
        # engine WE spawn (including after a crash) is ours to quit.
        self._owns_engine = False

    def _ensure_engine(self) -> "object":
        if self._engine is None:
            import chess.engine

            self._engine = chess.engine.SimpleEngine.popen_uci(shlex.split(self.cmd))
            self._owns_engine = True
            for opt, val in self.options.items():
                try:
                    self._engine.configure({opt: val})
                except chess.engine.EngineError:
                    logger.debug("engine %s has no option %s", self.cmd, opt)
        return self._engine

    def _handle_engine_death(self, exc: Exception) -> None:
        """Drop a dead engine so the next call respawns instead of reusing it."""
        logger.warning("search engine terminated (%s); will respawn on next call", exc)
        self._engine = None
        self._owns_engine = False

    def _limit(self) -> "object":
        import chess.engine

        if self.nodes:
            return chess.engine.Limit(nodes=self.nodes)
        return chess.engine.Limit(depth=self.depth, time=self.timeout_s)

    def score(self, board: "object", candidates: Sequence[str]) -> list[MoveScore]:
        import chess

        moves = []
        for u in candidates:
            if not u:
                continue
            try:
                mv = chess.Move.from_uci(u)
            except ValueError:
                continue
            if board.is_legal(mv):
                moves.append(mv)
        if not moves:
            return []
        import chess.engine

        engine = self._ensure_engine()
        try:
            infos = engine.analyse(
                board, self._limit(), root_moves=moves, multipv=len(moves)
            )
        except chess.engine.EngineTerminatedError as exc:
            self._handle_engine_death(exc)
            raise
        if isinstance(infos, dict):
            infos = [infos]
        out: list[MoveScore] = []
        for rank, info in enumerate(infos):
            pv = info.get("pv")
            if not pv:
                continue
            sc = info["score"].pov(board.turn).score(mate_score=MATE_SCORE)
            out.append(
                MoveScore(
                    uci=pv[0].uci(),
                    cp=int(max(-MATE_SCORE, min(MATE_SCORE, sc))),
                    rank=rank + 1,
                    depth=int(info.get("depth", 0)) or None,
                )
            )
        return out

    def close(self) -> None:
        # Quit and drop only an engine we spawned; an injected engine is
        # caller-owned and stays attached, so a reused controller keeps
        # honoring the shared-engine contract instead of respawning its own.
        if self._engine is not None and self._owns_engine:
            try:
                self._engine.quit()
            except Exception:  # already dead is fine
                logger.debug("engine quit failed", exc_info=True)
            self._engine = None
            self._owns_engine = False


class DumbSearchController(UciSearchController):
    """The docs' toy example: like Stockfish, but deliberately scatterbrained.

    Searches each candidate *separately* at a random shallow depth (lots of
    low-depth looks rather than one deep shared tree). Exists to demonstrate
    that any budget-allocation policy over the candidates — even a silly one —
    plugs into the same seam; a smarter controller would spend depth on the
    moves that look most interesting for a human in the position.
    """

    def __init__(
        self,
        cmd: str = "stockfish",
        *,
        min_depth: int = 2,
        max_depth: int = 6,
        seed: int = 0,
        **kwargs: object,
    ) -> None:
        super().__init__(cmd, **kwargs)  # type: ignore[arg-type]
        self.min_depth = int(min_depth)
        self.max_depth = int(max_depth)
        self._rng = random.Random(seed)

    def score(self, board: "object", candidates: Sequence[str]) -> list[MoveScore]:
        import chess
        import chess.engine

        engine = self._ensure_engine()
        out: list[MoveScore] = []
        for u in candidates:
            if not u:
                continue
            try:
                mv = chess.Move.from_uci(u)
            except ValueError:
                continue
            if not board.is_legal(mv):
                continue
            depth = self._rng.randint(self.min_depth, self.max_depth)
            try:
                info = engine.analyse(
                    board, chess.engine.Limit(depth=depth), root_moves=[mv]
                )
            except chess.engine.EngineTerminatedError as exc:
                self._handle_engine_death(exc)
                raise
            if isinstance(info, list):
                info = info[0]
            pv = info.get("pv")
            if not pv:
                continue
            sc = info["score"].pov(board.turn).score(mate_score=MATE_SCORE)
            out.append(
                MoveScore(
                    uci=pv[0].uci(),
                    cp=int(max(-MATE_SCORE, min(MATE_SCORE, sc))),
                    depth=depth,
                )
            )
        return out  # ranks derived from cp ordering by scores_to_arrays
