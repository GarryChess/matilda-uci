from __future__ import annotations

import logging
import re
import sys
import threading
from typing import TextIO

import chess

from .policy import MovePolicy

logger = logging.getLogger(__name__)


class UciEngine:
    """A UCI *server*: drives a :class:`MovePolicy` over the UCI text protocol.

    Reads commands from an input stream and writes protocol replies to an output
    stream (stdin/stdout by default). The move policy is injected, so the same
    loop serves the Maia policy today and any future human-like model later.

    Search runs on a worker thread so ``go infinite`` can hold its ``bestmove``
    until ``stop`` while the main loop stays responsive to ``isready``/``stop``.
    """

    def __init__(
        self,
        policy: MovePolicy,
        *,
        name: str = "Matilda",
        author: str = "Garry Chess",
        out: TextIO | None = None,
    ) -> None:
        self.policy = policy
        self.name = name
        self.author = author
        self.board = chess.Board()
        self._out = out if out is not None else sys.stdout
        self._write_lock = threading.Lock()
        self._search_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._abort_event = threading.Event()

    # --- main loop -------------------------------------------------------------
    def run(self, in_stream: TextIO | None = None) -> None:
        in_stream = in_stream if in_stream is not None else sys.stdin
        try:
            for raw in in_stream:
                if not self.handle_command(raw.strip()):
                    break
        finally:
            self._abort_search()
            self.policy.close()

    def handle_command(self, line: str) -> bool:
        """Process one command. Returns ``False`` when the engine should exit."""
        parts = line.split()
        if not parts:
            return True
        cmd, args = parts[0], parts[1:]

        if cmd == "uci":
            self._cmd_uci()
        elif cmd == "isready":
            self._send("readyok")
        elif cmd == "ucinewgame":
            self._cmd_ucinewgame()
        elif cmd == "setoption":
            self._cmd_setoption(args)
        elif cmd == "position":
            self._cmd_position(args)
        elif cmd == "go":
            self._cmd_go(args)
        elif cmd == "stop" or cmd == "ponderhit":
            self._release_search()
        elif cmd == "quit":
            self._abort_search()
            return False
        else:
            logger.debug("ignoring unsupported command: %s", line)
        return True

    def join_search(self, timeout: float | None = None) -> None:
        """Block until the current search thread finishes (used by tests)."""
        thread = self._search_thread
        if thread is not None:
            thread.join(timeout)

    # --- command handlers ------------------------------------------------------
    def _cmd_uci(self) -> None:
        self._send(f"id name {self.name}")
        self._send(f"id author {self.author}")
        for option in self.policy.uci_options():
            self._send(option.declaration())
        self._send("uciok")

    def _cmd_ucinewgame(self) -> None:
        self._abort_search()
        self.board = chess.Board()
        self.policy.new_game()

    def _cmd_setoption(self, args: list[str]) -> None:
        # Serialize with any running search: options like EngineCmd/Device tear
        # down the model/controller the search thread is using; per the UCI spec
        # setoption is sent while the engine is idle, but GUIs violate that.
        self._abort_search()
        if "name" not in args:
            return
        rest = args[args.index("name") + 1:]
        if "value" in rest:
            split = rest.index("value")
            name = " ".join(rest[:split])
            value = " ".join(rest[split + 1:])
        else:
            name, value = " ".join(rest), ""
        if name:
            self.policy.set_option(name, value)

    def _cmd_position(self, args: list[str]) -> None:
        self._abort_search()
        if not args:
            return
        if args[0] == "startpos":
            board = chess.Board()
            move_tokens = args[2:] if len(args) > 1 and args[1] == "moves" else []
        elif args[0] == "fen":
            fen = " ".join(args[1:7])
            try:
                board = chess.Board(fen)
            except ValueError:
                logger.warning("invalid FEN in position command: %s", fen)
                return
            tail = args[7:]
            move_tokens = tail[1:] if tail and tail[0] == "moves" else []
        else:
            logger.warning("unsupported position spec: %s", " ".join(args))
            return

        for token in move_tokens:
            try:
                board.push_uci(token)
            except ValueError:
                logger.warning("illegal move '%s' in position; stopping replay", token)
                break
        self.board = board

    def _cmd_go(self, args: list[str]) -> None:
        self._abort_search()
        params = _parse_go(args)
        # Policies that condition on the clock (e.g. Matilda's time-control
        # latch) may expose observe_go; the MovePolicy protocol doesn't require it.
        observe = getattr(self.policy, "observe_go", None)
        if callable(observe):
            try:
                observe(params, self.board)
            except Exception:
                logger.exception("policy.observe_go failed; continuing")
        hold = "infinite" in params or "ponder" in params
        self._stop_event.clear()
        self._abort_event.clear()
        self._search_thread = threading.Thread(
            target=self._search, args=(hold,), daemon=True
        )
        self._search_thread.start()

    # --- search ----------------------------------------------------------------
    def _search(self, hold: bool) -> None:
        board = self.board.copy()
        try:
            result = self.policy.select(board)
        except Exception as exc:  # never leave the GUI hanging without a bestmove
            logger.exception("policy.select failed")
            # Surface the failure on the UCI channel too: a bare null move loops
            # silently in most GUIs, hiding config errors (bad checkpoint path,
            # missing runtime dependency) behind "illegal move" popups.
            self._send(f"info string error: {type(exc).__name__}: {exc}")
            self._send("bestmove 0000")
            return

        if result.best_uci is None:
            self._send("bestmove 0000")
            return

        info = ["info", "depth", "1"]
        if result.score_cp is not None:
            info += ["score", "cp", str(result.score_cp)]
        # Maia is a one-ply policy, so the principal variation is just the move.
        info += ["pv", result.best_uci]
        self._send(" ".join(info))
        if result.info:
            self._send(f"info string {result.info}")

        if hold:
            self._stop_event.wait()
            if self._abort_event.is_set():
                return
        self._send(f"bestmove {result.best_uci}")

    def _release_search(self) -> None:
        """Handle ``stop``/``ponderhit``: let a held search emit its bestmove."""
        self._stop_event.set()
        self.join_search(timeout=5.0)

    def _abort_search(self) -> None:
        """Tear down any running search without emitting a bestmove."""
        thread = self._search_thread
        if thread is not None and thread.is_alive():
            self._abort_event.set()
            self._stop_event.set()
            thread.join(timeout=5.0)
            if thread.is_alive():
                # A stuck controller/engine call outlived the join. Proceeding
                # (e.g. policy.close on quit) will kill its engine from under it,
                # which unblocks it with EngineTerminatedError — noisy but safe.
                logger.warning("search thread still running after 5s abort join")
        self._search_thread = None

    # --- io --------------------------------------------------------------------
    def _send(self, line: str) -> None:
        with self._write_lock:
            self._out.write(line + "\n")
            self._out.flush()


_GO_INT_KEYS = frozenset(
    ("wtime", "btime", "winc", "binc", "movestogo", "depth", "nodes", "mate", "movetime")
)
# A UCI move token: from-square, to-square, optional promotion piece — or the
# null move. Used to delimit `searchmoves`, whose move list ends at the first
# token that isn't a move (the UCI spec allows further parameters after it).
_MOVE_RE = re.compile(r"^(?:[a-h][1-8][a-h][1-8][qrbn]?|0000)$")


def _parse_go(args: list[str]) -> dict:
    """Parse ``go`` arguments into a dict (int values for clock/limit keys)."""
    params: dict = {}
    i = 0
    while i < len(args):
        key = args[i]
        if key in _GO_INT_KEYS and i + 1 < len(args):
            try:
                params[key] = int(args[i + 1])
            except ValueError:
                pass
            i += 2
        elif key == "searchmoves":  # consume move tokens only, then keep parsing
            moves = []
            i += 1
            while i < len(args) and _MOVE_RE.match(args[i]):
                moves.append(args[i])
                i += 1
            params[key] = moves
        else:  # flags: infinite, ponder
            params[key] = True
            i += 1
    return params
