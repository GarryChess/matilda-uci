"""The Maia-3 move vocabulary and the move-head log-probability objective.

The learned move head emits logits over the same **4352-move vocabulary** Maia-3
uses (so the head aligns with the ``maia3_logits`` observation and can be
warm-started from Maia-3's policy head). Layout, matching ``maia3.utils
.get_all_possible_moves``:

- indices ``0..4095``: from->to moves, ``index = from_square*64 + to_square``
  (square = ``rank*8 + file``, python-chess numbering);
- indices ``4096..4351``: 256 promotions ``f"{from_file}7{to_file}8{piece}"`` for
  ``piece in (q, r, b, n)`` — only the side-to-move's promotions, because Maia-3
  works in the **side-to-move frame** (the board is vertically mirrored for Black).

This module reimplements the enumeration independently (no hard ``maia3``
dependency) so the env can compute the reward without loading the model; a gated
test asserts it matches ``maia3`` exactly when that package is available.
"""

from __future__ import annotations

import numpy as np

VOCAB_SIZE = 4352
_PROMO_PIECES = ("q", "r", "b", "n")
_LOGP_FLOOR = float(np.log(1e-9))


def _build_vocab() -> list[str]:
    moves: list[str] = []
    for rank in range(8):
        for file in range(8):
            for trank in range(8):
                for tfile in range(8):
                    moves.append(
                        f"{chr(97 + file)}{rank + 1}{chr(97 + tfile)}{trank + 1}"
                    )
    for ff in "abcdefgh":
        for ft in "abcdefgh":
            for piece in _PROMO_PIECES:
                moves.append(f"{ff}7{ft}8{piece}")
    return moves


ALL_MOVES: list[str] = _build_vocab()
MOVE_TO_IDX: dict[str, int] = {m: i for i, m in enumerate(ALL_MOVES)}
assert len(ALL_MOVES) == VOCAB_SIZE


def mirror_uci(uci: str) -> str:
    """Vertically mirror a UCI move (rank r -> 9-r), as Maia-3 does for Black."""
    out = f"{uci[0]}{9 - int(uci[1])}{uci[2]}{9 - int(uci[3])}"
    return out + uci[4:]


def move_index(uci: str, white_to_move: bool) -> int | None:
    """Index of a real-board UCI move in the side-to-move-frame vocabulary."""
    key = uci if white_to_move else mirror_uci(uci)
    return MOVE_TO_IDX.get(key)


def legal_entries(board: "object") -> list[tuple[str, int]]:
    """``(real_uci, vocab_index)`` for every legal move of ``board``."""
    import chess

    white = board.turn == chess.WHITE
    entries: list[tuple[str, int]] = []
    for move in board.legal_moves:
        uci = move.uci()
        idx = move_index(uci, white)
        if idx is not None:
            entries.append((uci, idx))
    return entries


def legal_mask(board: "object") -> np.ndarray:
    """Boolean ``(4352,)`` mask of legal moves in the side-to-move frame."""
    mask = np.zeros(VOCAB_SIZE, dtype=bool)
    for _, idx in legal_entries(board):
        mask[idx] = True
    return mask


def move_head_logp(
    logits: np.ndarray, board: "object", target_uci: str
) -> tuple[float, str | None]:
    """Masked log-softmax of ``logits`` over legal moves.

    Returns ``(log p(target), argmax_move_uci)`` where the move distribution is the
    softmax of ``logits`` restricted to legal moves. ``log p`` is floored at
    ``log(1e-9)`` if the target is somehow not a legal move (malformed data).
    """
    entries = legal_entries(board)
    if not entries:
        return 0.0, None
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    sub = logits[np.fromiter((i for _, i in entries), dtype=np.int64)]
    m = float(sub.max())
    log_z = m + float(np.log(np.exp(sub - m).sum()))
    best_uci = entries[int(np.argmax(sub))][0]

    logp = _LOGP_FLOOR
    for pos, (uci, _) in enumerate(entries):
        if uci == target_uci:
            logp = float(sub[pos] - log_z)
            break
    return logp, best_uci
