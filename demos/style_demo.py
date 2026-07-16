#!/usr/bin/env python3
"""Show that player-style embeddings measurably change Matilda's play.

Loads the deploy base + the style-token overlay, then compares the style-free
distribution against several player identities on a fixed set of positions.
Reported per player:

  mean KL(styled || generic)   how far the player's distribution moves
  top-move changes             positions where the most-likely move differs

Run (needs the checkpoints and the maia3 runtime):

    PYTHONPATH=src .venv/bin/python demos/style_demo.py \
        --base checkpoints/base_3k.pt --style checkpoints/style_token_3k.pt \
        --pids 0 25 250 2500 --elo 2000

Paper-scale context: the style token is worth +0.41% move-prediction overall
and +1.1-1.5% on GM bands over the same frozen base (see the paper repo's
RESULTS.md); this demo shows the per-position mechanics of that gain.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from matilda_uci.matilda import MatildaModel  # noqa: E402
from matilda_uci.matilda_policy import board_history_fens  # noqa: E402

# A small, varied fixed position set: openings, middlegames, an endgame.
POSITIONS: list[list[str]] = [
    [],  # startpos
    ["e2e4", "c7c5"],  # open sicilian choice
    ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3"],  # indian systems junction
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"],  # ruy: exchange or retreat
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "b2b4"],  # evans gambit offer
    ["d2d4", "d7d5", "c2c4", "d5c4", "g1f3", "g8f6", "e2e3", "e7e6", "f1c4"],
    # the Berlin queenless endgame:
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "g8f6", "e1g1", "f6e4", "d2d4",
     "e4d6", "b5c6", "d7c6", "d4e5", "d6f5", "d1d8", "e8d8"],
]


def kl(p: dict[str, float], q: dict[str, float]) -> float:
    """KL(p || q) over the union support, with a tiny floor for stability."""
    eps = 1e-9
    return sum(pi * math.log(pi / max(q.get(m, 0.0), eps))
               for m, pi in p.items() if pi > eps)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="checkpoints/base_3k.pt")
    ap.add_argument("--style", default="checkpoints/style_token_3k.pt")
    ap.add_argument("--posthoc", default=None)
    ap.add_argument("--pids", type=int, nargs="+", default=[0, 25, 250, 2500])
    ap.add_argument("--elo", type=int, default=2000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model = MatildaModel(args.base, device=args.device)
    rows = model.load_style(args.style, args.posthoc)
    print(f"style table: {rows} player rows (pid 0 = generic fallback)\n")

    boards = []
    for moves in POSITIONS:
        b = chess.Board()
        for mv in moves:
            b.push_uci(mv)
        boards.append(b)

    # Style-free reference per position.
    generic = []
    for b in boards:
        pred = model.predict(b, board_history=board_history_fens(b),
                             elo_self=args.elo, elo_oppo=args.elo)
        generic.append(pred.move_probs)

    print(f"{'pid':>6} {'mean KL':>9} {'max KL':>8} {'top-move changes':>17}")
    for pid in args.pids:
        if not 0 <= pid < rows:
            print(f"{pid:>6}  (out of range, skipped)")
            continue
        kls, flips = [], 0
        for b, g in zip(boards, generic):
            pred = model.predict(b, board_history=board_history_fens(b),
                                 elo_self=args.elo, elo_oppo=args.elo, pid=pid)
            kls.append(kl(pred.move_probs, g))
            top_s = max(pred.move_probs, key=pred.move_probs.get)
            top_g = max(g, key=g.get)
            flips += top_s != top_g
        print(f"{pid:>6} {sum(kls) / len(kls):>9.4f} {max(kls):>8.4f} "
              f"{flips:>10}/{len(boards)}")

    print("\nnon-zero KL = the identity vector genuinely conditions the policy;")
    print("top-move changes = positions where imitating that player picks a different move.")
    model.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
