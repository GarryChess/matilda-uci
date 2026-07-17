#!/usr/bin/env python3
"""Aggregate a directory tree of demo-game PGNs into a results report.

Reports W/D/L from Matilda's perspective (overall and by color), score with a
95% Wilson interval, an implied Elo difference, game-length stats, and
termination kinds.

    .venv/bin/python demos/aggregate_results.py demos/games/lvl8_sims
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import chess.pgn


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def implied_elo(score: float) -> float:
    """Elo difference implied by an expected score (clamped away from 0/1)."""
    s = min(max(score, 1e-3), 1 - 1e-3)
    return -400.0 * math.log10(1.0 / s - 1.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", help="directory tree containing .pgn files")
    ap.add_argument("--report", default=None, help="optional markdown output path")
    args = ap.parse_args()

    games = []
    for path in sorted(Path(args.root).rglob("*.pgn")):
        with open(path) as fh:
            game = chess.pgn.read_game(fh)
        if game is None:
            continue
        h = game.headers
        # "our" side is any Matilda-family engine (Matilda, Maia-3 baseline, ...)
        m_white = h.get("White", "").startswith(("Matilda", "Maia"))
        plies = sum(1 for _ in game.mainline_moves())
        games.append({
            "file": str(path),
            "m_white": m_white,
            "result": h.get("Result", "*"),
            "plies": plies,
            "termination": h.get("Termination", ""),
        })
    if not games:
        print("no games found", file=sys.stderr)
        return 2

    finished = [g for g in games if g["result"] in ("1-0", "0-1", "1/2-1/2")]
    unfinished = len(games) - len(finished)
    w = d = losses = 0
    by_color = {True: [0, 0, 0], False: [0, 0, 0]}  # white/black -> [w, d, l]
    for g in finished:
        if g["result"] == "1/2-1/2":
            d += 1
            by_color[g["m_white"]][1] += 1
        elif (g["result"] == "1-0") == g["m_white"]:
            w += 1
            by_color[g["m_white"]][0] += 1
        else:
            losses += 1
            by_color[g["m_white"]][2] += 1
    n = len(finished)
    score = (w + 0.5 * d) / n if n else 0.0
    lo, hi = wilson(score, n)
    plies = sorted(g["plies"] for g in finished)
    mean_plies = sum(plies) / len(plies)
    median_plies = plies[len(plies) // 2]

    lines = [
        f"# Results: {Path(args.root).name}",
        "",
        f"Games: {len(games)} ({n} finished, {unfinished} unfinished/excluded)",
        "",
        "| | W | D | L | score | 95% CI | implied Elo diff |",
        "|---|---|---|---|---|---|---|",
        f"| overall | {w} | {d} | {losses} | {score:.1%} | [{lo:.1%}, {hi:.1%}] "
        f"| {implied_elo(score):+.0f} [{implied_elo(lo):+.0f}, {implied_elo(hi):+.0f}] |",
        f"| as White | {by_color[True][0]} | {by_color[True][1]} | {by_color[True][2]} "
        f"| {(by_color[True][0] + 0.5 * by_color[True][1]) / max(1, sum(by_color[True])):.1%} | | |",
        f"| as Black | {by_color[False][0]} | {by_color[False][1]} | {by_color[False][2]} "
        f"| {(by_color[False][0] + 0.5 * by_color[False][1]) / max(1, sum(by_color[False])):.1%} | | |",
        "",
        f"Game length: mean {mean_plies:.0f} plies, median {median_plies}, "
        f"range {plies[0]}-{plies[-1]}.",
    ]
    out = "\n".join(lines)
    print(out)
    if args.report:
        Path(args.report).write_text(out + "\n")
        print(f"\nreport -> {args.report}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
