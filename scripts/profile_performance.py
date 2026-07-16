#!/usr/bin/env python3
"""Profile Matilda's inference throughput by game phase, with rliable CIs.

Two phases:

**fetch** — stream a byte-range of a lichess monthly database dump (rated
blitz games from 2026), replay each game, and sample positions into ply
tranches: opening (ply <= 16), middlegame (17-60), endgame (> 60). Each sample
keeps its real 8-position history, both player Elos, and the game's time
control, so the profiled call is exactly the live inference path. The sample
set is written to JSON for reproducible reruns.

    .venv/bin/python scripts/profile_performance.py fetch \
        --month 2026-06 --per-tranche 50 --out demos/data/blitz_positions.json

**run** — time `MatildaModel.predict` (full chain: Maia-3 -> featurize ->
re-ranker) per position, for each requested torch thread count, and aggregate
predictions/second per tranche with rliable's stratified-bootstrap interval
estimates (mean + IQM). Writes a plot and a markdown report.

    .venv/bin/python scripts/profile_performance.py run \
        --positions demos/data/blitz_positions.json --threads 1 4 8 \
        --plot docs/profiling.png --report docs/profiling.md
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

TRANCHES = (("opening", 1, 16), ("middlegame", 17, 60), ("endgame", 61, 10_000))


# --- fetch --------------------------------------------------------------------
def cmd_fetch(args: argparse.Namespace) -> int:
    import io

    import chess.pgn

    url = f"https://database.lichess.org/standard/lichess_db_standard_rated_{args.month}.pgn.zst"
    if shutil.which("zstd") is None:
        print("error: zstd is required for the fetch phase", file=sys.stderr)
        return 2
    print(f"streaming first {args.bytes:,} bytes of {url}")
    curl = subprocess.run(
        ["curl", "-s", "-r", f"0-{args.bytes}", url],
        check=True, capture_output=True,
    )
    # A truncated zstd stream decompresses up to the cut; ignore the tail error.
    zstd = subprocess.run(
        ["zstd", "-d", "-c"], input=curl.stdout, capture_output=True,
    )
    text = zstd.stdout.decode("utf-8", errors="replace")
    print(f"decompressed {len(text):,} chars of PGN")

    rng = random.Random(args.seed)
    quota = {name: args.per_tranche for name, _, _ in TRANCHES}
    samples: list[dict] = []
    dates: set[str] = set()
    stream = io.StringIO(text)
    games = 0
    while any(quota.values()):
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        h = game.headers
        if "Rated Blitz" not in h.get("Event", ""):
            continue
        try:
            welo, belo = int(h["WhiteElo"]), int(h["BlackElo"])
        except (KeyError, ValueError):
            continue
        tc = h.get("TimeControl", "180+0")
        try:
            tc_base, tc_inc = (int(x) for x in tc.split("+"))
        except ValueError:
            continue
        games += 1

        board = game.board()
        fens = [board.fen()]
        for move in game.mainline_moves():
            board.push(move)
            fens.append(board.fen())
        n_plies = len(fens) - 1
        if n_plies < 2:
            continue

        # At most one sample per tranche per game, at a random in-tranche ply.
        for name, lo, hi in TRANCHES:
            if quota[name] <= 0:
                continue
            lo_i, hi_i = max(lo, 1), min(hi, n_plies)
            if lo_i > hi_i:
                continue
            ply = rng.randint(lo_i, hi_i)
            # Never sample terminal positions: predict() short-circuits on them
            # in microseconds, and a single such 1/dt outlier poisons the mean.
            if not any(chess.Board(fens[ply]).legal_moves):
                continue
            elo_self = welo if ply % 2 == 0 else belo  # side to move at `ply`
            samples.append({
                "fen": fens[ply],
                "history": fens[max(0, ply - 8):ply],
                "ply": ply,
                "tranche": name,
                "elo_self": elo_self,
                "elo_oppo": belo if ply % 2 == 0 else welo,
                "tc_base": tc_base,
                "tc_inc": tc_inc,
            })
            quota[name] -= 1
        if h.get("UTCDate"):
            dates.add(h["UTCDate"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "source": url,
        "bytes": args.bytes,
        "games_scanned": games,
        "dates": sorted(dates),
        "seed": args.seed,
    }
    out.write_text(json.dumps({"meta": meta, "positions": samples}, indent=1))
    counts = {name: sum(s["tranche"] == name for s in samples) for name, _, _ in TRANCHES}
    print(f"wrote {len(samples)} positions {counts} from {games} games "
          f"({min(dates, default='?')}..{max(dates, default='?')}) -> {out}")
    return 0


# --- run ----------------------------------------------------------------------
def cmd_run(args: argparse.Namespace) -> int:
    import numpy as np
    import torch

    import chess
    from matilda_uci.matilda import MatildaModel

    data = json.loads(Path(args.positions).read_text())
    positions = data["positions"]
    by_tranche: dict[str, list[dict]] = {}
    for s in positions:
        by_tranche.setdefault(s["tranche"], []).append(s)

    model = MatildaModel(args.checkpoint, device=args.device)
    # Warm up: model + Maia-3 load and first-call overheads stay out of timings.
    warm = chess.Board()
    for _ in range(3):
        model.predict(warm, elo_self=1500, elo_oppo=1500)

    results: dict[str, np.ndarray] = {}  # label -> (N, 1) predictions/sec
    for threads in args.threads:
        torch.set_num_threads(threads)
        for name, _, _ in TRANCHES:
            rows = by_tranche.get(name, [])
            speeds = []
            for s in rows:
                board = chess.Board(s["fen"])
                t0 = time.perf_counter()
                pred = model.predict(
                    board,
                    board_history=s["history"],
                    elo_self=s["elo_self"],
                    elo_oppo=s["elo_oppo"],
                    tc_base=s["tc_base"],
                    tc_inc=s["tc_inc"],
                )
                dt = time.perf_counter() - t0
                if pred is None:  # terminal position: no inference ran
                    continue
                speeds.append(1.0 / dt)
            label = f"{name} ({threads} thr)"
            results[label] = np.asarray(speeds).reshape(-1, 1)
            print(f"{label:24s} n={len(speeds):3d} mean={np.mean(speeds):6.2f} pred/s")
    model.close()

    # rliable: stratified bootstrap interval estimates over positions.
    from rliable import library as rly
    from rliable import metrics as rmetrics
    from rliable import plot_utils

    def aggregate(x):
        return np.array([rmetrics.aggregate_mean(x), rmetrics.aggregate_iqm(x)])

    point, cis = rly.get_interval_estimates(results, aggregate, reps=args.reps)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plot_utils.plot_interval_estimates(
        point, cis, metric_names=["Mean", "IQM"],
        algorithms=list(results), xlabel="predictions / second",
    )
    plot_path = Path(args.plot)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.gcf().savefig(plot_path, bbox_inches="tight", dpi=150)
    print(f"plot -> {plot_path}")

    # Markdown report.
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                          capture_output=True, text=True).stdout.strip() or platform.machine()
    meta = data.get("meta", {})
    lines = [
        "# Inference throughput by game phase",
        "",
        "Full-chain `MatildaModel.predict` (Maia-3 23M -> featurize -> re-ranker)",
        f"on {chip}, {platform.system()} {platform.release()}, "
        f"torch {torch.__version__}, device={args.device}.",
        f"Positions: rated blitz games, {meta.get('source', '?').split('/')[-1]}"
        f" ({', '.join(meta.get('dates', [])[:3])}...), sampled per ply tranche "
        "with real history/Elos/TC. Intervals: rliable stratified bootstrap "
        f"({args.reps} reps), mean [95% CI].",
        "",
        "| tranche | threads | n | pred/s mean [95% CI] | IQM [95% CI] |",
        "|---|---|---|---|---|",
    ]
    for label, arr in results.items():
        name, thr = label.rsplit(" (", 1)
        mean_lo, iqm_lo = cis[label][0]
        mean_hi, iqm_hi = cis[label][1]
        lines.append(
            f"| {name} | {thr.rstrip(') thr')} | {arr.shape[0]} "
            f"| {point[label][0]:.2f} [{mean_lo:.2f}, {mean_hi:.2f}] "
            f"| {point[label][1]:.2f} [{iqm_lo:.2f}, {iqm_hi:.2f}] |"
        )
    lines += ["", f"![throughput intervals]({plot_path.name})", ""]
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines))
    print(f"report -> {report}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="sample blitz positions from a lichess dump")
    f.add_argument("--month", default="2026-06")
    f.add_argument("--bytes", type=int, default=15_000_000)
    f.add_argument("--per-tranche", type=int, default=50)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--out", default="demos/data/blitz_positions.json")
    f.set_defaults(func=cmd_fetch)

    r = sub.add_parser("run", help="time predict() and build the rliable report")
    r.add_argument("--positions", default="demos/data/blitz_positions.json")
    r.add_argument("--checkpoint", default="checkpoints/base_3k.pt")
    r.add_argument("--device", default="cpu")
    r.add_argument("--threads", type=int, nargs="+", default=[1, 4, 8])
    r.add_argument("--reps", type=int, default=2000)
    r.add_argument("--plot", default="docs/profiling.png")
    r.add_argument("--report", default="docs/profiling.md")
    r.set_defaults(func=cmd_run)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
