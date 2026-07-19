#!/usr/bin/env python3
"""Play demo games: the real matilda-uci engine vs lichess-level Stockfish.

Drives the actual UCI subprocess (python -m matilda_uci) — the same thing a
GUI runs — against Stockfish configured to approximate lichess's AI levels
(the fishnet mapping of level -> Skill Level/depth), and writes PGNs plus a
results table.

    .venv/bin/python demos/play_vs_stockfish.py --games 2 \
        --pairings 1500:6 2000:7 2800:8 --out demos/games

Lichess-level approximation (fishnet): level 6 = skill 11 / depth 8,
level 7 = skill 16 / depth 13, level 8 = skill 20 / depth 22. Games are
capped at --max-plies and adjudicated as a draw beyond it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import chess
import chess.engine
import chess.pgn

REPO = Path(__file__).resolve().parent.parent

# fishnet's lichess AI level -> (Stockfish "Skill Level", search depth)
LICHESS_LEVELS = {1: (-9, 5), 2: (-5, 5), 3: (-1, 5), 4: (3, 5),
                  5: (7, 5), 6: (11, 8), 7: (16, 13), 8: (20, 22)}


def matilda_engine(
    elo: int, temperature: float, engine_cmd: str = "", seed: int | None = None,
    checkpoint: str = "", name: str = "Matilda", no_engine: bool = False,
) -> chess.engine.SimpleEngine:
    # Run straight from the checkout: no editable install required.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    argv = [sys.executable, "-m", "matilda_uci",
            "--elo", str(elo), "--temperature", str(temperature),
            "--name", name,
            "--checkpoint", checkpoint or str(REPO / "checkpoints" / "base_3k.pt")]
    if seed is not None:
        argv += ["--seed", str(seed)]
    if no_engine:
        argv += ["--no-engine"]
    elif engine_cmd:  # override the default auto-resolved stockfish
        argv += ["--engine-cmd", engine_cmd]
    return chess.engine.SimpleEngine.popen_uci(argv, cwd=REPO, env=env)


def stockfish_engine(cmd: str, level: int) -> tuple[chess.engine.SimpleEngine, int]:
    skill, depth = LICHESS_LEVELS[level]
    eng = chess.engine.SimpleEngine.popen_uci([cmd])
    eng.configure({"Skill Level": skill})
    return eng, depth


def play_game(white, black, white_limit, black_limit, max_plies: int) -> chess.pgn.Game:
    board = chess.Board()
    while not board.is_game_over() and board.ply() < max_plies:
        eng, limit = (white, white_limit) if board.turn else (black, black_limit)
        result = eng.play(board, limit)
        if result.move is None:
            break
        board.push(result.move)
    game = chess.pgn.Game.from_board(board)
    if not board.is_game_over() and board.ply() >= max_plies:
        # Don't fake a result: a queen-up position "drawn" by a ply cap lies.
        game.headers["Result"] = "*"
        game.headers["Termination"] = f"stopped at {max_plies} plies (unfinished)"
    return game


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairings", nargs="+", default=["1500:6", "2000:7", "2800:8"],
                    help="matildaElo:lichessLevel pairs")
    ap.add_argument("--games", type=int, default=2,
                    help="games per pairing (colors alternate)")
    ap.add_argument("--stockfish", default="stockfish")
    ap.add_argument("--temperature", type=float, default=0.3,
                    help="sampling temperature so repeat games vary")
    ap.add_argument("--sf-movetime", type=float, default=0.5)
    ap.add_argument("--matilda-movetime", type=float, default=3.0,
                    help="per-move seconds handed to Matilda (go movetime; its "
                         "TC budget takes the min with this)")
    ap.add_argument("--matilda-engine-cmd", default="",
                    help="Matilda's search controller (default: the engine's own "
                         "auto-resolved stockfish)")
    ap.add_argument("--matilda-no-engine", action="store_true",
                    help="run Matilda from the raw human prior, no search engine")
    ap.add_argument("--seed", type=int, default=None,
                    help="Matilda sampling seed (default: fresh per start; set "
                         "distinct seeds for parallel runs)")
    ap.add_argument("--matilda-checkpoint", default="",
                    help="re-ranker checkpoint override (e.g. a zero-delta one, "
                         "which plays exactly the raw Maia-3 prior)")
    ap.add_argument("--matilda-name", default="Matilda",
                    help="name used in PGN headers (e.g. 'Maia-3')")
    ap.add_argument("--max-plies", type=int, default=240)
    ap.add_argument("--out", default="demos/games")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().strftime("%Y.%m.%d")
    summary: list[str] = []

    for pairing in args.pairings:
        elo_s, lvl_s = pairing.split(":")
        elo, level = int(elo_s), int(lvl_s)
        matilda = matilda_engine(elo, args.temperature, args.matilda_engine_cmd,
                                 args.seed, args.matilda_checkpoint,
                                 args.matilda_name, args.matilda_no_engine)
        sf, depth = stockfish_engine(args.stockfish, level)
        sf_limit = chess.engine.Limit(depth=depth, time=args.sf_movetime)
        m_limit = chess.engine.Limit(time=args.matilda_movetime)
        score = {"matilda": 0.0, "stockfish": 0.0}
        unfinished = 0
        try:
            for g in range(args.games):
                m_white = g % 2 == 0
                white, black = (matilda, sf) if m_white else (sf, matilda)
                wl, bl = (m_limit, sf_limit) if m_white else (sf_limit, m_limit)
                game = play_game(white, black, wl, bl, args.max_plies)
                m_name = f"{args.matilda_name} (Elo {elo})"
                w_name = m_name if m_white else f"Stockfish lvl{level}"
                b_name = f"Stockfish lvl{level}" if m_white else m_name
                game.headers.update({
                    "Event": f"{args.matilda_name} demo: Elo {elo} vs lichess level {level}",
                    "Site": "local", "Date": today, "Round": str(g + 1),
                    "White": w_name, "Black": b_name,
                })
                res = game.headers["Result"]
                if res == "*":
                    unfinished += 1
                elif res in ("1-0", "0-1"):
                    m_won = (res == "1-0") == m_white
                    score["matilda" if m_won else "stockfish"] += 1.0
                else:  # a genuine draw on the board
                    score["matilda"] += 0.5
                    score["stockfish"] += 0.5
                path = out / f"matilda{elo}_vs_lvl{level}_g{g + 1}.pgn"
                path.write_text(str(game) + "\n")
                print(f"{path.name}: {w_name} vs {b_name} -> {res}")
        finally:
            matilda.quit()
            sf.quit()
        extra = f" ({unfinished} unfinished)" if unfinished else ""
        summary.append(
            f"| {args.matilda_name} @ {elo} | lichess level {level} (skill {LICHESS_LEVELS[level][0]}, "
            f"depth {LICHESS_LEVELS[level][1]}) | {score['matilda']:g} - "
            f"{score['stockfish']:g}{extra} |"
        )

    if args.matilda_no_engine:
        assist = "; Matilda unassisted (pure human prior)"
    else:
        assist = ("; Matilda search controller: "
                  f"{args.matilda_engine_cmd or 'auto (stockfish)'}")
    table = ["# Demo games", "",
             f"Generated {today} by `demos/play_vs_stockfish.py` "
             f"(temperature {args.temperature}; lichess levels via the fishnet "
             f"mapping{assist}).",
             "",
             "Pairings match each Matilda Elo against the lichess level of "
             "comparable strength (fishnet levels 6/7/8 sit roughly at "
             "2300/2700/3100), so a hold or better is the expectation in each row.",
             "", "| Matilda | Opponent | Score (M - SF) |", "|---|---|---|", *summary, ""]
    (out / "README.md").write_text("\n".join(table))
    print(f"\nsummary -> {out / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
