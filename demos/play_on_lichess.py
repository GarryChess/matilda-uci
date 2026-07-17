#!/usr/bin/env python3
"""Play REAL games on lichess against the server-side Stockfish AI levels.

Uses the lichess Board API (or Bot API — auto-detected from the account) to
challenge lichess's actual AI at a given level, stream the game, and answer
with Matilda's moves. Every game is a native lichess game with a permanent
URL — no imports, no local approximation. AI games are always casual, so the
account's rating is untouched.

    export LICHESS_TOKEN=lip_...        # scope: board:play (or bot:play)
    .venv/bin/python demos/play_on_lichess.py --levels 6 7 8 --games-per-level 2 \
        --elo 3200 --engine-cmd stockfish

Prints per-game URLs/results and a README-ready markdown table at the end.

Account note: relaying engine moves through a regular account is only
appropriate against the computer AI (casual). For anything beyond that,
upgrade a FRESH account to a BOT account first:
    curl -X POST https://lichess.org/api/bot/account/upgrade \
         -H "Authorization: Bearer $LICHESS_TOKEN"
This script works identically with either account type.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import chess
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from matilda_uci.matilda_policy import MatildaPolicy  # noqa: E402

API = "https://lichess.org"

# The model conditions on the opponent's Elo; rough human-equivalent strength
# of lichess's AI levels for that input (documented approximations).
AUTO_OPP_ELO = {1: 800, 2: 1100, 3: 1400, 4: 1700, 5: 2000, 6: 2300, 7: 2700, 8: 3100}


class Lichess:
    def __init__(self, token: str) -> None:
        self.s = requests.Session()
        self.s.headers["Authorization"] = f"Bearer {token}"

    def account(self) -> dict:
        r = self.s.get(f"{API}/api/account", timeout=30)
        r.raise_for_status()
        return r.json()

    def challenge_ai(self, level: int, limit: int, inc: int, color: str) -> dict:
        r = self.s.post(
            f"{API}/api/challenge/ai",
            data={"level": level, "clock.limit": limit, "clock.increment": inc,
                  "color": color},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def stream_game(self, prefix: str, gid: str) -> requests.Response:
        # Note the path shape: .../game/stream/{id} (not .../game/{id}/stream).
        r = self.s.get(f"{API}/api/{prefix}/game/stream/{gid}",
                       stream=True, timeout=300)
        if r.status_code != 200:
            raise RuntimeError(f"stream for {gid} refused: HTTP {r.status_code} {r.text[:200]}")
        return r

    def move(self, prefix: str, gid: str, uci: str) -> None:
        r = self.s.post(f"{API}/api/{prefix}/game/{gid}/move/{uci}", timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"move {uci} rejected: {r.status_code} {r.text}")


def play_game(cli: Lichess, prefix: str, policy: MatildaPolicy,
              level: int, color: str, clock: tuple[int, int],
              move_delay: float = 3.0) -> dict:
    game = cli.challenge_ai(level, clock[0], clock[1], color)
    gid = game["id"]
    url = f"{API}/{gid}"
    print(f"  game {url} (level {level}, requested color {color})", flush=True)

    my_white: bool | None = None
    last_moves_seen = -1
    with cli.stream_game(prefix, gid) as resp:
        for raw in resp.iter_lines():
            if not raw:
                continue
            ev = json.loads(raw)
            if ev.get("type") == "gameFull":
                my_white = "aiLevel" not in ev.get("white", {})
                state = ev["state"]
            elif ev.get("type") == "gameState":
                state = ev
            else:
                continue  # chatLine etc.

            moves = state.get("moves", "").split()
            status = state.get("status", "started")
            if status != "started":
                return {"id": gid, "url": url, "status": status,
                        "winner": state.get("winner"), "my_white": my_white,
                        "plies": len(moves), "level": level}
            if len(moves) == last_moves_seen:
                continue  # clock tick or duplicate event
            last_moves_seen = len(moves)

            board = chess.Board()
            for m in moves:
                board.push_uci(m)
            if board.turn == (chess.WHITE if my_white else chess.BLACK):
                # Human-ish pacing: instant replies correlate with the server
                # AI stalling (observed 2026-07; a 4s-paced control ran fine).
                # Clock-aware: never spend more than ~1/35th of our remaining
                # time on the pause, so the pacing can't flag us.
                my_ms = state.get("wtime" if my_white else "btime") or 60_000
                time.sleep(min(move_delay, max(0.3, my_ms / 1000 / 35)))
                result = policy.select(board)
                if result.best_uci is None:
                    continue  # terminal; the final gameState will follow
                cli.move(prefix, gid, result.best_uci)
    if my_white is None:
        raise RuntimeError(
            f"stream for game {gid} ended before a gameFull event — the game "
            "was not playable over this API (check clock/API restrictions)."
        )
    # Stream closed without a terminal status (rare); report what we know.
    return {"id": gid, "url": url, "status": "streamEnded", "winner": None,
            "my_white": my_white, "plies": last_moves_seen, "level": level}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--token", default=os.environ.get("LICHESS_TOKEN", ""),
                    help="lichess API token (or set $LICHESS_TOKEN)")
    ap.add_argument("--levels", type=int, nargs="+", default=[6, 7, 8])
    ap.add_argument("--games-per-level", type=int, default=2)
    ap.add_argument("--elo", type=int, default=3200)
    ap.add_argument("--opp-elo", type=int, default=0,
                    help="0 = auto per level (rough human-equivalents)")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--engine-cmd", default="stockfish",
                    help="Matilda's local search controller ('' to disable)")
    ap.add_argument("--engine-depth", type=int, default=12)
    ap.add_argument("--checkpoint", default="checkpoints/base_3k.pt")
    ap.add_argument("--clock", default="180+0", help="base+inc seconds, e.g. 180+0")
    ap.add_argument("--pause", type=float, default=10.0,
                    help="seconds between games (be polite to lichess)")
    ap.add_argument("--move-delay", type=float, default=3.0,
                    help="seconds to wait before each of our moves")
    args = ap.parse_args()

    if not args.token:
        print("error: no token; create one at lichess.org/account/oauth/token "
              "with the board:play scope and pass --token or $LICHESS_TOKEN",
              file=sys.stderr)
        return 2
    base, _, inc = args.clock.partition("+")
    clock = (int(base), int(inc or 0))

    cli = Lichess(args.token)
    acct = cli.account()
    prefix = "bot" if acct.get("title") == "BOT" else "board"
    print(f"account: {acct['username']} (API: {prefix}), clock {clock[0]}+{clock[1]}")

    results: list[dict] = []
    try:
        for level in args.levels:
            opp = args.opp_elo or AUTO_OPP_ELO.get(level, 1500)
            policy = MatildaPolicy(
                checkpoint=args.checkpoint, elo_self=args.elo, elo_oppo=opp,
                temperature=args.temperature, seed=args.seed,
                tc_base=float(clock[0]), tc_inc=float(clock[1]),
                engine_cmd=args.engine_cmd, engine_depth=args.engine_depth,
            )
            try:
                for g in range(args.games_per_level):
                    color = "white" if g % 2 == 0 else "black"
                    res = play_game(cli, prefix, policy, level, color, clock,
                                    move_delay=args.move_delay)
                    me = "white" if res["my_white"] else "black"
                    outcome = ("draw" if res["winner"] is None else
                               "WIN" if res["winner"] == me else "loss")
                    res["outcome"] = outcome
                    results.append(res)
                    print(f"    -> {res['status']}, winner={res['winner']} "
                          f"({outcome} for Matilda), {res['plies']} plies", flush=True)
                    time.sleep(args.pause)
            finally:
                policy.close()
    except KeyboardInterrupt:
        print("interrupted; reporting finished games", file=sys.stderr)

    # README-ready table.
    print("\n=== README markdown ===\n")
    print("| Opponent | Games | Score |")
    print("|---|---|---|")
    for level in args.levels:
        rows = [r for r in results if r["level"] == level]
        if not rows:
            continue
        links, pts = [], 0.0
        for r in rows:
            me = "White" if r["my_white"] else "Black"
            tag = {"WIN": "1-0" if r["my_white"] else "0-1",
                   "loss": "0-1" if r["my_white"] else "1-0",
                   "draw": "1/2"}[r["outcome"]]
            pts += {"WIN": 1.0, "draw": 0.5, "loss": 0.0}[r["outcome"]]
            links.append(f"[as {me} ({tag})]({r['url']})")
        print(f"| lichess level {level} | {' · '.join(links)} "
              f"| **{pts:g} – {len(rows) - pts:g}** |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
