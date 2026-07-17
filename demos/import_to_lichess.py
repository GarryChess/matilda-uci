#!/usr/bin/env python3
"""Import demo PGNs to lichess and print README-ready markdown links.

Uses the public https://lichess.org/api/import endpoint (anonymous, rate
limited — the script paces itself). Each import creates a permanent
lichess game page with an interactive board, ideal for linking from the
README.

    .venv/bin/python demos/import_to_lichess.py demos/games/matched/*.pgn
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def import_pgn(pgn_text: str, *, attempts: int = 5, backoff: float = 65.0) -> dict:
    """POST one game; returns lichess's {'id':..., 'url':...}.

    Retries on rate limiting (HTTP 429) — lichess asks for a full minute of
    silence after a 429, so the backoff must be generous.
    """
    for attempt in range(attempts):
        proc = subprocess.run(
            ["curl", "-s", "-w", "\n%{http_code} %{redirect_url}", "-X", "POST",
             "https://lichess.org/api/import",
             "--data-urlencode", f"pgn={pgn_text}"],
            capture_output=True, text=True, check=True,
        )
        body, _, status = proc.stdout.rpartition("\n")
        code, _, redirect = status.partition(" ")
        if code == "200":
            return json.loads(body)
        if code == "303" and redirect:
            # Anonymous imports answer with a redirect to the created game
            # (identical PGNs map to the same game, so retries are safe).
            return {"id": redirect.rstrip("/").rsplit("/", 1)[-1], "url": redirect}
        if code == "429" and attempt < attempts - 1:
            print(f"    rate limited; waiting {backoff:.0f}s", file=sys.stderr)
            time.sleep(backoff)
            continue
        raise RuntimeError(f"lichess import returned HTTP {code}")
    raise RuntimeError("unreachable")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pgns", nargs="+", help="PGN files to import")
    ap.add_argument("--pause", type=float, default=8.0,
                    help="seconds between imports (anonymous rate limit)")
    args = ap.parse_args()

    lines = []
    for i, path in enumerate(sorted(args.pgns)):
        pgn = Path(path).read_text()
        if i > 0:
            time.sleep(args.pause)  # pace every request, successful or not
        try:
            res = import_pgn(pgn)
            url = res["url"]
        except Exception as exc:  # malformed / gave up — report and continue
            print(f"  ! {path}: import failed ({exc})", file=sys.stderr)
            continue
        # Pull White/Black/Result out of the headers for the link text.
        headers = dict(
            line[1:-1].split(" ", 1) for line in pgn.splitlines()
            if line.startswith("[") and " " in line
        )
        white = headers.get("White", "?").strip('"')
        black = headers.get("Black", "?").strip('"')
        result = headers.get("Result", "?").strip('"')
        lines.append(f"- [{white} vs {black}: {result}]({url})")
        print(f"  imported {Path(path).name} -> {url}", file=sys.stderr)

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
