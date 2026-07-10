from __future__ import annotations

import argparse
import logging
import sys
from typing import Sequence

from .engine import UciEngine
from .policy import MaiaPolicy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="matilda-uci",
        description="Matilda: a free human-like UCI chess engine (Maia-2 backed).",
    )
    parser.add_argument("--maia-type", default="rapid", choices=["rapid", "blitz"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--elo", type=int, default=1500, help="Engine (self) Elo for Maia.")
    parser.add_argument("--opp-elo", type=int, default=1500, help="Opponent Elo for Maia.")
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="0 = always the most human-likely move; >0 samples for variety.",
    )
    parser.add_argument("--name", default="Matilda", help="Engine name shown to the GUI.")
    parser.add_argument(
        "--log-level", default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Logs MUST go to stderr; stdout is the UCI protocol channel.
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        stream=sys.stderr,
        format="%(name)s: %(message)s",
    )
    policy = MaiaPolicy(
        elo_self=args.elo,
        elo_oppo=args.opp_elo,
        maia_type=args.maia_type,
        device=args.device,
        temperature=args.temperature,
    )
    engine = UciEngine(policy, name=args.name)
    try:
        engine.run()
    except (KeyboardInterrupt, EOFError):
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
