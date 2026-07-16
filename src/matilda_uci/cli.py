from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Sequence

from .engine import UciEngine

_MATILDA_DEVICES = ("cpu", "mps", "cuda")
_MAIA2_DEVICES = ("cpu", "gpu")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="matilda-uci",
        description=(
            "Matilda: a free human-like UCI chess engine "
            "(Maia-3 + trained re-ranker; legacy Maia-2 backend available)."
        ),
    )
    parser.add_argument(
        "--backend", default="matilda", choices=["matilda", "maia2"],
        help="matilda = Maia-3 + the paper's re-ranker (default); maia2 = legacy v1.",
    )
    parser.add_argument("--elo", type=int, default=1500, help="Engine (self) Elo.")
    parser.add_argument("--opp-elo", type=int, default=1500, help="Opponent Elo.")
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="0 = always the most human-likely move; >0 samples for variety.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for temperature sampling (distinct seeds -> distinct games).",
    )
    parser.add_argument("--name", default="Matilda", help="Engine name shown to the GUI.")
    parser.add_argument(
        "--log-level", default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--device", default="cpu",
        help="matilda: cpu/mps/cuda; maia2: cpu/gpu.",
    )

    matilda = parser.add_argument_group("matilda backend")
    matilda.add_argument(
        "--checkpoint", default="checkpoints/base_3k.pt",
        help="TXTC re-ranker checkpoint (the deploy model).",
    )
    matilda.add_argument("--maia3-model", default="23m", dest="maia3_model")
    matilda.add_argument(
        "--tc-base", type=float, default=180.0,
        help="Time-control base seconds fed to the model (default 180+0 blitz).",
    )
    matilda.add_argument("--tc-inc", type=float, default=0.0)
    matilda.add_argument(
        "--no-auto-tc", action="store_true",
        help="Do not latch the real TC from the first `go` clock times.",
    )
    matilda.add_argument(
        "--style-checkpoint", default="",
        help="Optional style-token overlay (e.g. checkpoints/style_token_3k.pt).",
    )
    matilda.add_argument(
        "--style-posthoc", default="",
        help="Optional post-hoc new-player embeddings (posthoc_lostyle_3k*.pt).",
    )
    matilda.add_argument(
        "--style-player-id", type=int, default=-1,
        help="Player row to imitate (-1 = style-free; 0 = generic player).",
    )
    matilda.add_argument(
        "--engine-cmd", default="",
        help="Optional search controller command (e.g. 'stockfish' or "
             "'lc0 --weights=...'); empty = play from the human prior alone.",
    )
    matilda.add_argument("--engine-depth", type=int, default=12)
    matilda.add_argument(
        "--engine-nodes", type=int, default=0,
        help=">0 switches the controller to a fixed node budget (Lc0-style).",
    )

    maia2 = parser.add_argument_group("maia2 backend (legacy)")
    # default=None so we can tell "left alone" from "explicitly requested" and
    # reject the flag under the matilda backend instead of ignoring it.
    maia2.add_argument("--maia-type", default=None, choices=["rapid", "blitz"])
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Fail fast at startup on configuration the engine cannot run with.

    Anything that slips past here only surfaces at the first ``go`` — and the
    engine answers a failed ``go`` with a null move, which GUIs render as a
    cryptic illegal-move loop. A clean argparse error beats that every time.
    """
    if args.backend == "matilda":
        if args.device not in _MATILDA_DEVICES:
            parser.error(
                f"--device {args.device!r} is not valid for the matilda backend "
                f"(choose from {', '.join(_MATILDA_DEVICES)})"
            )
        if args.maia_type is not None:
            parser.error(
                "--maia-type only applies to the legacy Maia-2 backend; "
                "add --backend maia2 (the default backend is the Maia-3 based matilda)"
            )
        if not Path(args.checkpoint).is_file():
            parser.error(
                f"re-ranker checkpoint not found: {args.checkpoint!r} — pass "
                "--checkpoint /path/to/base_3k.pt (see README for where to get it)"
            )
        for flag, path in (("--style-checkpoint", args.style_checkpoint),
                           ("--style-posthoc", args.style_posthoc)):
            if path and not Path(path).is_file():
                parser.error(f"{flag} file not found: {path!r}")
        if importlib.util.find_spec("maia3") is None:
            parser.error(
                "the 'maia3' package is required for the matilda backend; install "
                "the pinned revision:\n  pip install 'maia3 @ git+https://github.com/"
                "CSSLab/maia3.git@1e13597c42d4858b7cfd7cfdae01e297263364b2'"
            )
    else:  # maia2
        if args.device not in _MAIA2_DEVICES:
            parser.error(
                f"--device {args.device!r} is not valid for the maia2 backend "
                f"(choose from {', '.join(_MAIA2_DEVICES)})"
            )


def build_policy(args: argparse.Namespace):
    if args.backend == "maia2":
        from .policy import MaiaPolicy

        return MaiaPolicy(
            elo_self=args.elo,
            elo_oppo=args.opp_elo,
            maia_type=args.maia_type or "rapid",
            device=args.device,
            temperature=args.temperature,
            seed=args.seed,
        )
    from .matilda_policy import MatildaPolicy

    return MatildaPolicy(
        checkpoint=args.checkpoint,
        device=args.device,
        maia3_model=args.maia3_model,
        elo_self=args.elo,
        elo_oppo=args.opp_elo,
        tc_base=args.tc_base,
        tc_inc=args.tc_inc,
        auto_latch_tc=not args.no_auto_tc,
        temperature=args.temperature,
        style_checkpoint=args.style_checkpoint,
        style_posthoc=args.style_posthoc,
        style_player_id=args.style_player_id,
        engine_cmd=args.engine_cmd,
        engine_depth=args.engine_depth,
        engine_nodes=args.engine_nodes,
        seed=args.seed,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    # Logs MUST go to stderr; stdout is the UCI protocol channel.
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        stream=sys.stderr,
        format="%(name)s: %(message)s",
    )
    engine = UciEngine(build_policy(args), name=args.name)
    try:
        engine.run()
    except (KeyboardInterrupt, EOFError):
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
