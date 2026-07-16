#!/usr/bin/env python3
"""Generate checkpoint files in the exact formats Matilda ships.

Uses the faithful ``TXTC`` port in ``matilda_uci.matilda`` (verified against the
read-only paper repo — see ``scripts/verify_checkpoint.py``) to write fresh
(untrained / random-weight) checkpoints whose *structure* — key names, tensor
shapes, dtypes, and the wrapper metadata — is identical to the real files, so
they load and round-trip the same way. Three formats:

    base      TXTC state_dict          == base_3k.pt / base_hi.pt
    style     player-style overlay     == style_token_3k.pt
    posthoc   new-player embeddings    == posthoc_lostyle_3k.pt

Examples
--------
    python scripts/generate_checkpoint.py base    --out /tmp/base.pt
    python scripts/generate_checkpoint.py style   --n-players 6998 --base checkpoints/base_3k.pt --out /tmp/style.pt
    python scripts/generate_checkpoint.py posthoc --n-new 4745 --n-old 6999 --out /tmp/posthoc.pt

The residual head (``dl``) is zero-initialised by default (as in training), so a
generated ``base`` checkpoint is behaviourally *exactly Maia-3* — the documented
untrained state. ``n_players`` follows the paper's convention: the style table
gets ``n_players + 1`` rows (row 0 is the generic fallback).
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import torch

# Make the package importable when run straight from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from matilda_uci.matilda.model import STYLE_DIM, TXTC  # noqa: E402


def generate_base(args: argparse.Namespace) -> object:
    """A TXTC state_dict, like ``base_3k.pt`` (a bare ``OrderedDict``)."""
    torch.manual_seed(args.seed)
    model = TXTC(drop=args.dropout, n_players=args.n_players)
    if args.random_head:
        torch.nn.init.normal_(model.dl.weight, std=0.02)
    return OrderedDict(model.state_dict())


def generate_style(args: argparse.Namespace) -> object:
    """A style-token overlay, like ``style_token_3k.pt``.

    Layout: ``{"state_dict": {spe.weight, spp.weight, spp.bias}, "sdim",
    "n_players", "base"}``. ``spe`` has ``n_players + 1`` rows (row 0 = generic
    fallback); only the style params are saved, to be overlaid on a frozen base.
    """
    torch.manual_seed(args.seed)
    model = TXTC(n_players=args.n_players)
    full = model.state_dict()
    state = OrderedDict(
        (k, full[k]) for k in ("spe.weight", "spp.weight", "spp.bias")
    )
    return {
        "state_dict": state,
        "sdim": STYLE_DIM,
        "n_players": args.n_players,
        "base": args.base,
    }


def generate_posthoc(args: argparse.Namespace) -> object:
    """Post-hoc new-player embeddings, like ``posthoc_lostyle_3k.pt``.

    Layout: ``{"spe_new": (n_new, 32), "n_old": <first new row index>,
    "manifest": <name>, "k": <k-shot count, 0 = full fit>}``. Row ``i`` of
    ``spe_new`` becomes player id ``n_old + i`` when written into a style table.
    """
    torch.manual_seed(args.seed)
    spe_new = torch.randn(args.n_new, STYLE_DIM)
    return {
        "spe_new": spe_new,
        "n_old": args.n_old,
        "manifest": args.manifest,
        "k": args.k,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for weights")
    sub = parser.add_subparsers(dest="format", required=True)

    p_base = sub.add_parser("base", help="TXTC state_dict (base_3k.pt format)")
    p_base.add_argument("--out", required=True)
    p_base.add_argument("--n-players", type=int, default=1)
    p_base.add_argument("--dropout", type=float, default=0.1)
    p_base.add_argument(
        "--random-head", action="store_true",
        help="randomise the residual head instead of zeroing it (default: zero == Maia-3)",
    )
    p_base.set_defaults(func=generate_base)

    p_style = sub.add_parser("style", help="style overlay (style_token_3k.pt format)")
    p_style.add_argument("--out", required=True)
    p_style.add_argument("--n-players", type=int, default=6998)
    p_style.add_argument("--base", default="checkpoints/base_3k.pt")
    p_style.set_defaults(func=generate_style)

    p_ph = sub.add_parser("posthoc", help="new-player embeddings (posthoc format)")
    p_ph.add_argument("--out", required=True)
    p_ph.add_argument("--n-new", type=int, default=4745)
    p_ph.add_argument("--n-old", type=int, default=6999)
    p_ph.add_argument("--manifest", default="lostyle_manifest.json")
    p_ph.add_argument("--k", type=int, default=0)
    p_ph.set_defaults(func=generate_posthoc)

    args = parser.parse_args(argv)
    obj = args.func(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, out)
    print(f"wrote {args.format} checkpoint -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
