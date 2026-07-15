#!/usr/bin/env python3
"""Prove the ported model matches the shipped checkpoints AND the paper code.

Checks, all against the *real* files in ``checkpoints/``:

1. **Strict load** — a fresh ``TXTC`` loads ``base_3k.pt`` with ``strict=True``
   (every key + shape identical; zero missing/unexpected).
2. **Round-trip structure** — a generated checkpoint (all three formats) has
   the same key/shape/dtype layout as the real one.
3. **Zero-delta == Maia-3** — with the residual head zeroed, ``forward``
   reproduces Maia-3's legal-masked log-softmax exactly (the paper's
   untrained-model property).
4. **Paper forward parity** — the paper repo's ``TXTC`` (read-only clone of
   github.com/GarryChess/matilda1-paper) and our port produce IDENTICAL outputs
   from the same ``base_3k.pt`` weights and the same random inputs, for both
   the style-free and style paths. Skipped if the clone is absent.
5. **Paper featurizer parity** — our ``tensors_sf`` reproduces the paper's
   ``tensors_sf`` bit-for-bit on random inputs (with and without engine
   features). Skipped if the clone is absent.

Usage:

    PYTHONPATH=src .venv/bin/python scripts/verify_checkpoint.py [checkpoints_dir] \
        [--paper-repo /path/to/matilda1-paper]

The paper repo path can also come from $MATILDA1_PAPER_DIR.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from matilda_uci.matilda.features import tensors_sf  # noqa: E402
from matilda_uci.matilda.model import (  # noqa: E402
    N_CANDIDATES,
    VOCAB_SIZE,
    TXTC,
    maia3_reference_logprob,
)


def _layout(sd: dict) -> dict[str, tuple]:
    return {k: (tuple(v.shape), str(v.dtype)) for k, v in sd.items() if hasattr(v, "shape")}


def _ok(msg: str) -> None:
    print(f"  \033[32mPASS\033[0m {msg}")


def _fail(msg: str) -> None:
    print(f"  \033[31mFAIL\033[0m {msg}")


def _random_inputs(n: int = 4, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    ml = torch.randn(n, VOCAB_SIZE, generator=g)
    hid = torch.randn(n, 512, generator=g)
    imp = torch.randn(n, 64, generator=g)
    ci = torch.stack(
        [torch.randperm(VOCAB_SIZE, generator=g)[:N_CANDIDATES] for _ in range(n)]
    )
    lm = torch.zeros(n, VOCAB_SIZE)
    for r in range(n):
        lm[r, ci[r]] = 1.0
        lm[r, torch.randperm(VOCAB_SIZE, generator=g)[:20]] = 1.0
    tok = torch.randn(n, N_CANDIDATES, 9, generator=g)
    val = torch.ones(n, N_CANDIDATES)
    val[:, 13:] = 0.0  # a few padded candidate slots
    tcf = torch.randn(n, 2, generator=g)
    return ml, hid, imp, lm, ci, tok, val, tcf


def check_strict_load(ckpt_dir: Path) -> bool:
    print("[1] strict load of base_3k.pt into TXTC")
    real = torch.load(ckpt_dir / "base_3k.pt", map_location="cpu")
    n_players = real["spe.weight"].shape[0] - 1  # paper convention: rows = n+1
    model = TXTC(n_players=n_players)
    result = model.load_state_dict(real, strict=True)
    if result.missing_keys or result.unexpected_keys:
        _fail(f"missing={result.missing_keys} unexpected={result.unexpected_keys}")
        return False
    n_params = sum(p.numel() for p in model.parameters())
    _ok(f"strict=True load clean; {n_params:,} params, spe rows={n_players + 1}")
    return True


def check_roundtrip(ckpt_dir: Path) -> bool:
    print("[2] generated checkpoints match shipped structure")
    from generate_checkpoint import generate_base, generate_posthoc, generate_style

    class NS:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    real_base = _layout(torch.load(ckpt_dir / "base_3k.pt", map_location="cpu"))
    gen_base = _layout(
        generate_base(NS(seed=0, n_players=1, dropout=0.1, random_head=False))
    )
    ok = True
    if gen_base == real_base:
        _ok(f"base: {len(gen_base)} tensors identical (names/shapes/dtypes)")
    else:
        only_real = set(real_base) - set(gen_base)
        only_gen = set(gen_base) - set(real_base)
        diff = {k for k in set(real_base) & set(gen_base) if real_base[k] != gen_base[k]}
        _fail(f"base mismatch: only_real={only_real} only_gen={only_gen} shape_diff={diff}")
        ok = False

    style_path = ckpt_dir / "style_token_3k.pt"
    if style_path.exists():
        real_style = torch.load(style_path, map_location="cpu")
        gen_style = generate_style(
            NS(seed=0, n_players=real_style["n_players"], base=real_style["base"])
        )
        keys_match = set(gen_style) == set(real_style)
        state_match = _layout(gen_style["state_dict"]) == _layout(real_style["state_dict"])
        meta_match = gen_style["sdim"] == real_style["sdim"]
        if keys_match and state_match and meta_match:
            _ok(f"style: top keys {sorted(gen_style)}, spe rows={real_style['n_players'] + 1}")
        else:
            _fail(f"style mismatch keys={keys_match} state={state_match} meta={meta_match}")
            ok = False
    else:
        print("  SKIP style_token_3k.pt not staged (roadmap-only file)")

    ph_path = ckpt_dir / "posthoc_lostyle_3k.pt"
    if ph_path.exists():
        real_ph = torch.load(ph_path, map_location="cpu")
        gen_ph = generate_posthoc(
            NS(seed=0, n_new=real_ph["spe_new"].shape[0], n_old=real_ph["n_old"],
               manifest=real_ph.get("manifest", ""), k=real_ph.get("k", 0))
        )
        if set(gen_ph) == set(real_ph) and gen_ph["spe_new"].shape == real_ph["spe_new"].shape:
            _ok(f"posthoc: keys {sorted(gen_ph)}, spe_new{tuple(real_ph['spe_new'].shape)}")
        else:
            _fail(f"posthoc mismatch: gen_keys={sorted(gen_ph)} real_keys={sorted(real_ph)}")
            ok = False
    else:
        print("  SKIP posthoc_lostyle_3k.pt not staged (roadmap-only file)")
    return ok


def check_zero_delta(ckpt_dir: Path) -> bool:
    print("[3] zero-delta output == Maia-3 masked log-softmax")
    real = torch.load(ckpt_dir / "base_3k.pt", map_location="cpu")
    model = TXTC(n_players=real["spe.weight"].shape[0] - 1)
    model.load_state_dict(real, strict=True)
    model.reset_nudge_head()  # force the untrained invariant regardless of training
    model.eval()

    ml, hid, imp, lm, ci, tok, val, tcf = _random_inputs()
    with torch.no_grad():
        out = model(ml, hid, imp, lm, ci, tok, val, tcf, pid=None, style=False)
        ref = maia3_reference_logprob(ml, lm)
    if torch.equal(out, ref):
        _ok("identical to Maia-3 with zeroed head (bitwise)")
        return True
    max_abs = (out - ref).abs().max().item()
    _fail(f"diverges from Maia-3: max|delta|={max_abs:.3e}")
    return False


def _load_paper_module(paper_repo: Path, name: str):
    for sub in ("pipeline/train", "pipeline/lib"):
        p = str(paper_repo / sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    import importlib

    return importlib.import_module(name)


def check_paper_forward_parity(ckpt_dir: Path, paper_repo: Path) -> bool | None:
    print("[4] forward parity vs the paper repo's TXTC (same weights, same inputs)")
    if not (paper_repo / "pipeline/train/train_base_tc.py").exists():
        print(f"  SKIP paper repo not found at {paper_repo}")
        return None
    import torch.nn as nn

    paper = _load_paper_module(paper_repo, "train_base_tc")
    sd = torch.load(ckpt_dir / "base_3k.pt", map_location="cpu")

    # Paper-side construction, exactly as the checkpoint READMEs prescribe.
    pm = paper.TXTC(heads=8)
    pm.tp = nn.Linear(4 + 5 + 32, pm.tp.out_features)
    pm.load_state_dict(sd, strict=True)
    pm.eval()

    ours = TXTC()  # defaults are the shipped configuration
    ours.load_state_dict(sd, strict=True)
    ours.eval()

    ml, hid, imp, lm, ci, tok, val, tcf = _random_inputs()
    ok = True
    with torch.no_grad():
        a = pm(ml, hid, imp, lm, ci, tok, val, tcf, pid=None, style=False)
        b = ours(ml, hid, imp, lm, ci, tok, val, tcf, pid=None, style=False)
    if torch.equal(a, b):
        _ok("style-free outputs bitwise identical")
    else:
        _fail(f"style-free outputs differ: max|delta|={(a - b).abs().max().item():.3e}")
        ok = False

    # Style path: the shipped base has a zero (frozen) style projection, so give
    # both models the same random style params to exercise the path for real.
    g = torch.Generator().manual_seed(7)
    spe = torch.randn(sd["spe.weight"].shape, generator=g)
    spp_w = torch.randn(sd["spp.weight"].shape, generator=g)
    spp_b = torch.randn(sd["spp.bias"].shape, generator=g)
    for m in (pm, ours):
        with torch.no_grad():
            m.spe.weight.copy_(spe)
            m.spp.weight.copy_(spp_w)
            m.spp.bias.copy_(spp_b)
    pid = torch.tensor([0, 1, 1, 0])
    with torch.no_grad():
        a = pm(ml, hid, imp, lm, ci, tok, val, tcf, pid=pid, style=True)
        b = ours(ml, hid, imp, lm, ci, tok, val, tcf, pid=pid, style=True)
    if torch.equal(a, b):
        _ok("style-conditioned outputs bitwise identical")
    else:
        _fail(f"style outputs differ: max|delta|={(a - b).abs().max().item():.3e}")
        ok = False
    return ok


def check_paper_featurizer_parity(paper_repo: Path) -> bool | None:
    print("[5] featurizer parity vs the paper repo's tensors_sf")
    if not (paper_repo / "pipeline/train/train_sf_smoke.py").exists():
        print(f"  SKIP paper repo not found at {paper_repo}")
        return None
    paper_sf = _load_paper_module(paper_repo, "train_sf_smoke")

    rng = np.random.RandomState(3)
    n = 32
    d = {
        "maia_logits": rng.randn(n, VOCAB_SIZE).astype(np.float32),
        "maia_hidden": rng.randn(n, 512).astype(np.float32),
        "maia_importance": rng.randn(n, 8, 8).astype(np.float32),
        "legal_mask": (rng.rand(n, VOCAB_SIZE) < 0.01).astype(np.int8),
        "cand_idx": rng.randint(-1, VOCAB_SIZE, (n, N_CANDIDATES)).astype(np.int64),
        "cand_logit": (-rng.rand(n, N_CANDIDATES) * 8).astype(np.float32),
        "cand_valid": (rng.rand(n, N_CANDIDATES) < 0.9).astype(np.float32),
        "target": rng.randint(0, VOCAB_SIZE, n).astype(np.int64),
        "elo_self": rng.randint(1000, 3200, n).astype(np.int32),
        "tc_base": rng.choice([60, 180, 300, 600], n).astype(np.int32),
        "tc_inc": rng.choice([0, 1, 2, 5], n).astype(np.int32),
        "sf_cp": np.where(rng.rand(n, N_CANDIDATES) < 0.7,
                          rng.randint(-800, 800, (n, N_CANDIDATES)),
                          -32001).astype(np.int32),
        "sf_rank": rng.randint(0, 17, (n, N_CANDIDATES)).astype(np.int8),
        "sf_valid": (rng.rand(n) < 0.8).astype(np.int8),
    }
    ok = True
    for use_sf in (True, False):
        theirs = paper_sf.tensors_sf(d, use_sf)
        mine = tensors_sf(d, use_sf)
        for key in ("ml", "hid", "imp", "lm", "ci", "tok", "val", "tcf", "tg"):
            if not torch.equal(theirs[key], mine[key]):
                delta = (theirs[key].float() - mine[key].float()).abs().max().item()
                _fail(f"use_sf={use_sf} key '{key}' differs (max|delta|={delta:.3e})")
                ok = False
    if ok:
        _ok("all tensor outputs bitwise identical (use_sf=True and False)")
    return ok


def main(argv: list[str] | None = None) -> int:
    default_paper = os.environ.get(
        "MATILDA1_PAPER_DIR",
        str(Path(__file__).resolve().parent.parent.parent / "matilda1-paper"),
    )
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ckpt_dir", nargs="?", default="checkpoints")
    parser.add_argument("--paper-repo", default=default_paper,
                        help="path to a read-only clone of matilda1-paper")
    args = parser.parse_args(argv)

    ckpt_dir = Path(args.ckpt_dir)
    if not (ckpt_dir / "base_3k.pt").exists():
        print(f"error: {ckpt_dir}/base_3k.pt not found", file=sys.stderr)
        return 2
    paper_repo = Path(args.paper_repo)

    results = [
        check_strict_load(ckpt_dir),
        check_roundtrip(ckpt_dir),
        check_zero_delta(ckpt_dir),
        check_paper_forward_parity(ckpt_dir, paper_repo),
        check_paper_featurizer_parity(paper_repo),
    ]
    print()
    hard = [r for r in results if r is not None]
    skipped = results.count(None)
    if all(hard):
        note = f" ({skipped} skipped)" if skipped else ""
        print(f"\033[32mALL CHECKS PASSED\033[0m{note} — port is faithful.")
        return 0
    print("\033[31mSOME CHECKS FAILED\033[0m")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
