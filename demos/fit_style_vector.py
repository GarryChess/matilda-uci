#!/usr/bin/env python3
"""Fit a 32-d style vector for a player from a PGN of their games.

The third file of the personalization design: the base model and the style
transformation ship with Matilda; this script produces the player-specific
embedding. Only the single 32-d vector is optimized — the base model and the
transformation stay frozen, exactly like the paper's post-hoc embedding fits
(break-even around ~60 player moves, clearly positive by ~100+).

    .venv/bin/python demos/fit_style_vector.py games.pgn --player "Tal, Mihail" \
        --out demos/style_vectors/tal.pt

Then play as that player:

    matilda-uci --style-checkpoint checkpoints/style_token_3k.pt \
                --style-vector demos/style_vectors/tal.pt

A held-out split reports the honest gain: NLL / top-1 of the styled model vs
the style-free base on moves the fit never saw.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from matilda_uci.matilda.model import N_CANDIDATES, TXTC  # noqa: E402
from matilda_uci.matilda.move_vocab import legal_mask, move_index  # noqa: E402


def collect_decisions(pgn_paths: list[str], player: str, max_moves: int,
                      default_elo: int) -> list[dict]:
    """(board, history, played move, elos) for every move by ``player``."""
    import chess.pgn

    decisions: list[dict] = []
    for path in pgn_paths:
        with open(path, encoding="utf-8", errors="replace") as fh:
            while len(decisions) < max_moves:
                game = chess.pgn.read_game(fh)
                if game is None:
                    break
                h = game.headers
                white = h.get("White", "")
                black = h.get("Black", "")
                if player.lower() in white.lower():
                    ours = True
                elif player.lower() in black.lower():
                    ours = False
                else:
                    continue

                def elo(tag: str) -> int:
                    try:
                        return int(h.get(tag, ""))
                    except ValueError:
                        return default_elo

                board = game.board()
                fens = [board.fen()]
                for move in game.mainline_moves():
                    my_turn = board.turn if ours else not board.turn
                    if my_turn:
                        decisions.append({
                            "fen": board.fen(),
                            "history": fens[-9:-1] if len(fens) > 1 else [],
                            "played": move.uci(),
                            "elo_self": elo("WhiteElo" if ours else "BlackElo"),
                            "elo_oppo": elo("BlackElo" if ours else "WhiteElo"),
                        })
                    board.push(move)
                    fens.append(board.fen())
                    if len(decisions) >= max_moves:
                        break
    return decisions


def featurize(decisions: list[dict], maia3_model: str, tc: tuple[float, float],
              device: str) -> dict:
    """Maia-3 features for every decision, matching the training featurizer."""
    import chess

    from matilda_uci.matilda.maia3_wrapper import Maia3Wrapper

    wrap = Maia3Wrapper(model=maia3_model, device=device, top_k=N_CANDIDATES,
                        capture_hidden=True, capture_importance=True)
    cols: dict[str, list] = {k: [] for k in (
        "maia_logits", "maia_hidden", "maia_importance", "legal_mask",
        "cand_idx", "cand_logit", "cand_valid", "target", "tc_base", "tc_inc")}
    kept = 0
    for i, dec in enumerate(decisions):
        board = chess.Board(dec["fen"])
        white = board.turn == chess.WHITE
        target = move_index(dec["played"], white)
        if target is None:
            continue
        r = wrap.infer(board, elo_self=dec["elo_self"], elo_oppo=dec["elo_oppo"],
                       board_history=dec["history"])
        if r.hidden is None:
            continue
        cand = [u for u, _ in r.top_moves[:N_CANDIDATES]]
        idx = np.full(N_CANDIDATES, -1, np.int64)
        lg = np.zeros(N_CANDIDATES, np.float32)
        valid = np.zeros(N_CANDIDATES, np.float32)
        for j, u in enumerate(cand):
            vi = move_index(u, white)
            if vi is None:
                continue
            idx[j] = vi
            lg[j] = r.logit(u)
            valid[j] = 1.0
        if valid.sum() == 0:
            continue
        imp = (np.asarray(r.importance, np.float32).reshape(8, 8)
               if r.importance is not None else np.zeros((8, 8), np.float32))
        cols["maia_logits"].append(np.asarray(r.logits, np.float16).reshape(-1)[:4352])
        cols["maia_hidden"].append(np.asarray(r.hidden, np.float16).reshape(-1))
        cols["maia_importance"].append(imp.astype(np.float16))
        cols["legal_mask"].append(legal_mask(board).astype(np.int8))
        cols["cand_idx"].append(idx)
        cols["cand_logit"].append(lg)
        cols["cand_valid"].append(valid)
        cols["target"].append(int(target))
        cols["tc_base"].append(tc[0])
        cols["tc_inc"].append(tc[1])
        kept += 1
        if kept % 100 == 0:
            print(f"  featurized {kept}/{len(decisions)}", flush=True)
    wrap.close()
    return {k: np.stack(v) if k != "target" else np.asarray(v, np.int64)
            for k, v in cols.items()}


def fit(d: dict, base_ckpt: str, style_ckpt: str, *, holdout: float,
        steps: int, lr: float, seed: int) -> tuple:
    """Optimize the single embedding row; return (vector, metrics)."""
    import torch

    from matilda_uci.matilda.features import tensors_sf

    tok = torch.load(style_ckpt, map_location="cpu")
    sd = torch.load(base_ckpt, map_location="cpu")
    model = TXTC(sdim=int(tok["sdim"]), n_players=1)
    model.load_state_dict({k: v for k, v in sd.items() if not k.startswith("sp")},
                          strict=False)
    spp_only = {k: v for k, v in tok["state_dict"].items() if not k.startswith("spe")}
    model.load_state_dict(spp_only, strict=False)
    with torch.no_grad():
        model.spe.weight[0] = tok["state_dict"]["spe.weight"][0]
        model.spe.weight[1].zero_()  # start from no style
    model.eval()  # dropout off: only one row trains, features are fixed

    for p in model.parameters():
        p.requires_grad = False
    model.spe.weight.requires_grad = True

    T = tensors_sf(d, use_sf=True)
    tg = T["tg"]
    n = len(tg)
    rng = np.random.RandomState(seed)
    order = rng.permutation(n)
    n_hold = max(1, int(n * holdout))
    hold, train = order[:n_hold], order[n_hold:]
    pid = torch.ones(n, dtype=torch.long)

    def logp(ix):
        return model(T["ml"][ix], T["hid"][ix], T["imp"][ix], T["lm"][ix],
                     T["ci"][ix], T["tok"][ix], T["val"][ix], T["tcf"][ix],
                     pid=pid[ix], style=True)

    def nll(ix, styled=True):
        with torch.no_grad():
            lp = logp(ix) if styled else model(
                T["ml"][ix], T["hid"][ix], T["imp"][ix], T["lm"][ix], T["ci"][ix],
                T["tok"][ix], T["val"][ix], T["tcf"][ix], pid=None, style=False)
            per = -lp.gather(1, tg[ix, None]).squeeze(1)
            top1 = (lp.argmax(1) == tg[ix]).float().mean().item()
        return per.mean().item(), top1

    base_nll, base_top1 = nll(hold, styled=False)
    opt = torch.optim.Adam([model.spe.weight], lr=lr)
    batch = min(256, len(train))
    for step in range(steps):
        ix = torch.tensor(rng.choice(train, size=batch, replace=False))
        loss = -logp(ix).gather(1, tg[ix, None]).squeeze(1).mean()
        opt.zero_grad()
        loss.backward()
        # only the player's row may move
        model.spe.weight.grad[0].zero_()
        opt.step()
        if (step + 1) % 50 == 0:
            h_nll, h_top1 = nll(hold)
            print(f"  step {step + 1}: train_loss={loss.item():.4f} "
                  f"holdout_nll={h_nll:.4f} top1={h_top1:.3f}", flush=True)

    styled_nll, styled_top1 = nll(hold)
    vector = model.spe.weight[1].detach().clone()
    metrics = {
        "moves_total": n, "moves_train": len(train), "moves_holdout": len(hold),
        "holdout_nll_stylefree": round(base_nll, 4),
        "holdout_nll_styled": round(styled_nll, 4),
        "nll_gain_pct": round(100 * (base_nll - styled_nll) / base_nll, 2),
        "holdout_top1_stylefree": round(base_top1, 4),
        "holdout_top1_styled": round(styled_top1, 4),
    }
    return vector, metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pgns", nargs="+", help="PGN files containing the player's games")
    ap.add_argument("--player", required=True,
                    help="substring matched against the White/Black headers")
    ap.add_argument("--out", required=True, help="output .pt for the 32-d vector")
    ap.add_argument("--base", default="checkpoints/base_3k.pt")
    ap.add_argument("--style-checkpoint", default="checkpoints/style_token_3k.pt")
    ap.add_argument("--max-moves", type=int, default=2000)
    ap.add_argument("--default-elo", type=int, default=2700,
                    help="Elo assumed when PGN headers lack ratings (OTB games)")
    ap.add_argument("--tc", default="5400+30",
                    help="time control fed to the model (classical default)")
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--maia3-model", default="23m")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import torch

    base, _, inc = args.tc.partition("+")
    decisions = collect_decisions(args.pgns, args.player, args.max_moves,
                                  args.default_elo)
    if len(decisions) < 30:
        print(f"error: only {len(decisions)} moves found for {args.player!r}; "
              "need at least ~60 to break even", file=sys.stderr)
        return 2
    print(f"{len(decisions)} decisions by {args.player!r}; featurizing via Maia-3...")
    d = featurize(decisions, args.maia3_model, (float(base), float(inc or 0)),
                  args.device)
    print(f"fitting 32-d vector on {len(d['target'])} featurized moves...")
    vector, metrics = fit(d, args.base, args.style_checkpoint,
                          holdout=args.holdout, steps=args.steps, lr=args.lr,
                          seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vector, out)
    print("\n=== results (held-out moves) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\nvector -> {out}")
    print(f"play it: matilda-uci --style-checkpoint {args.style_checkpoint} "
          f"--style-vector {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
