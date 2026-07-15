# Matilda

A free human-like UCI chess engine.

Unlike Stockfish, Matilda doesn't try to play the *best* move — it plays the
move a **human at a given Elo** would most likely play. It speaks the standard
[UCI](https://www.chessprogramming.org/UCI) protocol, so it plugs into any
chess GUI (CuteChess, Arena, BanksiaGUI, En Croissant) or engine runner that
can talk to Stockfish.

The default backend is **Matilda**: frozen [Maia-3](https://github.com/CSSLab/maia3)
(23M) re-ranked by the paper's trained set-transformer, with optional engine
features from any plugged-in search controller (Stockfish, Lc0, or your own).
The legacy [Maia-2](https://github.com/CSSLab/maia2) backend remains available
via `--backend maia2`.

## Install & run

```bash
pip install -e ".[dev]"    # from a clone (PyPI release TBD)

# the matilda backend additionally needs the pinned Maia-3 runtime:
pip install 'maia3 @ git+https://github.com/CSSLab/maia3.git@1e13597c42d4858b7cfd7cfdae01e297263364b2'

matilda-uci --elo 1500 --checkpoint checkpoints/base_3k.pt
```

Maia-3's 23M weights auto-download from HuggingFace on first use. The
re-ranker checkpoint (`base_3k.pt`, ~6 MB) is required for the default
backend and validated at startup; ask the maintainers or see the paper repo
for the released weights. The UCI handshake itself is instant — models load
on the first move request.

The legacy backend needs neither: `matilda-uci --backend maia2 --maia-type rapid`.

## CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--backend` | `matilda` | `matilda` (Maia-3 + re-ranker) or `maia2` (legacy) |
| `--elo` | 1500 | Playing strength (the Elo the engine imitates) |
| `--opp-elo` | 1500 | The opponent Elo the model conditions on |
| `--temperature` | 0.0 | 0 = always the most-human move; higher = sample |
| `--device` | `cpu` | matilda: `cpu`/`mps`/`cuda`; maia2: `cpu`/`gpu` |
| `--name` | `Matilda` | Engine name reported to the GUI |
| `--log-level` | `WARNING` | Logging goes to stderr (stdout is the UCI channel) |

Matilda-backend flags:

| Flag | Default | Meaning |
|---|---|---|
| `--checkpoint` | `checkpoints/base_3k.pt` | The trained re-ranker weights |
| `--tc-base` / `--tc-inc` | 180 / 0 | Time control fed to the model (blitz default) |
| `--no-auto-tc` | off | Don't latch the real TC from the first `go` clocks |
| `--style-checkpoint` | — | Optional style-token overlay (personalization) |
| `--style-posthoc` | — | Optional post-hoc new-player embeddings |
| `--style-player-id` | -1 | Player row to imitate (-1 = style-free) |
| `--engine-cmd` | — | Search controller, e.g. `stockfish` or `lc0 --weights=...` |
| `--engine-depth` / `--engine-nodes` | 12 / 0 | Controller search budget (nodes>0 = node limit) |
| `--maia3-model` | `23m` | Maia-3 variant |

Legacy maia2-backend flag: `--maia-type` (`rapid`/`blitz`) — only valid with
`--backend maia2`.

## UCI options (matilda backend)

Set from any GUI's engine-options dialog: `UCI_LimitStrength`, `UCI_Elo`
(1000–3200), `OpponentElo`, `TimeControlBase`/`TimeControlInc`/`AutoLatchTC`,
`Temperature`, `Checkpoint`, `StyleCheckpoint`/`StylePosthoc`/`StylePlayerId`,
`EngineCmd`/`EngineDepth`/`EngineNodes`, and `Device`.

The model is rating-conditioned, so `UCI_Elo` maps directly onto playing style
and strength: the engine plays like a human of that rating, including
human-characteristic mistakes. Time control is a real model input — the same
position is played differently at 60+0 than at 900+10 — and is latched
automatically from the clock at the first `go` of each game.

## Architecture

```
UciEngine (protocol loop) ──drives──> MovePolicy (swap seam)
                                        ├── MatildaPolicy (default)
                                        │     └── MatildaModel
                                        │           ├── Maia3Wrapper (frozen 23M prior)
                                        │           ├── TXTC re-ranker (base_3k.pt)
                                        │           └── SearchController (optional:
                                        │               Stockfish / Lc0 / custom)
                                        └── MaiaPolicy (Maia-2, legacy)
```

`MovePolicy` is a small protocol and the deliberate extension point. The
`SearchController` protocol is the second seam: any engine that can score up
to 16 candidate moves plugs in, and without one the model degrades gracefully
to the pure human prior. `go infinite` runs on a worker thread and holds
`bestmove` until `stop`, per the UCI spec.

## Development

```bash
pip install -e ".[dev]"
python -m pytest
python -m ruff check .
PYTHONPATH=src python scripts/verify_checkpoint.py checkpoints  # model parity
```

The test-suite injects fake policies/models/wrappers, so it runs without the
model runtimes or downloads. `scripts/verify_checkpoint.py` proves the ported
model matches the paper implementation bitwise (requires the checkpoints and,
for the parity checks, a clone of the paper repo).

## License

GPL-3.0 — see LICENSE.
