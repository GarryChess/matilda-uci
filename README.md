# Matilda

<img src="assets/botvinnik_1962.jpg" align="right" width="180" alt="Mikhail Botvinnik, 1962">

Long before brute-force search won, world champion **Mikhail Botvinnik**
argued that a chess program should work the way a master does: don't examine
every move — build a small candidate set the way a human would, and spend
your calculation only there. He devoted his post-championship career (the
PIONEER project; *Computers, Chess and Long-Range Planning*, 1970) to
formalizing that selective, human-feature-focused search. Hardware went the
other way, and Botvinnik's programme was shelved.

Matilda's architecture is that idea, made practical by learning the human
part instead of hand-coding it. A frozen human-move model (Maia-3) proposes
the up-to-16 moves a human would actually consider in the position — the
candidate set Botvinnik wanted — and the pluggable search controller
decides how to spend engine effort across *only those moves*. Stockfish and
Lc0 plug in today; the `SearchController` API (see
[developer.md](developer.md)) exists precisely so anyone can implement their
own allocation policy — deeper on the moves that look most interesting for a
human, shallower elsewhere. The re-ranker then blends what the engine found
back into the human distribution. It should "just work" with any controller:
that composability, not any single engine, is the point.

Matilda speaks the standard
[UCI](https://www.chessprogramming.org/UCI) protocol, so it plugs into any
chess GUI (CuteChess, Arena, BanksiaGUI, En Croissant) or engine runner that
can talk to Stockfish. The default backend architecture is a frozen [Maia-3](https://github.com/CSSLab/maia3)
(23M) re-ranked by the paper's trained set-transformer, with optional engine
features from any plugged-in search controller (Stockfish, Lc0, or your own). By 
default we use Stockfish, but we encourage developers to make the engine search build 
off of Maia's attention layers to focus on more human-like engine searches and not always 
do max depth searches.

<sub>Photo: Harry Pot / Anefo, Dutch National Archives — licensed
[CC BY-SA 3.0 NL](https://creativecommons.org/licenses/by-sa/3.0/nl/deed.en).</sub>


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

To use it from a chess GUI, point the "add engine" dialog at the
`matilda-uci` executable — walkthroughs for CuteChess, cutechess-cli, and
En Croissant are in [docs/gui-demo.md](docs/gui-demo.md).


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
| `--style-checkpoint` | — | Style transformation weights (pairs with `--style-vector`) |
| `--style-vector` | — | A 32-d player embedding to imitate (`demos/fit_style_vector.py`) |
| `--engine-cmd` | — | Search controller, e.g. `stockfish` or `lc0 --weights=...` |
| `--engine-depth` / `--engine-nodes` / `--engine-movetime` | 12 / 0 / 0 | Controller search budget |
| `--threads` / `--cache-size` | 0 / 4096 | torch threads; prediction-cache entries |
| `--maia3-model` | `23m` | Maia-3 variant (non-23m warns: untrained-against) |

Legacy maia2-backend flag: `--maia-type` (`rapid`/`blitz`) — only valid with
`--backend maia2`.

## UCI options (matilda backend)

Set from any GUI's engine-options dialog: `UCI_LimitStrength`, `UCI_Elo`
(1000–3200), `OpponentElo`, `TimeControlBase`/`TimeControlInc`/`AutoLatchTC`,
`Temperature`, `Checkpoint`, `StyleCheckpoint`/`StyleVector`,
`EngineCmd`/`EngineDepth`/`EngineNodes`/`EngineMovetime`, `Device`,
`Threads`, and `CacheSize`.

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

## Real games on lichess: Matilda at full strength vs the AI levels

Played **live on lichess.org** against the real server-side AI (via the Board
API — `demos/play_on_lichess.py`): Matilda at Elo 3200 with the Stockfish
search controller, 5+2 clock, one game each color per level.

| Opponent | Games | Score |
|---|---|---|
| lichess level 6 | [win as White, mate](https://lichess.org/UZStPZRC) · [win as Black, mate](https://lichess.org/qzQLv7mU) | **2 – 0** |
| lichess level 7 | [win as White, mate](https://lichess.org/FzmvD0kZ) · [win as Black, mate](https://lichess.org/DZNi54vz) | **2 – 0** |
| lichess level 8 | [draw as White](https://lichess.org/yrGCakOj) · [draw as Black](https://lichess.org/5Zq9shsE) | **1 – 1** |

5/6 against the machine spectrum, all decided over the board (mates and a
193-ply stalemate grind) — checkmating levels 6–7 outright and holding
level 8, essentially full-strength Stockfish, to two draws. For local
engine-vs-engine testing without a lichess account, `demos/play_vs_stockfish.py`
reproduces the same opponents offline; sample PGNs in [demos/games/](demos/games/).

## Demos, docs, numbers

- **[developer.md](developer.md)** — Python API, custom search controllers,
  style personalization, embedding in your own service.
- **[docs/gui-demo.md](docs/gui-demo.md)** — playing it from CuteChess & co.
- **[docs/profiling.md](docs/profiling.md)** — inference throughput by game
  phase (~100 predictions/s on an Apple M3 Pro CPU), rliable bootstrap CIs.
- **[demos/games/](demos/games/)** — sample games vs lichess-level Stockfish;
  regenerate with `demos/play_vs_stockfish.py`.
- **[demos/style_demo.py](demos/style_demo.py)** — measure how player-style
  embeddings condition the policy.

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

## Citation

If you use Matilda in your research, please cite:

```bibtex
@article{carlson2026matilda,
  title  = {Matilda: Engine-Agnostic Search with Human Policy Guidance},
  author = {Carlson, Jason},
  year   = {2026},
  note   = {Preprint},
}
```

To cite this UCI engine / its documentation specifically:

```bibtex
@software{matilda_uci_2026,
  title   = {matilda-uci: a human-like UCI chess engine},
  author  = {Carlson, Jason and Hartfield, Justin},
  year    = {2026},
  url     = {https://github.com/GarryChess/matilda-uci},
}
```
