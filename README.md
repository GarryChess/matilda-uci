# Matilda

A free human-like UCI chess engine.

Unlike Stockfish, Matilda doesn't try to play the *best* move — it plays the
move a **human at a given Elo** would most likely play. It speaks the standard
[UCI](https://www.chessprogramming.org/UCI) protocol, so it plugs into any
chess GUI (CuteChess, Arena, BanksiaGUI, En Croissant) or engine runner that
can talk to Stockfish. Today's backend is the rating-conditioned
[Maia-2](https://github.com/CSSLab/maia2) human-move model.

## Install & run

```bash
pip install -e ".[dev]"    # from a clone (PyPI release TBD)
matilda-uci --elo 1500 --maia-type rapid
```

The process speaks UCI on stdin/stdout. Point your GUI's "add engine" dialog at
the `matilda-uci` executable. The first move request downloads/loads the Maia-2
weights; the UCI handshake itself is instant.

## CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--elo` | 1500 | Playing strength (the Elo the engine imitates) |
| `--opp-elo` | 1500 | The opponent Elo the model conditions on |
| `--maia-type` | `rapid` | `rapid` or `blitz` model variant |
| `--device` | `cpu` | `cpu` or `gpu` |
| `--temperature` | 0.0 | 0 = always the most-human move; higher = sample the human distribution |
| `--name` | `Matilda` | Engine name reported to the GUI |
| `--log-level` | `WARNING` | Logging goes to stderr (stdout is the UCI channel) |

## UCI options

Set from any GUI's engine-options dialog:

| Option | Type | Notes |
|---|---|---|
| `UCI_LimitStrength` | check | Standard flag; on by default |
| `UCI_Elo` | spin | Playing strength, 1100–2000 |
| `OpponentElo` | spin | Opponent conditioning |
| `MaiaType` | combo | `rapid` / `blitz` |
| `Device` | combo | `cpu` / `gpu` |
| `Temperature` | string | Same as the CLI flag |

Maia is rating-conditioned, so `UCI_Elo` maps directly onto playing style and
strength: the engine plays like a human of that rating, including
human-characteristic mistakes.

## Architecture

```
UciEngine (protocol loop)  ──drives──>  MovePolicy (swap seam)
                                            └── MaiaPolicy (Maia-2, v1)
```

`MovePolicy` is a small protocol — `select(board) -> PolicyResult` plus option
plumbing — and is the deliberate extension point: future human-like backends
(Maia-3, style-conditioned models) implement the same surface and slot into the
engine unchanged. `go infinite` runs on a worker thread and holds `bestmove`
until `stop`, per the UCI spec.

## Development

```bash
pip install -e ".[dev]"
python -m pytest
python -m ruff check .
```

The test-suite injects fake policies/wrappers, so it runs without the Maia
runtime or downloads.

## License

TBD.
