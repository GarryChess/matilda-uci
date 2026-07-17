# Using Matilda from a chess GUI

Matilda is a normal UCI engine, so anything that can run Stockfish can run it.
Below: BanksiaGUI, CuteChess (the most common engine tester), cutechess-cli
for automated matches, and En Croissant.

## The one thing to know first

From a checkout, point your GUI at **`bin/matilda-uci-local`** — a launcher
that needs no pip install and resolves the checkpoint path itself. (With an
installed package, the executable is `matilda-uci`; then either set the
working directory to the repo or pass `--checkpoint /absolute/path/to/base_3k.pt`.)
Bad configuration fails at startup with a clear message — check the GUI's
engine-log/stderr pane if the engine won't start. The first move takes a few
seconds (model load; Maia-3 downloads from HuggingFace once); the handshake
itself is instant.

## BanksiaGUI

1. **Engines tab (sidebar) → `+` Add** (protocol: UCI).
2. *Command/file*: browse to `<repo>/bin/matilda-uci-local`.
3. Optional *arguments*: e.g. `--elo 3200 --engine-cmd stockfish` for the
   engine-assisted maximum strength, or leave empty and set everything as UCI
   options instead (`UCI_Elo`, `OpponentElo`, `EngineCmd`, `Temperature`, ...)
   in the engine's option editor — Matilda's options appear there after the
   first handshake.
4. To reproduce the **lichess levels** as opponents, add Stockfish three times
   and name the entries "Lichess lvl6/7/8"; in each entry's option editor set
   `Skill Level` to 11 / 16 / 20 respectively (the fishnet mapping; lichess
   also caps depth at 8 / 13 / 22 — use a fixed-depth or short-movetime time
   control in the match dialog to approximate it).
5. **New game / New match**: pick Matilda vs a lichess-level entry, choose the
   time control, and let them play — or seat yourself as one side to play
   against Matilda directly.
6. Give Matilda a real clock (e.g. 3+0): it latches the time control off the
   clock at the first move and genuinely plays differently at different TCs.

Tip: for engine-vs-engine matches set Matilda's `Temperature` to ~0.3 so
repeat games vary; at 0 it deterministically plays the single most-human move.

## CuteChess (GUI)

1. **Tools → Settings → Engines → Add.**
2. *Command*: the full path to `matilda-uci` (find it with `which matilda-uci`;
   it lives in your venv's `bin/`).
3. *Arguments*: e.g. `--elo 1500 --checkpoint /path/to/checkpoints/base_3k.pt`
4. *Working directory*: the repo checkout (then `--checkpoint` can be omitted).
5. OK → **Game → New** → play against it, or pit it against Stockfish.

Engine options (Elo, opponent Elo, time control, temperature, the search
controller, style player) are all editable in the engine's *Configure* dialog —
they're standard UCI options.

## cutechess-cli (automated matches)

```bash
cutechess-cli \
  -engine name=Matilda cmd=/path/to/venv/bin/matilda-uci \
          arg=--elo arg=1800 dir=/path/to/matilda-uci \
  -engine name=Stockfish cmd=stockfish option."Skill Level"=11 \
  -each proto=uci tc=40/60 -rounds 10 -pgnout matilda_vs_sf.pgn
```

Notes:
- `dir=` sets the working directory so the default checkpoint path resolves.
- Matilda reads the real time control off the clock at the first `go` of each
  game (`AutoLatchTC`), so `tc=` genuinely changes how it plays, not just how
  fast it moves.
- Add `arg=--temperature arg=0.3` for variety across rounds — at temperature 0
  it plays the single most-human move every time, so repeat games are identical.

For a ready-made demo match against lichess-level Stockfish (no cutechess
needed, python-chess orchestrates), see `demos/play_vs_stockfish.py`; sample
output PGNs live in `demos/games/`.

## En Croissant

Engines → Add engine → Local → point at `matilda-uci`, set arguments as above.
The analysis view works too: Matilda's "evaluation" is the human-expected
score at the configured Elo (converted to centipawns), and its move list is
the human-probability ranking — a rating-conditioned second opinion next to
Stockfish's objective line.

## Playing it yourself at different strengths

The default is `--elo 1500` (a solid club player); the trained range is
1000–3200. `--elo` changes *style*, not just strength — expect the 1200 to
grab material and miss back-rank ideas, and the 2800 to play principled,
theory-heavy chess. When you don't pass `--opp-elo`, the model assumes the
opponent is your own strength.

```bash
matilda-uci                             # default: Elo 1500, no engine assist
matilda-uci --elo 1200                  # club beginner, mistakes included
matilda-uci --elo 2000 --opp-elo 2400   # club player facing a stronger opponent
matilda-uci --elo 2800 --engine-cmd stockfish --engine-depth 16   # top band
```

Search-controller budgets compose freely (depth, nodes, or a time cap):

```bash
# fixed depth (how the training annotations were made; depth 12 is the default)
matilda-uci --elo 3200 --engine-cmd stockfish --engine-depth 21

# fixed node budget — deterministic cost, the natural mode for Lc0
matilda-uci --elo 3200 --engine-cmd stockfish --engine-nodes 50000

# wall-clock cap per position, alone or combined with a depth ceiling
matilda-uci --elo 3200 --engine-cmd stockfish --engine-movetime 0.5
matilda-uci --elo 3200 --engine-cmd stockfish --engine-depth 21 --engine-movetime 1.0
```

Engine-performance knobs: `--threads N` sets torch's inference threads and
`--cache-size N` the prediction cache (repeated positions skip inference);
both are also UCI options (`Threads`, `CacheSize`) alongside
`EngineDepth`/`EngineNodes`/`EngineMovetime`.

## Bring your own search engine

Any UCI engine can be the search controller — the model consumes only
(centipawns, rank, scored-flag) per candidate, so the backend is swappable:

```bash
# Stockfish (default recommendation)
matilda-uci --elo 3200 --engine-cmd stockfish

# Lc0 — use a node budget; tested with lc0 v0.32 on Metal
matilda-uci --elo 3200 --engine-cmd 'lc0 --weights=/path/to/net.pb.gz' --engine-nodes 800

# anything else that speaks UCI
matilda-uci --elo 3200 --engine-cmd '/path/to/your-engine --your-flags'
```

Notes for non-Stockfish engines:

- Matilda passes **no engine-specific options** to the controller — only the
  moves to score and the search limit — so engines that lack `Hash`/`Threads`
  or other Stockfish-isms work unmodified (option probes are best-effort and
  ignored on failure).
- Lc0 wants `--engine-nodes` rather than depth (its depth semantics differ);
  the training-time Lc0 annotations used 800 nodes.
- Controllers run as separate OS processes on the CPU; `--device` only moves
  Matilda's own weights (Maia-3 + re-ranker + style) to `mps`/`cuda`.
- For a custom *allocation policy* (different budgets per candidate — the
  Botvinnik idea), implement the two-method `SearchController` protocol in
  Python instead: see [developer.md](../developer.md).

## Playing as a specific player (style vectors)

Personalization takes three files: the base model and the **style
transformation** (`style_token_3k.pt`) ship with Matilda; the third is a
32-dimensional embedding of the player you want, supplied at runtime. Fit one
from a PGN of their games:

```bash
.venv/bin/python demos/fit_style_vector.py tal_games.pgn \
    --player "Tal" --out tal.pt
matilda-uci --elo 2700 \
    --style-checkpoint checkpoints/style_token_3k.pt --style-vector tal.pt
```

Measured on held-out moves the fit never saw: the Tal vector predicts Tal's
moves **7.8% better** (NLL) than the style-free base (top-1 60.8% → 62.8%,
2000 OTB moves), and a Magnus vector fitted on his lichess blitz account's
games gains **2.7%** (top-1 59.8% → 61.5%). Without a style vector the engine
runs entirely style-free — the "no style" case needs no weights at all. Fits
break even around ~60 moves of games and are clearly positive past ~100.
