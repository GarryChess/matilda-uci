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

```bash
matilda-uci --elo 1200    # club beginner: plays human 1200 moves, mistakes included
matilda-uci --elo 2000    # strong club player
matilda-uci --elo 2800 --engine-cmd stockfish --engine-depth 16   # top-band, engine-assisted
```

`--elo` changes *style*, not just strength — expect the 1200 to grab material
and miss back-rank ideas, and the 2800 to play principled, theory-heavy chess.
