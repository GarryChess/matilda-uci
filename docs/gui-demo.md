# Using Matilda from a chess GUI

Matilda is a normal UCI engine, so anything that can run Stockfish can run it.
Below: CuteChess (the most common engine tester), cutechess-cli for automated
matches, and En Croissant.

## The one thing to know first

The engine executable is `matilda-uci` (or `python -m matilda_uci` from a
checkout). It needs to find `checkpoints/base_3k.pt`; either start it from the
repo directory or pass `--checkpoint /absolute/path/to/base_3k.pt` as an
engine argument. Bad configuration fails at startup with a clear message —
check the GUI's engine-log/stderr pane if the engine won't start. The first
move takes a few seconds (model load; Maia-3 downloads from HuggingFace once);
the handshake itself is instant.

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
