# Demo games: the Elo × level matrix

Local simulations (clearly not lichess.org games — for those, see the
README's linked live games), generated 2026.07.19 by
`demos/play_vs_stockfish.py`: the real matilda-uci UCI subprocess with its
default configuration (auto-resolved Stockfish controller, depth 22,
TC-derived move budgets, temperature 0.3), two games per cell with colors
alternating, 3s per move for Matilda and the fishnet approximation of the
lichess AI levels for the opponent.

| Matilda \ Opponent | level 6 (skill 11, depth 8) | level 7 (skill 16, depth 13) | level 8 (skill 20, depth 22) |
|---|---|---|---|
| Matilda @ 1500 | 0 - 2 | 0 - 2 | 0 - 2 |
| Matilda @ 2000 | 1 - 1 | 0 - 2 | 0 - 2 |
| Matilda @ 2800 | 2 - 0 | 0 - 2 | 0 - 2 |
| Matilda @ 3200 | 2 - 0 | 2 - 0 | 1 - 1 |

Fishnet levels 6/7/8 sit roughly at 2300/2700/3100 strength, so the
expected picture is exactly this staircase: an Elo-conditioned engine
should lose below the diagonal, contest it, and win above it. Matilda @
1500 losing every game is the point, not a failure — it plays like a 1500,
and a 1500 does not beat a 2300. Two games per cell is a small sample
(temperature 0.3 trades strength for variety), so single-cell swings of a
full point are normal; the 3200 row matches the live lichess results
(sweeps 6 and 7, splits with 8).

Regenerate any row with e.g.:

    python demos/play_vs_stockfish.py --pairings 2800:6 2800:7 2800:8 \
        --games 2 --seed 303 --out demos/games/matched
