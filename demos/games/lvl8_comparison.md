# Matilda vs raw Maia-3 against lichess level 8 — 100 games each

Identical setup both runs: Elo 3200, temperature 0.3, OpponentElo left at its
1500 default, opponent = Stockfish as fishnet configures lichess level 8
(Skill Level 20, depth 22, 0.5 s/move), colors alternating, 4 seeds.
"Maia-3" = a zero-delta re-ranker checkpoint, which provably plays the raw
Maia-3 prior; Matilda = `base_3k.pt` + Stockfish depth-12 candidate features.

| engine | W | D | L | score | 95% CI | mean plies |
|---|---|---|---|---|---|---|
| raw Maia-3 prior | 0 | 0 | 100 | 0.0% | [0.0%, 3.7%] | 77 |
| **Matilda (+re-ranker, +engine features)** | **1** | **12** | **87** | **7.0%** | [3.4%, 13.7%] | **116** |

The confidence intervals don't overlap: the re-ranker plus a modest depth-12
engine block converts a total collapse (0/100, games over in ~77 plies) into
real resistance — 13 non-lost games and ~40 extra plies of survival against a
full-strength engine. This is the paper's top-band gain expressed in game
results rather than move-prediction accuracy.

Raw data: `lvl8_sims/` (Matilda) and `maia3_lvl8_sims/` (baseline), each with
per-run `RESULTS.md` from `demos/aggregate_results.py`.
