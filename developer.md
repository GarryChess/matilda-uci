# Developer guide

Integrating Matilda into your own environment — as a Python library, a UCI
subprocess, or a serving backend. For the research background see the README;
for the model's provenance see `src/matilda_uci/matilda/model.py`.

## The layers

```
matilda_uci.UciEngine          UCI text protocol server (stdin/stdout or any streams)
matilda_uci.MatildaPolicy      UCI options/clock latching around the model
matilda_uci.matilda.MatildaModel   the inference facade  <- start here from Python
  ├── Maia3Wrapper             frozen Maia-3 (23M): human prior + features
  ├── TXTC                     the trained 1.7M re-ranker (base_3k.pt)
  └── SearchController         optional engine features (yours to implement)
```

Everything above `MatildaModel` is UCI plumbing; everything below it is the
paper's model, ported and verified bitwise (`scripts/verify_checkpoint.py`).

## Python quickstart

```python
import chess
from matilda_uci.matilda import MatildaModel

model = MatildaModel("checkpoints/base_3k.pt")   # device="mps" works on Apple Silicon

board = chess.Board()
for mv in ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"]:
    board.push_uci(mv)

pred = model.predict(
    board,
    board_history=[],        # preceding FENs, oldest first (see History below)
    elo_self=1500,           # the human being imitated
    elo_oppo=1500,
    tc_base=180, tc_inc=0,   # time control in seconds — a real model input
)
print(sorted(pred.move_probs.items(), key=lambda kv: -kv[1])[:5])
# [('f8c5', 0.327), ('g8f6', 0.302), ('h7h6', 0.145), ...]
print(pred.win_prob)         # Maia-3 expected score for the side to move
```

`predict` returns `None` only when the position has no legal moves. The
probabilities cover every legal move and sum to 1.

**History.** Maia-3 conditions on the trailing 8 positions. If your `board`
carries its `move_stack` (you pushed the moves), use
`matilda_uci.matilda_policy.board_history_fens(board)` to build the list; a
bare FEN with no history is also fine (matches early-game training rows).

**Elos and TC.** Both players' Elos (1000–3200 is in-distribution) and the
time control pair are genuine model inputs. Maia-3 is blitz-trained; 180+0 is
the sane default when the real TC is unknown.

## Custom search controllers

The model's optional engine-feature block consumes, per candidate move:
`(scored?, centipawns, cp-loss-vs-best, rank, is-engine-best)`. Depth and time
are **not** features — which is the entire point: how you spend search budget
across the up-to-16 human candidates is your policy choice.

```python
from matilda_uci.matilda import MoveScore, UciSearchController, DumbSearchController

# Any UCI engine binary; one shared search ranks all candidates (how the
# training annotations were made). nodes>0 switches to a node budget (Lc0).
sf  = UciSearchController("stockfish", depth=12)
lc0 = UciSearchController("lc0 --weights=/path/net.pb.gz", nodes=800)

# The deliberately silly example: every candidate at a random shallow depth.
dumb = DumbSearchController("stockfish", min_depth=2, max_depth=6)

pred = model.predict(board, controller=sf)
print(pred.engine_used)   # True when the controller scored this position
```

To write your own, implement two methods (`matilda_uci.matilda.SearchController`
is the protocol):

```python
class MyController:
    def score(self, board, candidates):    # candidates: list[str] UCIs, best-first
        # Spend your budget however you like; score any subset.
        # cp is from the SIDE TO MOVE's point of view, mates clamped to +/-32000.
        return [MoveScore(uci=candidates[0], cp=42, depth=18)]

    def close(self):
        pass
```

Rules the featurizer enforces for you: unscored candidates get the sentinel
automatically; ranks are derived 1-based by cp descending unless you supply a
complete ranking; a controller that raises degrades that move to the pure
human prior (and a crashed engine subprocess is respawned on the next call).

From the UCI side the same seam is the `EngineCmd`/`EngineDepth`/`EngineNodes`
options.

## Player-style personalization

Three files make a personalized Matilda:

1. **the base model** (`base_3k.pt`) — ships with Matilda;
2. **the style transformation** (`style_token_3k.pt`) — the trained projection
   that turns an embedding into a context nudge; ships with Matilda;
3. **the player's 32-d embedding** — yours to supply at runtime. This is the
   only artifact you produce for a new player, and it "just works":

```python
model = MatildaModel("checkpoints/base_3k.pt")
pid = model.load_style_vector("checkpoints/style_token_3k.pt", "tal.pt")
pred = model.predict(board, pid=pid)     # imitate the supplied player
pred = model.predict(board)              # style-free: no weights needed at all
```

Fit a vector from a PGN of the player's games with
`demos/fit_style_vector.py` (only the 32-d row optimizes; base and
transformation stay frozen — the paper's post-hoc fit; break-even ≈60 moves,
clearly positive past ≈100). Measured on held-out moves: a Tal vector (2000
OTB moves) predicts Tal **7.8% better** (NLL) than the style-free base; a
Magnus vector from his lichess blitz account gains **2.7%**. Via UCI, the
same is `StyleCheckpoint` + `StyleVector`; `demos/style_demo.py` visualizes
per-position distribution shifts.

(Research note: `MatildaModel.load_style` can still mount the paper's full
per-player training tables, but those embeddings were fitted on modest data —
treat them as a rough lower bound, not vendable player models.)

## Driving the UCI engine programmatically

`matilda-uci` is a normal UCI engine, so python-chess can drive it:

```python
import chess.engine
eng = chess.engine.SimpleEngine.popen_uci(["matilda-uci", "--elo", "1800"])
eng.configure({"OpponentElo": 2000, "EngineCmd": "stockfish"})
result = eng.play(chess.Board(), chess.engine.Limit(time=1.0))
```

Notes for integrators:

- **stdout is the protocol channel**; all logging goes to stderr.
- Config errors fail at startup with a clean argparse message; anything that
  slips through surfaces as `info string error: ...` before `bestmove 0000`.
- The engine latches the real time control from the clocks of the first `go`
  of each game (`AutoLatchTC`); send `ucinewgame` between games so it resets.
- Model load happens on the first `go`, not during the handshake — budget a
  few seconds for the first move (Maia-3 downloads from HuggingFace once).

## Embedding in a service

`MatildaModel` is the intended serving surface (one instance per process;
`predict` is not thread-safe — serialize calls or shard instances). The
wrapper/model/controller are all lazily constructed, so cold-start cost is
paid on first prediction. Inject `wrapper=` (any object with the
`Maia3Wrapper.infer` signature) to fake Maia-3 in tests — the whole test
suite runs without model runtimes this way.

Runtime requirements: `torch`, `numpy`, `python-chess`, plus the pinned
Maia-3 package:

```
pip install 'maia3 @ git+https://github.com/CSSLab/maia3.git@1e13597c42d4858b7cfd7cfdae01e297263364b2'
```

Weights: `base_3k.pt` (~6.4 MB, the re-ranker) local; Maia-3 23M (~88 MB)
auto-downloads to `HF_HOME` on first use.
