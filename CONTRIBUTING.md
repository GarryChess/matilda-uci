# Contributing

Bug reports and fixes are welcome as ordinary issues and PRs. This page is
mostly about the contribution we want to encourage: getting your search
engine natively supported by Matilda.

## Adding a search engine

Matilda's whole design bet is that the search layer is pluggable: Maia-3
proposes the up-to-16 moves a human would consider, and a search controller
decides how to spend engine effort across only those moves. Stockfish and
Lc0 work out of the box through `UciSearchController`; anything that can
score candidate moves can slot in — including allocation policies that
search the human-likely moves deeper instead of running one fixed-depth
search (see the `DumbSearchController` example for the shape of a custom
policy).

What the model consumes per candidate is deliberately narrow: a centipawn
score from the side to move's point of view (mates clamped to ±32000), a
1-based rank, and a scored flag. Depth and time are *not* features — how you
spend the budget is your policy choice. The full protocol and rules are in
[developer.md](developer.md#custom-search-controllers).

### If your engine speaks UCI

You may not need any code: `--engine-cmd "yourengine --your-flags"` already
drives any UCI engine through the shared-search controller. If that works,
a PR adding your engine to the tested list is still valuable — see below.

### Native support

To have your controller ship in this repo:

1. Implement the `SearchController` protocol (`score(board, candidates)` +
   `close()`) in `src/matilda_uci/matilda/search.py` or a new module next
   to it.
2. Add tests that run without your engine installed: the suite fakes
   engines everywhere (see `tests/test_matilda_policy.py` for the pattern) —
   protocol conformance, cp/rank/sentinel conventions, and crash behavior
   (a controller that raises must degrade that move to the human prior, not
   kill the game).
3. If it needs binaries or weights, document where they come from in
   [docs/gui-demo.md](docs/gui-demo.md); never vendor them.
4. Run `python -m pytest` and `python -m ruff check .` — both must be clean.
5. Open a PR labeled **`engine-integration`** describing: what the engine
   is, how the budget is spent across candidates, and a few sample games or
   score() outputs on known positions.

Small, reviewable PRs merge fastest. The repo takes changes through PRs
only (branch protection), so fork or branch as needed.

## Questions

Open an issue, or ask in the [Discord](https://discord.com/invite/RtWdaky4f).
