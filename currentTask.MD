# matilda-uci — current tasks

Goal: `pip install matilda-uci` then `matilda-uci --elo 1500` just works — engine
loaded, sane time budgets, no manual maia3 install, no checkpoint hunting. The
README has to sell the project on its own (people decide whether to build on it
from that page alone).

Hard deadline: **UCI + Python API solid by Sunday night 2026-07-19** so Justin can
wire it to the website, for the **GM demo Monday 2026-07-20** (play Matilda styled
as Magnus / Tal / Kasparov).

Status context: v0.1.0 is live on PyPI as `matilda-uci`. PR #4 (Jason's review
fixes) is merged. Repo ruleset requires a PR per change; current work branch is
`uci-engine-work`.

---

## Track A — engine & time-control correctness (P0, demo-critical)

- [ ] **Require a search engine at startup; stockfish is the default.**
      `--engine-cmd` defaults to `auto` → resolve `stockfish` on PATH, fail at
      startup with an install link if missing. Explicit opt-out: `--no-engine`
      (CLI) / `engine_cmd=None` (Python). Same semantics in `MatildaPolicy`.
      Why: evals for 3000+ vs 2500 look completely different if the engine
      silently isn't loaded. Default `--engine-depth` goes 12 → 22.
      *Done when: `matilda-uci --elo 1500` with stockfish installed uses it with
      no flags; without stockfish it errors immediately with the install link.*
- [ ] **Time control drives the engine's per-move search time.** New config
      class: classify the TC (base + 40×inc, lichess convention) into
      bullet/blitz/rapid/classical with per-move caps ~2s/15s/30s/60s
      (overridable). If `EngineMovetime` is also set, use the **min** of the
      two. In games, also respect the remaining clock (never more than
      remaining/30) and a GUI `go movetime`. Today the TC only conditions
      Maia-3 and never touches the search budget.
      *Done when: a 60+0 game caps stockfish at ~2s/move, a 300+2 game at
      15s/move, and `go movetime 500` caps it at 0.5s — verified by tests.*
- [ ] **Seeds: fresh entropy per startup, reproducible on request.** Default
      seed becomes None → draw a random seed at init (CLI and Python API) and
      log it; `--seed N` reproduces a run. Today the default is a fixed 0, so
      every game with temperature > 0 replays identically.
- [ ] **Kasparov style vector** for the demo set (have Tal + Magnus already):
      source PGNs (pgnmentor per-player archive), fit with
      `demos/fit_style_vector.py`, save `demos/style_vectors/kasparov.pt`,
      verify it measurably shifts play like the other two.
- [ ] **API smoke for the Sunday hookup:** run the developer.md quickstart
      end-to-end against the new defaults and fix any doc drift.

## Track B — zero-friction install (P1)

- [ ] **Kill the manual maia3 git install.** Publish the pinned maia3 commit to
      PyPI as a credited AGPL package (working name `maia3-runtime`) and depend
      on it in pyproject. PyPI rejects git-URL deps, so this is the clean path;
      no relicensing of matilda-uci needed (GPLv3 + AGPLv3 coexist as separate
      works). **Decision for Jason:** package name, and whether to ping CSSLab
      first.
- [ ] **Checkpoints resolve without a checkout.** First-run auto-download of
      `base_3k.pt` / `style_token_3k.pt` (GitHub release assets →
      `~/.cache/matilda-uci`), so the `--checkpoint` default works after a bare
      pip install. Add git-lfs (`.gitattributes`) for the three small
      checkpoints (~14 MB total) so a git clone works offline.
- [ ] **Maia-3 weights distribution — decision for Jason.** Request was git-lfs
      for the 88 MB maia3 .pt too, but GitHub LFS free bandwidth (~1 GB/month)
      dies at ~11 clones/month. Recommendation: keep the HF auto-download
      (already how the maia3 package works) or mirror as a GitHub release
      asset (unlimited bandwidth). LFS only for our small checkpoints.
- [ ] Version bump + tag when B lands (release.yml publishes on `v*` tags).

## Track C — README & docs (P2)

- [ ] Top-of-README link row like maia3's repo: paper (arXiv) / garrychess
      website / Discord (https://discord.com/invite/RtWdaky4f).
- [ ] Matilda-vs-Maia3 move-prediction numbers by Elo band (from the paper's
      RESULTS.md) near the top.
- [ ] Stockfish install pointers per OS (link to stockfishchess.org/download),
      next to the new engine-required behavior.
- [ ] De-AI formatting pass: kill the odd bolding/highlighting, read it as a
      human wrote it.
- [ ] developer.md: document the release process (tag `vX.Y.Z` → release.yml →
      PyPI trusted publisher; v0.1.0 already live).
- [ ] CONTRIBUTING doc for third-party search-engine integrations: implement
      `SearchController`, open a PR with the `engine-integration` label, what
      tests we expect.

## Track D — demo matrix & evals (P2, after Track A lands)

- [ ] Rerun local demo games as a full matrix: Matilda {1500, 2000, 2800, 3200}
      × lichess levels {6, 7, 8}, 2 games each (24 games) via
      `demos/play_vs_stockfish.py`. Previous numbers were handicapped: opp-elo
      defaulted to 1500 regardless of --elo (fixed in PR #4), and some runs had
      no engine attached — which is exactly the Track A failure mode.
- [ ] Update `demos/games/matched/README.md` + main README table. Keep local
      sims clearly labeled as sims; the real-lichess-games section stays
      separate.

## Track E — website backend (separate repo, with Jason)

Jason's spec, recorded here so it doesn't get lost (implementation lives in the
backend repo, not matilda-uci):

- New lambdas `matilda_inference` / `matilda_training` (Python 3.12 zip +
  SnapStart), replacing byom inference/training when ready. The inference
  lambda privately calls the existing optimize-Stockfish lambda, and holds
  Maia-3 23M + Matilda's ~1.7M-param re-ranker + the style projection.
- Style embeddings live in S3, referenced by v3 byom entries (schema like the
  existing one but pointing at a style embedding instead of LoRA weights);
  fetched on demand into a large (~10k) LRU cache.
- API surface should match byom inference for game start + state semantics.
  **Action item: diff the state semantics against existing byom inference and
  flag any differences to Jason before switching traffic.**
- Migration of existing community bots: scrape the exact S3 files used to
  train the old style embeddings, fit Matilda embeddings from the same data,
  new Dynamo tables. Jason helps here (AWS-heavy / data-sensitive).
