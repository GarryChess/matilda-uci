    Use those models in the UCI-engine
    I think you just need the elite one and not the style embeddings but double check the readme
    Style embeddings would be user-supplied which can be a separate roadmap feature
    I’ll send you the architecture diagram in a few so you can get a feel

It should have some demo of using it with a tool ppl commonly use to test uci engines, a way for people to include their own style embeddings with some stats showing it works, checks for correct maia-3 variant features / warnings if supplied wrong, works with external stockfish, shares stockfish if it creates one.

it should optionally allow one to override stockfish with Lc0 or a custom search engine that follows a given api… double check architecture etc to know what this api should be but basically they can provide up to 16 moves (not necessarily top) for a given position each with different depths and one other thing (maybe time limit used along with the centipawn (maybe also win percent score) score obtained from each respective move. Those end up as features to the set transformer.

The big idea with the architecture is anyone can plug in any search controller like stockfish and it should “just work” so people can eventually implement their own custom search protocols based on the moves that look most interesting for a human in the given position. In the readme make sure this is clear and relate it historically to Mikhail botvirnnik’s grand vision for how chess can be programmed using this idea of human feature focused search.

some examples of it being used with python and installation instructions / examples
example of the different engines including a custom “dumb one” that is like stockfish but it does like random depths for each of the top 16 or something (lots of low depth)

performance profiling on ur machine with X threads (predictions per second for opening vs middle game vs endgame ply tranches from random rated blitz game positions from 2026 in any given month-day… not that much data is needed for it to be statistically significant, use rliable for generating stats and plots).

developer.md for developer integration into their envs.

some game demos of you playing against it tuned at various Elos / having it play against lichess’ lvl6-8

citation at the bottom of the readme (I think it has this already)

reddit post when ready (once AAAI submission is accepted)

Botvinniki photo maybe included with credit acknowledgment for the photo (https://en.wikipedia.org/wiki/Mikhail_Botvinnik#/media/File:Mikhail_Botvinnik_1962.jpg)

Move history: yes, but only through Maia-3. The featurizer feeds Maia-3 the position plus the trailing 8 positions (board_history, HIST=8); Maia-3 (the Chessformer) conditions on that history. Our re-ranker never sees history directly — it arrives pre-digested in Maia-3's logits/hidden/attention. UCI handles this for free: every go comes after position startpos moves e2e4 ..., so the engine always has the move list and the wrapper reconstructs the last 8 positions. The only edge case is analysis from a bare FEN with no history — then the history is empty, which is exactly what early-game training rows look like, so degradation is graceful and in-distribution.

Time control: yes, directly into our model — and Maia-3 does not take it. The TC pair (log1p(base), log1p(inc)) is two of the context encoder's input dims; Maia-3 consumes only position + history + the two Elos. For the UCI engine: the protocol never transmits base TC explicitly, but at move one wtime ≈ base and winc = inc, so the engine can latch them at game start; a hosted lichess bot gets TC from game metadata anyway; and a setoption fallback plus a sane default (e.g. 180+0) covers bare analysis — TC sensitivity is real but modest. One disambiguation worth engraving: time control (base+inc) is an input; remaining clock time (clk_self) is not — we only recorded clock for the time-pressure analyses. And a third input in the same category you didn't ask about but your friend will hit: Maia-3 needs both player Elos, which UCI also doesn't provide — same solution (options/metadata, defaults).

￼

We’d just make sure it’s in parity for some data with the paper model

Based on the checkpoint files, write me python files to generate checkpoint files like that and we need to spin up a virtual env to test these.

The matilda-uci package should be able to be installed with pip install matilda. Here is the details from Pypi:

Pending project name
Publisher
Details
matilda-uci
GitHub
Repository: jcarlson212/matilda-uci (https://github.com/jcarlson212/matilda-uci)
Workflow: release.yml
Environment name: (Any)
Remove (https://pypi.org/manage/account/publishing/#remove-publisher-ada272a4-b658-4831-9faf-8ea7e8c46d60)

Main idea: just tell fable to study read only the Matilda1 paper repo and then implement x,y,x inference engine in Matilda-uci
