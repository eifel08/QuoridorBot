# Bot Evolution: design analysis and candidate selection (Phase 2)

Status: **plan only, no implementation yet.** Waiting for review before Phase 3.

## 1. What the current engine already does

I read `engine/src/*`, `engine/README.md`, `quoridor/bots/*`, `quoridor/ml/*`, `tools/*` and
the Colab notebook. The C++ engine is already strong and well instrumented:

| Area | Already in the baseline |
|---|---|
| Board | 81-bit bitboards, make/unmake, incremental Zobrist, bitboard flood-fill BFS, legality shortcut via shortest-path cut masks, planar `may_disconnect` test |
| Search | Fail-soft negamax alpha-beta, iterative deepening, 2-way-bucket TT (64 MB), TT-move, shortest-path pawn moves, killers, history, lazy staged generation |
| Pruning | Wall filter (attack + local/global/funnel defense), route-zone relevance, futility pruning, decided-race cutoffs, lock proofs, one-ply leaf quiescence ("use it or lose it") |
| Eval | `10 * path diff` + blockability-aware wall value (bottleneck/soft layers, greedy chain of up to 3 simulated blocks, capped reserve, surplus/def/tie values) + tempo |
| ML | Python AlphaZero-style ResNet (6x64) + MCTS, imitation warm-up then RL against the Python depth-4 bot. Best result in its README: **~49% vs the *Python* depth-4 bot**, which is far below the C++ engine. No trained weights are committed. |

### What it does NOT have (measured or read)

1. **Wastes almost half its clock.** Iterative deepening refuses to start a new depth after 50%
   of the limit. Measured at 5 s/move: start position finishes depth 11 at **2.86 s**, then
   stops; a middlegame position stops at 3.5 s. There's no estimate of the next iteration's cost
   and no use of partial iterations.
2. **Odd/even score oscillation.** Start position: depth 1-11 scores are `20, 5, 20, 5, ... 19`.
   The tempo term flips with the parity of the horizon, which hurts TT reuse across iterations and
   makes aspiration windows useless as things stand.
3. **No PVS, aspiration windows, LMR, null move, or extensions.** With a branching factor of
   ~30-130 (pawn moves plus filtered walls), late-move reductions alone are typically worth more
   than a ply.
4. **Single-threaded**, on a 4-core machine.
5. **No opening book.** Every game starts with an ~11-ply search from scratch, and the opening
   is where it's shallowest relative to the game's length.
6. **Eval weights are hand-set**, tuned one at a time with 200-400-game matches. There's no joint
   tuning against game outcomes.
7. **No repetition detection** (games end at a ply cap as a "draw").
8. **No ML eval inside the fast engine.** The ML agent lives in Python with ~100 evals/s, against
   ~4-5 M nodes/s in C++.
9. **Build flags**: `-O3` only, so no `-march=native`, LTO or PGO.

## 2. ~50 improvement ideas, scored

**Elo** is a prior estimate of the gain at 5 s/move against the current baseline, before
measuring. Treat it as a ranking signal, not a promise. **Cost** is the runtime overhead per node
(L = negligible, M = noticeable, H = large). **Feas.** is implementation feasibility in this
project (5 = easy, 1 = research project). **Tier**: A = build it, B = build it if cheap or as
part of a variant, C = rejected or deferred.

### A. Search and speed

| # | Idea | Elo | Cost | Feas. | Tier | Notes |
|---|---|---|---|---|---|---|
| 1 | **Smart time management**: predict the next iteration's time from the effective branching factor, use up to ~95% of 5 s, keep the best move of a partial iteration once the PV move is resolved | +40-70 | L | 5 | **A** | Measured waste of 40-45% of the clock. Cheapest big win. |
| 2 | **Principal Variation Search** (null-window re-search) | +15-30 | L | 5 | **A** | Standard, safe with the existing TT. |
| 3 | **Aspiration windows** around the previous score | +5-15 | L | 4 | **A** | Needs #4 (parity fix) to pay off. |
| 4 | **Parity-consistent tempo** (score always from the same side-to-move convention at the horizon / half-tempo at odd leaves) | +5-15 | L | 5 | **A** | Fixes the 20/5 oscillation; also makes TT hits across depths consistent. |
| 5 | **Late Move Reductions** for late quiet walls (non-attack, non-defense-top) and non-shortening pawn moves; re-search on fail-high | +40-80 | L | 4 | **A** | Biggest search gain expected; must not reduce attack walls, the TT move or killers. |
| 6 | Null-move pruning (R=2-3), disabled when the opponent can block a fragile path or walls are low | +0-30 | L | 3 | B | Quoridor has no pass move, but "having the move is good" mostly holds. Zugzwang-like wall races make it risky; gate it and measure. |
| 7 | Countermove + continuation history | +5-15 | L | 4 | B | Cheap ordering gain. |
| 8 | TT upgrade: 4-way buckets, 256 MB, store static eval, better replacement | +5-15 | L | 5 | **A** | Longer thinks at 5 s fill 64 MB. |
| 9 | Compiler: `-march=native`, LTO, PGO | +5-15 | none | 5 | **A** | 10-25% nps for free. |
| 10 | **Lazy SMP** (4 threads, shared TT) | +60-100 | none | 4 | B | Real gain, but only fair if every bot gets the same cores; see the open decisions. |
| 11 | Pondering (think on the opponent's time) | +40-70 | none | 3 | C | Unfair in a shared-CPU tournament; skip. |
| 12 | "Must-block" extension: extend by 1 ply when the opponent wins the race in <= 2 and I have walls | +10-25 | L | 4 | **A** | Tactical horizon fix for last-chance blocks. |
| 13 | Singular extension on the TT move | +0-15 | M | 3 | C | Complex; small expected gain here. |
| 14 | Internal iterative reduction when there's no TT move | +5-10 | L | 5 | B | |
| 15 | ProbCut / multi-cut | +0-15 | M | 3 | C | |
| 16 | Repetition detection (two-fold in search = draw score) | +0-10 | L | 5 | **A** | Correctness; stops pawn shuffles that run into the ply cap. |
| 17 | Incremental shortest-path DAG maintenance (dynamic BFS) instead of a fresh flood per node | +5-15 | - | 2 | C | Floods are already bitboard-fast; complex for a small gain. |
| 18 | AVX2 flood fill (both 64-bit halves in one 128-bit register) | +0-10 | - | 3 | C | Marginal; compiler flags get most of it. |
| 19 | Root re-search of the top 2 moves with an extra ply when they're close | +5-10 | M | 4 | B | |
| 20 | Move-count pruning of very late quiet walls at depth <= 2 | +10-25 | L | 4 | B | Pairs with LMR. |

### B. Heuristics and domain knowledge

| # | Idea | Elo | Cost | Feas. | Tier | Notes |
|---|---|---|---|---|---|---|
| 21 | **Texel tuning**: jointly fit all eval weights by logistic regression of game outcome on the eval, using a large set of quiet self-play positions | +30-60 | none | 5 | **A** | The weights were tuned one at a time; joint tuning is the classic big win. |
| 22 | **Path flexibility**: width of the shortest-path DAG (min edge cut between layers) as a continuous term, also for paths longer than `smart_dist` (now assumed "rich") | +15-30 | M | 4 | **A** | Extends the existing blockability analysis to the opening and middlegame. |
| 23 | Number of distinct shortest paths (log count via DAG DP) | +5-15 | L | 5 | B | Cheap, but correlates with #22. |
| 24 | Second-best path length (detour cost if the best route is cut) | +10-20 | M | 4 | B | Generalises the greedy chain to long paths. |
| 25 | **Phase-tapered eval**: separate weights by total walls left (opening/middlegame/race) | +15-30 | L | 5 | **A** | Tuned with #21. |
| 26 | Exact race parity term when both sides have few walls (who moves first in a tie) | +5-10 | L | 5 | B | Partly covered by `decided`. |
| 27 | Pawn centrality / column-freedom term (opening) | +5-15 | L | 5 | B | Mostly matters before the walls start; the book covers it too. |
| 28 | Attack mobility: count of legal walls that lengthen the opponent's path, and their total gain | +10-20 | M | 4 | B | Feature for the ML eval. |
| 29 | Own-lane / corridor building bonus (locked or unblockable own path) | +5-15 | L | 4 | B | Partly via `locked`. |
| 30 | Face-off / jump term (adjacent pawns; who gets the jump) | +5-10 | L | 4 | B | |
| 31 | Wall-waste forcing: credit positions where the opponent's remaining walls are locked out | +5-10 | L | 4 | B | Exists via `lock`; tune its value. |
| 32 | Minimum walls needed to make the opponent's path longer than mine (min-cut style) | +10-30 | H | 2 | C | Too expensive per node. |
| 33 | Cut-vertex / articulation-square detection on the route zone | +5-15 | M | 3 | C | Overlaps with bottleneck analysis. |

### C. Opening book

| # | Idea | Elo | Cost | Feas. | Tier | Notes |
|---|---|---|---|---|---|---|
| 34 | **Engine-generated book**: expand a tree from the start position (both colours) for 8-12 plies; each node is scored by a long search (30-60 s, depth ~14) and kept if within a margin of the best | +30-60 | none | 4 | **A** | The real source of opening strength; uses offline compute. |
| 35 | **Theory seed lines**: encode the known named lines (standard 3-step centre march, Shiller-style early vertical wall, side-wall, Rush, Reed-style double horizontal, box/gap) as candidate branches that #34 must evaluate, plus refutation candidates for each | +10-20 | none | 4 | **A** | Named lines are seeds only. They're **verified by deep search** and dropped if they score badly; I won't trust my memory of notation blindly. |
| 36 | Left-right mirror + colour-flip canonicalisation | (size) | none | 5 | **A** | Halves or quarters the book. |
| 37 | Self-play statistics on book nodes (win rate from fast games from each leaf) mixed with the search score | +5-15 | none | 4 | B | Catches lines that score well but lose in practice. |
| 38 | Weighted random choice among near-equal book moves | (variety) | none | 5 | **A** | Avoids being predictable or exploitable, and gives game variety. |

### D. Machine learning

| # | Idea | Elo | Cost | Feas. | Tier | Notes |
|---|---|---|---|---|---|---|
| 39 | **NNUE-style value net** inside the C++ engine: sparse inputs (pawn square x player, 128 wall slots, walls-left one-hot) feeding an incrementally updated accumulator (~128-256 hidden), plus a small dense head with the classic features (distances, blockability, flexibility). Hand-written int16/float inference, updated in make/unmake. | +40-120 | M | 3 | **A** | Fast enough for alpha-beta (~0.1-0.3 µs/eval). Training data from engine self-play, labels = blend(outcome, depth-8 search score). |
| 40 | Small dense MLP on ~40 handcrafted features (no incremental accumulator) | +20-60 | M | 4 | B (fallback) | Simpler fallback if NNUE doesn't beat the classic eval. |
| 41 | GBDT (LightGBM) eval exported to C++ code | +10-40 | M-H | 3 | C | Tree walks are slower than a small MLP for similar accuracy. |
| 42 | Policy net for move ordering and LMR decisions at high-depth nodes only | +20-50 | H | 2 | C | Expensive inference; revisit after #39. |
| 43 | C++ AlphaZero/MCTS with the net | ? | H | 1 | C | Months of compute to beat a 10 M nps alpha-beta. |
| 44 | Existing Python ResNet+MCTS agent as a tournament entrant | n/a | - | 3 | C | ~50% vs the *Python* depth 4; would lose every game to the C++ engine and costs hours to train. Mentioned in the report as a reference point only. |
| 45 | Iterated distillation: NNUE gen-2 trained on games of NNUE gen-1 | +10-30 | none | 4 | B | If time allows after gen 1. |
| 46 | Train in Colab / cloud GPU (scaled-up notebook) | (scale) | - | 3 | B | Notebook shipped alongside; the actual run happens here on the cloud container's CPU (see the open decisions). |

### E. Hybrid / architecture

| # | Idea | Elo | Cost | Feas. | Tier | Notes |
|---|---|---|---|---|---|---|
| 47 | **3-phase hybrid**: book -> NNUE alpha-beta midgame -> deep exact endgame (classic eval + decided cutoffs + endgame solver) | combines | - | 4 | **A** | Required candidate. Phase switch on "out of book" and on total walls left / race decided. |
| 48 | **Exact endgame solver** when total walls left <= 2 (or one side has none): deep search with the race solver, no futility/LMR, all walls considered | +10-30 | - | 4 | **A** | Endgame is where exactness wins. |
| 49 | Phase-aware time allocation inside the 5 s cap: instant book moves, full 5 s for volatile midgame moves, early stop when the root move has been stable for several iterations | +10-20 | - | 5 | **A** | |
| 50 | Eval ensemble: blend NNUE and the classic eval by phase | +0-20 | M | 4 | B | Safety net if NNUE is weaker in some phase. |
| 51 | Exact pawn-only race solver (both walls = 0) with full jump rules, via retrograde over 81x81x2 states | +0-5 | L | 5 | B | Basically covered by `decided`. |

## 3. Selected bot variants

Every variant is the same C++ binary with a different **profile** (option set), so they can play
each other with `tools.match` / the new tournament runner, and each can be picked in the GUI and
Auto-Play via a bot spec.

| ID | Name | Contents | Why it's in |
|---|---|---|---|
| **B0** | `baseline` | Current `main` engine, current defaults, `-O3` | Benchmark (required) |
| **B1** | `turbo` | B0 eval + search upgrades: #1 time management, #2 PVS, #3 aspiration, #4 parity fix, #5 LMR, #8 TT, #9 build flags, #12 must-block extension, #16 repetition | Isolates the value of search/speed work |
| **B2** | `book-turbo` | B1 + #21/#25 Texel-tuned tapered eval + #22 path flexibility + #34-38 opening book | Required "Opening Book + upgraded C++ search/heuristic" bot |
| **B3** | `nnue` | B1 search + #39 NNUE eval trained on self-play (real trained weights committed) | Required cloud-trained ML bot |
| **B4** | `hybrid` | #47: book (phase 1) -> NNUE alpha-beta (phase 2) -> #48 deep exact endgame with classic eval (phase 3) + #49 phase-aware time | Required hybrid 3-phase bot |
| **B5** | `tuned-classic` | B1 search + Texel-tuned tapered classic eval + flexibility, **no book** | Isolates heuristic tuning from the book (and is the "additional top C++ heuristic variant") |
| (B6) | `smp` | B2 with Lazy SMP, 4 threads | Optional. Only if you want a multi-core bot in the field (fairness question below) |

So the comparison is designed to attribute gains. B0->B1 is search, B1->B5 is heuristics,
B5->B2 is the book, B1->B3 is NNUE vs the classic eval, and B3/B2->B4 is the hybrid.

## 4. Training plan (B3 / B4)

1. **Data**: about 1-2 M positions from C++ self-play at fast time controls, with randomised
   openings and occasional random moves for diversity. Every sampled position is labelled with a
   fixed-depth search score (depth ~8) and the game result.
2. **Model**: NNUE-style sparse accumulator (290 binary inputs -> 128 hidden, clipped ReLU) plus
   ~20 dense classic features -> 32 -> 1. Trained in PyTorch on a loss of
   `lambda * outcome + (1 - lambda) * sigmoid(score / K)`. Horizontal mirroring is used as
   augmentation.
3. **Export** to a small binary weights file (`engine/nets/*.nnue`, a few hundred KB), loaded by
   the engine. Inference is float or int16 with incremental accumulator updates in make/unmake.
4. **Gate**: the net is kept only if it beats B1 at equal time in a 200-game fast match. If not,
   it's retrained (more data, gen-2 distillation) before the tournament.
5. A Colab notebook for the same pipeline is shipped, so a bigger run can be done on GPU later.

## 5. Tournament plan

* Round robin, every pair of B0-B5 (15 pairs). Colours alternate per opening, and each bot has
  its own engine process.
* **Hard cap of 5 s per move**, enforced by the runner: the bot's own limit is <= 5 s, and a move
  that takes more than 5.5 s wall-clock is logged as a time violation and loses the game.
* Openings: N distinct starting positions (0-2 reasonable plies, curated), each played twice with
  colours swapped. Opening books still apply from those positions.
* Recorded per game: winner, plies, per-move time, depth, nodes and nodes/s, book hits, and phase
  switches.
* Report: win matrix, W/L per bot, Elo estimate with error bars (BayesElo-style), average game
  length, nodes/s, depth per phase, and post-mortems of the best and worst bot built from
  annotated losses.

## 6. Open decisions before Phase 3

1. **CPU fairness.** I recommend every bot runs **single-threaded** and 4 games run in parallel on
   the 4 cores. Optionally, B6 (Lazy SMP) plays a separate sequential mini-match.
2. **Tournament size.** At 5 s/move a game takes ~3-5 min, so 15 pairs x 16 games ≈ 240 games ≈
   3-4 h with 4 games in parallel. That's only ±12% error per pair. I recommend also running a
   **0.5 s/move supplementary round robin (100+ games per pair)** for statistically solid ranking,
   with the 5 s event as the headline.
3. **Training compute.** I can't log into your Google Colab. I recommend training on this cloud
   container (4 vCPU, 15 GB RAM, PyTorch CPU from PyPI). It's enough for an NNUE of this size, but
   data generation takes 1-3 h. I'll also ship the Colab notebook for larger runs.
4. **Branch.** Phase 1 fast-forwarded `main` to include all 9 other branches (it was a linear
   chain, with no conflicts and nothing lost) and pushed it. Experiments live on
   `claude/zealous-goodall-2f2e8o`, the only branch this session may push to, instead of
   `feature/bot-evolution-tournament`.
