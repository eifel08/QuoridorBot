# Bot evolution: tournament and engineering report

This report covers Phases 3–4 of [bot-evolution-plan.md](bot-evolution-plan.md). It describes six bots built on one C++ engine. Each bot adds one layer on top of the previous one. They played two single-threaded round-robins, with 4 games running in parallel on 4 cores:

| Tournament | Time per move | Games | Games per pair | Openings |
|---|---|---|---|---|
| Headline | **5 s** | 240 | 16 | 8 openings, each played with both colours |
| High-volume | **0.5 s** | 1,500 | 100 | 50 openings, each played with both colours |

Raw data, per-move logs and the generated tables are in:
- [`results/rr_5s/`](../results/rr_5s/report.md)
- [`results/rr_0.5s/`](../results/rr_0.5s/report.md)
- [`results/postmortem.md`](../results/postmortem.md)

**Bottom line:** the two neural-evaluation bots, **hybrid** and **nnue**, are about **+210 Elo** stronger than the engine `main` had before this work, at 5 s per move. At 0.5 s per move the gain is about **+170–200 Elo**.
- **Best bot:** hybrid, by a hair. It is first at 5 s (+222 ±84) and second at 0.5 s (+170 ±36).
- **Runner-up:** nnue, statistically tied with hybrid. It wins their head-to-head 10–6 at 5 s and 50–49 at 0.5 s.
- **Weakest:** the old **baseline**. Over all 1,740 games it is last, and it is joint-last at 5 s together with **turbo**.

## 1. The bots

| Bot | What it is |
|---|---|
| **baseline** | The `main` engine before this work, frozen in `engine/baseline/` and built with `-O3`. It has: <br>• alpha-beta with iterative deepening, a transposition table and move ordering<br>• wall filtering, futility pruning and quiescence search<br>• the hand-written "classic" evaluation |
| **turbo** | baseline plus search upgrades: <br>• smart time management (it predicts the next iteration's cost, keeps partial iterations, and has a hard deadline)<br>• PVS and aspiration windows<br>• late-move reductions (LMR)<br>• detecting repeated positions as draws<br>• a 128 MB transposition table<br>• `-march=native -flto` |
| **tuned** | turbo with a **Texel-tuned linear evaluation**: 10 features that blend from middlegame to endgame weights as walls run out, including two new path-shape features. It was fitted on 237k self-play positions. Validation log-loss dropped from 0.584 to 0.562. |
| **book** | tuned plus a **563-position opening book**. Each position was searched for 15 s, plus 8 s for a second-best alternative. The book covers centre-march, Shiller-style, Reed-style, side-wall and rush lines. |
| **nnue** | turbo search with a **trained neural evaluation** (NNUE-style): 312 sparse inputs plus 13 dense features, then layers of 64 and 16 units, then one output. It was trained on 400k self-play positions on the container's CPU; Colab wasn't needed. C++ inference matches PyTorch to within 0.5 eval units. |
| **hybrid** | Three phases: <br>1. opening book<br>2. neural evaluation in the midgame<br>3. the tuned linear evaluation once 6 or fewer walls are left, and an exact deep endgame search (no LMR, no futility) once 4 or fewer remain |

All bots use 1 thread. Every move in both tournaments met the time limit. The slowest move was 5.03 s, by the baseline, which has no hard deadline; the allowed limit is 1.1·t + 0.05 s.

## 2. Headline results: 5 s per move

**Results matrix** (row bot's wins – column bot's wins, 16 games per pair):

| | hybrid | nnue | tuned | book | baseline | turbo | Total W-L-D | Score |
|---|---|---|---|---|---|---|---|---|
| **hybrid** | - | 6-10 | 14-2 | 11-5 | 14-2 | 12-3 | 57-22-1 | 71.9% |
| **nnue** | 10-6 | - | 11-5 | 10-6 | 12-4 | 13-3 | 56-24-0 | 70.0% |
| **tuned** | 2-14 | 5-11 | - | 8-8 | 8-7 | 11-5 | 34-45-1 | 43.1% |
| **book** | 5-11 | 6-10 | 8-8 | - | 7-8 | 8-8 | 34-45-1 | 43.1% |
| **baseline** | 2-14 | 4-12 | 7-8 | 8-7 | - | 8-8 | 29-49-2 | 37.5% |
| **turbo** | 3-12 | 3-13 | 5-11 | 8-8 | 8-8 | - | 27-52-1 | 34.4% |

**Ranking and engine statistics:**

| Rank | Bot | Elo vs baseline (±2σ) | Avg game length (plies) | Avg depth | Nodes/s | Avg s/move | Book moves |
|---|---|---|---|---|---|---|---|
| 1 | **hybrid** | **+222 (±84)** | 76.4 | 17.0 | 1.38 M | 2.53 | 180 |
| 2 | **nnue** | +209 (±90) | 74.0 | 15.9 | 1.15 M | 2.53 | 0 |
| 3 | tuned | +36 (±76) | 76.2 | 17.8 | 1.91 M | 2.54 | 0 |
| 4 | book | +36 (±75) | 77.7 | 18.3 | 1.85 M | 2.42 | 177 |
| 5 | baseline | 0 | 75.9 | 17.0 | 2.46 M | 2.49 | 0 |
| 6 | turbo | −20 (±101) | 76.3 | 17.3 | 2.49 M | 2.52 | 0 |

Game endings: 237 reached the goal; 3 hit the 200-ply cap and were draws. Red won 120 of the 237 decided games, so the openings are colour-balanced.

## 3. High-volume results: 0.5 s per move

| | nnue | hybrid | tuned | book | turbo | baseline | Total W-L-D | Score |
|---|---|---|---|---|---|---|---|---|
| **nnue** | - | 49-50 | 60-40 | 62-38 | 66-34 | 80-20 | 317-182-1 | 63.5% |
| **hybrid** | 50-49 | - | 60-40 | 54-46 | 63-37 | 68-32 | 295-204-1 | 59.1% |
| **tuned** | 40-60 | 40-60 | - | 55-44 | 57-42 | 72-28 | 264-234-2 | 53.0% |
| **book** | 38-62 | 46-54 | 44-55 | - | 57-42 | 68-32 | 253-245-2 | 50.8% |
| **turbo** | 34-66 | 37-63 | 42-57 | 42-57 | - | 55-45 | 210-288-2 | 42.2% |
| **baseline** | 20-80 | 32-68 | 28-72 | 32-68 | 45-55 | - | 157-343-0 | 31.4% |

| Rank | Bot | Elo vs baseline (±2σ) | Avg game length | Avg depth | Nodes/s |
|---|---|---|---|---|---|
| 1 | **nnue** | **+197 (±36)** | 69.7 | 13.0 | 1.24 M |
| 2 | hybrid | +170 (±33) | 71.2 | 13.1 | 1.44 M |
| 3 | tuned | +133 (±34) | 72.3 | 13.9 | 1.92 M |
| 4 | book | +120 (±37) | 72.3 | 14.2 | 1.90 M |
| 5 | turbo | +68 (±37) | 71.7 | 13.9 | 2.58 M |
| 6 | baseline | 0 | 71.1 | 12.4 | 2.54 M |

With 100 games per pair, these error bars are tight. Every layer adds strength in the expected order. The only exception is the opening book, which gives nothing measurable over tuned.

## 4. Post-mortem

All numbers in this section come from `tools/postmortem.py`, which reads the per-move logs: score, depth and time for every move. The full tables are in [`results/postmortem.md`](../results/postmortem.md).

### Why the neural bots dominated

**1. They judge the position better, and the gap opens early.** This is the decisive factor. The table shows how often a bot's own score points to the eventual winner, by phase:

| Eval accuracy at 5 s | plies 0–19 | plies 20–39 | plies 40+ |
|---|---|---|---|
| hybrid | **76%** | 86% | 94% |
| nnue | 71% | 84% | 92% |
| tuned | 68% | 86% | 94% |
| baseline | 62% | 84% | 90% |

From ply 40 onwards every bot is about 90–94% accurate, because the search can see the race to the goal by then. The difference is in the first 20 plies, when the wall battle is being set up. The hand-written evaluation can't tell which wall structures will pay off ten moves later. The network learned this from 400k self-play outcomes.

**2. When two bots disagree, the neural bot is usually right.** A *dispute* is when both bots report a score above +10 on consecutive plies, between plies 12 and 39; in other words, each thinks it is winning. In those disputes:

| Bot | Right at 0.5 s | Right at 5 s |
|---|---|---|
| hybrid | 53% | **81%** (30/37) |
| nnue | **65%** (240/370) | 56% |
| tuned | 50% | 20% |
| baseline | **25%** | **26%** |

nnue gets into far more disputes than anyone else (370 at 0.5 s), because it sees the position differently. It wins most of them. Turbo and baseline share the classic evaluation and almost never dispute each other, which confirms the measure works.

A typical game, nnue against baseline at 5 s (baseline played Red, opening 3):
- **Plies 14–42:** baseline rates itself between +11 and +35 while it spends its walls on the left side (h22, h24, v15, h04, h13, h70, h72, v32). nnue calmly routes around them.
- **Ply 43:** nnue's score jumps to +76, then +91 and +167.
- **Ply 49:** nnue announces a forced win.
- **Ply 52:** baseline sees that it has lost, three plies later.

The baseline wasn't out-searched; it reached depth 13–22. It was out-judged: its walls lengthened nnue's path less than its evaluation believed.

**3. It costs search speed and still wins.** nnue searches only 1.15 M nodes/s, against 2.46 M for baseline, and about one ply shallower (15.9 against 17.0 at 5 s). Better judgement more than pays for that.

**4. Why hybrid edges out nnue at 5 s.** Each component contributes something:

| Component | What it adds |
|---|---|
| Exact endgame search | Hybrid sees its own forced losses latest of any bot (ply 48.9 against 45.3). It keeps setting the hardest problems instead of giving up, and it was right in 81% of disputes. |
| Switch to the tuned linear evaluation with ≤ 6 walls | The linear evaluation is about 40% faster, so hybrid reaches depth 17.0 against nnue's 15.9. |
| Opening book | It saves time in the opening, but only about 2 book moves per game get played (180 in 80 games). |

At 0.5 s these advantages disappear and nnue leads by +27, inside the error bars. The 16-game head-to-head at 5 s went 10–6 to nnue. **The two are effectively equal.** Hybrid is the better choice for long time controls, and plain nnue for blitz.

### Why the weakest bots failed

**baseline (last overall).** It has two problems:

1. **A blunt evaluation.** It was right in only 25% of disputes at either time control, and its early-game accuracy is lowest (57% at 0.5 s, 62% at 5 s). Games are typically decided around ply 19–23, when the winner's score last changes sign, but baseline only sees its forced loss at about ply 45. In other words, it walks into lost positions believing it is ahead.
2. **Shallow search at short time controls.** At 0.5 s it reaches depth 12.4, against 13.9 for turbo. It has no move-time prediction, so it often abandons an iteration it had already paid for. That explains turbo's +68 at 0.5 s.

**turbo (last at 5 s, −20 ±101: statistically level with baseline).** Turbo's improvements are all search-side. At 0.5 s they are worth +68 Elo, because they buy about 1.5 extra plies. At 5 s they buy only 0.3 ply (17.3 against 17.0): Quoridor's branching factor is roughly 130 moves before wall filtering, so it absorbs a 2× speed-up in under a ply. LMR reduces quiet wall moves, which may also cost some tactics. The result is the same evaluation at almost the same depth: baseline against turbo went 8–8, and the two never disagree.

The general lesson: **at long time controls the evaluation is what separates bots, not the search.** Search tricks pay off mainly in blitz.

### Other findings

- **The opening book didn't help.** book against tuned was 8–8 at 5 s and 44–55 at 0.5 s. A 10-ply book carries a bot through the quiet pawn-march opening, where every bot already plays sensibly. The real fight starts with the first walls, around plies 6–14, and then the position is out of book within 2–4 moves.
- **Spare walls aren't an advantage in themselves.** The side with more walls left at ply 40 won only 48–49% of decided games. The tuned evaluation still gives a high endgame weight to "wall surplus" (26) and "full walls" (30). As a result, tuned ends lost games with 1.0–1.6 walls unused, which is a wasted resource. The neural net learned to spend walls on time.
- **Overconfidence ("surprised losses").** A surprised loss is one where the bot still scored itself at +15 or better shortly before seeing the forced loss. These were rare at 5 s, 0–8% for most bots. nnue had the most, at 17% (4 of 24 losses): its occasional misjudgements are confident ones.

## 5. Engineering notes

- **Correctness checks:**
  - With every new option off, the new engine reproduces the baseline search exactly (0 mismatches over 710 positions at depth 4).
  - Linear mode with the default weights reproduces the classic evaluation.
  - The engine self-test passes.
- **Linear eval speed:** it was first slower than the classic evaluation, at 1.7 M against 2.9 M nodes/s. Two fixes:
  - replacing a path-counting dynamic programme with a layer-width count;
  - limiting the evaluation cache to the neural modes, because cache misses hurt the cheap linear evaluation.
- **Neural net training:** it overfits after one epoch, because self-play positions from the same game are correlated, so training keeps the best validation epoch automatically. More varied data (the next generation of self-play) is the obvious next lever.
- **Not implemented:**
  - plan idea #12, extending the search for must-block situations;
  - profile-guided optimisation (PGO);
  - 4-way transposition-table buckets;
  - a 4-thread exhibition match, as you requested.
- **Colab:** not used. The container's CPU trained the net in 38 s, and self-play took about 1 hour.

## 6. Recommended next steps

1. **A second generation of self-play data from nnue/hybrid games, and retraining.** This is the biggest expected gain, since better evaluation is what wins. Also try a wider network (128 units) once there is enough data.
2. **Retune the linear endgame weights on nnue-quality data**, so the wall-surplus weight comes down.
3. **Give hybrid a deeper book built with the nnue profile**, aimed at the wall phase (plies 8–20) rather than the pawn march.
4. **A must-block extension and PGO** for the search.

## 7. Reproducing

```bash
make -C engine && make -C engine baseline
python -m tools.selfplay_data ...           # self-play data (see the script's --help)
python -m tools.texel_tune ...              # writes engine/nets/linear_weights.txt
python -m tools.train_nnue ...              # writes engine/nets/nnue.txt
python -m tools.build_book --profile nnue   # writes engine/book/book.json
nohup setsid bash tools/run_tournaments.sh > results/run.log 2>&1 &
python -m tools.postmortem results/rr_0.5s results/rr_5s --md results/postmortem.md
```

To play against a bot in the GUI or CLI, use the bot spec `profile:name=hybrid,time=5`; the presets "Hybrid", "NNUE", "Book" and "Turbo" do the same.
