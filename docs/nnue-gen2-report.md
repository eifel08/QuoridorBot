# NNUE generation 2

The generation-2 network is now the default (`engine/nets/nnue.txt`) for the **NNUE** and
**Hybrid** bots. Generation 1 is frozen as `engine/nets/nnue_gen1.txt` and can still be played as
the profiles `nnue-gen1` and `hybrid-gen1`. Re-run everything with `python -m tools.gen2_pipeline`,
which is resumable and skips a stage once its output exists.

## Pipeline

1. **Self-play:** 7,000 games between the generation-1 bots `nnue` and `hybrid`, each side
   picked at random per game, at 0.12 s per move. That gives about 468k positions in
   `data/selfplay_gen2.jsonl`.
2. **Training:** three candidates, each validated on held-out games of the generation-2 data:

   | Candidate | Data | Net | Val log-loss | Score vs gen-1 nnue (200 games, 0.2 s) |
   |---|---|---|---|---|
   | gen2a | gen-2 only (421k positions) | 64x16 | 0.4231 | 49.5% |
   | **gen2b** | gen-1 + gen-2 (821k positions) | 64x16 | **0.4210** | **65.0%** |
   | gen2c | gen-1 + gen-2 | 128x16, weight decay 0.01 | 0.4268 | 60.5% |

   gen2b was kept as `nnue_gen2`. Training on generation-2 data alone overfits after about 2
   epochs and only matches generation 1. Mixing in the older data is what produced the gain.
3. **Benchmark:** a round-robin of `nnue`, `hybrid`, `nnue-gen2` and `hybrid-gen2`. Each game
   was single-threaded, with 4 games running in parallel.

## Results

Elo is measured against generation-1 `nnue`, with ±2σ bootstrap error bars.

### 0.5 s per move (600 games, 50 openings x 2 colours)

| Rank | Bot | Elo | Score | Avg depth | Nodes/s |
|---|---|---|---|---|---|
| 1 | nnue-gen2 | +61 (±43) | 57.0% | 13.6 | 1.48 M |
| 2 | hybrid-gen2 | +47 (±36) | 54.3% | 14.2 | 1.79 M |
| 3 | nnue | 0 | 45.3% | 13.5 | 1.52 M |
| 4 | hybrid | -11 (±43) | 43.3% | 13.6 | 1.76 M |

### 5 s per move (96 games, 8 openings x 2 colours)

| Rank | Bot | Elo | Score | Avg depth | Nodes/s |
|---|---|---|---|---|---|
| 1 | nnue-gen2 | +107 (±92) | 64.6% | 17.4 | 1.40 M |
| 2 | hybrid-gen2 | +33 (±94) | 50.0% | 19.8 | 1.67 M |
| 3 | nnue | 0 | 43.8% | 18.7 | 1.37 M |
| 4 | hybrid | -28 (±94) | 39.6% | 18.1 | 1.69 M |

### Head to head (gen 2 vs gen 1)

| Pair | 0.5 s | 5 s |
|---|---|---|
| nnue-gen2 vs nnue | **64-36** | **12-4** |
| hybrid-gen2 vs hybrid | **62-38** | **10-6** |
| nnue-gen2 vs hybrid | 51-49 | 10-6 |
| hybrid-gen2 vs nnue | 57-43 | 7-8 (1 draw) |
| nnue-gen2 vs hybrid-gen2 | 56-44 | 9-7 |

There were no time violations. The new network is as fast as the old one, so the gain comes
entirely from evaluation quality.

## Takeaways

- The gain is clear at 0.5 s (64-36 and 62-38 in the same-bot pairs, 2.4-2.8σ above 50%) and points the same
  way at 5 s, where the sample is smaller.
- With generation-2 weights, NNUE slightly edges Hybrid (56-44 and 9-7). That is not
  significant, so the GUI ranking stays Hybrid first. Next steps worth trying: a larger 5 s
  sample, and retuning `hybrid_walls` for the new network, because the tuned-eval fallback in
  the late game may now be weaker than the network.
- Detailed logs are in `results/gen2/`: `pipeline.log`, `gate_*`, `train_*.log`, and
  `rr_0.5s/` and `rr_5s/` (games, `report.md`, `postmortem.md`).
