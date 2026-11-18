## Post-mortem `rr_5s` (5.0 s/move, 96 games)

| Bot | Eval accuracy plies 0-19 / 20-39 / 40+ | Forced win seen at ply | Lead over opponent (plies) | Forced loss seen at ply | Surprised losses | Walls placed plies 0-19 / 20-39 | Walls left in wins / losses | Right when both claim the edge |
|---|---|---|---|---|---|---|---|---|
| **nnue** | 70% / 82% / 95% | 44.7 | +0.5 | 45.0 | 1/26 (4%) | 3.2 / 3.5 | 0.9 / 0.9 | 8/29 (28%) |
| **hybrid** | 77% / 84% / 92% | 46.6 | +1.1 | 47.7 | 1/29 (3%) | 3.5 / 3.3 | 0.5 / 0.5 | 8/14 (57%) |
| **nnue-gen2** | 74% / 82% / 99% | 41.8 | +0.6 | 45.4 | 1/17 (6%) | 3.4 / 3.8 | 1.1 / 0.6 | 30/38 (79%) |
| **hybrid-gen2** | 72% / 85% / 97% | 45.3 | +1.2 | 41.8 | 0/23 (0%) | 3.6 / 3.5 | 1.0 / 0.5 | 1/13 (8%) |

The side with more walls left at ply 40 won 35/74 (47%) of decided games with a wall difference.
