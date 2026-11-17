## Post-mortem `rr_0.5s` (0.5 s/move, 1500 games)

| Bot | Eval accuracy plies 0-19 / 20-39 / 40+ | Forced win seen at ply | Lead over opponent (plies) | Forced loss seen at ply | Surprised losses | Walls placed plies 0-19 / 20-39 | Walls left in wins / losses | Right when both claim the edge |
|---|---|---|---|---|---|---|---|---|
| **baseline** | 57% / 77% / 87% | 46.3 | -0.0 | 45.2 | 13/343 (4%) | 3.8 / 3.3 | 0.7 / 0.8 | 51/207 (25%) |
| **turbo** | 61% / 77% / 86% | 43.2 | +0.7 | 45.2 | 7/288 (2%) | 4.1 / 3.3 | 0.8 / 0.5 | 68/135 (50%) |
| **tuned** | 59% / 80% / 89% | 45.1 | +1.1 | 45.4 | 34/234 (15%) | 3.1 / 3.0 | 1.3 / 1.6 | 89/179 (50%) |
| **book** | 61% / 76% / 86% | 45.1 | +1.2 | 45.6 | 22/245 (9%) | 3.1 / 2.9 | 1.3 / 1.6 | 85/191 (45%) |
| **nnue** | 68% / 81% / 89% | 43.9 | +1.1 | 44.7 | 25/182 (14%) | 3.5 / 3.4 | 1.0 / 0.9 | 240/370 (65%) |
| **hybrid** | 68% / 79% / 88% | 45.0 | +0.7 | 47.5 | 22/204 (11%) | 3.3 / 3.3 | 1.1 / 1.0 | 145/274 (53%) |

The side with more walls left at ply 40 won 619/1277 (48%) of decided games with a wall difference.

## Post-mortem `rr_5s` (5.0 s/move, 240 games)

| Bot | Eval accuracy plies 0-19 / 20-39 / 40+ | Forced win seen at ply | Lead over opponent (plies) | Forced loss seen at ply | Surprised losses | Walls placed plies 0-19 / 20-39 | Walls left in wins / losses | Right when both claim the edge |
|---|---|---|---|---|---|---|---|---|
| **baseline** | 62% / 84% / 90% | 45.3 | -0.4 | 45.1 | 1/49 (2%) | 3.9 / 3.2 | 0.9 / 0.6 | 8/31 (26%) |
| **turbo** | 65% / 85% / 94% | 44.5 | +0.6 | 45.6 | 4/52 (8%) | 3.8 / 3.5 | 0.6 / 0.6 | 7/18 (39%) |
| **tuned** | 68% / 86% / 94% | 45.8 | +0.8 | 44.6 | 0/45 (0%) | 3.3 / 2.6 | 1.6 / 1.0 | 4/20 (20%) |
| **book** | 67% / 82% / 93% | 45.5 | +1.4 | 46.9 | 2/45 (4%) | 3.2 / 2.9 | 1.1 / 1.1 | 10/17 (59%) |
| **nnue** | 71% / 84% / 92% | 43.9 | +1.1 | 45.3 | 4/24 (17%) | 3.3 / 3.3 | 1.0 / 1.0 | 23/41 (56%) |
| **hybrid** | 76% / 86% / 94% | 45.6 | +0.8 | 48.9 | 1/22 (5%) | 3.5 / 2.9 | 1.2 / 0.7 | 30/37 (81%) |

The side with more walls left at ply 40 won 103/211 (49%) of decided games with a wall difference.
