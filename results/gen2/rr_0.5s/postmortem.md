## Post-mortem `rr_0.5s` (0.5 s/move, 600 games)

| Bot | Eval accuracy plies 0-19 / 20-39 / 40+ | Forced win seen at ply | Lead over opponent (plies) | Forced loss seen at ply | Surprised losses | Walls placed plies 0-19 / 20-39 | Walls left in wins / losses | Right when both claim the edge |
|---|---|---|---|---|---|---|---|---|
| **nnue** | 61% / 79% / 93% | 43.9 | +1.0 | 45.2 | 22/164 (13%) | 3.4 / 3.3 | 0.8 / 1.2 | 76/200 (38%) |
| **hybrid** | 65% / 75% / 89% | 44.0 | +0.7 | 45.2 | 21/170 (12%) | 3.4 / 3.4 | 1.0 / 1.1 | 71/165 (43%) |
| **nnue-gen2** | 64% / 79% / 92% | 42.8 | +0.7 | 42.3 | 16/129 (12%) | 3.9 / 3.3 | 0.8 / 0.8 | 129/222 (58%) |
| **hybrid-gen2** | 66% / 79% / 92% | 44.2 | +0.6 | 44.5 | 15/137 (11%) | 3.9 / 3.3 | 0.8 / 0.6 | 99/163 (61%) |

The side with more walls left at ply 40 won 216/503 (43%) of decided games with a wall difference.
