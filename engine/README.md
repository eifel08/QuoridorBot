# Quoridor C++ search engine

A fast replacement for the search in `quoridor/bots/alphabeta.py`. The rules are exactly those of
`quoridor/board.py` (verified move-for-move, see [Verification](#verification)). The GUI, CLI and
bot-match runner use it automatically once it is built; without it they fall back to the Python bot.

## Build

Any C++20 compiler works (g++ 10+, clang 12+, MSVC 2019+). Pick one:

```bash
python build_engine.py
```

```bash
make -C engine
```

```bash
cmake -S engine -B engine/build -DCMAKE_BUILD_TYPE=Release
```

```bash
cmake --build engine/build --config Release
```

All of them produce `engine/build/quoridor_engine` (`.exe` on Windows; CMake with Visual Studio puts
it in `engine/build/Release/`, which is also searched). Optimisation level is `-O3`. To use a binary
somewhere else, set the `QUORIDOR_ENGINE` environment variable to its path.

## Run

```bash
python main.py                                  # menu; bots now think with the C++ engine
python main.py --mode bot --depth 30 --time-limit 2
python main.py --mode bots --bot1 ab:depth=30,time=1 --bot2 ab:depth=30,time=1,futility=0
python main.py --mode bot --walls attack        # walls considered: attack | both (default) | all
python main.py --engine py                      # force the old Python search
```

`--depth` is the maximum depth and `--time-limit` the seconds per move; the engine plays the best
move of the deepest iteration it completed. With the C++ engine a high depth plus a time limit
(e.g. depth 30, 1-5 s) is the sensible setting.

Bot specs (`--bot1/--bot2`, `python -m quoridor.botmatch`) accept every engine option below, e.g.
`ab:depth=6,wall_filter=0,futility=0`, plus `engine=cpp|py|auto` and `walls=attack|both|all`
(the same as `wall_filter=2|1|0`). The wall mode can also be picked per bot in the GUI menu and
the `--cli` questions.

## Architecture

```
engine/src/board.h, board.cpp    position, make/unmake, BFS, legal moves, Zobrist keys
engine/src/search.h, search.cpp  negamax, TT, move ordering, filtering, pruning, evaluation
engine/src/main.cpp              text protocol on stdin/stdout, self-test
quoridor/bots/cpp_engine.py      Python side: CppEngine (process wrapper), CppBot (a Player)
tools/                           parity, verify_search, bench, match, calibration, eval_accuracy
tests/test_cpp_engine.py         unit tests (skipped if the engine is not built)
```

### Why a separate executable instead of pybind11

* The existing bot interface is already "here is the complete position, give me a move"
  (`Player.get_move(BoardState)`), which maps one-to-one onto two lines of text. `CppBot` subclasses
  `AlphaBetaBot`, so the GUI change is two lines (`AlphaBetaBot(...)` -> `make_search_bot(...)`).
* No coupling to the Python version or ABI: the same binary works with every Python (the project
  targets 3.14 on Windows, where a pybind11 module needs a matching MSVC toolchain and a rebuild
  per Python version). Only a C++ compiler is needed, no Python headers and no extra pip package.
* The GUI runs bots in a background thread; a subprocess makes that trivially non-blocking, and a
  crash or runaway search in the engine cannot take the window down (the process is killed on exit).
* The binary doubles as a command-line tool for debugging (`echo "go depth 8" | quoridor_engine`).

The cost is one pipe round trip per move (well under a millisecond), irrelevant next to the search.

### Board

* Squares are `row * 9 + col`. A set of squares is an 81-bit bitboard (`BB`, two `uint64_t`; a plain
  struct rather than `__int128` so MSVC can build it).
* Walls are two 64-bit masks (`hw`, `vw`), one bit per anchor. "Which slots are still free" is a few
  shifts: `free_h = ~(hw | hw<<1 | hw>>1 | vw)`.
* `open[d]` holds, for each of the 4 directions, the squares from which a pawn may step that way.
  Placing a wall clears 4 bits, removing it sets them again.
* **make/unmake** on a single board: `make()` returns the mover's previous square, `unmake()` takes
  it back. The Zobrist hash is updated incrementally in both. Nothing is copied or allocated in the
  search; all per-node scratch data are fixed-size arrays on the stack.
* **BFS** is a bitboard flood fill: one BFS level is `expand(frontier) & ~seen`, four shift+AND
  operations for the whole frontier at once. No queue, no per-square loop.
* **Wall legality**: one flood from each goal row gives the set of edges on *any* shortest path of
  each player. A wall that blocks none of a player's shortest-path edges cannot disconnect that
  player (and leaves the distance unchanged), so the path-existence BFS is skipped for it.

### Search

* **Negamax alpha-beta** (fail-soft), **iterative deepening** with a time limit and/or a maximum
  depth. An iteration that runs out of time is discarded and the move of the last completed depth
  is played. A new iteration is not started once half the time is used (it could not finish).
* **Zobrist hashing**: 64-bit keys for each pawn square per player, each wall slot per orientation,
  each walls-remaining count per player, and the side to move.
* **Transposition table**: fixed array of 16-byte entries (full key, value, best move, depth, bound
  flag, age) in buckets of two, slot 0 depth-preferred, slot 1 always replaced. A stored value is
  used only if its depth >= the remaining depth, respecting EXACT/LOWER/UPPER; the stored move is
  always tried first. Win scores are stored relative to the node so they stay correct at another ply.
* **Move ordering**: (1) TT move, (2) pawn moves along a shortest path, (3) killer moves, then
  (4) attack walls, (5) defense walls, (6) everything else (other walls, pawn moves that do not
  shorten my path). Inside each tier: most steps added to the opponent's path first, then fewest
  steps added to mine, then the history heuristic. With `wall_filter=0` the attack and defense
  walls are therefore still searched before the rest. The stages are lazy: walls are only
  generated if the TT move and the shortest-path pawn moves did not already cause a cutoff.
  Ordering only changes speed, never the result: alpha-beta returns the minimax value whatever
  the order (checked by `tools.verify_search`).
* The shortest-path lengths of both players are passed down the tree (a pawn move's new distance
  is read from the goal flood, a wall's from its legality check), so leaf nodes do no BFS at all.
* **Win/loss scores** are `WIN - ply` / `-WIN + ply`: faster wins and slower losses are preferred.

### Wall filtering (`wall_filter`)

Pawn moves are always generated. Walls are limited to:

* **attack walls**: block an edge on any shortest path of the opponent;
* **defense walls**: overlap or cross a wall the opponent could place to block an edge on any of
  *my* shortest paths (they pre-empt that wall).

Instead of two full BFS runs per player the engine floods once from the goal row and then walks
from the pawn through decreasing distance layers; this visits exactly the edges `u->v` with
`dist_from_pawn[u] + 1 + dist_to_goal[v] == shortest_distance`.

If the filter leaves a player without any move, all legal walls are generated.

### Evaluation

```
eval = (opp_dist - my_dist) * dist_weight + (my_walls' - opp_walls') * wall_weight + tempo_bonus

my_walls'  = min(my_walls,  opp_dist + wall_cap_slack)     (with wall_cap on, the default)
opp_walls' = min(opp_walls, my_dist  + wall_cap_slack)
```

from the side to move's point of view. **Dynamic wall value**: a player can place at most one wall
per turn, and only on the turns it still gets before the opponent arrives, so a reserve only counts
up to the opponent's remaining distance (+1 turn of slack). Example: Red 4 steps from its goal with
6 walls, Blue 2 steps away with none: the old formula counted all 6 walls (+8.1 steps for Red), the
capped one counts 3 (+3.3). Far from the goal the cap changes nothing.

### Decided positions (`decided`)

A player without walls cannot lengthen the other one's path, so the game is a race. If the opponent
has no walls and `my_dist <= opp_dist` (my move), the node returns a win; if I have no walls and
`opp_dist < my_dist`, a loss. The score is `WIN - (ply + plies the race still takes)`.

The pawns meeting can change a race by a step (a jump gains one). The engine therefore checks
whether the loser can get next to any square of the winner's shortest paths by the time the winner
is there. If not, the result is exact. If the pawns can meet, the result is only declared with an
extra `decided_margin` (default 1) steps of lead; otherwise the normal search continues.

### Futility pruning (`futility`)

At every node except the root and decided positions:

```
if eval + my_walls * wall_recovery + depth_margin * remaining_depth < alpha:  prune
```

It is skipped when alpha is a win/loss score.

## Options

Set with `set <name> <value>` in the protocol, `CppBot(options={...})`, bot specs, or
`tools.match --a/--b`.

| Option | Default | Meaning |
|---|---|---|
| `dist_weight` | 10 | evaluation: per step of path difference |
| `wall_weight` | 16 | evaluation: per wall of reserve difference |
| `tempo_bonus` | 5 | evaluation: bonus for the side to move |
| `wall_cap` | 1 | evaluation: count a wall reserve only up to the opponent's distance + slack |
| `wall_cap_slack` | 1 | extra turns allowed by the cap |
| `wall_filter` | 1 | 0 = all legal walls, 1 = attack + defense walls, 2 = attack walls only |
| `decided` | 1 | decided-position cutoffs |
| `decided_margin` | 1 | extra lead required when the pawns can still meet |
| `futility` | 1 | futility pruning |
| `wall_recovery` | 5 | futility margin per wall the side to move has left |
| `depth_margin` | 5 | futility margin per remaining ply |
| `tt` | 1 | transposition table |
| `tt_exact_depth` | 0 | reuse TT values only at exactly the stored depth (verification) |
| `alphabeta` | 1 | 0 = plain minimax (verification) |
| `killers`, `history` | 1 | move-ordering heuristics |
| `seed` | 0 | non-zero: random tie-breaking between equally ordered root moves |
| `hash_mb` | 64 | transposition table size in MB |
| `verbose` | 0 | print an `info` line per completed depth |

Changing any option that affects scores clears the transposition table.

## Protocol

One command per line; each is answered by exactly one line (`ok`, `error ...`, or data).

```
position <turn> <r0> <c0> <r1> <c1> <walls0> <walls1> [h<r><c> | v<r><c>]...
go [depth N] [time SECONDS]   -> bestmove m74 depth 9 score 21 nodes 1234567 time 0.4991
moves                         -> every legal move (never filtered)
genmoves                      -> the moves the search tries, in order (filter applied)
play <move> | dist | eval | hash | set <name> <value> | newgame | selftest [games] [seed] | quit
```

Moves are `m<row><col>` (pawn to that square), `h<row><col>` / `v<row><col>` (wall anchor), the same
coordinates as `PawnMove` / `WallMove`.

## Verification

```bash
python -m tools.parity --games 300          # rules: Python vs C++ legal moves at every position
```

```bash
python -m tools.verify_search --positions 150 --tt-depth 5
```

```bash
python -m tools.bench --seconds 2           # nodes/s and depth: Python bot vs C++
```

```bash
python -m tools.match --a futility=1 --b futility=0 --time 0.05 --games 200
```

```bash
python -m unittest tests.test_cpp_engine    # includes a full game through the pygame GUI
```

* **Rules parity** plays random games with `quoridor/board.py` and compares, at every position,
  the full set of legal moves, both shortest-path lengths, and the incrementally updated hash.
* **TT correctness** (futility and filtering off, fixed depth): with `tt_exact_depth 1` the TT is a
  pure cache and the root value must equal the no-TT value; this is asserted. With the default rule
  (stored depth >= remaining depth) a transposition can hand a node the result of a *deeper* search
  than it would do itself, so small differences are legitimate; they are counted and reported.
* **Alpha-beta correctness**: the root value equals plain minimax.

## Results

Measured on a 4-core cloud VM (g++ 13, `-O3`), `python -m tools.bench --seconds 2 --depth 5`.

### Speed: depth reached in 2 seconds

| Position | Python (old bot) | C++ default | C++ filter + futility off |
|---|---|---|---|
| opening | depth 8, 4.6 k nodes/s | depth 11, 4.7 M nodes/s | depth 6, 10.7 M nodes/s |
| middlegame | depth 10, 6.1 k nodes/s | depth 17, 4.5 M nodes/s | depth 7, 12.6 M nodes/s |
| endgame | depth 21, 22 k nodes/s | depth 30 (forced loss proven) | depth 30 (forced loss proven) |

On average the C++ engine searches about 500x more nodes per second than the Python bot
(about 1000x with filtering and futility off, where nodes are cheaper because less of each node's
work is wall scoring). The depths are not strictly comparable: the Python bot only searches walls
that cut ONE shortest path and improve the score (5-20 per node), the C++ engine the wider
attack + defense set (or every wall). Iterative deepening stops early once half the time is used,
so a 2 s search typically finishes after 1-1.5 s.

### Strength against the old Python bot

`python -m quoridor.botmatch --bot1 ab:depth=30,time=0.3,engine=cpp --bot2 ab:depth=30,time=0.3,engine=py --games 40`:
the C++ engine wins **34 of 40 games (85%)**, 75% as Red and 95% as Blue.

### Strength (`tools.match`, 0.05 s per move, 200 games, colours alternated per opening)

| A | B | A wins |
|---|---|---|
| `futility=1` (margins 5 / 5) | `futility=0` | **73.0%** (depth 10.6 vs 8.6) |
| futility margins 5 / 5 | margins 20 / 10 | 73.5% |
| futility margins 20 / 10 | `futility=0` | 47.5% (too wide to prune much) |
| `wall_filter=1` (attack + defense) | `wall_filter=0` (all walls) | **78.5%** (depth 8.5 vs 6.2) |
| `wall_filter=2` (attack only) | `wall_filter=1` | 75.5% (depth 10.7 vs 8.5) |
| `decided=1` | `decided=0` | 53.0% (within noise) |
| `wall_weight=16` | `wall_weight=8` / `12` / `20` / `24` (depth 5) | 64% / 61% / 53% / 59% |

### Wall weight and the dynamic wall value (0.05 s per move, colours alternated)

| A | B | A wins |
|---|---|---|
| `wall_cap=1,wall_cap_slack=1` (weight 16) | no cap (weight 16) | **69.5%** (400 games) |
| `wall_cap=1,wall_cap_slack=0` (weight 16) | no cap (weight 16) | 57.5% (400 games) |
| cap, slack 2 | cap, slack 1 | 46.8% (400 games) |
| cap + slack 1, weight 20 / 12 | cap + slack 1, weight 16 | 48.2% / 34.8% (400 games) |
| weight 6 / 10 (no cap) | weight 16 (no cap) | 33.0% / 46.5% (200 games) |

Lowering the wall weight was tried to stop the eval from overrating a big wall reserve in a race
the opponent is about to win. It made the bots clearly weaker and did not make the eval more
accurate: on 500 fixed positions labelled with their real winner (`python -m tools.eval_accuracy`)
every weight from 3 to 16 predicts the winner about 80% of the time, and on the race positions
where weight 16 wrongly claimed a win, weight 6 showed the same scores. The dynamic cap removes
exactly the surplus walls a player cannot use before the race ends, keeps the weight that plays
best, and wins 69.5% against the uncapped evaluation. `python -m tools.calibration` measures how
often a bot's own eval claims a win it then loses in self-play.

`tempo_bonus` makes no measurable difference at a fixed depth (every leaf of one iteration has the
same side to move, so it shifts all scores equally); it only matters when comparing positions at
different depths through the TT.

### Wall modes at a fixed depth (no time limit)

`tools.match --depth D`, same seed per row; "A wins" counts decisive games (draws = move cap).

| Depth | All vs Attack + defense | Attack + defense vs Attack only | All vs Attack only | Nodes per move (all / both / attack) |
|---|---|---|---|---|
| 4 | 51.3% | 52.0% | 51.0% | 16.9 k / 2.2 k / 0.4 k |
| 5 | 47.0% | 49.7% | 52.0% | 84 k / 8.4-9.1 k / 1.1 k |
| 6 | 52.0% | 51.0% | - | 364 k / 25-29 k / 3.8 k |

At the same depth the wider modes are at most slightly stronger: every result is within the 95%
noise band of a few hundred games. Each wider mode sees a superset of the walls, so it is never
worse in principle, but walls that do not cut a shortest path rarely change the best move within
a 4-6 ply horizon. It costs roughly 8-10x more nodes per step up, so with a time limit the
narrower modes reach deeper and win (see the table above).

Note that at short time controls **attack-only filtering** (`wall_filter=2`) is clearly stronger
than the attack + defense default: the extra depth is worth more than the defensive walls it
leaves out. The default is the attack + defense filter as specified; switch with
`ab:...,wall_filter=2` or `set wall_filter 2`.
