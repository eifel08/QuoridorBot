# QuoridorBot

A 2-player [Quoridor](https://en.wikipedia.org/wiki/Quoridor) game emulator in Python, with
three front-ends: a **browser version** (works from a plain link, also on phones), a pygame
desktop GUI and a terminal CLI. Play **Human vs Human** on one device, or **Human vs Bot** against
an alpha-beta search bot with an adjustable search depth (1-30).

![GUI screenshot](docs/screenshot.png)

## Play in the browser

**Online:** once GitHub Pages is on (see below), the game is at
**https://eifel08.github.io/QuoridorBot/**. Share that link and play on one device (pass and play).
Names can be preset in the URL: `https://eifel08.github.io/QuoridorBot/?p1=Alice&p2=Bob`.

**Locally** (Python only, nothing to install):

```bash
python3 main.py --web
```

That serves the `web/` folder at http://localhost:8000 and opens a browser tab
(`--port 9000` to change the port, `--no-browser` to skip opening a tab). Any static file
server works too, e.g. `cd web && python3 -m http.server`.

The web version is plain HTML/CSS/JavaScript in `web/` with no build step and no external
libraries. `web/engine.js` is a port of the Python rules, and `tests/test_web_engine.py`
checks that it computes exactly the same legal moves, legal walls and path lengths as the
Python engine on a few hundred random positions. It also has *Undo*, editable player names,
and a console API (`Quoridor.state()` returns the same dict as `BoardState.to_dict()`).

### Turning on the GitHub Pages link (one-time)

1. Merge this work into the `main` branch.
2. On GitHub: **Settings -> Pages -> Build and deployment -> Source: GitHub Actions**.
3. Every push to `main` then runs the tests and publishes `web/` via
   `.github/workflows/pages.yml` (it can also be started by hand from the *Actions* tab).

## Game modes

Every front-end lets you pick the mode before a game:

| | Browser | Desktop window | Terminal |
|---|---|---|---|
| Choose the mode | *Human vs Human / Human vs Bot / Bot vs Bot* switch at the top | start menu | prompt at start |
| Search depth (1-30) | slider | slider (or arrow keys) | prompt |
| Think-time cap | dropdown (1 s ... no limit) | buttons | prompt |
| Play Red (first) or Blue | dropdown | buttons | prompt |

- **Human vs Human**: two people take turns on the same computer.
- **Human vs Bot**: you play against the alpha-beta bot. After each bot move, the depth it reached,
  the number of positions it searched, the time it took and its score are shown (under the board in
  the browser, in the side panel on the desktop, after the move in the terminal).
  In the browser, *Undo* takes back your move and the bot's reply.
- **Bot vs Bot**: two bots with independent settings play each other while you watch, or run
  hundreds of games without a board. See the next section.

Shortcuts: `python3 main.py --mode bot --depth 6 --time-limit 3 --play-as 2` skips the desktop menu
(add `--cli` for the terminal). In the browser, `?mode=bot&depth=6` preselects the mode and depth.

## Bot vs Bot

Set up two bots with different settings (e.g. depth 2 against depth 4, or a trained ML agent
against alpha-beta) and either **watch** a game move by move or run a fast **batch** of games for
statistics.

### Watching a game

**Browser** (`python3 main.py --web` or the GitHub Pages link): choose *Bot vs Bot*, then set:
- a depth slider (1-30) and a think-time limit for the Red bot and, separately, for the Blue bot;
- *Speed*: the pause between moves (Slow / Normal / Fast / Instant);
- *Vary games*: the bots pick at random between equally good moves, so repeated games differ.
  The alpha-beta bot is otherwise fully deterministic;
- *Play continuously*: starts the next game automatically and keeps a running score;
- *Pause* / *Step* / *Undo*: stop, play one move at a time, or go back one move (Space also
  pauses).

`?mode=bots&red=2&blue=5` in the URL preselects the mode and both depths.

**Desktop window**: `python3 main.py`, then *Bot vs Bot* in the menu. Each bot has its own depth
slider and think-time buttons, and there's a choice of pause between moves. While a game runs:
Space pauses or resumes, R goes back to the menu. To skip the menu:

```bash
python3 main.py --mode bots --bot1 ab:depth=2 --bot2 ab:depth=4 --delay 0.5
```

**Terminal**: `python3 main.py --cli`, choose `3`, or add the same flags with `--cli`. The board
is printed after every move.

### Batch mode (fast, no board)

```bash
python3 -m quoridor.botmatch --bot1 ab:depth=2 --bot2 ab:depth=4 --games 100
```

(`python3 main.py --mode bots --headless --games 100 ...` does the same.) Example output:

```
        spec                                wins   win %  CI low  as Red  as Blue  s/move  walls
Bot 1   ab:depth=2                            33   33.0%   24.6%   42.0%    24.0%   0.004    7.4
Bot 2   ab:depth=4                            67   67.0%   57.3%   76.0%    58.0%   0.054    9.7
100 games, 0 draws, 65 moves per game on average, 51.0s total
```

- **Speed**: games run in parallel on all CPU cores (`--workers N` to change). The 100 games above
  took 51 s on 4 cores.
- **Fairness**: the bots **swap colours every game** (moving first is an advantage; `--no-swap`
  keeps Bot 1 on Red).
- **Varied openings**: each game starts with 0-6 random moves (`--opening-plies`, 0 = normal start).
  Without them, two deterministic bots would play the same game every time. `--seed` fixes the
  openings, so different settings can be compared on identical games.
- **Columns**: `CI low` is the 95% lower confidence bound of the win rate. `s/move` is the average
  thinking time per move, and `walls` the average number of walls placed per game.
- **Saving results**: `--json results.json` saves the summary plus every game, and
  `--csv games.csv` saves one row per game.
- Games that reach 200 moves count as draws.

### Bot settings

`--bot1` / `--bot2` take a short spec (the same format everywhere):

| Spec | Bot |
|---|---|
| `ab` | alpha-beta, depth 4, no time limit |
| `ab:depth=2` | alpha-beta, depth 2 |
| `ab:depth=8,time=2` | alpha-beta, depth 8, at most 2 s per move |
| `ab:depth=4,random=1` | alpha-beta that picks at random between equally good moves |
| `ab:depth=30,time=1,futility=0` | C++ engine with a flag changed (flags: [engine/README.md](engine/README.md)) |
| `ab:depth=4,engine=py` | force the Python search (`engine=cpp` forces C++; default: C++ if built) |
| `ml:model=runs/exp1/best.pt` | trained ML agent (needs PyTorch), 64 MCTS simulations per move |
| `ml:model=best.pt,sims=0` | ML agent playing its network's first choice without search (fast) |

Example, checking a trained model against the bot it was trained on:

```bash
python3 -m quoridor.botmatch --bot1 ml:model=runs/exp1/best.pt --bot2 ab:depth=4 --games 200
```

## The C++ search engine

The bots think with a C++ engine (`engine/`) when it has been built, and with the Python search
described in the next section otherwise. Build it once:

```bash
python -m quoridor.bots.cpp_engine build
```

It implements the same rules, searches millions of positions per second, and adds a
transposition table, wall filtering, decided-position cutoffs and futility pruning. With it, set a
high depth and let the think time decide (e.g. `python main.py --mode bot --depth 30 --time-limit 2`).
`python main.py --engine py` forces the Python search. Architecture, flags, benchmarks and the
verification tools are documented in [engine/README.md](engine/README.md).

## The alpha-beta bot (Python reference)

Code: `quoridor/bots/alphabeta.py` (`AlphaBetaBot`), ported line by line to `web/bot.js` for the
browser. A test checks that both versions pick the same move, with the same score and the same
number of searched positions.

**Heuristic.** For a position, with both step counts measured as BFS shortest paths (pawns ignored):

```
Score = (steps the bot needs to reach its goal) - (steps the opponent needs to reach its goal)
```

The bot wants the Score as **low** as possible: few steps for itself, many for the opponent. The
search is written as *negamax*: each position is scored for the side to move as `-Score`
(opponent steps - own steps) and both sides maximise. This is the same minimax with alpha-beta
pruning, just with one function instead of separate min and max functions. A win found during the
search scores +/- 1,000,000; faster wins score slightly higher.

**Search.**
- *Iterative deepening*: it searches depth 1, then 2, ... up to the chosen depth. If the think-time
  cap runs out, it plays the best move from the deepest depth it fully finished. The depth slider
  is the maximum; the cap keeps deep settings responsive. Pick "no limit" to always search the
  full depth.
- *Move ordering*: the best move from the previous depth (stored in a transposition table) is tried
  first, then pawn steps that shorten the bot's own path, then walls by how much they help.
  Good ordering lets alpha-beta skip more of the tree.
- *Wall pruning*: only walls that cut one of the opponent's current shortest paths **and** improve
  the score are considered (~5-20 instead of ~130). This is what makes depths beyond 3 practical.
  The trade-off is that walls placed purely to prepare a later trap are not seen.
- *Stateless*: `get_move(board_state)` copies the read-only `BoardState` into a scratch board,
  searches, and throws it away. The bot keeps no memory between turns.

**Speed vs depth** (one mid-game position, no time cap; the browser is ~6x faster than Python):

| Depth | Positions | Python | Browser (JS) |
|---|---|---|---|
| 4 | ~100 | 0.02 s | <0.01 s |
| 6 | ~840 | 0.2 s | 0.05 s |
| 8 | ~7,400 | 1.7 s | 0.35 s |
| 10 | ~63,000 | 16 s | 2.4 s |
| 12+ | grows ~8x per 2 plies | use the time cap | use the time cap |

Depth 1 only looks at its own next move, so it walks straight and never defends. Depths 4-8 play
sensibly and answer fast. Above ~10, the time cap decides how deep it actually gets. Even depths
tend to be more cautious than odd ones, because an even depth always ends the search on the
opponent's reply.

Use the bot from code:

```python
from quoridor import Game
from quoridor.bots import AlphaBetaBot, search

game = Game(AlphaBetaBot("Fast", depth=2), AlphaBetaBot("Deep", depth=6, time_limit=2))
winner = game.run()

move, info = search(game.state, depth=6, time_limit=None)  # one-off analysis
```

## Training an ML agent (Google Colab)

The `quoridor/ml/` package trains a neural-network agent whose goal is to **win at least 95% of
games against the Depth-4 alpha-beta bot**. Training stops by itself and saves
`milestone_95.pt` once it gets there.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eifel08/QuoridorBot/blob/feature/ml-training-pipeline/notebooks/train_colab.ipynb)

### Step by step in Colab

1. Click the badge above, or go to [colab.research.google.com](https://colab.research.google.com),
   choose *File → Open notebook → GitHub*, paste `eifel08/QuoridorBot`, select the branch
   `feature/ml-training-pipeline` and open `notebooks/train_colab.ipynb`.
   (Once this branch is merged into `main`, change `blob/feature/ml-training-pipeline` in the
   link to `blob/main` and `BRANCH` in the first cell to `main`.)
2. *Runtime → Change runtime type → T4 GPU* (or any GPU). Games run on the CPU, so a runtime
   with more CPU cores (Colab Pro, High-RAM) trains faster. The notebook prints how many cores
   you got.
3. *Runtime → Run all*. Allow access to Google Drive when asked. The notebook:
   - clones this repository and installs `requirements-ml.txt` (PyTorch is already on Colab);
   - runs a one-minute self-test;
   - opens TensorBoard;
   - starts training with checkpoints in `MyDrive/QuoridorBot/runs/<RUN_NAME>/`.
4. Watch the progress (next section). When the session ends or disconnects, reconnect and run all
   cells again: training continues from the last checkpoint in the same `RUN_NAME` folder.
5. At the end, the notebook re-tests the final model on 200 new openings and offers it for
   download. Play against it with `python main.py --ml-model milestone_95.pt`.

### Monitoring

- **Training output**, one line per iteration, plus an evaluation every few iterations:
  ```
  [ 0.19h it   18] train: agent won 17% of 24 games, loss p=0.992 v=0.142, 77s
  [ 0.21h it   18] EVAL vs Depth-4: win rate 65.0% (26/40, 95% CI low 49.5%, red 70% / blue 60%)
  ```
- **TensorBoard** (started in the notebook): `eval/win_rate` is the number that has to reach 0.95.
  Also useful: `train/train_win_rate`, `train/policy_loss`, `train/value_loss`.
- **`metrics.jsonl`** in the run folder, one JSON record per event. Notebook step 8 plots it,
  even after a disconnect.
- **Files** in the run folder:

  | File | Contents |
  |---|---|
  | `latest.pt` | last iteration's model and optimizer state, used to resume |
  | `best.pt` | the model with the best evaluation win rate so far |
  | `milestone_95.pt` | written when 95% is reached and confirmed |
  | `buffer.npz` | recent training positions, so a resumed run doesn't start with empty data |

### How the training works

- **Opponent and evaluation.** The Depth-4 bot is deterministic, so every game starts from a
  *seeded random opening* (0-6 random moves). Otherwise each evaluation game would be identical and
  the win rate would always be 0% or 100%. An evaluation is `--eval-games` games (default 100)
  against the deterministic bot on the same fixed openings each time, so checkpoints are compared
  fairly. The agent plays Red in half of them and Blue in the other half.
- **Stopping condition.** When an evaluation reaches `--target-win-rate` (default 0.95), the model
  plays `--confirm-games` (default 200) *new* openings, to rule out a lucky run on the fixed set.
  If that also reaches the target, it is saved as `milestone_95.pt` and training stops.
  `--keep-training` continues after that. The output also shows a 95% confidence lower bound
  ("CI low") for each win rate.
- **Agent.** A residual convolutional network (6 blocks × 64 channels, ~0.5 M parameters).
  - *Input*: pawns, walls, walls left and both players' BFS distance maps, always encoded from the
    side to move. Inside the network input only, Blue's view is mirrored so it also "moves up";
    the game itself is never flipped.
  - *Outputs*: a move distribution over 209 actions (81 pawn squares, 64 horizontal and 64
    vertical walls), with illegal moves masked out, and a value in [-1, 1] estimating who wins.
  - *Playing*: an MCTS (PUCT) search with `--sims` simulations per move picks the move.
- **Phase 1, warm-up by imitation.** `--warmup-games` games of alpha-beta against alpha-beta. The
  network learns to copy the bot's moves and predict the winner, so the next phase doesn't start
  from random play.
- **Phase 2, RL against the Depth-4 bot (expert iteration).** Each iteration:
  - the agent plays `--games-per-iter` games against the bot, with exploration noise and random
    moves early in the game;
  - its positions get the MCTS visit counts as the move target and the game result as the value
    target. The bot's own moves are also imitated (`--learn-opponent-moves`);
  - `--train-steps` training steps on a replay buffer, with left-right mirroring as augmentation;
  - `latest.pt` is saved.
- **Parallelism.** Games run on all CPU cores (`--workers`), and the network trains on the GPU if
  there is one.

### Expectations (read this before a long run)

Deeper alpha-beta with the same heuristic does **not** come close to 95% against Depth 4.
Measured with `python -m quoridor.ml.evaluate --baseline D` (40 games each, random openings):

| Agent | Win rate vs Depth 4 |
|---|---|
| alpha-beta depth 1 | 5% |
| alpha-beta depth 2 | 35% |
| alpha-beta depth 3 | 55% |
| alpha-beta depth 4 | 45% (mirror match) |
| alpha-beta depth 5 | 50% |
| alpha-beta depth 6 | 50% |

So the ML agent has to learn the Depth-4 bot's specific blind spots: it only considers walls that
immediately lengthen the opponent's path, and it doesn't look past 4 moves. That is learnable, but
it takes many games. Expect hours to days of compute rather than minutes, and there is no
guarantee a given run reaches 95%. `best.pt` always holds the strongest model so far.

**What a short local test run did** (4 CPU cores, no GPU, about 1.3 hours, 500 warm-up games,
24 games per iteration, 48 simulations per move):

| Stage | Win rate vs Depth 4 |
|---|---|
| imitation warm-up only, plain MCTS, first version | 0% (0/40) |
| after adding the wall-gain input planes and the 50/50 heuristic mix | 12.5% (iteration 2) |
| RL, learning rate 1e-3 | rose to 42.5% (iteration 12), then fell to 5% (iteration 20) |
| RL from the best checkpoint with `--rl-lr 2e-4` | 30-65% per 40-game evaluation |
| that best checkpoint, re-tested on 100 fresh openings | **49%** (95% CI lower bound 39%) |

Lessons that are now built into the defaults:
- The **wall-gain planes** and **heuristic mix** were what made the agent competitive at all.
- A lower **RL learning rate** plus **`--revert-drop`** fight the collapses.
- 40-game evaluations swing by ±15 points, and the best of many such evaluations overstates the
  true strength (65% measured, 49% on re-test). This is why the default evaluation uses 100
  games and why the 95% stop is confirmed on 200 fresh openings.

To get from about 50% towards 95%, give it far more games: a Colab run with more cores for many
hours. You can also try `--sims 96`, `--games-per-iter 96`, a bigger network
(`--channels 96 --blocks 8`) or a lower `--heuristic-mix` once the value loss has dropped.


### Running locally and main options

```bash
pip install -r requirements-ml.txt
```
```bash
python -m quoridor.ml.train --out runs/exp1                    # start or resume
```
```bash
python -m quoridor.ml.train --out runs/exp1 --preset smoke     # 2-minute pipeline check
```
```bash
python -m quoridor.ml.evaluate --model runs/exp1/best.pt --games 200
```

| Option | Default | Meaning |
|---|---|---|
| `--preset` | none | `smoke` (tiny test), `colab-free` (lighter iterations), `full` |
| `--target-win-rate` | 0.95 | stop when this win rate vs Depth 4 is reached and confirmed |
| `--eval-games` / `--eval-every` | 100 / 2 | evaluation size and frequency (in iterations) |
| `--confirm-games` | 200 | fresh-opening games to confirm the target (0 = skip) |
| `--games-per-iter` / `--train-steps` | 48 / 300 | games and training steps per iteration |
| `--sims` / `--eval-sims` | 64 / 64 | MCTS simulations per move in training / evaluation |
| `--warmup-games` / `--warmup-steps` | 1000 / 4000 | imitation warm-up size |
| `--heuristic-mix` | 0.5 | MCTS value = (1 − mix) × network + mix × path heuristic (0 = network only) |
| `--lr` / `--rl-lr` | 1e-3 / 2e-4 | learning rate for the warm-up / for the RL phase |
| `--revert-drop` | 0.25 | reload `best.pt` if an evaluation drops this far below the best (0 = off) |
| `--channels` / `--blocks` | 64 / 6 | network size |
| `--opponent-depth` | 4 | which alpha-beta depth to train against and evaluate on |
| `--max-hours` | none | save and stop cleanly after this long (resume later) |
| `--workers` | all cores | parallel game processes |
| `--fresh` / `--keep-training` | off | ignore an existing run / keep going after the milestone |

`python -m quoridor.ml.train --help` lists all options.

## Desktop and terminal versions

Requires Python 3.9+. Install the dependency (only needed for the desktop GUI):

```bash
pip install -r requirements.txt
```

This installs [pygame-ce](https://pyga.me/), the community edition of pygame. The code still
does `import pygame`; pygame-ce is used because it ships prebuilt wheels for new Python
versions (e.g. 3.14 on Windows), whereas plain `pygame` tries to compile from source there and
fails. If you previously installed plain `pygame`, run `pip uninstall pygame` first, since the
two can't be installed side by side.

| Command | Version |
|---|---|
| `python3 main.py` | pygame desktop window (starts with the mode menu) |
| `python3 main.py --p1 Alice --p2 Bob` | desktop window with names |
| `python3 main.py --cli` | terminal (no dependencies; asks for the mode) |
| `python3 main.py --mode bot --depth 6` | desktop window, straight into Human vs Bot |
| `python3 main.py --ml-model best.pt` | play against a trained ML agent (needs PyTorch) |
| `python3 main.py --mode bots --bot1 ab:depth=2 --bot2 ab:depth=4` | watch Bot vs Bot |
| `python3 -m quoridor.botmatch --games 100` | fast headless Bot vs Bot batch with statistics |
| `python3 main.py --web` | browser version on localhost |

Run the tests (GUI tests run headless and are skipped without pygame; the web engine test
needs Node.js and is skipped without it):

```bash
python3 -m unittest -v
```

## Controls (browser and desktop GUI)

- **Goal rows** are tinted: the top row (row 0) in red is Player 1's goal, the bottom row
  (row 8) in blue is Player 2's goal. The active player's pawn pulses and their card is outlined.
- **Move the pawn**: click your pawn to show grey dots on every legal destination (jumps
  included), then click one. You can also click a legal square directly, or drag your pawn
  onto it. An illegal drop slides the pawn back.
- **Place a wall**: drag one from your reserve (your card in the side panel) onto the board.
  A semi-transparent preview snaps to the nearest groove:
  - over a **horizontal** groove the wall covers the hovered square and the one **to its left**;
  - over a **vertical** groove it covers the hovered square and the one **below**;
  - if that spot is illegal, the preview tries the spot shifted by one square; if both are
    illegal the preview turns red and dropping sends the wall back with the reason.
- **Right-click / Esc** cancels a drag. Desktop: **R** or *New game* restarts.
  Browser: *New game*, *Undo* (Ctrl+Z); touch works the same as the mouse.

## Rules implemented

- 9x9 board. **Player 1** (`1`) starts at row 8, col 4 and must reach **row 0**.
  **Player 2** (`2`) starts at row 0, col 4 and must reach **row 8**.
- Each player has **10 walls**.
- On a turn you either move your pawn or place a wall.
- Pawns move one square orthogonally. If the opponent is adjacent you may jump
  straight over them; if that jump is blocked by a wall or the board edge you may
  jump diagonally to either side of them instead (unless a wall blocks that side).
- Walls are 2 squares long, horizontal or vertical. They cannot go out of bounds,
  overlap another wall, or cross another wall at its center.
- A wall may not cut off **either** player from their goal row (checked with BFS).

## CLI input format

Coordinates are `row col`, both 0-based; row 0 is the top of the board.

| Command | Meaning | Example |
|---|---|---|
| `m <row> <col>` | Move your pawn to that square (jumps included) | `m 7 4` |
| `w <row> <col> h` | Horizontal wall under squares `(row,col)` and `(row,col+1)` | `w 3 4 h` |
| `w <row> <col> v` | Vertical wall right of squares `(row,col)` and `(row+1,col)` | `w 3 4 v` |
| `moves` | Show your legal pawn moves (marked `*`) | |
| `help` | Show commands | |
| `quit` | End the game | |

For walls, `(row, col)` is the top-left square of the 2x2 block the wall sits in,
so both values range from 0 to 7. Walls are drawn with `#`:

```
      0   1   2   3   4   5   6   7   8
    +---+---+---+---+---+---+---+---+---+
 3  | . | . | . | . | . | . | . | . | . |
    +---+---+---+---+#######+---+---+---+     <- w 3 4 h
 4  | . | . | . | . | . | . | . | . | . |
```

Invalid input or illegal moves are rejected with an explanation and you are asked again.

## Code layout

```
main.py                Launcher: desktop GUI, --cli, --web
requirements.txt       pygame-ce (desktop GUI only)
web/                   Browser version (static site, deployed to GitHub Pages)
  index.html, style.css, app.js   UI: SVG board, drag-and-drop, animations
  engine.js            JS port of the rules (checked against Python by tests)
  bot.js, bot-worker.js  JS port of the alpha-beta bot, run in a Web Worker
.github/workflows/     Tests + GitHub Pages deployment
engine/                C++ search engine (see engine/README.md): src/, Makefile, CMakeLists.txt
tools/                 engine checks: parity, verify_search, bench, match
quoridor/
  moves.py             PawnMove / WallMove / Orientation value types
  state.py             BoardState: immutable, serializable snapshot given to players
  board.py             Mutable rules engine: movement/jumps, wall validation, BFS
  player.py            Player (abstract base), CLI HumanPlayer, command parser
  bots/alphabeta.py    AlphaBetaBot: negamax alpha-beta search, heuristic, iterative deepening
  bots/cpp_engine.py   bridge to the C++ engine: CppEngine (process), CppBot (drop-in AlphaBetaBot)
  bots/factory.py      bot spec strings ("ab:depth=4", "ml:model=...") -> players / fast policies
  botmatch.py          headless Bot vs Bot batch runner (parallel, statistics, JSON/CSV)
  ml/                  ML agent + training pipeline (PyTorch, see requirements-ml.txt)
    arena.py           fast game runner, seeded random openings, win-rate statistics
    encoding.py        board -> network planes, 209-action space, mirroring
    model.py           policy-value ResNet, checkpoints
    mcts.py            PUCT Monte Carlo tree search
    selfplay.py        warm-up / training / evaluation games (worker processes)
    train.py           training loop, evaluation, 95% stopping condition, resume
    evaluate.py        standalone win-rate check
    agent.py           MLPlayer: a trained model as a normal Player
notebooks/train_colab.ipynb  Google Colab notebook for cloud training
requirements-ml.txt  torch, numpy, tensorboard (training only)
  game.py              Game engine + GameObserver hooks for views
  renderer.py, cli.py  ASCII rendering and terminal front-end
  gui/
    layout.py          Pure pixel geometry (squares, grooves, wall snapping)
    app.py             QuoridorGUI: pygame drawing, mouse input, animations
    anim.py, theme.py  Tweens and colors
    menu.py            Start menu: mode, depth sliders, think times, side, move delay
    player.py          GuiHumanPlayer, GuiBotPlayer (runs a bot in a background thread)
tests/                 unittest suite
```

### State vs. rendering

The engine owns the only mutable `Board`. Each turn it builds a frozen `BoardState` and
hands it to the player (`get_move(board_state)`) and to observers such as the GUI. Views only
read snapshots, so drawing can never change the game, and coordinates are never flipped:
row 0 is always the top row for every player.

### Writing a bot

Subclass `Player` and implement `get_move(board_state) -> Move`. The `BoardState` holds
everything needed, so the bot doesn't have to remember anything between turns:

| Field / method | Content |
|---|---|
| `current_player`, `opponent` | Your index (0 or 1) and the other player's |
| `pawns`, `goal_rows`, `walls_left` | Per-player tuples |
| `walls` | `(row, col, Orientation)` for every placed wall |
| `valid_pawn_moves`, `valid_walls` | All legal moves for the player to move |
| `shortest_paths` | BFS distance of each player to their goal |
| `wall_matrix(o)`, `pawn_matrix(p)`, `to_planes()` | 8x8 / 9x9 0-1 grids for ML models |
| `to_dict()`, `to_json()`, `from_dict()` | Serialization |
| `to_board()` | A mutable `Board` copy for search (`apply_move`, `wall_error`, ...) |

`Player.observe(state)` is also called after every move if a bot wants to think on the
opponent's turn.
