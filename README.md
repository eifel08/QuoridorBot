# QuoridorBot

A 2-player [Quoridor](https://en.wikipedia.org/wiki/Quoridor) game emulator in Python.
Phase 1: the core rules engine and a Human vs. Human command-line interface.
A `BotPlayer` will be added later behind the same `Player` interface.

## Running

Requires Python 3.9+ and no third-party packages.

```bash
python3 main.py
```

Run the tests:

```bash
python3 -m unittest -v
```

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

## Input format

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
main.py              CLI entry point (Human vs. Human)
quoridor/
  moves.py           PawnMove / WallMove / Orientation value types
  board.py           Board state, movement + jump rules, wall validation, BFS
  player.py          Player (abstract base), HumanPlayer, command parser
  game.py            Game engine: turn loop, validation, win detection
  renderer.py        ASCII board rendering
tests/               unittest suite
```

### Adding a bot

Subclass `Player` and implement `get_move(board_state) -> Move`. The engine passes a
copy of the `Board`; `self.index` (0 or 1) tells the player which pawn it controls.
Helpers such as `board.valid_pawn_moves(i)`, `board.is_valid_wall(i, wall)` and
`board.shortest_path_length(i)` are available for search/evaluation.
