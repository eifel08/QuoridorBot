"""Alpha-beta pruning bot (no machine learning).

Heuristic (from the bot's point of view)::

    Score = (steps the bot needs to reach its goal)
          - (steps the opponent needs to reach its goal)

Both step counts are BFS shortest-path lengths. The bot wants this score to
be as *small* as possible (few steps for itself, many for the opponent).
Internally the search is written as *negamax*: every position is scored for
the side to move as ``-Score`` (= opponent steps - own steps), so each
player maximises and alpha-beta pruning applies symmetrically. The value
reported in :attr:`AlphaBetaBot.last_info` is converted back to the
``Score`` convention above.

Search features:

* iterative deepening from depth 1 up to ``depth`` (1-30), stopping early
  when the optional ``time_limit`` runs out; the best move of the deepest
  *completed* iteration is played;
* move ordering (transposition-table move first, then pawn steps that
  shorten the own path, then walls by how much they help);
* a Zobrist-hashed transposition table (local to one ``get_move`` call);
* wall moves are restricted to walls that cut the opponent's current
  shortest path and improve the score. This keeps the branching factor
  around 5-20 instead of ~130 and is what makes depths > 3 practical.

The bot is stateless between turns: ``get_move`` builds a scratch copy of the
read-only :class:`~quoridor.state.BoardState` and throws it away afterwards.
"""

from __future__ import annotations

import random
import time
from collections import deque
from typing import Optional

from ..moves import Move, Orientation, PawnMove, WallMove
from ..player import Player
from ..state import BoardState

N = 9
NW = N - 1  # wall anchors per row/column
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
STEP = (-N, N, -1, 1)
OPPOSITE = (DOWN, UP, RIGHT, LEFT)

WIN = 1_000_000
EXACT, LOWER, UPPER = 0, 1, 2
MAX_DEPTH = 30

# Internal move encoding: ("p", square) or ("h"/"v", anchor) with
# square = row * 9 + col and anchor = row * 8 + col.
IMove = tuple[str, int]

_rng = random.Random(20240611)
Z_PAWN = [[_rng.getrandbits(64) for _ in range(N * N)] for _ in range(2)]
Z_WALL = {o: [_rng.getrandbits(64) for _ in range(NW * NW)] for o in "hv"}
Z_LEFT = [[_rng.getrandbits(64) for _ in range(11)] for _ in range(2)]
Z_TURN = _rng.getrandbits(64)


class SearchTimeout(Exception):
    pass


class FastBoard:
    """Compact mutable position used only inside one search."""

    def __init__(self, state: BoardState) -> None:
        self.pawns = [r * N + c for r, c in state.pawns]
        self.goal = list(state.goal_rows)
        self.walls_left = list(state.walls_left)
        self.turn = state.current_player
        self.h: set[int] = set()
        self.v: set[int] = set()
        # open_[sq][d] is True when the pawn can step from sq in direction d.
        self.open_ = [[True] * 4 for _ in range(N * N)]
        for sq in range(N * N):
            r, c = divmod(sq, N)
            if r == 0:
                self.open_[sq][UP] = False
            if r == N - 1:
                self.open_[sq][DOWN] = False
            if c == 0:
                self.open_[sq][LEFT] = False
            if c == N - 1:
                self.open_[sq][RIGHT] = False
        for r, c, o in state.walls:
            self._set_wall(o.value if isinstance(o, Orientation) else o, r * NW + c, False)
        self.hash = self._full_hash()

    # ------------------------------------------------------------------ #
    def _full_hash(self) -> int:
        h = Z_PAWN[0][self.pawns[0]] ^ Z_PAWN[1][self.pawns[1]]
        h ^= Z_LEFT[0][self.walls_left[0]] ^ Z_LEFT[1][self.walls_left[1]]
        for a in self.h:
            h ^= Z_WALL["h"][a]
        for a in self.v:
            h ^= Z_WALL["v"][a]
        if self.turn:
            h ^= Z_TURN
        return h

    @staticmethod
    def wall_edges(o: str, a: int) -> tuple[tuple[int, int], ...]:
        """(square, direction) pairs whose step the wall blocks (both ways)."""
        r, c = divmod(a, NW)
        s = r * N + c
        if o == "h":
            return ((s, DOWN), (s + N, UP), (s + 1, DOWN), (s + 1 + N, UP))
        return ((s, RIGHT), (s + 1, LEFT), (s + N, RIGHT), (s + N + 1, LEFT))

    def _set_wall(self, o: str, a: int, value: bool) -> None:
        """Remove (value=True) or place (value=False) a wall."""
        for sq, d in self.wall_edges(o, a):
            self.open_[sq][d] = value
        anchors = self.h if o == "h" else self.v
        if value:
            anchors.discard(a)
        else:
            anchors.add(a)

    def overlaps(self, o: str, a: int) -> bool:
        c = a % NW
        if o == "h":
            return (a in self.h or a in self.v or (c > 0 and a - 1 in self.h)
                    or (c < NW - 1 and a + 1 in self.h))
        return (a in self.v or a in self.h or (a >= NW and a - NW in self.v)
                or (a < NW * (NW - 1) and a + NW in self.v))

    # ------------------------------------------------------------------ #
    def distance(self, player: int) -> Optional[int]:
        start, goal = self.pawns[player], self.goal[player]
        if start // N == goal:
            return 0
        open_ = self.open_
        dist = {start: 0}
        q = deque([start])
        while q:
            sq = q.popleft()
            nd = dist[sq] + 1
            o = open_[sq]
            for d in range(4):
                if o[d]:
                    n = sq + STEP[d]
                    if n not in dist:
                        if n // N == goal:
                            return nd
                        dist[n] = nd
                        q.append(n)
        return None

    def shortest_path(self, player: int) -> list[int]:
        """Squares of one shortest path (start first), [] if none."""
        start, goal = self.pawns[player], self.goal[player]
        parent = {start: -1}
        q = deque([start])
        end = start if start // N == goal else -1
        while q and end < 0:
            sq = q.popleft()
            o = self.open_[sq]
            for d in range(4):
                if o[d]:
                    n = sq + STEP[d]
                    if n not in parent:
                        parent[n] = sq
                        if n // N == goal:
                            end = n
                            break
                        q.append(n)
        if end < 0:
            return []
        path = [end]
        while parent[path[-1]] != -1:
            path.append(parent[path[-1]])
        return path[::-1]

    def pawn_moves(self, player: int) -> list[int]:
        me, opp = self.pawns[player], self.pawns[1 - player]
        open_ = self.open_
        out: list[int] = []
        for d in range(4):
            if not open_[me][d]:
                continue
            n = me + STEP[d]
            if n != opp:
                out.append(n)
                continue
            if open_[opp][d]:
                out.append(opp + STEP[d])
                continue
            for sd in ((LEFT, RIGHT) if d in (UP, DOWN) else (UP, DOWN)):
                if open_[opp][sd]:
                    s = opp + STEP[sd]
                    if s != me and s not in out:
                        out.append(s)
        return list(dict.fromkeys(out))

    # ------------------------------------------------------------------ #
    def make(self, m: IMove) -> None:
        kind, x = m
        p = self.turn
        if kind == "p":
            self.hash ^= Z_PAWN[p][self.pawns[p]] ^ Z_PAWN[p][x]
            self.pawns[p] = x
        else:
            self._set_wall(kind, x, False)
            self.hash ^= Z_WALL[kind][x] ^ Z_LEFT[p][self.walls_left[p]] ^ Z_LEFT[p][self.walls_left[p] - 1]
            self.walls_left[p] -= 1
        self.turn = 1 - p
        self.hash ^= Z_TURN

    def unmake(self, m: IMove, prev_square: int) -> None:
        self.turn = 1 - self.turn
        self.hash ^= Z_TURN
        kind, x = m
        p = self.turn
        if kind == "p":
            self.hash ^= Z_PAWN[p][x] ^ Z_PAWN[p][prev_square]
            self.pawns[p] = prev_square
        else:
            self._set_wall(kind, x, True)
            self.hash ^= Z_WALL[kind][x] ^ Z_LEFT[p][self.walls_left[p]] ^ Z_LEFT[p][self.walls_left[p] + 1]
            self.walls_left[p] += 1

    def won(self, player: int) -> bool:
        return self.pawns[player] // N == self.goal[player]

    # ------------------------------------------------------------------ #
    # Helpers used by the ML pipeline (quoridor/ml)
    # ------------------------------------------------------------------ #
    @classmethod
    def initial(cls) -> "FastBoard":
        """Start position (Player 1 at (8, 4) to move)."""
        from ..board import Board

        return cls(Board().snapshot(0))

    def copy(self) -> "FastBoard":
        other = FastBoard.__new__(FastBoard)
        other.pawns = list(self.pawns)
        other.goal = list(self.goal)
        other.walls_left = list(self.walls_left)
        other.turn = self.turn
        other.h = set(self.h)
        other.v = set(self.v)
        other.open_ = [list(o) for o in self.open_]
        other.hash = self.hash
        return other

    def legal_walls(self, player: int) -> list[tuple[str, int]]:
        """All legal wall placements for ``player`` as ("h"|"v", anchor).

        Same result as ``Board.valid_walls`` but much faster: a wall that does
        not cut an edge of a player's current shortest path cannot disconnect
        that player, so only walls touching one of the two paths need a BFS.
        """
        if self.walls_left[player] <= 0:
            return []
        risky: set[tuple[str, int]] = set()
        for p in (0, 1):
            path = self.shortest_path(p)
            for a_sq, b_sq in zip(path, path[1:]):
                ra, ca = divmod(a_sq, N)
                rb, cb = divmod(b_sq, N)
                if ra == rb:
                    c = min(ca, cb)
                    for r in (ra - 1, ra):
                        if 0 <= r < NW and c < NW:
                            risky.add(("v", r * NW + c))
                else:
                    r = min(ra, rb)
                    for c in (ca - 1, ca):
                        if 0 <= c < NW and r < NW:
                            risky.add(("h", r * NW + c))
        out: list[tuple[str, int]] = []
        for o in "hv":
            for a in range(NW * NW):
                if self.overlaps(o, a):
                    continue
                if (o, a) in risky:
                    self._set_wall(o, a, False)
                    ok = self.distance(0) is not None and self.distance(1) is not None
                    self._set_wall(o, a, True)
                    if not ok:
                        continue
                out.append((o, a))
        return out

    def legal_moves(self) -> list[tuple[str, int]]:
        """Every legal move for the side to move (pawn moves first)."""
        me = self.turn
        return [("p", sq) for sq in self.pawn_moves(me)] + self.legal_walls(me)

    def play(self, m: tuple[str, int]) -> Optional[int]:
        """Apply a move for the side to move; returns the winner index or None."""
        mover = self.turn
        self.make(m)
        return mover if m[0] == "p" and self.won(mover) else None


def evaluate(board: FastBoard) -> int:
    """Negamax value for the side to move: opponent steps - own steps (= -Score)."""
    me = board.turn
    return board.distance(1 - me) - board.distance(me)  # type: ignore[operator]


class AlphaBetaSearch:
    def __init__(self, board: FastBoard, deadline: Optional[float],
                 rng: Optional[random.Random] = None) -> None:
        self.b = board
        self.deadline = deadline
        self.rng = rng
        self.nodes = 0
        self.tt: dict[int, tuple[int, int, int, Optional[IMove]]] = {}

    # ------------------------------------------------------------------ #
    def ordered_moves(self, tt_move: Optional[IMove], shuffle: bool = False) -> list[IMove]:
        b = self.b
        me, opp = b.turn, 1 - b.turn
        my_d = b.distance(me)
        opp_d = b.distance(opp)
        assert my_d is not None and opp_d is not None
        scored: list[tuple[int, IMove]] = []

        start = b.pawns[me]
        for sq in b.pawn_moves(me):
            b.pawns[me] = sq
            d = b.distance(me)
            b.pawns[me] = start
            scored.append((opp_d - d, ("p", sq)))  # type: ignore[operator]

        if b.walls_left[me] > 0:
            path = b.shortest_path(opp)
            cands: dict[IMove, None] = {}
            for a_sq, b_sq in zip(path, path[1:]):
                ra, ca = divmod(a_sq, N)
                rb, cb = divmod(b_sq, N)
                if ra == rb:  # horizontal step -> vertical wall
                    c = min(ca, cb)
                    for r in (ra - 1, ra):
                        if 0 <= r < NW and c < NW:
                            cands[("v", r * NW + c)] = None
                else:  # vertical step -> horizontal wall
                    r = min(ra, rb)
                    for c in (ca - 1, ca):
                        if 0 <= c < NW and r < NW:
                            cands[("h", r * NW + c)] = None
            base = opp_d - my_d
            for m in cands:
                kind, a = m
                if b.overlaps(kind, a):
                    continue
                b._set_wall(kind, a, False)
                nd_opp = b.distance(opp)
                nd_me = b.distance(me)
                b._set_wall(kind, a, True)
                if nd_opp is None or nd_me is None:
                    continue  # would block a player completely: illegal
                gain = nd_opp - nd_me
                if gain > base:
                    scored.append((gain, m))

        if shuffle and self.rng is not None:
            self.rng.shuffle(scored)  # random tie-breaking among equally ordered moves
        scored.sort(key=lambda t: -t[0])  # stable: pawn moves win ties
        moves = [m for _, m in scored]
        if tt_move is not None and tt_move in moves:
            moves.remove(tt_move)
            moves.insert(0, tt_move)
        return moves

    def negamax(self, depth: int, alpha: int, beta: int, ply: int) -> int:
        self.nodes += 1
        if self.deadline is not None and (self.nodes & 255) == 0 and time.perf_counter() > self.deadline:
            raise SearchTimeout()
        b = self.b
        if depth == 0:
            return evaluate(b)

        alpha0 = alpha
        entry = self.tt.get(b.hash)
        tt_move = None
        if entry is not None:
            e_depth, e_val, e_flag, tt_move = entry
            if e_depth >= depth:
                if e_flag == EXACT:
                    return e_val
                if e_flag == LOWER and e_val >= beta:
                    return e_val
                if e_flag == UPPER and e_val <= alpha:
                    return e_val

        me = b.turn
        moves = self.ordered_moves(tt_move)
        if not moves:  # boxed in (very rare): score the position as it stands
            return evaluate(b)
        best_val, best_move = -WIN * 2, None
        for m in moves:
            prev = b.pawns[me]
            b.make(m)
            if m[0] == "p" and b.won(me):
                val = WIN - (ply + 1)
            else:
                val = -self.negamax(depth - 1, -beta, -alpha, ply + 1)
            b.unmake(m, prev)
            if val > best_val:
                best_val, best_move = val, m
            if val > alpha:
                alpha = val
            if alpha >= beta:
                break

        flag = EXACT
        if best_val <= alpha0:
            flag = UPPER
        elif best_val >= beta:
            flag = LOWER
        self.tt[b.hash] = (depth, best_val, flag, best_move)
        return best_val

    def root(self, depth: int, pv_move: Optional[IMove]) -> tuple[int, IMove]:
        b = self.b
        me = b.turn
        alpha, beta = -WIN * 2, WIN * 2
        best_val, best_move = -WIN * 2, None
        for m in self.ordered_moves(pv_move, shuffle=True):
            prev = b.pawns[me]
            b.make(m)
            if m[0] == "p" and b.won(me):
                val = WIN - 1
            else:
                val = -self.negamax(depth - 1, -beta, -alpha, 1)
            b.unmake(m, prev)
            if val > best_val:
                best_val, best_move = val, m
            alpha = max(alpha, val)
        assert best_move is not None
        return best_val, best_move


def search(state: BoardState, depth: int, time_limit: Optional[float] = None,
           rng: Optional[random.Random] = None) -> tuple[Move, dict]:
    """Iterative-deepening alpha-beta search. Returns (move, info).

    ``rng`` (optional) breaks ties between equally good moves at random;
    without it the bot is fully deterministic.
    """
    best_move, info = search_board(FastBoard(state), depth, time_limit, rng)
    kind, x = best_move
    if kind == "p":
        move: Move = PawnMove(*divmod(x, N))
    else:
        r, c = divmod(x, NW)
        move = WallMove(r, c, Orientation(kind))
    info["move"] = str(move)
    return move, info


def search_board(board: FastBoard, depth: int, time_limit: Optional[float] = None,
                 rng: Optional[random.Random] = None) -> tuple[IMove, dict]:
    """Same as :func:`search` on a :class:`FastBoard` (left unchanged).

    Returns the internal move ``("p", square)`` / ``("h"|"v", anchor)``.
    """
    depth = max(1, min(int(depth), MAX_DEPTH))
    start = time.perf_counter()
    deadline = start + time_limit if time_limit else None
    s = AlphaBetaSearch(board.copy(), deadline, rng)
    best_move: Optional[IMove] = None
    best_val = 0
    reached = 0
    for d in range(1, depth + 1):
        try:
            best_val, best_move = s.root(d, best_move)
        except SearchTimeout:
            break
        reached = d
        if abs(best_val) >= WIN - 1000:
            break  # forced win/loss found: deeper search cannot change it

    if best_move is None:  # timed out before depth 1 finished: take the best-ordered move
        best_move = s.ordered_moves(None)[0]

    info = {
        "depth_requested": depth,
        "depth_reached": reached,
        "nodes": s.nodes,
        "seconds": round(time.perf_counter() - start, 3),
        # Score = own steps - opponent steps (lower is better for the bot).
        "score": -best_val if abs(best_val) < WIN - 1000 else None,
        "forced": None if abs(best_val) < WIN - 1000 else ("win" if best_val > 0 else "loss"),
        # Backed-up search value in steps for the side to move (higher is better), or the
        # number of half-moves to a forced win/loss; see quoridor.evaluation.
        "eval_steps": best_val if reached and abs(best_val) < WIN - 1000 else None,
        "plies": WIN - abs(best_val) if abs(best_val) >= WIN - 1000 else None,
    }
    return best_move, info


class AlphaBetaBot(Player):
    """Player that picks moves with alpha-beta search.

    Args:
        name: Display name.
        depth: Maximum search depth in plies (1-30).
        time_limit: Optional seconds per move; the search stops after the
            deepest iteration that finished in time. ``None`` = no limit.
    """

    def __init__(self, name: str = "Bot", depth: int = 3, time_limit: Optional[float] = 5.0) -> None:
        super().__init__(name)
        self.depth = max(1, min(int(depth), MAX_DEPTH))
        self.time_limit = time_limit
        self.rng: Optional[random.Random] = None  # set for random tie-breaking (varied games)
        self.last_info: dict = {}

    def get_move(self, board_state: BoardState) -> Move:
        move, self.last_info = search(board_state, self.depth, self.time_limit, self.rng)
        return move
