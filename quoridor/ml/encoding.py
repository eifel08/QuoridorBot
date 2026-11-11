"""Neural-network input/output encoding.

Everything is encoded from the point of view of the *side to move* ("me"):
if Player 2 is to move, the board is mirrored top-to-bottom so that "me"
always starts near row 8 and walks towards row 0. This only happens inside
the network's input tensor; the real game state is never flipped.

Input planes (``N_PLANES x 9 x 9`` float32):
    0  my pawn                       4  my walls left / 10 (constant plane)
    1  opponent pawn                 5  opponent walls left / 10
    2  horizontal walls (8x8, padded)  6  my BFS distance to my goal / 20
    3  vertical walls (8x8, padded)    7  opponent distance to their goal / 20

Actions (``N_ACTIONS = 209``, canonical coordinates):
    0..80     move pawn to square r*9+c
    81..144   horizontal wall at anchor r*8+c
    145..208  vertical wall at anchor r*8+c
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ..bots.alphabeta import N, NW, STEP, FastBoard

N_PLANES = 8
N_ACTIONS = N * N + 2 * NW * NW  # 209
H_OFFSET = N * N
V_OFFSET = N * N + NW * NW
UNREACHABLE = 1.0

IMove = tuple[str, int]


# ---------------------------------------------------------------------- #
# Coordinate transforms: vertical flip (perspective) and horizontal mirror
# (data augmentation). Both are their own inverse.
# ---------------------------------------------------------------------- #
def _sq(sq: int, flip: bool, mirror: bool) -> int:
    r, c = divmod(sq, N)
    if flip:
        r = N - 1 - r
    if mirror:
        c = N - 1 - c
    return r * N + c


def _anchor(a: int, flip: bool, mirror: bool) -> int:
    # A wall anchor (r, c) spans rows r..r+1 and cols c..c+1, so both
    # flips map it to (7 - r) / (7 - c).
    r, c = divmod(a, NW)
    if flip:
        r = NW - 1 - r
    if mirror:
        c = NW - 1 - c
    return r * NW + c


def move_to_action(move: IMove, flip: bool, mirror: bool = False) -> int:
    kind, x = move
    if kind == "p":
        return _sq(x, flip, mirror)
    a = _anchor(x, flip, mirror)
    return (H_OFFSET if kind == "h" else V_OFFSET) + a


def action_to_move(action: int, flip: bool, mirror: bool = False) -> IMove:
    if action < H_OFFSET:
        return ("p", _sq(action, flip, mirror))
    if action < V_OFFSET:
        return ("h", _anchor(action - H_OFFSET, flip, mirror))
    return ("v", _anchor(action - V_OFFSET, flip, mirror))


def mirror_action(action: int) -> int:
    """Left-right mirror of a canonical action (for augmentation)."""
    return move_to_action(action_to_move(action, False, True), False)


MIRROR_PERM = np.array([mirror_action(a) for a in range(N_ACTIONS)], dtype=np.int64)


def mirror_batch(planes: np.ndarray, pis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Left-right mirror augmentation (the game is symmetric)."""
    out = planes[..., ::-1].copy()
    # Wall planes are 8x8 maps padded to 9x9: mirror only the 8x8 part.
    out[:, 2:4, :8, :8] = planes[:, 2:4, :8, :8][..., ::-1]
    out[:, 2:4, :, 8] = 0
    return out, pis[:, MIRROR_PERM]


# ---------------------------------------------------------------------- #


# ---------------------------------------------------------------------- #
def _distance_map(board: FastBoard, goal_row: int) -> np.ndarray:
    """BFS distance from every square to ``goal_row`` (pawns ignored)."""
    dist = np.full(N * N, -1, dtype=np.int16)
    q: deque[int] = deque()
    for c in range(N):
        sq = goal_row * N + c
        dist[sq] = 0
        q.append(sq)
    open_ = board.open_
    while q:
        sq = q.popleft()
        for d in range(4):
            if open_[sq][d]:
                n = sq + STEP[d]
                if dist[n] < 0:
                    dist[n] = dist[sq] + 1
                    q.append(n)
    out = np.where(dist < 0, UNREACHABLE * 20, dist).astype(np.float32) / 20.0
    return out.reshape(N, N)


def encode(board: FastBoard, mirror: bool = False) -> np.ndarray:
    """Planes for the side to move. ``mirror`` = left-right augmentation."""
    me = board.turn
    opp = 1 - me
    flip = me == 1
    x = np.zeros((N_PLANES, N, N), dtype=np.float32)
    r, c = divmod(_sq(board.pawns[me], flip, mirror), N)
    x[0, r, c] = 1
    r, c = divmod(_sq(board.pawns[opp], flip, mirror), N)
    x[1, r, c] = 1
    for a in board.h:
        r, c = divmod(_anchor(a, flip, mirror), NW)
        x[2, r, c] = 1
    for a in board.v:
        r, c = divmod(_anchor(a, flip, mirror), NW)
        x[3, r, c] = 1
    x[4] = board.walls_left[me] / 10.0
    x[5] = board.walls_left[opp] / 10.0
    for plane, player in ((6, me), (7, opp)):
        dm = _distance_map(board, board.goal[player])
        if flip:
            dm = dm[::-1]
        if mirror:
            dm = dm[:, ::-1]
        x[plane] = dm
    return x


def legal_mask(board: FastBoard, moves: list[IMove] | None = None) -> np.ndarray:
    """Boolean mask over canonical actions for the side to move."""
    flip = board.turn == 1
    mask = np.zeros(N_ACTIONS, dtype=bool)
    for m in (moves if moves is not None else board.legal_moves()):
        mask[move_to_action(m, flip)] = True
    return mask
