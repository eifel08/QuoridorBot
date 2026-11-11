// Quoridor rules engine for the browser.
//
// A direct port of quoridor/board.py + quoridor/state.py. Coordinates are the
// same as the Python engine: (row, col), row 0 at the top. Player 0 (Player 1)
// starts at (8, 4) aiming for row 0; player 1 (Player 2) starts at (0, 4)
// aiming for row 8. Walls are [row, col, "h"|"v"] anchored at the top-left
// square of the 2x2 block they touch (0..7).
//
// tests/test_web_engine.py checks this file against the Python engine.

export const SIZE = 9;
export const WALLS_PER_PLAYER = 10;
export const GOAL_ROWS = [0, SIZE - 1];
const DIRS = [[-1, 0], [1, 0], [0, -1], [0, 1]];

const key = (r, c, o) => `${r},${c},${o}`;

export function newGame() {
  return {
    pawns: [[SIZE - 1, 4], [0, 4]],
    walls: new Set(),
    wallsLeft: [WALLS_PER_PLAYER, WALLS_PER_PLAYER],
    current: 0,
    winner: null,
    moveNumber: 0,
  };
}

export function cloneGame(g) {
  return {
    pawns: g.pawns.map((p) => [...p]),
    walls: new Set(g.walls),
    wallsLeft: [...g.wallsLeft],
    current: g.current,
    winner: g.winner,
    moveNumber: g.moveNumber,
  };
}

const inBounds = (r, c) => r >= 0 && r < SIZE && c >= 0 && c < SIZE;

export function isBlocked(walls, a, b) {
  const [r1, c1] = a;
  const [r2, c2] = b;
  if (r1 === r2) {
    const c = Math.min(c1, c2);
    return walls.has(key(r1, c, "v")) || walls.has(key(r1 - 1, c, "v"));
  }
  const r = Math.min(r1, r2);
  return walls.has(key(r, c1, "h")) || walls.has(key(r, c1 - 1, "h"));
}

function canStep(walls, a, b) {
  if (!inBounds(b[0], b[1])) return false;
  if (Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]) !== 1) return false;
  return !isBlocked(walls, a, b);
}

const same = (a, b) => a[0] === b[0] && a[1] === b[1];

export function validPawnMoves(g, player) {
  const me = g.pawns[player];
  const opp = g.pawns[1 - player];
  const out = [];
  const seen = new Set();
  const add = (sq) => {
    const k = `${sq[0]},${sq[1]}`;
    if (!seen.has(k)) { seen.add(k); out.push(sq); }
  };
  for (const [dr, dc] of DIRS) {
    const step = [me[0] + dr, me[1] + dc];
    if (!canStep(g.walls, me, step)) continue;
    if (!same(step, opp)) { add(step); continue; }
    const jump = [opp[0] + dr, opp[1] + dc];
    if (canStep(g.walls, opp, jump)) { add(jump); continue; }
    for (const [sdr, sdc] of [[dc, dr], [-dc, -dr]]) {
      const side = [opp[0] + sdr, opp[1] + sdc];
      if (canStep(g.walls, opp, side) && !same(side, me)) add(side);
    }
  }
  return out;
}

export function conflictingWalls(r, c, o) {
  if (o === "h") return [key(r, c, "h"), key(r, c - 1, "h"), key(r, c + 1, "h"), key(r, c, "v")];
  return [key(r, c, "v"), key(r - 1, c, "v"), key(r + 1, c, "v"), key(r, c, "h")];
}

export function shortestPath(g, player, walls = g.walls) {
  const start = g.pawns[player];
  const goal = GOAL_ROWS[player];
  if (start[0] === goal) return 0;
  const seen = new Set([`${start[0]},${start[1]}`]);
  let frontier = [start];
  let dist = 0;
  while (frontier.length) {
    dist += 1;
    const next = [];
    for (const sq of frontier) {
      for (const [dr, dc] of DIRS) {
        const n = [sq[0] + dr, sq[1] + dc];
        const k = `${n[0]},${n[1]}`;
        if (seen.has(k) || !canStep(walls, sq, n)) continue;
        if (n[0] === goal) return dist;
        seen.add(k);
        next.push(n);
      }
    }
    frontier = next;
  }
  return null;
}

/** Reason the wall is illegal for `player`, or null if it is legal. */
export function wallError(g, player, [r, c, o]) {
  if (g.wallsLeft[player] <= 0) return "You have no walls left.";
  if (!(r >= 0 && r < SIZE - 1 && c >= 0 && c < SIZE - 1)) return "Wall is out of bounds.";
  if (conflictingWalls(r, c, o).some((k) => g.walls.has(k))) {
    return "That wall overlaps or crosses an existing wall.";
  }
  const walls = new Set(g.walls);
  walls.add(key(r, c, o));
  for (const p of [0, 1]) {
    if (shortestPath(g, p, walls) === null) {
      return `That wall would completely block ${p === player ? "you" : "your opponent"} from reaching the goal.`;
    }
  }
  return null;
}

export function validWalls(g, player) {
  if (g.wallsLeft[player] <= 0) return [];
  const out = [];
  for (const o of ["h", "v"]) {
    for (let r = 0; r < SIZE - 1; r++) {
      for (let c = 0; c < SIZE - 1; c++) {
        if (wallError(g, player, [r, c, o]) === null) out.push([r, c, o]);
      }
    }
  }
  return out;
}

/** Reason the move is illegal for the player to move, or null. */
export function moveError(g, move) {
  if (g.winner !== null) return "The game is over.";
  if (move.type === "pawn") {
    const ok = validPawnMoves(g, g.current).some((sq) => same(sq, [move.row, move.col]));
    return ok ? null : "Your pawn cannot reach that square this turn.";
  }
  if (move.type === "wall") return wallError(g, g.current, [move.row, move.col, move.orientation]);
  return "Unknown move type.";
}

/** Returns a new game with the move applied (throws if illegal). */
export function applyMove(g, move) {
  const err = moveError(g, move);
  if (err) throw new Error(err);
  const n = cloneGame(g);
  const p = n.current;
  if (move.type === "pawn") {
    n.pawns[p] = [move.row, move.col];
  } else {
    n.walls.add(key(move.row, move.col, move.orientation));
    n.wallsLeft[p] -= 1;
  }
  n.moveNumber += 1;
  if (n.pawns[p][0] === GOAL_ROWS[p]) n.winner = p;
  else n.current = 1 - p;
  return n;
}

export function wallList(g) {
  return [...g.walls]
    .map((k) => { const [r, c, o] = k.split(","); return [Number(r), Number(c), o]; })
    .sort((a, b) => (a[2] < b[2] ? -1 : a[2] > b[2] ? 1 : a[0] - b[0] || a[1] - b[1]));
}

/** Same shape as Python's BoardState.to_dict(), so bots can share code/data. */
export function snapshot(g) {
  const over = g.winner !== null;
  return {
    size: SIZE,
    current_player: g.current,
    pawns: g.pawns.map((p) => [...p]),
    goal_rows: [...GOAL_ROWS],
    walls: wallList(g),
    walls_left: [...g.wallsLeft],
    valid_pawn_moves: over ? [] : validPawnMoves(g, g.current),
    valid_walls: over ? [] : validWalls(g, g.current),
    shortest_paths: [shortestPath(g, 0), shortestPath(g, 1)],
    move_number: g.moveNumber,
    winner: g.winner,
  };
}

/** Build a game from a snapshot dict (inverse of snapshot()). */
export function fromSnapshot(s) {
  return {
    pawns: s.pawns.map((p) => [...p]),
    walls: new Set(s.walls.map(([r, c, o]) => key(r, c, o))),
    wallsLeft: [...s.walls_left],
    current: s.current_player,
    winner: s.winner ?? null,
    moveNumber: s.move_number ?? 0,
  };
}
