// Alpha-beta pruning bot for the browser: a line-by-line port of
// quoridor/bots/alphabeta.py (same move ordering, so both pick the same
// moves; tests/test_web_engine.py checks this).
//
// Heuristic for the bot:  Score = (bot steps to goal) - (opponent steps to goal),
// both BFS shortest paths; the bot minimises it. The search is negamax, so
// each node is scored for the side to move as -Score and maximised.

const N = 9, NW = 8;
const UP = 0, DOWN = 1, LEFT = 2, RIGHT = 3;
const STEP = [-N, N, -1, 1];
const WIN = 1000000;
const EXACT = 0, LOWER = 1, UPPER = 2;
export const MAX_DEPTH = 30;

class Timeout extends Error {}

// Seeded 32-bit PRNG so hashes are reproducible.
function mulberry32(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return (t ^ (t >>> 14)) >>> 0;
  };
}
const rnd = mulberry32(20240611);
const zt = (n) => Array.from({ length: n }, () => [rnd(), rnd()]);
const Z_PAWN = [zt(N * N), zt(N * N)];
const Z_WALL = { h: zt(NW * NW), v: zt(NW * NW) };
const Z_LEFT = [zt(11), zt(11)];
const Z_TURN = [rnd(), rnd()];

class FastBoard {
  constructor(s) {
    this.pawns = s.pawns.map(([r, c]) => r * N + c);
    this.goal = [...s.goal_rows];
    this.wallsLeft = [...s.walls_left];
    this.turn = s.current_player;
    this.h = new Set();
    this.v = new Set();
    this.open = [];
    for (let sq = 0; sq < N * N; sq++) {
      const r = Math.floor(sq / N), c = sq % N;
      this.open.push([r > 0, r < N - 1, c > 0, c < N - 1]);
    }
    for (const [r, c, o] of s.walls) this.setWall(o, r * NW + c, false);
    this.h1 = 0; this.h2 = 0;
    this.xor(Z_PAWN[0][this.pawns[0]]); this.xor(Z_PAWN[1][this.pawns[1]]);
    this.xor(Z_LEFT[0][this.wallsLeft[0]]); this.xor(Z_LEFT[1][this.wallsLeft[1]]);
    for (const a of this.h) this.xor(Z_WALL.h[a]);
    for (const a of this.v) this.xor(Z_WALL.v[a]);
    if (this.turn) this.xor(Z_TURN);
  }

  xor([a, b]) { this.h1 ^= a; this.h2 ^= b; }
  get key() { return (this.h1 >>> 0) * 2097152 + (this.h2 & 0x1fffff); }

  static wallEdges(o, a) {
    const r = Math.floor(a / NW), c = a % NW, s = r * N + c;
    return o === "h"
      ? [[s, DOWN], [s + N, UP], [s + 1, DOWN], [s + 1 + N, UP]]
      : [[s, RIGHT], [s + 1, LEFT], [s + N, RIGHT], [s + N + 1, LEFT]];
  }

  setWall(o, a, value) {
    for (const [sq, d] of FastBoard.wallEdges(o, a)) this.open[sq][d] = value;
    const set = o === "h" ? this.h : this.v;
    if (value) set.delete(a); else set.add(a);
  }

  overlaps(o, a) {
    const c = a % NW;
    if (o === "h") {
      return this.h.has(a) || this.v.has(a) || (c > 0 && this.h.has(a - 1)) || (c < NW - 1 && this.h.has(a + 1));
    }
    return this.v.has(a) || this.h.has(a) || (a >= NW && this.v.has(a - NW)) || (a < NW * (NW - 1) && this.v.has(a + NW));
  }

  distance(player) {
    const start = this.pawns[player], goal = this.goal[player];
    if (Math.floor(start / N) === goal) return 0;
    const dist = new Int8Array(N * N).fill(-1);
    dist[start] = 0;
    const q = [start];
    for (let i = 0; i < q.length; i++) {
      const sq = q[i], nd = dist[sq] + 1, o = this.open[sq];
      for (let d = 0; d < 4; d++) {
        if (!o[d]) continue;
        const n = sq + STEP[d];
        if (dist[n] !== -1) continue;
        if (Math.floor(n / N) === goal) return nd;
        dist[n] = nd;
        q.push(n);
      }
    }
    return null;
  }

  shortestPath(player) {
    const start = this.pawns[player], goal = this.goal[player];
    const parent = new Int16Array(N * N).fill(-2);
    parent[start] = -1;
    const q = [start];
    let end = Math.floor(start / N) === goal ? start : -1;
    for (let i = 0; i < q.length && end < 0; i++) {
      const sq = q[i], o = this.open[sq];
      for (let d = 0; d < 4; d++) {
        if (!o[d]) continue;
        const n = sq + STEP[d];
        if (parent[n] !== -2) continue;
        parent[n] = sq;
        if (Math.floor(n / N) === goal) { end = n; break; }
        q.push(n);
      }
    }
    if (end < 0) return [];
    const path = [end];
    while (parent[path[path.length - 1]] !== -1) path.push(parent[path[path.length - 1]]);
    return path.reverse();
  }

  pawnMoves(player) {
    const me = this.pawns[player], opp = this.pawns[1 - player];
    const out = [];
    for (let d = 0; d < 4; d++) {
      if (!this.open[me][d]) continue;
      const n = me + STEP[d];
      if (n !== opp) { out.push(n); continue; }
      if (this.open[opp][d]) { out.push(opp + STEP[d]); continue; }
      for (const sd of (d === UP || d === DOWN) ? [LEFT, RIGHT] : [UP, DOWN]) {
        if (this.open[opp][sd]) {
          const s = opp + STEP[sd];
          if (s !== me && !out.includes(s)) out.push(s);
        }
      }
    }
    return [...new Set(out)];
  }

  make([kind, x]) {
    const p = this.turn;
    if (kind === "p") {
      this.xor(Z_PAWN[p][this.pawns[p]]); this.xor(Z_PAWN[p][x]);
      this.pawns[p] = x;
    } else {
      this.setWall(kind, x, false);
      this.xor(Z_WALL[kind][x]);
      this.xor(Z_LEFT[p][this.wallsLeft[p]]); this.xor(Z_LEFT[p][this.wallsLeft[p] - 1]);
      this.wallsLeft[p] -= 1;
    }
    this.turn = 1 - p;
    this.xor(Z_TURN);
  }

  unmake([kind, x], prev) {
    this.turn = 1 - this.turn;
    this.xor(Z_TURN);
    const p = this.turn;
    if (kind === "p") {
      this.xor(Z_PAWN[p][x]); this.xor(Z_PAWN[p][prev]);
      this.pawns[p] = prev;
    } else {
      this.setWall(kind, x, true);
      this.xor(Z_WALL[kind][x]);
      this.xor(Z_LEFT[p][this.wallsLeft[p]]); this.xor(Z_LEFT[p][this.wallsLeft[p] + 1]);
      this.wallsLeft[p] += 1;
    }
  }

  won(player) { return Math.floor(this.pawns[player] / N) === this.goal[player]; }
}

function evaluate(b) {
  const me = b.turn;
  return b.distance(1 - me) - b.distance(me);
}

const sameMove = (a, b) => a && b && a[0] === b[0] && a[1] === b[1];

class Search {
  constructor(board, deadline) {
    this.b = board;
    this.deadline = deadline;
    this.nodes = 0;
    this.tt = new Map();
  }

  orderedMoves(ttMove) {
    const b = this.b, me = b.turn, opp = 1 - me;
    const myD = b.distance(me), oppD = b.distance(opp);
    const scored = [];
    const start = b.pawns[me];
    for (const sq of b.pawnMoves(me)) {
      b.pawns[me] = sq;
      const d = b.distance(me);
      b.pawns[me] = start;
      scored.push([oppD - d, ["p", sq]]);
    }
    if (b.wallsLeft[me] > 0) {
      const path = b.shortestPath(opp);
      const cands = new Map();
      const add = (o, a) => { const k = `${o}${a}`; if (!cands.has(k)) cands.set(k, [o, a]); };
      for (let i = 0; i + 1 < path.length; i++) {
        const ra = Math.floor(path[i] / N), ca = path[i] % N;
        const rb = Math.floor(path[i + 1] / N), cb = path[i + 1] % N;
        if (ra === rb) {
          const c = Math.min(ca, cb);
          for (const r of [ra - 1, ra]) if (r >= 0 && r < NW && c < NW) add("v", r * NW + c);
        } else {
          const r = Math.min(ra, rb);
          for (const c of [ca - 1, ca]) if (c >= 0 && c < NW && r < NW) add("h", r * NW + c);
        }
      }
      const base = oppD - myD;
      for (const m of cands.values()) {
        const [kind, a] = m;
        if (b.overlaps(kind, a)) continue;
        b.setWall(kind, a, false);
        const nOpp = b.distance(opp), nMe = b.distance(me);
        b.setWall(kind, a, true);
        if (nOpp === null || nMe === null) continue;
        const gain = nOpp - nMe;
        if (gain > base) scored.push([gain, m]);
      }
    }
    scored.sort((x, y) => y[0] - x[0]);
    const moves = scored.map((t) => t[1]);
    if (ttMove) {
      const i = moves.findIndex((m) => sameMove(m, ttMove));
      if (i >= 0) { moves.splice(i, 1); moves.unshift(ttMove); }
    }
    return moves;
  }

  negamax(depth, alpha, beta, ply) {
    this.nodes++;
    if (this.deadline !== null && (this.nodes & 255) === 0 && performance.now() > this.deadline) throw new Timeout();
    const b = this.b;
    if (depth === 0) return evaluate(b);
    const alpha0 = alpha;
    const key = b.key;
    const entry = this.tt.get(key);
    let ttMove = null;
    if (entry) {
      const [eDepth, eVal, eFlag, eMove] = entry;
      ttMove = eMove;
      if (eDepth >= depth) {
        if (eFlag === EXACT) return eVal;
        if (eFlag === LOWER && eVal >= beta) return eVal;
        if (eFlag === UPPER && eVal <= alpha) return eVal;
      }
    }
    const me = b.turn;
    const moves = this.orderedMoves(ttMove);
    if (!moves.length) return evaluate(b);
    let bestVal = -WIN * 2, bestMove = null;
    for (const m of moves) {
      const prev = b.pawns[me];
      b.make(m);
      const val = (m[0] === "p" && b.won(me)) ? WIN - (ply + 1) : -this.negamax(depth - 1, -beta, -alpha, ply + 1);
      b.unmake(m, prev);
      if (val > bestVal) { bestVal = val; bestMove = m; }
      if (val > alpha) alpha = val;
      if (alpha >= beta) break;
    }
    const flag = bestVal <= alpha0 ? UPPER : bestVal >= beta ? LOWER : EXACT;
    this.tt.set(key, [depth, bestVal, flag, bestMove]);
    return bestVal;
  }

  root(depth, pvMove) {
    const b = this.b, me = b.turn;
    let alpha = -WIN * 2;
    const beta = WIN * 2;
    let bestVal = -WIN * 2, bestMove = null;
    for (const m of this.orderedMoves(pvMove)) {
      const prev = b.pawns[me];
      b.make(m);
      const val = (m[0] === "p" && b.won(me)) ? WIN - 1 : -this.negamax(depth - 1, -beta, -alpha, 1);
      b.unmake(m, prev);
      if (val > bestVal) { bestVal = val; bestMove = m; }
      alpha = Math.max(alpha, val);
    }
    return [bestVal, bestMove];
  }
}

/**
 * Iterative-deepening alpha-beta search.
 * @param snapshot  engine snapshot (same shape as Python BoardState.to_dict())
 * @param depth     max depth in plies, 1..30
 * @param timeLimitMs  optional time budget in ms (null = none)
 * @returns {move, info} with move = {type, row, col, orientation?}
 */
export function search(snapshot, depth, timeLimitMs = null) {
  depth = Math.max(1, Math.min(Math.floor(depth), MAX_DEPTH));
  const t0 = performance.now();
  const s = new Search(new FastBoard(snapshot), timeLimitMs ? t0 + timeLimitMs : null);
  let bestMove = null, bestVal = 0, reached = 0;
  for (let d = 1; d <= depth; d++) {
    try {
      [bestVal, bestMove] = s.root(d, bestMove);
    } catch (e) {
      if (e instanceof Timeout) break;
      throw e;
    }
    reached = d;
    if (Math.abs(bestVal) >= WIN - 1000) break;
  }
  if (!bestMove) bestMove = s.orderedMoves(null)[0];
  const [kind, x] = bestMove;
  const move = kind === "p"
    ? { type: "pawn", row: Math.floor(x / N), col: x % N }
    : { type: "wall", row: Math.floor(x / NW), col: x % NW, orientation: kind };
  const forced = Math.abs(bestVal) >= WIN - 1000;
  return {
    move,
    info: {
      depth_requested: depth,
      depth_reached: reached,
      nodes: s.nodes,
      seconds: Math.round(performance.now() - t0) / 1000,
      score: forced ? null : -bestVal,
      forced: forced ? (bestVal > 0 ? "win" : "loss") : null,
    },
  };
}
