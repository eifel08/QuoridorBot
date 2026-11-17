#include "board.h"
#include <algorithm>

namespace quoridor {

Zobrist ZOB;

namespace {

// Edges removed by each wall slot: for a horizontal wall the squares that lose
// their SOUTH / NORTH step, for a vertical wall those losing EAST / WEST.
BB WALL_CLEAR_A[2][NWALL];  // [0] horizontal: SOUTH, [1] vertical: EAST
BB WALL_CLEAR_B[2][NWALL];  // [0] horizontal: NORTH, [1] vertical: WEST
BB OPEN_EMPTY[4];           // open masks of a board without walls
BB GOAL[2];
uint64_t TOUCH[2][NWALL][2];
Points WALL_PTS[2][NWALL];   // the 3 lattice points of each slot
Points BORDER_PTS;  // [orientation][anchor][orientation of the touching slot]
bool tables_ready = false;

constexpr uint64_t COL0 = 0x0101010101010101ULL;  // anchor column 0
constexpr uint64_t COL7 = 0x8080808080808080ULL;  // anchor column 7
constexpr int STEP[4] = {-N, N, -1, 1};

uint64_t splitmix64(uint64_t& s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

}  // namespace

void Board::init_tables() {
    if (tables_ready) return;
    tables_ready = true;
    for (int sq = 0; sq < NSQ; ++sq) {
        int r = sq / N, c = sq % N;
        if (r > 0) OPEN_EMPTY[NORTH] |= BB::square(sq);
        if (r < N - 1) OPEN_EMPTY[SOUTH] |= BB::square(sq);
        if (c > 0) OPEN_EMPTY[WEST] |= BB::square(sq);
        if (c < N - 1) OPEN_EMPTY[EAST] |= BB::square(sq);
        if (r == 0) GOAL[0] |= BB::square(sq);
        if (r == N - 1) GOAL[1] |= BB::square(sq);
    }
    for (int a = 0; a < NWALL; ++a) {
        int s = (a / NW) * N + (a % NW);  // top-left square of the 2x2 block
        WALL_CLEAR_A[0][a] = BB::square(s) | BB::square(s + 1);          // lose SOUTH
        WALL_CLEAR_B[0][a] = BB::square(s + N) | BB::square(s + N + 1);  // lose NORTH
        WALL_CLEAR_A[1][a] = BB::square(s) | BB::square(s + N);          // lose EAST
        WALL_CLEAR_B[1][a] = BB::square(s + 1) | BB::square(s + N + 1);  // lose WEST
    }
    // Lattice points (row, col) of the 10 x 10 corner grid a wall slot runs through.
    auto points = [](int o, int a, int out[3][2]) {
        const int r = a / NW, c = a % NW;
        for (int k = 0; k < 3; ++k) {
            out[k][0] = o == 0 ? r + 1 : r + k;
            out[k][1] = o == 0 ? c + k : c + 1;
        }
    };
    for (int o = 0; o < 2; ++o)
        for (int a = 0; a < NWALL; ++a) {
            int p[3][2];
            points(o, a, p);
            for (int o2 = 0; o2 < 2; ++o2)
                for (int b = 0; b < NWALL; ++b) {
                    if (o2 == o && b == a) continue;
                    int q[3][2];
                    points(o2, b, q);
                    bool touch = false;
                    for (auto& x : p)
                        for (auto& y : q) touch |= x[0] == y[0] && x[1] == y[1];
                    if (touch) TOUCH[o][a][o2] |= 1ULL << b;
                }
            for (auto& x : p) {
                const int i = x[0] * 10 + x[1];
                (i < 64 ? WALL_PTS[o][a].lo : WALL_PTS[o][a].hi) |= 1ULL << (i & 63);
            }
        }
    for (int i = 0; i < 10; ++i)
        for (int j = 0; j < 10; ++j)
            if (i == 0 || j == 0 || i == 9 || j == 9) {
                const int k = i * 10 + j;
                (k < 64 ? BORDER_PTS.lo : BORDER_PTS.hi) |= 1ULL << (k & 63);
            }
    uint64_t seed = 0x51ED270B0A7D1E55ULL;
    for (auto& p : ZOB.pawn)
        for (auto& k : p) k = splitmix64(seed);
    for (auto& w : ZOB.wall)
        for (auto& k : w) k = splitmix64(seed);
    for (auto& w : ZOB.walls_left)
        for (auto& k : w) k = splitmix64(seed);
    ZOB.turn = splitmix64(seed);
}

BB Board::goal_mask(int player) { return GOAL[player]; }

void Board::reset() { set(0, (N - 1) * N + N / 2, N / 2, MAX_WALLS, MAX_WALLS, 0, 0); }

void Board::set(int turn_, int p0, int p1, int wl0, int wl1, uint64_t hw_, uint64_t vw_) {
    for (int d = 0; d < 4; ++d) open[d] = OPEN_EMPTY[d];
    hw = hw_;
    vw = vw_;
    for (uint64_t b = hw; b; b &= b - 1) {
        int a = std::countr_zero(b);
        open[SOUTH] &= ~WALL_CLEAR_A[0][a];
        open[NORTH] &= ~WALL_CLEAR_B[0][a];
    }
    for (uint64_t b = vw; b; b &= b - 1) {
        int a = std::countr_zero(b);
        open[EAST] &= ~WALL_CLEAR_A[1][a];
        open[WEST] &= ~WALL_CLEAR_B[1][a];
    }
    pawn[0] = uint8_t(p0);
    pawn[1] = uint8_t(p1);
    walls_left[0] = uint8_t(wl0);
    walls_left[1] = uint8_t(wl1);
    turn = uint8_t(turn_);
    hash = compute_hash();
}

uint64_t Board::compute_hash() const {
    uint64_t h = ZOB.pawn[0][pawn[0]] ^ ZOB.pawn[1][pawn[1]];
    h ^= ZOB.walls_left[0][walls_left[0]] ^ ZOB.walls_left[1][walls_left[1]];
    for (uint64_t b = hw; b; b &= b - 1) h ^= ZOB.wall[0][std::countr_zero(b)];
    for (uint64_t b = vw; b; b &= b - 1) h ^= ZOB.wall[1][std::countr_zero(b)];
    if (turn) h ^= ZOB.turn;
    return h;
}

// --------------------------------------------------------------------------
// make / unmake
// --------------------------------------------------------------------------
int Board::make(Move m) {
    const int p = turn;
    const int prev = pawn[p];
    if (is_pawn(m)) {
        hash ^= ZOB.pawn[p][prev] ^ ZOB.pawn[p][m];
        pawn[p] = m;
    } else {
        if (is_hwall(m)) {
            int a = m - H_BASE;
            hw |= 1ULL << a;
            open[SOUTH] &= ~WALL_CLEAR_A[0][a];
            open[NORTH] &= ~WALL_CLEAR_B[0][a];
            hash ^= ZOB.wall[0][a];
        } else {
            int a = m - V_BASE;
            vw |= 1ULL << a;
            open[EAST] &= ~WALL_CLEAR_A[1][a];
            open[WEST] &= ~WALL_CLEAR_B[1][a];
            hash ^= ZOB.wall[1][a];
        }
        hash ^= ZOB.walls_left[p][walls_left[p]] ^ ZOB.walls_left[p][walls_left[p] - 1];
        --walls_left[p];
    }
    turn ^= 1;
    hash ^= ZOB.turn;
    return prev;
}

void Board::unmake(Move m, int prev_sq) {
    turn ^= 1;
    hash ^= ZOB.turn;
    const int p = turn;
    if (is_pawn(m)) {
        hash ^= ZOB.pawn[p][m] ^ ZOB.pawn[p][prev_sq];
        pawn[p] = uint8_t(prev_sq);
    } else {
        // Legal walls never share a blocked edge, so OR-ing the edges back is exact.
        if (is_hwall(m)) {
            int a = m - H_BASE;
            hw &= ~(1ULL << a);
            open[SOUTH] |= WALL_CLEAR_A[0][a];
            open[NORTH] |= WALL_CLEAR_B[0][a];
            hash ^= ZOB.wall[0][a];
        } else {
            int a = m - V_BASE;
            vw &= ~(1ULL << a);
            open[EAST] |= WALL_CLEAR_A[1][a];
            open[WEST] |= WALL_CLEAR_B[1][a];
            hash ^= ZOB.wall[1][a];
        }
        hash ^= ZOB.walls_left[p][walls_left[p]] ^ ZOB.walls_left[p][walls_left[p] + 1];
        ++walls_left[p];
    }
}

// --------------------------------------------------------------------------
// Path finding: breadth-first flood fill on bitboards. Each BFS level is a
// handful of 128-bit shifts/ANDs, with no queue and no per-square work.
// --------------------------------------------------------------------------
int flood_distance(const BB* open, int from, BB goal) {
    BB front = BB::square(from);
    if ((front & goal).any()) return 0;
    BB seen = front;
    for (int d = 1;; ++d) {
        BB next = ((front & open[NORTH]).shr(N) | (front & open[SOUTH]).shl(N) |
                   (front & open[WEST]).shr(1) | (front & open[EAST]).shl(1)) & ~seen;
        if (!next.any()) return -1;
        if ((next & goal).any()) return d;
        seen |= next;
        front = next;
    }
}

void clear_wall(BB* o, Move wall) {
    if (is_hwall(wall)) {
        const int a = wall - H_BASE;
        o[SOUTH] &= ~WALL_CLEAR_A[0][a];
        o[NORTH] &= ~WALL_CLEAR_B[0][a];
    } else {
        const int a = wall - V_BASE;
        o[EAST] &= ~WALL_CLEAR_A[1][a];
        o[WEST] &= ~WALL_CLEAR_B[1][a];
    }
}

void apply_wall(Layout& lay, Move wall) {
    clear_wall(lay.open, wall);
    if (is_hwall(wall)) lay.hw |= 1ULL << (wall - H_BASE);
    else lay.vw |= 1ULL << (wall - V_BASE);
}

uint64_t free_h_of(uint64_t hw, uint64_t vw) { return ~(hw | ((hw & ~COL7) << 1) | ((hw & ~COL0) >> 1) | vw); }
uint64_t free_v_of(uint64_t hw, uint64_t vw) { return ~(vw | (vw << NW) | (vw >> NW) | hw); }

Points barrier_points(uint64_t hw, uint64_t vw) {
    Points p = BORDER_PTS;
    for (uint64_t b = hw; b; b &= b - 1) {
        const Points& w = WALL_PTS[0][std::countr_zero(b)];
        p.lo |= w.lo;
        p.hi |= w.hi;
    }
    for (uint64_t b = vw; b; b &= b - 1) {
        const Points& w = WALL_PTS[1][std::countr_zero(b)];
        p.lo |= w.lo;
        p.hi |= w.hi;
    }
    return p;
}

bool may_disconnect(const Points& barrier, Move wall) {
    const Points& w = is_hwall(wall) ? WALL_PTS[0][wall - H_BASE] : WALL_PTS[1][wall - V_BASE];
    return std::popcount(w.lo & barrier.lo) + std::popcount(w.hi & barrier.hi) >= 2;
}

void touching_slots(Move m, uint64_t& h, uint64_t& v) {
    const int o = is_hwall(m) ? 0 : 1;
    const int a = o == 0 ? m - H_BASE : m - V_BASE;
    h = TOUCH[o][a][0];
    v = TOUCH[o][a][1];
}

int Board::distance(int player) const { return flood_distance(open, pawn[player], GOAL[player]); }

int Board::distance_with_wall(int player, Move wall) const {
    BB o[4] = {open[0], open[1], open[2], open[3]};
    clear_wall(o, wall);
    return flood_distance(o, pawn[player], GOAL[player]);
}

void Board::goal_layers(int player, GoalLayers& out, int stop_sq) const { goal_layers_open(open, player, out, stop_sq); }

void goal_layers_open(const BB* open, int player, GoalLayers& out, int stop_sq) {
    BB front = GOAL[player], seen = front;
    int k = 0;
    out.layer[0] = front;
    while (true) {
        if (stop_sq >= 0 && front.test(stop_sq)) break;
        BB next = expand_open(open, front) & ~seen;
        if (!next.any()) break;
        out.layer[++k] = next;
        seen |= next;
        front = next;
    }
    out.count = k + 1;
}

BB route_zone(const BB* open, int pawn, const GoalLayers& gl, int slack) {
    const int D = gl.dist(pawn);
    if (D < 0) return BB{0, 0};
    const int lim = D + std::max(0, slack);
    const int top = std::min(lim, gl.count - 1);
    BB pre[NSQ + 1];  // pre[j]: squares at most j steps from the goal
    BB acc{0, 0};
    for (int j = 0; j <= top; ++j) {
        acc |= gl.layer[j];
        pre[j] = acc;
    }
    BB front = BB::square(pawn), seen = front, zone{0, 0};
    for (int k = 0; k <= lim && front.any(); ++k) {  // front: squares exactly k steps from the pawn
        zone |= front & pre[std::min(lim - k, top)];
        const BB next = expand_open(open, front) & ~seen;
        seen |= next;
        front = next;
    }
    return zone;
}

void zone_slots(BB zone, uint64_t& h, uint64_t& v) {
    h = v = 0;
    for (int r = 0; r < NW; ++r) {
        const uint32_t a = zone.row(r), b = zone.row(r + 1);
        const uint32_t e = a & b;                           // edges (r,c)-(r+1,c) inside the zone
        const uint32_t f = (a & (a >> 1)) | (b & (b >> 1));  // edges (r,c)-(r,c+1) / (r+1,c)-(r+1,c+1)
        h |= uint64_t((e | (e >> 1)) & 0xFF) << (r * NW);
        v |= uint64_t(f & 0xFF) << (r * NW);
    }
}

PathCut Board::shortest_path_cuts(int player, const GoalLayers& gl) const {
    return path_cuts_open(open, pawn[player], gl);
}

PathCut path_cuts_open(const BB* open, int pawn_sq, const GoalLayers& gl) {
    PathCut cut;
    const int D = gl.dist(pawn_sq);
    if (D <= 0) return cut;
    // Walk from the pawn, always stepping one layer closer to the goal. The
    // squares visited at step d are exactly those with dist_from_pawn == d and
    // dist_to_goal == D - d, i.e. the squares on some shortest path.
    BB vert{}, horz{};  // edges used: indexed by the upper square / the left square
    BB cur = BB::square(pawn_sq);
    for (int d = 0; d < D; ++d) {
        const BB target = gl.layer[D - d - 1];
        BB n = (cur & open[NORTH]).shr(N) & target;
        BB s = (cur & open[SOUTH]).shl(N) & target;
        BB w = (cur & open[WEST]).shr(1) & target;
        BB e = (cur & open[EAST]).shl(1) & target;
        vert |= n | s.shr(N);
        horz |= w | e.shr(1);
        cur = n | s | w | e;
    }
    // A vertical step below square (r, c) is blocked by horizontal walls
    // anchored at (r, c - 1) and (r, c).
    for (int r = 0; r < NW; ++r) {
        uint32_t bits = vert.row(r);
        cut.h |= uint64_t((bits | (bits >> 1)) & 0xFF) << (r * NW);
    }
    // A horizontal step right of square (r, c) is blocked by vertical walls
    // anchored at (r - 1, c) and (r, c).
    for (int r = 0; r < N; ++r) {
        uint64_t bits = horz.row(r) & 0xFF;
        if (r < NW) cut.v |= bits << (r * NW);
        if (r > 0) cut.v |= bits << ((r - 1) * NW);
    }
    return cut;
}

// --------------------------------------------------------------------------
// Blockability (see board.h). Cost: one goal flood, one walk over the
// shortest-path DAG, and one or two early-exit BFS per bottleneck slot.
// --------------------------------------------------------------------------
Blockability analyse_blockability(const Layout& lay, int target, int limit, bool exact) {
    Blockability res;
    const int other = target ^ 1;
    GoalLayers gl;
    goal_layers_open(lay.open, target, gl, lay.pawn[target]);
    const int D = gl.dist(lay.pawn[target]);
    if (D <= 0) return res;  // on the goal row (or cut off): nothing left to block
    const uint64_t fh = free_h_of(lay.hw, lay.vw), fv = free_v_of(lay.hw, lay.vw);
    const Points barrier = barrier_points(lay.hw, lay.vw);
    bool legal_cut = false, blocked_by_other = false;

    // Tests one slot. Returns the new distance (> D), or -1 if the wall is illegal.
    auto test = [&](Move m, bool& sure_legal) -> int {
        const bool risky = may_disconnect(barrier, m);
        sure_legal = !risky;
        BB t[4] = {lay.open[0], lay.open[1], lay.open[2], lay.open[3]};
        clear_wall(t, m);
        const int nd = flood_distance(t, lay.pawn[target], GOAL[target]);
        if (nd < 0) return -1;  // cuts the target off: illegal, and stays illegal forever
        legal_cut = true;
        if (exact && risky && flood_distance(t, lay.pawn[other], GOAL[other]) < 0) {
            blocked_by_other = true;  // illegal only while the other player needs this edge
            return -1;
        }
        return nd;
    };

    BB cur = BB::square(lay.pawn[target]);
    for (int d = 0; d < D; ++d) {
        const BB tl = gl.layer[D - d - 1];
        const BB n = (cur & lay.open[NORTH]).shr(N) & tl;
        const BB s = (cur & lay.open[SOUTH]).shl(N) & tl;
        const BB w = (cur & lay.open[WEST]).shr(1) & tl;
        const BB e = (cur & lay.open[EAST]).shl(1) & tl;
        cur = n | s | w | e;
        // Edges of this layer: vertical ones by their upper square, horizontal by their left square.
        const BB vert = n | s.shr(N), horz = w | e.shr(1);
        const int nv = popcount(vert), nh = popcount(horz);
        // Every free slot cutting at least one edge of the layer ...
        uint64_t ah = 0, av = 0;
        for (BB x = vert; x.any();) {
            const int q = lowest_square(x), r = q / N, c = q % N;
            x = x & ~BB::square(q);
            if (c > 0) ah |= 1ULL << (r * NW + c - 1);  // horizontal walls at (r, c-1), (r, c)
            if (c < NW) ah |= 1ULL << (r * NW + c);
        }
        for (BB x = horz; x.any();) {
            const int q = lowest_square(x), r = q / N, c = q % N;
            x = x & ~BB::square(q);
            if (r > 0) av |= 1ULL << ((r - 1) * NW + c);  // vertical walls at (r-1, c), (r, c)
            if (r < NW) av |= 1ULL << (r * NW + c);
        }
        ah &= fh;
        av &= fv;
        // ... and the ones cutting ALL of them (a bottleneck: one edge, or two side by side).
        uint64_t ch = 0, cv = 0;
        if (nv + nh == 1) {
            ch = ah;
            cv = av;
        } else if (nv == 2 && nh == 0) {
            const int q = lowest_square(vert), r = q / N, c = q % N;
            if (c < NW && vert == (BB::square(q) | BB::square(q + 1))) ch = ah & (1ULL << (r * NW + c));
        } else if (nh == 2 && nv == 0) {
            const int q = lowest_square(horz), r = q / N, c = q % N;
            if (r < NW && horz == (BB::square(q) | BB::square(q + N))) cv = av & (1ULL << (r * NW + c));
        }
        bool found = false;
        for (int o = 0; o < 2 && !(found && !exact); ++o)
            for (uint64_t b = o == 0 ? ch : cv; b; b &= b - 1) {
                const Move m = o == 0 ? hwall_move(std::countr_zero(b)) : vwall_move(std::countr_zero(b));
                bool sure;
                if (!exact && !may_disconnect(barrier, m)) {  // legal for sure, and it cuts every path: gain >= 1
                    legal_cut = found = true;
                    if (res.best_gain < 1) {
                        res.best_gain = 1;
                        res.best_wall = m;
                    }
                    break;
                }
                const int nd = test(m, sure);
                if (nd < 0) continue;
                found = true;
                if (nd - D > res.best_gain) {
                    res.best_gain = nd - D;
                    res.best_wall = m;
                }
                if (!exact) break;
            }
        if (found) {
            ++res.opportunities;
        } else {
            // No single wall closes this layer. If a legal wall still cuts part of it,
            // two or more walls can (a "soft" layer: several routes side by side).
            for (int o = 0; o < 2; ++o) {
                bool soft = false;
                for (uint64_t b = (o == 0 ? ah & ~ch : av & ~cv); b; b &= b - 1) {
                    const Move m = o == 0 ? hwall_move(std::countr_zero(b)) : vwall_move(std::countr_zero(b));
                    if (!may_disconnect(barrier, m)) {
                        soft = legal_cut = true;
                        break;
                    }
                    bool sure;
                    if (test(m, sure) >= 0) {
                        soft = true;
                        break;
                    }
                }
                if (soft) {
                    ++res.soft;
                    break;
                }
            }
        }
        if (res.opportunities + res.soft >= limit || res.soft >= 2) return res;
    }
    // Proof of a permanent lock: no free slot cutting any shortest-path edge keeps the
    // target connected. Walls are only ever added, so none of them ever will.
    res.locked = !legal_cut && !blocked_by_other;
    return res;
}

// --------------------------------------------------------------------------
// Move generation
// --------------------------------------------------------------------------
int Board::pawn_moves(int player, Move* out) const {
    const int me = pawn[player], opp = pawn[1 - player];
    int n = 0;
    auto add = [&](int sq) {
        for (int i = 0; i < n; ++i)
            if (out[i] == sq) return;
        out[n++] = Move(sq);
    };
    for (int d = 0; d < 4; ++d) {
        if (!open[d].test(me)) continue;
        const int step = me + STEP[d];
        if (step != opp) {
            add(step);
            continue;
        }
        // The opponent is adjacent in this direction: straight jump ...
        if (open[d].test(opp)) {
            add(opp + STEP[d]);
            continue;
        }
        // ... or, if a wall or the board edge is behind it, the two diagonals.
        const int s0 = d < 2 ? WEST : NORTH;
        for (int sd = s0; sd < s0 + 2; ++sd)
            if (open[sd].test(opp)) add(opp + STEP[sd]);
    }
    return n;
}

uint64_t Board::free_h() const {
    return ~(hw | ((hw & ~COL7) << 1) | ((hw & ~COL0) >> 1) | vw);
}

uint64_t Board::free_v() const {
    return ~(vw | (vw << NW) | (vw >> NW) | hw);
}

void Board::legal_moves(MoveList& out) const {
    const int me = turn;
    out.n = pawn_moves(me, out.m);
    if (walls_left[me] == 0) return;
    // A wall can only disconnect a player if it cuts a shortest path of that
    // player, so only those walls need a path-existence BFS.
    PathCut cut[2];
    for (int p = 0; p < 2; ++p) {
        GoalLayers gl;
        goal_layers(p, gl, pawn[p]);
        cut[p] = shortest_path_cuts(p, gl);
    }
    for (int o = 0; o < 2; ++o) {
        const uint64_t c0 = o == 0 ? cut[0].h : cut[0].v;
        const uint64_t c1 = o == 0 ? cut[1].h : cut[1].v;
        for (uint64_t b = o == 0 ? free_h() : free_v(); b; b &= b - 1) {
            const int a = std::countr_zero(b);
            const Move m = o == 0 ? hwall_move(a) : vwall_move(a);
            if (((c0 >> a) & 1) && distance_with_wall(0, m) < 0) continue;
            if (((c1 >> a) & 1) && distance_with_wall(1, m) < 0) continue;
            out.add(m);
        }
    }
}

void Board::legal_moves_slow(MoveList& out) const {
    const int me = turn;
    out.n = pawn_moves(me, out.m);
    if (walls_left[me] == 0) return;
    for (int o = 0; o < 2; ++o) {
        for (int a = 0; a < NWALL; ++a) {
            const int c = a % NW;
            bool clash;
            if (o == 0)
                clash = ((hw >> a) & 1) || ((vw >> a) & 1) || (c > 0 && ((hw >> (a - 1)) & 1)) ||
                        (c < NW - 1 && ((hw >> (a + 1)) & 1));
            else
                clash = ((vw >> a) & 1) || ((hw >> a) & 1) || (a >= NW && ((vw >> (a - NW)) & 1)) ||
                        (a + NW < NWALL && ((vw >> (a + NW)) & 1));
            if (clash) continue;
            const Move m = o == 0 ? hwall_move(a) : vwall_move(a);
            if (distance_with_wall(0, m) < 0 || distance_with_wall(1, m) < 0) continue;
            out.add(m);
        }
    }
}

bool Board::is_legal(Move m) const {
    const int me = turn;
    if (m >= NMOVES) return false;
    if (is_pawn(m)) {
        Move pm[8];
        const int n = pawn_moves(me, pm);
        for (int i = 0; i < n; ++i)
            if (pm[i] == m) return true;
        return false;
    }
    if (walls_left[me] == 0) return false;
    const bool h = is_hwall(m);
    const int a = h ? m - H_BASE : m - V_BASE;
    if (!(((h ? free_h() : free_v()) >> a) & 1)) return false;
    return distance_with_wall(0, m) >= 0 && distance_with_wall(1, m) >= 0;
}

}  // namespace quoridor
