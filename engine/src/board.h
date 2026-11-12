// Quoridor board: bitboard representation, make/unmake, flood-fill BFS and
// legal move generation. The rules are exactly those of quoridor/board.py.
//
// Squares are numbered row * 9 + col (row 0 at the top). Player 0 starts on
// row 8 and must reach row 0; player 1 starts on row 0 and must reach row 8.
// A wall is anchored at (row, col) with 0 <= row, col <= 7:
//   horizontal: lies below squares (row, col) and (row, col + 1)
//   vertical:   lies right of squares (row, col) and (row + 1, col)
#pragma once

#include <bit>
#include <cstdint>

namespace quoridor {

constexpr int N = 9;
constexpr int NSQ = N * N;      // 81 squares
constexpr int NW = N - 1;       // wall anchors per row / column
constexpr int NWALL = NW * NW;  // 64 wall slots per orientation
constexpr int MAX_WALLS = 10;

enum Dir { NORTH = 0, SOUTH = 1, WEST = 2, EAST = 3 };

// --------------------------------------------------------------------------
// 81-bit square set (bit i = square i). Plain struct instead of __int128 so
// the engine also builds with MSVC.
// --------------------------------------------------------------------------
struct BB {
    uint64_t lo, hi;

    BB() = default;  // uninitialised; write BB{} for the empty set
    constexpr BB(uint64_t l, uint64_t h) : lo(l), hi(h) {}

    static constexpr BB square(int sq) {
        return sq < 64 ? BB(1ULL << sq, 0) : BB(0, 1ULL << (sq - 64));
    }
    constexpr bool test(int sq) const {
        return sq < 64 ? (lo >> sq) & 1 : (hi >> (sq - 64)) & 1;
    }
    constexpr bool any() const { return (lo | hi) != 0; }
    constexpr BB operator&(BB o) const { return {lo & o.lo, hi & o.hi}; }
    constexpr BB operator|(BB o) const { return {lo | o.lo, hi | o.hi}; }
    constexpr BB operator~() const { return {~lo, ~hi & 0x1FFFFULL}; }
    constexpr BB& operator|=(BB o) { lo |= o.lo; hi |= o.hi; return *this; }
    constexpr BB& operator&=(BB o) { lo &= o.lo; hi &= o.hi; return *this; }
    constexpr bool operator==(const BB& o) const = default;
    // Shifts by 1 (one column) or 9 (one row); callers mask out board edges first.
    constexpr BB shl(int k) const { return {lo << k, ((hi << k) | (lo >> (64 - k))) & 0x1FFFFULL}; }
    constexpr BB shr(int k) const { return {(lo >> k) | (hi << (64 - k)), hi >> k}; }
    // The 9 bits of one board row.
    constexpr uint32_t row(int r) const {
        int s = r * N;
        uint64_t v = s < 64 ? (lo >> s) | (s ? hi << (64 - s) : 0) : hi >> (s - 64);
        return uint32_t(v) & 0x1FF;
    }
};

// --------------------------------------------------------------------------
// Moves: one byte. 0..80 pawn to that square, 81..144 horizontal wall at
// anchor (m - 81), 145..208 vertical wall at anchor (m - 145).
// --------------------------------------------------------------------------
using Move = uint8_t;
constexpr Move MOVE_NONE = 255;
constexpr int H_BASE = NSQ, V_BASE = NSQ + NWALL, NMOVES = NSQ + 2 * NWALL;  // 209 codes
constexpr int MAX_MOVES = 5 + 2 * NWALL;  // at most 5 pawn moves + every wall slot

constexpr bool is_pawn(Move m) { return m < H_BASE; }
constexpr bool is_hwall(Move m) { return m >= H_BASE && m < V_BASE; }
constexpr Move pawn_move(int sq) { return Move(sq); }
constexpr Move hwall_move(int a) { return Move(H_BASE + a); }
constexpr Move vwall_move(int a) { return Move(V_BASE + a); }

struct MoveList {
    Move m[MAX_MOVES];
    int n = 0;
    void add(Move x) { m[n++] = x; }
};

// Result of flooding outwards from a player's goal row: layer[k] is the set
// of squares exactly k steps from the goal (pawns are ignored, as in Python).
struct GoalLayers {
    BB layer[NSQ + 1];
    int count;  // number of non-empty layers
    // Distance of a square to the goal, -1 if it cannot reach it.
    int dist(int sq) const {
        for (int k = 0; k < count; ++k)
            if (layer[k].test(sq)) return k;
        return -1;
    }
};

// Wall slots (as 64-bit anchor masks) that would block at least one edge
// lying on some shortest path of a player.
struct PathCut {
    uint64_t h = 0, v = 0;
};

struct Board {
    BB open[4];           // open[d]: squares from which a pawn may step in direction d
    uint64_t hw = 0, vw = 0;  // placed walls, bit = anchor row * 8 + col
    uint8_t pawn[2] = {0, 0};
    uint8_t walls_left[2] = {0, 0};
    uint8_t turn = 0;
    uint64_t hash = 0;

    static void init_tables();  // call once at startup

    void reset();  // start position
    // Set up an arbitrary position (does not validate it).
    void set(int turn_, int p0, int p1, int wl0, int wl1, uint64_t hw_, uint64_t vw_);
    uint64_t compute_hash() const;

    static BB goal_mask(int player);
    bool won(int player) const { return goal_mask(player).test(pawn[player]); }

    // ---- make / unmake (incremental hash, no copies) ----
    // Returns the mover's previous pawn square, to be handed back to unmake().
    int make(Move m);
    void unmake(Move m, int prev_sq);

    // ---- path finding ----
    BB expand(BB front) const {
        return (front & open[NORTH]).shr(N) | (front & open[SOUTH]).shl(N) |
               (front & open[WEST]).shr(1) | (front & open[EAST]).shl(1);
    }
    // Shortest number of steps from the player's pawn to its goal row, -1 if none.
    int distance(int player) const;
    // Same, with one extra wall (not yet on the board) taken into account.
    int distance_with_wall(int player, Move wall) const;
    // Flood from the goal row. If stop_sq >= 0 the flood stops once that square
    // has been reached (its layer is the last one); otherwise it covers the board.
    void goal_layers(int player, GoalLayers& out, int stop_sq = -1) const;
    // Walls cutting an edge of ANY shortest path of `player`. `gl` must come
    // from goal_layers(player, ...) and reach the pawn. Uses the identity
    // "u->v is on a shortest path iff dist_from_pawn[u] + 1 + dist_to_goal[v] == D".
    PathCut shortest_path_cuts(int player, const GoalLayers& gl) const;

    // ---- move generation ----
    int pawn_moves(int player, Move* out) const;  // returns the count (<= 5)
    // Wall slots that do not overlap/cross an existing wall.
    uint64_t free_h() const;
    uint64_t free_v() const;
    // Every legal move for the side to move (pawn moves first).
    void legal_moves(MoveList& out) const;
    // Slow reference implementation (one BFS per wall and player); tests only.
    void legal_moves_slow(MoveList& out) const;
    bool is_legal(Move m) const;
};

// Zobrist keys (fixed seed, so hashes are reproducible between runs).
struct Zobrist {
    uint64_t pawn[2][NSQ];
    uint64_t wall[2][NWALL];  // [0] horizontal, [1] vertical
    uint64_t walls_left[2][MAX_WALLS + 1];
    uint64_t turn;
};
extern Zobrist ZOB;

}  // namespace quoridor
