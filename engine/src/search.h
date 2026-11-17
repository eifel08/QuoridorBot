// Negamax alpha-beta search with iterative deepening, a Zobrist transposition
// table, staged move ordering, wall filtering, decided-position cutoffs and
// futility pruning. See README.md for the description of every flag.
#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include "board.h"

namespace quoridor {

constexpr int WIN = 30000;
constexpr int WIN_THRESHOLD = WIN - 1000;  // |score| >= this means a decided game
constexpr int INF = 32000;
constexpr int MAX_PLY = 96;
constexpr int MAX_DEPTH = 64;

enum WallFilter { FILTER_OFF = 0, FILTER_ATTACK_DEFENSE = 1, FILTER_ATTACK_ONLY = 2 };

struct Config {
    // ---- evaluation (Step 4) ----
    int dist_weight = 10;  // per step of shortest-path difference
    int wall_weight = 16;  // per wall of reserve difference
    int tempo_bonus = 5;   // side to move is effectively half a step ahead
    // Dynamic wall value: a player can only place walls on its own turns before the
    // opponent arrives, so its reserve counts for at most (opponent's distance + slack)
    // walls. Surplus walls are worthless in a race that is about to end.
    bool wall_cap = true;
    int wall_cap_slack = 1;
    // Blockability-aware wall value (see README "Evaluation"): a wall is only worth
    // what it can still do to the opponent's path.
    bool smart_walls = true;
    int wall_tie = 1;      // a wall that cannot lengthen the opponent's path at all (tie-breaker only)
    int wall_low = 10;     // a wall that can only gain 1 step (a block of 2+ steps keeps wall_weight)
    int wall_rich = 3;     // this many separate blocking bottlenecks: walls keep the full wall_weight
    int wall_chain = 3;    // greedy blocks simulated when the path is only barely blockable
    int wall_def = 6;      // a wall kept back to defend while the opponent can still block me
    int smart_dist = 6;    // analyse only paths at most this long (longer ones count as richly blockable)
    // Walls beyond the cap are not worthless: they are kept for later in the game. Each
    // counts this much, so spending one on a wall that changes nothing is never free.
    int wall_surplus = 1;
    bool lock_decided = true;  // race cutoffs also when the opponent's walls provably cannot touch my path

    // ---- search features ----
    bool use_tt = true;           // transposition table
    bool tt_exact_depth = false;  // only reuse entries of exactly the remaining depth (verification)
    bool alphabeta = true;        // false = plain minimax (verification)
    bool killers = true;          // killer-move ordering
    bool history = true;          // history-heuristic ordering
    int wall_filter = FILTER_ATTACK_DEFENSE;  // Step 3
    // Defensive walls beyond "overlaps the attacking wall" (filter 1 only):
    bool def_global = true;   // walls that make an attacking wall illegal (path-existence rule)
    bool def_funnel = true;   // walls extending such a barrier (one follow-up wall from sealing it)
    int def_threats = 2;      // strongest opponent walls against me that are analysed
    int def_min_depth = 1;    // remaining depth needed for global defense
    int def_funnel_depth = 3; // remaining depth needed for funnelling walls
    // Wall relevance: walls that are neither attack walls nor global defense must block an
    // edge on a route at most this many steps longer than some player's shortest path.
    // -1: only prune walls in dead regions (squares neither pawn can reach any more).
    int wall_zone = 4;
    // Leaf quiescence ("use it or lose it"): when the opponent's block against the side
    // to move is fragile (barely blockable path), the side to move first gets to try the
    // replies that kill it: walls that make the blocking walls illegal or pre-empt them,
    // and pawn steps into a corridor no wall can lengthen any more.
    bool qsearch = true;
    bool decided = true;          // Step 5: exact race cutoffs
    int decided_margin = 1;       // extra steps required when the pawns can still meet
    bool futility = true;         // Step 6
    int wall_recovery = 5;        // futility margin per wall the side to move still has
    int depth_margin = 5;         // futility margin per remaining ply

    // ---- bot evolution (see docs/bot-evolution-plan.md); all off = the baseline search ----
    bool smart_time = false;  // predicted iteration cost, up to ~95% of the limit, keep partial iterations
    bool pvs = false;         // principal variation search (null-window scouts + re-search)
    int aspiration = 0;       // > 0: aspiration window half-width around the score of depth - 2
    bool lmr = false;         // late move reductions for quiet / late moves
    int lmr_min_moves = 3;    // moves tried at a node before reductions start
    int lmr_min_depth = 3;    // remaining depth needed to reduce
    bool repetition = false;  // a position repeated on the search path scores as a draw
    // Evaluation: 0 = classic (hand-set), 1 = tuned linear features with phase taper,
    // 2 = neural (NNUE-style, loadnet), 3 = hybrid (neural while total walls >= hybrid_walls,
    // tuned linear below).
    int eval_mode = 0;
    int hybrid_walls = 6;
    // Endgame phase: with at most this many walls left in total (or one side out of walls) the
    // root disables the unsound prunings (LMR, futility) and widens the wall zone: a deep,
    // exact alpha-beta for the race. -1 = never.
    int endgame_walls = -1;

    uint64_t seed = 0;  // != 0: break ties between equally ordered root moves at random
    bool verbose = false;  // print one "info" line per completed depth
};

// Linear evaluation features (eval_mode 1), all "me minus opponent" from the side to move.
enum Feature {
    F_DIST = 0,     // opponent distance - my distance
    F_WALL_FULL,    // walls counted at full value (rich path / block of 2+ steps)
    F_WALL_LOW,     // walls that block exactly one step
    F_WALL_DEF,     // non-attacking walls kept back for defense
    F_WALL_TIE,     // walls that cannot lengthen the opponent's path
    F_WALL_SURPLUS, // walls above the cap
    F_TEMPO,        // constant 1 (side to move)
    F_NARROW,       // single-square layers of the opponent's shortest-path DAG minus mine
    F_PATHS,        // shortest-path layers with 3+ squares (many routes), mine minus opponent's
    F_DIST_PROD,    // (d_opp - d_me) scaled by closeness: 10 * diff / (d_me + d_opp + 2)
    NFEAT
};

// Small NNUE-style network: sparse binary inputs (pawn squares, wall slots and wall
// counts, side-to-move perspective) into an accumulator, concatenated with the
// linear features, then one hidden layer. Output in evaluation units.
struct Net {
    static constexpr int NSPARSE = 2 * NSQ + 2 * NWALL + 2 * (MAX_WALLS + 1);  // 312
    static constexpr int NDENSE = NFEAT + 3;
    int h1 = 0, h2 = 0;
    std::vector<float> w1, b1, w2, b2, w3;  // w1: NSPARSE x h1, w2: (h1 + NDENSE) x h2, w3: h2
    float b3 = 0, out_scale = 1;
    bool loaded() const { return h1 > 0; }
    bool load(const char* path);
};

struct SearchResult {
    Move best = MOVE_NONE;
    int score = 0;      // from the side to move's point of view
    int depth = 0;      // deepest completed iteration
    uint64_t nodes = 0;
    double seconds = 0;
};

class Search {
public:
    Config cfg;
    Board board;

    explicit Search(int hash_mb = 64);
    void resize_tt(int hash_mb);
    void new_game();  // forget everything learnt (TT, history, killers)

    // Iterative deepening up to max_depth; time_limit <= 0 means no limit.
    SearchResult go(int max_depth, double time_limit);

    int evaluate(int me, int d_me, int d_opp) const;
    int evaluate_classic(int me, int d_me, int d_opp) const;
    // Linear features of the position (eval_mode 1) and the phase (total walls left, 0..20).
    void features(int me, int d_me, int d_opp, int* f) const;
    int phase() const { return board.walls_left[0] + board.walls_left[1]; }
    int evaluate_linear(int me, int d_me, int d_opp) const;
    int evaluate_net(int me, int d_me, int d_opp) const;
    // Tuned weights: weights[0] at phase 20 (all walls), weights[1] at phase 0.
    int lin_w[2][NFEAT] = {{10, 16, 10, 6, 1, 1, 5, 0, 0, 0}, {10, 16, 10, 6, 1, 1, 5, 0, 0, 0}};
    Net net;
    // Shape of a player's shortest-path DAG: single-square layers and log2 of the path count.
    void path_shape(int target, int& narrow, int& log_paths) const;

    // Cached blockability of `target`'s path: the greedy chain of single-wall
    // blocks the other player could make (see board.h, Blockability).
    struct PathInfo {
        uint8_t trunc_dep;  // cache-internal: key-1 entry whose answer is under key 2
        uint8_t locked, nsteps, trunc;
        uint8_t gain[4];  // steps added by block i (0..nsteps-1)
        uint8_t rich[4];  // block i had >= wall_rich opportunities: stop, walls keep full value
    };
    PathInfo path_info(int target, int d_target = -1) const;
    // Value of `attacker`'s wall reserve against `target` (eval units). `w_eff`:
    // walls that count at all, `useful`: of those, the ones that can lengthen the path.
    int wall_attack_value(int attacker, int d_target, int& w_eff, int& useful) const;
    bool root_exclude_[NMOVES + 1] = {};  // root moves the next go() must not play (book building)
    // Moves in the order the search would try them at the root (debugging / tests).
    void ordered_moves(MoveList& out);

private:
    struct TTEntry {
        uint64_t key;
        int16_t value;
        Move move;
        int8_t depth;
        uint8_t flag;  // 0 = empty, else EXACT / LOWER / UPPER
        uint8_t age;
        uint8_t pad[2];
    };
    struct Cand {  // a move with everything the child node needs
        int32_t score;
        Move move;
        uint8_t d_me, d_opp;  // shortest-path lengths after the move
    };

    struct EvalEntry {
        uint64_t key;
        PathInfo info;
        uint8_t pad[4];
    };
    // Defensive analysis of a position for the side to move (cached: interior nodes recur
    // every iteration). thr: the opponent's single-wall threats (legal for them, lengthen my
    // path); def: global defense (walls making a top threat illegal); fun: funnelling walls.
    struct DefInfo {
        uint64_t thr[2] = {0, 0}, def[2] = {0, 0}, fun[2] = {0, 0};
        uint8_t level = 0;  // 0 = threats only, 1 = + global defense, 2 = + funnel
    };
    struct DefEntry {
        uint64_t key = 0;
        DefInfo info;
    };
    struct ShapeEntry {
        uint64_t key = 0;
        int8_t narrow = 0, log_paths = 0;
    };
    mutable std::vector<ShapeEntry> scache_;
    struct EvCacheEntry {
        uint64_t key = 0;
        int32_t value = 0;
    };
    mutable std::vector<EvCacheEntry> evcache_;  // static evals (eval_mode >= 1)
    uint64_t path_[MAX_PLY + 2];  // hashes along the search path (repetition)
    int root_scores_[MAX_DEPTH + 2];
    bool root_ok_ = false;  // root_best_ of the running iteration is a real (non fail-low) result
    int root_alpha_ = -INF;
    mutable std::vector<DefEntry> dcache_;
    mutable std::vector<EvalEntry> ecache_;  // allocated once; no allocation during the search
    uint64_t ecache_mask_ = 0;

    std::vector<TTEntry> tt_;
    uint64_t tt_mask_ = 0;
    uint8_t age_ = 0;
    int32_t history_[2][NMOVES];
    Move killer_[MAX_PLY][2];

    uint64_t nodes_ = 0;
    bool stop_ = false;
    bool timed_ = false;
    std::chrono::steady_clock::time_point deadline_;
    Move root_best_ = MOVE_NONE;   // best root move of the running iteration
    Move root_hint_ = MOVE_NONE;   // best root move of the previous iteration
    uint64_t rng_ = 0;

    int negamax(int depth, int alpha, int beta, int ply, int d_me, int d_opp);
    int decided_score(int me, int d_me, int d_opp, int ply) const;
    bool can_interact(int winner, int d_winner) const;
    int gen_walls(int me, int d_me, int d_opp, const GoalLayers& gl_me, int filter, int depth, Cand* out) const;
    void defense(int me, int d_me, const PathCut& cut_me, const PathCut& cut_opp, uint64_t free_h, uint64_t free_v,
                 int level, DefInfo& out) const;
    void global_defense(const Move* threat, int nt, uint64_t free_h, uint64_t free_v, bool funnel, DefInfo& di) const;
    int quiesce(int alpha, int beta, int ply, int d_me, int d_opp);
    int leaf_value(int ply, int d_me, int d_opp) const;  // decided score or static eval, side to move
    TTEntry* tt_probe(uint64_t key);
    void tt_store(uint64_t key, int depth, int value, int flag, Move move);
};

}  // namespace quoridor
