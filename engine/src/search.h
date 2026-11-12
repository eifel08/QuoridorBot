// Negamax alpha-beta search with iterative deepening, a Zobrist transposition
// table, staged move ordering, wall filtering, decided-position cutoffs and
// futility pruning. See README.md for the description of every flag.
#pragma once

#include <chrono>
#include <cstdint>
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

    // ---- search features ----
    bool use_tt = true;           // transposition table
    bool tt_exact_depth = false;  // only reuse entries of exactly the remaining depth (verification)
    bool alphabeta = true;        // false = plain minimax (verification)
    bool killers = true;          // killer-move ordering
    bool history = true;          // history-heuristic ordering
    int wall_filter = FILTER_ATTACK_DEFENSE;  // Step 3
    bool decided = true;          // Step 5: exact race cutoffs
    int decided_margin = 1;       // extra steps required when the pawns can still meet
    bool futility = true;         // Step 6
    int wall_recovery = 5;        // futility margin per wall the side to move still has
    int depth_margin = 5;         // futility margin per remaining ply

    uint64_t seed = 0;  // != 0: break ties between equally ordered root moves at random
    bool verbose = false;  // print one "info" line per completed depth
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
    int gen_walls(int me, int d_me, int d_opp, const GoalLayers& gl_me, int filter, Cand* out) const;
    TTEntry* tt_probe(uint64_t key);
    void tt_store(uint64_t key, int depth, int value, int flag, Move move);
};

}  // namespace quoridor
