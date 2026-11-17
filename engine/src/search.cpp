#include "search.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace quoridor {

namespace {

enum : uint8_t { TT_EMPTY = 0, TT_EXACT = 1, TT_LOWER = 2, TT_UPPER = 3 };

constexpr uint64_t COL0 = 0x0101010101010101ULL;
constexpr uint64_t COL7 = 0x8080808080808080ULL;

// Ordering scores of the "everything after the shortest-path pawn moves" stage.
// Moves are tried tier by tier: killers, attack walls, defense walls, then the
// rest (other walls and pawn moves that do not shorten my path). Inside a tier:
// most steps added to the opponent's path, fewest added to mine, then history.
constexpr int32_t SCORE_KILLER = 0x70000000;
constexpr int32_t TIER_ATTACK = 0x40000000;   // blocks an edge on a shortest path of the opponent
constexpr int32_t TIER_DEFENSE = 0x20000000;  // pre-empts a wall that would block one of my shortest paths
constexpr int32_t TIER_FUNNEL = 0x10000000;   // extends a barrier one wall away from pre-empting one
constexpr int32_t SCORE_PER_STEP = 1 << 18;   // per step added to the opponent's path
constexpr int32_t SCORE_OWN_STEP = 1 << 14;   // per step added to my own path (penalty)
constexpr int32_t HISTORY_MAX = SCORE_OWN_STEP - 1;
constexpr int32_t SCORE_ROOT_PAWN = 0x7FFFFF00;  // ordered_moves(): shortest-path pawn moves first

// Win scores are stored relative to the node ("win in k plies from here") so
// that an entry stays correct when the position is reached at another ply.
inline int value_to_tt(int v, int ply) {
    return v >= WIN_THRESHOLD ? v + ply : v <= -WIN_THRESHOLD ? v - ply : v;
}
inline int value_from_tt(int v, int ply) {
    return v >= WIN_THRESHOLD ? v - ply : v <= -WIN_THRESHOLD ? v + ply : v;
}

inline uint64_t xorshift(uint64_t& s) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
}

}  // namespace

Search::Search(int hash_mb) {
    Board::init_tables();
    board.reset();
    resize_tt(hash_mb);
    ecache_.assign(1 << 15, EvalEntry{});  // 768 KB: stays in the CPU cache
    ecache_mask_ = ecache_.size() - 1;
    dcache_.assign(1 << 14, DefEntry{});  // 640 KB
    new_game();
}

void Search::resize_tt(int hash_mb) {
    uint64_t want = (uint64_t(std::max(1, hash_mb)) << 20) / sizeof(TTEntry);
    uint64_t size = 2;
    while (size * 2 <= want) size *= 2;
    tt_.assign(size, TTEntry{});
    tt_mask_ = size - 2;  // buckets of two entries: index is always even
}

void Search::new_game() {
    std::fill(tt_.begin(), tt_.end(), TTEntry{});
    std::fill(ecache_.begin(), ecache_.end(), EvalEntry{});
    std::fill(dcache_.begin(), dcache_.end(), DefEntry{});
    std::memset(history_, 0, sizeof(history_));
    std::memset(killer_, MOVE_NONE, sizeof(killer_));
    age_ = 0;
}

// --------------------------------------------------------------------------
// Transposition table: buckets of two entries. Slot 0 is depth-preferred,
// slot 1 is always replaced. Entries of earlier searches (older age) are
// treated as replaceable.
// --------------------------------------------------------------------------
Search::TTEntry* Search::tt_probe(uint64_t key) {
    TTEntry* e = &tt_[key & tt_mask_];
    if (e[0].flag != TT_EMPTY && e[0].key == key) return &e[0];
    if (e[1].flag != TT_EMPTY && e[1].key == key) return &e[1];
    return nullptr;
}

void Search::tt_store(uint64_t key, int depth, int value, int flag, Move move) {
    TTEntry* e = &tt_[key & tt_mask_];
    TTEntry* slot;
    if (e[0].flag != TT_EMPTY && e[0].key == key)
        slot = &e[0];
    else if (e[1].flag != TT_EMPTY && e[1].key == key)
        slot = &e[1];
    else if (e[0].flag == TT_EMPTY || e[0].age != age_ || depth >= e[0].depth)
        slot = &e[0];
    else
        slot = &e[1];
    if (move == MOVE_NONE && slot->flag != TT_EMPTY && slot->key == key) move = slot->move;
    slot->key = key;
    slot->value = int16_t(value);
    slot->move = move;
    slot->depth = int8_t(depth);
    slot->flag = uint8_t(flag);
    slot->age = age_;
}

// --------------------------------------------------------------------------
// Evaluation (Step 4), from the side to move's point of view.
// --------------------------------------------------------------------------
Search::PathInfo Search::path_info(int target, int d_target) const {
    // Fast path, no flood at all: a distance equal to the row distance means the
    // only shortest path is straight along the pawn's column (every step must
    // gain a row), so each of its edges is a bottleneck. Richly blockable as soon
    // as `wall_rich` of them have a free covering slot that cannot disconnect.
    {
        const int r = board.pawn[target] / N, c = board.pawn[target] % N;
        const int rows = target == 0 ? r : N - 1 - r;
        if (d_target >= 0 && d_target == rows && rows >= cfg.wall_rich) {
            const uint64_t fh = board.free_h();
            uint64_t slots = 0;  // free covering slots, one bit per step
            int n = 0;
            for (int k = 0; k < rows; ++k) {
                const int er = target == 0 ? r - 1 - k : r + k;  // upper row of the k-th edge
                const uint64_t cover = ((c > 0 ? 1ULL << (er * NW + c - 1) : 0) | (c < NW ? 1ULL << (er * NW + c) : 0)) & fh;
                if (cover) {
                    slots |= cover;
                    ++n;
                }
            }
            if (n >= cfg.wall_rich) {
                const Points barrier = barrier_points(board.hw, board.vw);
                n = 0;
                for (int k = 0; k < rows && n < cfg.wall_rich; ++k) {
                    const int er = target == 0 ? r - 1 - k : r + k;
                    for (int dc = -1; dc <= 0; ++dc) {
                        const int cc = c + dc;
                        if (cc < 0 || cc >= NW || !((slots >> (er * NW + cc)) & 1)) continue;
                        if (!may_disconnect(barrier, hwall_move(er * NW + cc))) {
                            ++n;
                            break;
                        }
                    }
                }
                if (n >= cfg.wall_rich) {
                    PathInfo pi{};
                    pi.nsteps = 1;
                    pi.rich[0] = 1;
                    pi.gain[0] = 1;
                    return pi;
                }
            }
        }
    }
    // Two cache levels. Key 1 is the walls plus the target's pawn: the fast pass
    // (target connectivity only) settles most positions, "richly blockable" or
    // "no usable bottleneck at all", and that answer does not depend on the
    // other pawn, so the attacker's pawn moves keep hitting it. Only barely
    // blockable paths need the exact chain, cached under key 2 (+ other pawn).
    static constexpr uint64_t SALT[2] = {0x9E3779B97F4A7C15ULL, 0xC2B2AE3D27D4EB4FULL};
    const int other = target ^ 1;
    // The incremental hash minus pawns, reserves and turn: the walls alone.
    const uint64_t walls_key = board.hash ^ ZOB.pawn[0][board.pawn[0]] ^ ZOB.pawn[1][board.pawn[1]] ^
                               ZOB.walls_left[0][board.walls_left[0]] ^ ZOB.walls_left[1][board.walls_left[1]] ^
                               (board.turn ? ZOB.turn : 0);
    const uint64_t key1 = walls_key ^ ZOB.pawn[target][board.pawn[target]] ^ SALT[target];
    EvalEntry* e = &ecache_[key1 & ecache_mask_];
    PathInfo pi{};
    if (e->key == key1) {
        if (!e->info.trunc_dep) return e->info;
    } else {
        Layout lay = board.layout();
        const Blockability fast = analyse_blockability(lay, target, std::max(1, cfg.wall_rich), false);
        pi.locked = fast.locked;
        const bool none = fast.opportunities == 0 && fast.soft == 0;
        if (none || fast.opportunities + fast.soft >= cfg.wall_rich || fast.soft >= 2) {
            if (!none) {
                pi.nsteps = 1;
                pi.rich[0] = 1;
                pi.gain[0] = uint8_t(std::min(fast.best_gain, 255));
            }
            e->key = key1;
            e->info = pi;
            return pi;
        }
        e->key = key1;
        e->info = pi;
        e->info.trunc_dep = 1;  // depends on the other pawn: see key 2
    }
    const uint64_t key2 = key1 ^ ZOB.pawn[other][board.pawn[other]] ^ 0x5851F42D4C957F2DULL;
    e = &ecache_[key2 & ecache_mask_];
    if (e->key == key2) return e->info;
    pi = PathInfo{};
    Layout lay = board.layout();
    const int chain = std::clamp(cfg.wall_chain, 1, 4);
    for (int i = 0; i < chain; ++i) {
        const Blockability b = analyse_blockability(lay, target, std::max(1, cfg.wall_rich), true);
        if (i == 0) pi.locked = b.locked;
        if (b.opportunities == 0 && b.soft == 0) break;  // the next block is impossible: the chain ends
        pi.nsteps = uint8_t(i + 1);
        pi.gain[i] = uint8_t(std::min(b.best_gain, 255));
        if (b.opportunities + b.soft >= cfg.wall_rich || b.soft >= 2) {
            pi.rich[i] = 1;
            break;
        }
        if (b.opportunities == 0) {
            // Only a soft layer left: about one step per wall for the two walls that close it.
            pi.gain[i] = 1;
            if (i + 1 < 4) {
                pi.gain[i + 1] = 1;
                pi.nsteps = uint8_t(i + 2);
            }
            break;
        }
        if (i == chain - 1) {
            pi.trunc = 1;  // still blockable after `chain` blocks: assume it stays that way
            break;
        }
        apply_wall(lay, b.best_wall);  // the opponent's best detour is the next path to block
    }
    e->key = key2;
    e->info = pi;
    return pi;
}

int Search::wall_attack_value(int attacker, int d_target, int& w_eff, int& useful) const {
    const int w = board.walls_left[attacker];
    // I can use at most one wall per turn I still get before the opponent arrives.
    w_eff = cfg.wall_cap ? std::min(w, d_target + cfg.wall_cap_slack) : w;
    useful = 0;
    if (w_eff <= 0) return w_eff = 0;
    // Far from the goal a path is practically always richly blockable; the analysis
    // only pays off in races and endgames, where the target is close to its goal.
    if (!cfg.smart_walls || d_target > cfg.smart_dist) {
        useful = w_eff;
        return w_eff * cfg.wall_weight;
    }
    const PathInfo pi = path_info(attacker ^ 1, d_target);
    // A block worth 2+ steps keeps the full weight; a +1 block is worth less (it
    // barely beats spending the turn on a pawn step).
    auto value_of = [&](int gain) { return gain >= 2 ? cfg.wall_weight : gain == 1 ? cfg.wall_low : cfg.wall_tie; };
    int value = 0;
    for (int i = 0; i < pi.nsteps && useful < w_eff; ++i) {
        if (pi.rich[i]) {  // plenty of places left to block: every remaining wall counts in full
            value += (w_eff - useful) * cfg.wall_weight;
            useful = w_eff;
            break;
        }
        value += value_of(pi.gain[i]);
        ++useful;
    }
    if (useful < w_eff && pi.trunc) {  // still blockable after the simulated blocks: assume it stays so
        value += (w_eff - useful) * cfg.wall_weight;
        useful = w_eff;
    }
    return value;
}

int Search::evaluate(int me, int d_me, int d_opp) const {
    const int opp = me ^ 1;
    int my_eff, my_useful, opp_eff, opp_useful;
    int my_walls = wall_attack_value(me, d_opp, my_eff, my_useful);
    int opp_walls = wall_attack_value(opp, d_me, opp_eff, opp_useful);
    if (cfg.smart_walls) {
        // Walls that cannot attack are still worth something as long as the other side
        // can attack: kept back for defence, one per wall that can hurt us.
        const int my_rest = my_eff - my_useful, opp_rest = opp_eff - opp_useful;
        const int my_def = std::min(my_rest, opp_useful), opp_def = std::min(opp_rest, my_useful);
        my_walls += my_def * cfg.wall_def + (my_rest - my_def) * cfg.wall_tie;
        opp_walls += opp_def * cfg.wall_def + (opp_rest - opp_def) * cfg.wall_tie;
    }
    return (d_opp - d_me) * cfg.dist_weight + (my_walls - opp_walls) + cfg.tempo_bonus;
}

// --------------------------------------------------------------------------
// Decided positions (Step 5). A player without walls cannot lengthen the
// other one's path, so the game is a pure race the player who is ahead wins
// by walking. Returns INF if the position is not decided.
//
// The only thing that can still change the race is the pawns meeting (a jump
// gains a step, a pawn in a corridor can cost one). If the loser cannot get
// next to the winner's route in time the result is exact; otherwise an extra
// `decided_margin` steps are required, else the normal search continues.
// --------------------------------------------------------------------------
int Search::decided_score(int me, int d_me, int d_opp, int ply) const {
    const int opp = me ^ 1;
    // Walls that provably can never lengthen a path (locked corridor) count as no walls.
    auto harmless = [&](int attacker, int target) {
        if (board.walls_left[attacker] == 0) return true;
        const int d = target == me ? d_me : d_opp;
        return cfg.lock_decided && cfg.smart_walls && d <= cfg.smart_dist && path_info(target, d).locked;
    };
    if (d_me <= d_opp && harmless(opp, me)) {
        // I move first: my d_me-th move comes before the opponent's d_me-th.
        if (d_me + cfg.decided_margin <= d_opp || !can_interact(me, d_me))
            return WIN - (ply + 2 * d_me - 1);
    } else if (d_opp < d_me && harmless(me, opp)) {
        if (d_opp + cfg.decided_margin < d_me || !can_interact(opp, d_opp))
            return -(WIN - (ply + 2 * d_opp));
    }
    return INF;
}

bool Search::can_interact(int winner, int d_winner) const {
    const int loser = winner ^ 1;
    GoalLayers gl;
    board.goal_layers(winner, gl, board.pawn[winner]);
    BB route = BB::square(board.pawn[winner]);  // where the winner can be after i moves
    BB reach = BB::square(board.pawn[loser]);   // where the loser can be within i + 1 moves
    for (int i = 0; i < d_winner; ++i) {
        reach |= board.expand(reach);
        if (((route | board.expand(route)) & reach).any()) return true;
        route = board.expand(route) & gl.layer[d_winner - i - 1];
    }
    return false;
}

// --------------------------------------------------------------------------
// Wall generation (Step 3) with exact distance bookkeeping.
//
// Legality: a wall can only disconnect a player if it cuts one of that
// player's shortest paths, so the BFS is skipped for every other wall (and
// the player's distance is then known to be unchanged).
// --------------------------------------------------------------------------
// Global (topological) defense. For the opponent's strongest walls against
// me (W_atk: legal now, lengthens my path most), find every wall W_def that
// makes W_atk ILLEGAL: with both on the board some player has no path left.
// Only walls cutting a shortest path of that player in board + W_atk can
// disconnect it, so those are the only ones tested (one early-exit BFS each).
// With `funnel`, also the walls one step short of that: free slots touching
// W_atk or a W_def AND touching a placed wall or the board edge, i.e. that
// extend a barrier which one follow-up wall would close.
void Search::global_defense(int me, int d_me, const PathCut& cut_me, const PathCut& cut_opp, uint64_t free_h,
                            uint64_t free_v, bool funnel, uint64_t (&def)[2], uint64_t (&fun)[2]) const {
    const int opp = me ^ 1;
    constexpr int MAXT = 4;
    Move threat[MAXT];
    int loss[MAXT], nt = 0;
    const int want = std::clamp(cfg.def_threats, 1, MAXT);
    const Points now = barrier_points(board.hw, board.vw);
    for (int o = 0; o < 2; ++o) {
        const uint64_t mine = o == 0 ? cut_me.h & free_h : cut_me.v & free_v;
        const uint64_t theirs = o == 0 ? cut_opp.h : cut_opp.v;
        for (uint64_t b = mine; b; b &= b - 1) {
            const int a = std::countr_zero(b);
            const Move m = o == 0 ? hwall_move(a) : vwall_move(a);
            const int nd = board.distance_with_wall(me, m);
            if (nd <= d_me) continue;  // not a threat (or illegal: it would cut me off)
            if (((theirs >> a) & 1) && may_disconnect(now, m) && board.distance_with_wall(opp, m) < 0)
                continue;  // illegal for them
            const int l = nd - d_me;
            int k = nt < want ? nt++ : want;
            if (k == want && l <= loss[want - 1]) continue;
            if (k == want) k = want - 1;
            while (k > 0 && loss[k - 1] < l) {
                threat[k] = threat[k - 1];
                loss[k] = loss[k - 1];
                --k;
            }
            threat[k] = m;
            loss[k] = l;
        }
    }
    if (nt == 0) return;

    // Slots that already lean on something: a placed wall or the board edge.
    uint64_t anchored[2] = {COL0 | COL7, 0xFFULL | (0xFFULL << 56)};
    if (funnel) {
        for (int o = 0; o < 2; ++o)
            for (uint64_t b = o == 0 ? board.hw : board.vw; b; b &= b - 1) {
                uint64_t th, tv;
                touching_slots(o == 0 ? hwall_move(std::countr_zero(b)) : vwall_move(std::countr_zero(b)), th, tv);
                anchored[0] |= th;
                anchored[1] |= tv;
            }
    }

    for (int t = 0; t < nt; ++t) {
        Layout lay = board.layout();
        apply_wall(lay, threat[t]);
        const uint64_t fh = free_h_of(lay.hw, lay.vw) & free_h, fv = free_v_of(lay.hw, lay.vw) & free_v;
        const Points barrier = barrier_points(lay.hw, lay.vw);  // with W_atk
        uint64_t found[2] = {0, 0};
        for (int p = 0; p < 2; ++p) {
            GoalLayers gl;
            goal_layers_open(lay.open, p, gl, lay.pawn[p]);
            const PathCut c = path_cuts_open(lay.open, lay.pawn[p], gl);
            for (int o = 0; o < 2; ++o)
                for (uint64_t b = (o == 0 ? c.h & fh : c.v & fv) & ~(def[o] | found[o]); b; b &= b - 1) {
                    const int a = std::countr_zero(b);
                    const Move m = o == 0 ? hwall_move(a) : vwall_move(a);
                    if (!may_disconnect(barrier, m)) continue;  // closes no barrier curve: cannot cut anyone off
                    BB x[4] = {lay.open[0], lay.open[1], lay.open[2], lay.open[3]};
                    clear_wall(x, m);
                    if (flood_distance(x, lay.pawn[p], Board::goal_mask(p)) < 0) found[o] |= 1ULL << a;
                }
        }
        def[0] |= found[0];
        def[1] |= found[1];
        if (funnel) {
            uint64_t near[2];
            touching_slots(threat[t], near[0], near[1]);
            for (int o = 0; o < 2; ++o)
                for (uint64_t b = found[o]; b; b &= b - 1) {
                    uint64_t th, tv;
                    touching_slots(o == 0 ? hwall_move(std::countr_zero(b)) : vwall_move(std::countr_zero(b)), th, tv);
                    near[0] |= th;
                    near[1] |= tv;
                }
            fun[0] |= near[0] & anchored[0] & free_h;
            fun[1] |= near[1] & anchored[1] & free_v;
        }
    }
    fun[0] &= ~def[0];
    fun[1] &= ~def[1];
}

int Search::gen_walls(int me, int d_me, int d_opp, const GoalLayers& gl_me, int filter, int depth, Cand* out) const {
    const int opp = me ^ 1;
    GoalLayers gl_opp;
    board.goal_layers(opp, gl_opp, board.pawn[opp]);
    const PathCut cut_opp = board.shortest_path_cuts(opp, gl_opp);
    const PathCut cut_me = board.shortest_path_cuts(me, gl_me);
    const uint64_t free_h = board.free_h(), free_v = board.free_v();

    // Attack: walls that block an edge on any of the opponent's shortest paths.
    const uint64_t att_h = cut_opp.h & free_h, att_v = cut_opp.v & free_v;
    // Local defense: walls that overlap or cross a wall the opponent could use
    // to block an edge on any of my shortest paths.
    const uint64_t th = cut_me.h & free_h, tv = cut_me.v & free_v;
    uint64_t def_h = (((th & ~COL7) << 1) | ((th & ~COL0) >> 1) | tv) & free_h;
    uint64_t def_v = ((tv << NW) | (tv >> NW) | th) & free_v;
    // Global defense (path-existence) and funnelling walls; see global_defense().
    uint64_t gdef[2] = {0, 0}, fun[2] = {0, 0};
    if (filter == FILTER_ATTACK_DEFENSE && cfg.def_global && board.walls_left[opp] > 0 && depth >= cfg.def_min_depth) {
        const bool funnel = cfg.def_funnel && depth > cfg.def_min_depth;
        DefEntry& e = dcache_[board.hash & (dcache_.size() - 1)];
        if (e.key == board.hash && (e.funnel || !funnel)) {
            gdef[0] = e.def[0];
            gdef[1] = e.def[1];
            if (funnel) {
                fun[0] = e.fun[0];
                fun[1] = e.fun[1];
            }
        } else {
            global_defense(me, d_me, cut_me, cut_opp, free_h, free_v, funnel, gdef, fun);
            e = DefEntry{board.hash, {gdef[0], gdef[1]}, {fun[0], fun[1]}, uint8_t(funnel), {}};
        }
    }
    def_h |= gdef[0];
    def_v |= gdef[1];
    fun[0] &= ~(def_h | att_h);
    fun[1] &= ~(def_v | att_v);

    uint64_t cand_h = free_h, cand_v = free_v;
    if (filter == FILTER_ATTACK_ONLY) {
        cand_h = att_h;
        cand_v = att_v;
    } else if (filter == FILTER_ATTACK_DEFENSE) {
        cand_h = att_h | def_h | fun[0];
        cand_v = att_v | def_v | fun[1];
    }

    int n = 0;
    for (int o = 0; o < 2; ++o) {
        const uint64_t hits_opp = o == 0 ? cut_opp.h : cut_opp.v;
        const uint64_t hits_me = o == 0 ? cut_me.h : cut_me.v;
        for (uint64_t b = o == 0 ? cand_h : cand_v; b; b &= b - 1) {
            const int a = std::countr_zero(b);
            const Move m = o == 0 ? hwall_move(a) : vwall_move(a);
            int nd_opp = d_opp, nd_me = d_me;
            if ((hits_opp >> a) & 1) {
                nd_opp = board.distance_with_wall(opp, m);
                if (nd_opp < 0) continue;  // illegal: opponent cut off
            }
            if ((hits_me >> a) & 1) {
                nd_me = board.distance_with_wall(me, m);
                if (nd_me < 0) continue;  // illegal: I would be cut off
            }
            const int gain = nd_opp - d_opp, loss = nd_me - d_me;
            int32_t s = cfg.history ? history_[me][m] : 0;
            s += gain * SCORE_PER_STEP - loss * SCORE_OWN_STEP;
            const bool global = ((gdef[o] >> a) & 1) != 0;
            if ((hits_opp >> a) & 1) s += TIER_ATTACK + (global ? SCORE_PER_STEP : 0);  // attacks AND defends
            else if (((o == 0 ? def_h : def_v) >> a) & 1) s += TIER_DEFENSE + (global ? SCORE_PER_STEP : 0);
            else if ((fun[o] >> a) & 1) s += TIER_FUNNEL;
            out[n++] = Cand{s, m, uint8_t(nd_me), uint8_t(nd_opp)};
        }
    }
    return n;
}

// --------------------------------------------------------------------------
// Negamax
// --------------------------------------------------------------------------
int Search::negamax(int depth, int alpha, int beta, int ply, int d_me, int d_opp) {
    ++nodes_;
    if (timed_ && (nodes_ & 1023) == 0 && std::chrono::steady_clock::now() >= deadline_) stop_ = true;
    if (stop_) return 0;

    const int me = board.turn, opp = me ^ 1;
    const bool root = ply == 0;

    if (!root && cfg.decided) {
        const int v = decided_score(me, d_me, d_opp, ply);
        if (v != INF) return v;
    }
    if (depth <= 0 || ply >= MAX_PLY - 1) return evaluate(me, d_me, d_opp);

    // ---- transposition table ----
    Move tt_move = root ? root_hint_ : MOVE_NONE;
    if (cfg.use_tt) {
        if (const TTEntry* e = tt_probe(board.hash)) {
            tt_move = e->move;  // always used for ordering, whatever its depth
            const bool deep_enough = cfg.tt_exact_depth ? e->depth == depth : e->depth >= depth;
            if (!root && deep_enough) {
                const int v = value_from_tt(e->value, ply);
                if (e->flag == TT_EXACT) return v;
                if (cfg.alphabeta) {
                    if (e->flag == TT_LOWER && v >= beta) return v;
                    if (e->flag == TT_UPPER && v <= alpha) return v;
                }
            }
        }
    }

    // ---- futility pruning (Step 6) ----
    // A position this far below alpha can only be rescued by walls, so the
    // margin grows with the walls the side to move still has.
    if (cfg.futility && !root && alpha > -WIN_THRESHOLD && alpha < WIN_THRESHOLD) {
        const int bound = evaluate(me, d_me, d_opp) + int(board.walls_left[me]) * cfg.wall_recovery +
                          cfg.depth_margin * depth;
        if (bound < alpha) return bound;
    }

    const int alpha0 = alpha;
    int best = -INF;
    Move best_move = MOVE_NONE;
    int tried = 0;

    // Searches one move; returns true if the node is finished (cutoff or stop).
    auto search_move = [&](Move m, int nd_me, int nd_opp) -> bool {
        ++tried;
        int val;
        const int prev = board.make(m);
        if (is_pawn(m) && nd_me == 0)
            val = WIN - (ply + 1);  // reached the goal row
        else if (cfg.alphabeta)
            val = -negamax(depth - 1, -beta, -alpha, ply + 1, nd_opp, nd_me);
        else
            val = -negamax(depth - 1, -INF, INF, ply + 1, nd_opp, nd_me);
        board.unmake(m, prev);
        if (stop_) return true;
        if (val > best) {
            best = val;
            best_move = m;
            if (root) root_best_ = m;
        }
        if (val > alpha) alpha = val;
        if (cfg.alphabeta && alpha >= beta) {
            if (cfg.killers && killer_[ply][0] != m) {
                killer_[ply][1] = killer_[ply][0];
                killer_[ply][0] = m;
            }
            if (cfg.history) {
                int32_t& h = history_[me][m];
                h += depth * depth;
                if (h > HISTORY_MAX)
                    for (auto& side : history_)
                        for (auto& x : side) x /= 2;
            }
            return true;
        }
        return false;
    };

    // Distance of every square to my goal: gives the new distance of each pawn
    // move for free (the opponent's distance is unchanged by a pawn move).
    GoalLayers gl_me;
    board.goal_layers(me, gl_me);
    Move pm[8];
    int pd[8];
    const int npawn = board.pawn_moves(me, pm);
    for (int i = 0; i < npawn; ++i) pd[i] = gl_me.dist(pm[i]);

    bool done = false;

    // ---- stage 1: transposition-table move ----
    if (tt_move != MOVE_NONE) {
        bool legal = false;
        int nd_me = d_me, nd_opp = d_opp;
        if (is_pawn(tt_move)) {
            for (int i = 0; i < npawn; ++i)
                if (pm[i] == tt_move) {
                    legal = true;
                    nd_me = pd[i];
                }
        } else if (tt_move < NMOVES && board.walls_left[me] > 0) {
            const bool h = is_hwall(tt_move);
            const int a = h ? tt_move - H_BASE : tt_move - V_BASE;
            if (((h ? board.free_h() : board.free_v()) >> a) & 1) {
                nd_opp = board.distance_with_wall(opp, tt_move);
                nd_me = board.distance_with_wall(me, tt_move);
                legal = nd_opp >= 0 && nd_me >= 0;
            }
        }
        if (legal)
            done = search_move(tt_move, nd_me, nd_opp);
        else
            tt_move = MOVE_NONE;  // hash collision or stale hint: ignore it
    }

    // ---- stage 2: pawn moves along a shortest path (closest to the goal first) ----
    if (!done) {
        int order[8], n = 0;
        for (int i = 0; i < npawn; ++i)
            if (pd[i] < d_me && pm[i] != tt_move) order[n++] = i;
        if (root && cfg.seed)
            for (int i = n - 1; i > 0; --i) std::swap(order[i], order[xorshift(rng_) % (i + 1)]);
        std::stable_sort(order, order + n, [&](int a, int b) { return pd[a] < pd[b]; });
        for (int k = 0; k < n && !done; ++k) done = search_move(pm[order[k]], pd[order[k]], d_opp);
    }

    // ---- stage 3: killers, walls by how much they lengthen the opponent's
    //      path, then everything else by history ----
    if (!done) {
        Cand cand[MAX_MOVES];
        int n = 0;
        for (int i = 0; i < npawn; ++i)
            if (pd[i] >= d_me && pm[i] != tt_move)
                cand[n++] = Cand{(cfg.history ? history_[me][pm[i]] : 0) - (pd[i] - d_me) * SCORE_OWN_STEP, pm[i],
                                 uint8_t(pd[i]), uint8_t(d_opp)};
        if (board.walls_left[me] > 0) {
            int nw = gen_walls(me, d_me, d_opp, gl_me, cfg.wall_filter, depth, cand + n);
            // Never let the filter leave a player without any move.
            if (nw == 0 && npawn == 0 && cfg.wall_filter != FILTER_OFF)
                nw = gen_walls(me, d_me, d_opp, gl_me, FILTER_OFF, depth, cand + n);
            n += nw;
        }
        for (int i = 0; i < n; ++i) {
            if (cfg.killers) {
                if (cand[i].move == killer_[ply][0]) cand[i].score = SCORE_KILLER;
                else if (cand[i].move == killer_[ply][1]) cand[i].score = SCORE_KILLER - 1;
            }
            if (root && cfg.seed) cand[i].score += int32_t(xorshift(rng_) & 7);
        }
        for (int i = 0; i < n && !done; ++i) {
            int bi = i;
            for (int j = i + 1; j < n; ++j)
                if (cand[j].score > cand[bi].score) bi = j;
            std::swap(cand[i], cand[bi]);
            if (cand[i].move == tt_move) continue;
            done = search_move(cand[i].move, cand[i].d_me, cand[i].d_opp);
        }
    }

    if (stop_) return 0;
    if (tried == 0) return evaluate(me, d_me, d_opp);  // no legal move: score it as it stands

    if (cfg.use_tt) {
        const int flag = !cfg.alphabeta ? TT_EXACT : best <= alpha0 ? TT_UPPER : best >= beta ? TT_LOWER : TT_EXACT;
        tt_store(board.hash, depth, value_to_tt(best, ply), flag, best_move);
    }
    return best;
}

// --------------------------------------------------------------------------
// Iterative deepening
// --------------------------------------------------------------------------
SearchResult Search::go(int max_depth, double time_limit) {
    using clock = std::chrono::steady_clock;
    const auto start = clock::now();
    auto elapsed = [&] { return std::chrono::duration<double>(clock::now() - start).count(); };

    SearchResult res;
    max_depth = std::clamp(max_depth, 1, MAX_DEPTH);
    timed_ = time_limit > 0;
    if (timed_) deadline_ = start + std::chrono::duration_cast<clock::duration>(std::chrono::duration<double>(time_limit));
    stop_ = false;
    nodes_ = 0;
    ++age_;
    std::memset(killer_, MOVE_NONE, sizeof(killer_));
    rng_ = cfg.seed ? (cfg.seed * 0x9E3779B97F4A7C15ULL) ^ board.hash ^ 0x2545F4914F6CDD1DULL : 0;
    root_hint_ = MOVE_NONE;

    const int d_me = board.distance(board.turn), d_opp = board.distance(board.turn ^ 1);
    for (int depth = 1; depth <= max_depth; ++depth) {
        root_best_ = MOVE_NONE;
        const int score = negamax(depth, -INF, INF, 0, d_me, d_opp);
        if (stop_) break;  // unfinished iteration: keep the previous depth's move
        res.best = root_best_;
        res.score = score;
        res.depth = depth;
        root_hint_ = root_best_;
        if (cfg.verbose) {
            std::printf("info depth %d score %d nodes %llu time %.3f\n", depth, score,
                        static_cast<unsigned long long>(nodes_), elapsed());
            std::fflush(stdout);
        }
        if (res.best == MOVE_NONE) break;  // no legal move at all
        // A win/loss that lies inside the searched horizon is final.
        if (std::abs(score) >= WIN_THRESHOLD && WIN - std::abs(score) <= depth) break;
        // The next iteration would not finish in the remaining time anyway.
        if (timed_ && elapsed() >= 0.5 * time_limit) break;
    }
    if (res.best == MOVE_NONE) {  // out of time before depth 1 finished
        MoveList ml;
        ordered_moves(ml);
        if (ml.n > 0) res.best = ml.m[0];
    }
    res.nodes = nodes_;
    res.seconds = elapsed();
    return res;
}

void Search::ordered_moves(MoveList& out) {
    out.n = 0;
    const int me = board.turn;
    const int d_me = board.distance(me), d_opp = board.distance(me ^ 1);
    GoalLayers gl_me;
    board.goal_layers(me, gl_me);
    Move pm[8];
    const int npawn = board.pawn_moves(me, pm);
    Cand cand[MAX_MOVES];
    int n = 0;
    for (int i = 0; i < npawn; ++i) {
        const int d = gl_me.dist(pm[i]);
        cand[n++] = Cand{d < d_me ? SCORE_ROOT_PAWN - d : 0, pm[i], uint8_t(d), uint8_t(d_opp)};
    }
    if (board.walls_left[me] > 0) {
        int nw = gen_walls(me, d_me, d_opp, gl_me, cfg.wall_filter, MAX_DEPTH, cand + n);
        if (nw == 0 && npawn == 0 && cfg.wall_filter != FILTER_OFF)
            nw = gen_walls(me, d_me, d_opp, gl_me, FILTER_OFF, MAX_DEPTH, cand + n);
        n += nw;
    }
    std::stable_sort(cand, cand + n, [](const Cand& a, const Cand& b) { return a.score > b.score; });
    for (int i = 0; i < n; ++i) out.add(cand[i].move);
}

}  // namespace quoridor
