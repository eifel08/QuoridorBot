// Quoridor engine: line-based text protocol on stdin/stdout.
//
// Every command is answered with exactly one final line ("ok", "error ...",
// or a data line). "go" may print "info ..." lines before its final
// "bestmove ..." line when "set verbose 1" is active.
//
//   position <turn> <r0> <c0> <r1> <c1> <walls0> <walls1> [h<r><c> | v<r><c>]...
//   go [depth N] [time SECONDS]   -> bestmove <move> depth D score S nodes N time T
//   moves                         -> moves <every legal move>          (no filtering)
//   genmoves                      -> moves <moves in search order>     (filter applied)
//   play <move>                   -> ok        (applies a legal move to the position)
//   dist                          -> dist <d0> <d1>
//   eval                          -> eval <static evaluation for the side to move>
//   hash                          -> hash <zobrist key>
//   set <option> <value>          -> ok        (see README.md for the options)
//   newgame                       -> ok        (clears the transposition table)
//   selftest [games] [seed]       -> ok ...    (internal consistency checks)
//   quit
//
// Moves are written m<r><c> (pawn to row r, column c), h<r><c>, v<r><c>.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "board.h"
#include "search.h"

using namespace quoridor;

namespace {

std::string move_str(Move m) {
    if (m == MOVE_NONE) return "none";
    char buf[16];
    if (is_pawn(m))
        std::snprintf(buf, sizeof buf, "m%d%d", m / N, m % N);
    else if (is_hwall(m))
        std::snprintf(buf, sizeof buf, "h%d%d", (m - H_BASE) / NW, (m - H_BASE) % NW);
    else
        std::snprintf(buf, sizeof buf, "v%d%d", (m - V_BASE) / NW, (m - V_BASE) % NW);
    return buf;
}

Move parse_move(const std::string& s) {
    if (s.size() != 3 || s[1] < '0' || s[2] < '0') return MOVE_NONE;
    const int r = s[1] - '0', c = s[2] - '0';
    if (s[0] == 'm' && r < N && c < N) return pawn_move(r * N + c);
    if (r >= NW || c >= NW) return MOVE_NONE;
    if (s[0] == 'h') return hwall_move(r * NW + c);
    if (s[0] == 'v') return vwall_move(r * NW + c);
    return MOVE_NONE;
}

std::string list_str(const MoveList& ml) {
    std::string out = "moves";
    for (int i = 0; i < ml.n; ++i) out += " " + move_str(ml.m[i]);
    return out;
}

bool set_position(Board& b, std::istringstream& in, std::string& err) {
    int turn, r0, c0, r1, c1, w0, w1;
    if (!(in >> turn >> r0 >> c0 >> r1 >> c1 >> w0 >> w1)) {
        err = "position needs: turn r0 c0 r1 c1 walls0 walls1";
        return false;
    }
    auto bad = [](int v, int hi) { return v < 0 || v > hi; };
    if (bad(turn, 1) || bad(r0, N - 1) || bad(c0, N - 1) || bad(r1, N - 1) || bad(c1, N - 1) ||
        bad(w0, MAX_WALLS) || bad(w1, MAX_WALLS) || (r0 == r1 && c0 == c1)) {
        err = "position out of range";
        return false;
    }
    uint64_t hw = 0, vw = 0;
    std::string tok;
    while (in >> tok) {
        const Move m = parse_move(tok);
        if (m == MOVE_NONE || is_pawn(m)) {
            err = "bad wall '" + tok + "'";
            return false;
        }
        if (is_hwall(m)) hw |= 1ULL << (m - H_BASE);
        else vw |= 1ULL << (m - V_BASE);
    }
    b.set(turn, r0 * N + c0, r1 * N + c1, w0, w1, hw, vw);
    return true;
}

bool set_option(Search& s, const std::string& name, const std::string& value, std::string& err) {
    Config& c = s.cfg;
    const long v = std::strtol(value.c_str(), nullptr, 10);
    if (name == "dist_weight") c.dist_weight = int(v);
    else if (name == "wall_weight") c.wall_weight = int(v);
    else if (name == "tempo_bonus") c.tempo_bonus = int(v);
    else if (name == "wall_cap") c.wall_cap = v != 0;
    else if (name == "wall_cap_slack") c.wall_cap_slack = int(v);
    else if (name == "tt") c.use_tt = v != 0;
    else if (name == "tt_exact_depth") c.tt_exact_depth = v != 0;
    else if (name == "alphabeta") c.alphabeta = v != 0;
    else if (name == "killers") c.killers = v != 0;
    else if (name == "history") c.history = v != 0;
    else if (name == "wall_filter") c.wall_filter = int(std::clamp(v, 0L, 2L));
    else if (name == "decided") c.decided = v != 0;
    else if (name == "decided_margin") c.decided_margin = int(v);
    else if (name == "futility") c.futility = v != 0;
    else if (name == "wall_recovery") c.wall_recovery = int(v);
    else if (name == "depth_margin") c.depth_margin = int(v);
    else if (name == "seed") { c.seed = std::strtoull(value.c_str(), nullptr, 10); return true; }
    else if (name == "verbose") { c.verbose = v != 0; return true; }
    else if (name == "hash_mb") { s.resize_tt(int(v)); }
    else {
        err = "unknown option '" + name + "'";
        return false;
    }
    s.new_game();  // stored scores depend on the options
    return true;
}

// Internal consistency checks on random games: incremental hash, make/unmake
// restoring the position, fast legal move generation == slow reference.
std::string selftest(int games, uint64_t seed) {
    uint64_t s = seed * 0x9E3779B97F4A7C15ULL + 1;
    auto rnd = [&](int n) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return int(s % uint64_t(n));
    };
    long positions = 0;
    Board b;
    for (int g = 0; g < games; ++g) {
        b.reset();
        for (int ply = 0; ply < 300; ++ply) {
            MoveList fast, slow;
            b.legal_moves(fast);
            b.legal_moves_slow(slow);
            std::sort(fast.m, fast.m + fast.n);
            std::sort(slow.m, slow.m + slow.n);
            if (fast.n != slow.n || !std::equal(fast.m, fast.m + fast.n, slow.m)) return "error movegen mismatch";
            if (b.hash != b.compute_hash()) return "error incremental hash mismatch";
            for (int i = 0; i < fast.n; ++i) {
                if (!b.is_legal(fast.m[i])) return "error is_legal rejects a legal move";
                const Board before = b;
                const int prev = b.make(fast.m[i]);
                if (b.hash != b.compute_hash()) return "error hash after make";
                b.unmake(fast.m[i], prev);
                if (b.hash != before.hash || b.hw != before.hw || b.vw != before.vw ||
                    b.pawn[0] != before.pawn[0] || b.pawn[1] != before.pawn[1] ||
                    b.walls_left[0] != before.walls_left[0] || b.walls_left[1] != before.walls_left[1] ||
                    b.turn != before.turn || !(b.open[0] == before.open[0]) || !(b.open[1] == before.open[1]) ||
                    !(b.open[2] == before.open[2]) || !(b.open[3] == before.open[3]))
                    return "error unmake did not restore the position";
            }
            ++positions;
            if (fast.n == 0) break;
            // Walls half of the time so that mazes build up.
            int np = 0;
            while (np < fast.n && is_pawn(fast.m[np])) ++np;
            Move m;
            if (np < fast.n && (np == 0 || rnd(2))) m = fast.m[np + rnd(fast.n - np)];
            else m = fast.m[rnd(np)];
            const int mover = b.turn;
            b.make(m);
            if (b.won(mover)) break;
        }
    }
    return "ok " + std::to_string(positions) + " positions";
}

}  // namespace

int main() {
    std::ios::sync_with_stdio(false);
    Board::init_tables();
    Search search;
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream in(line);
        std::string cmd, err;
        if (!(in >> cmd)) continue;
        std::string out;
        if (cmd == "quit") break;
        if (cmd == "position") {
            out = set_position(search.board, in, err) ? "ok" : "error " + err;
        } else if (cmd == "go") {
            int depth = MAX_DEPTH;
            double time_limit = 0;
            std::string key;
            while (in >> key) {
                if (key == "depth") in >> depth;
                else if (key == "time") in >> time_limit;
            }
            const SearchResult r = search.go(depth, time_limit);
            char buf[160];
            std::snprintf(buf, sizeof buf, "bestmove %s depth %d score %d nodes %llu time %.4f",
                          move_str(r.best).c_str(), r.depth, r.score,
                          static_cast<unsigned long long>(r.nodes), r.seconds);
            out = buf;
        } else if (cmd == "moves") {
            MoveList ml;
            search.board.legal_moves(ml);
            out = list_str(ml);
        } else if (cmd == "genmoves") {
            MoveList ml;
            search.ordered_moves(ml);
            out = list_str(ml);
        } else if (cmd == "play") {
            std::string tok;
            in >> tok;
            const Move m = parse_move(tok);
            if (m != MOVE_NONE && search.board.is_legal(m)) {
                search.board.make(m);
                out = "ok";
            } else {
                out = "error illegal move '" + tok + "'";
            }
        } else if (cmd == "dist") {
            out = "dist " + std::to_string(search.board.distance(0)) + " " + std::to_string(search.board.distance(1));
        } else if (cmd == "eval") {
            const int me = search.board.turn;
            out = "eval " + std::to_string(search.evaluate(me, search.board.distance(me), search.board.distance(me ^ 1)));
        } else if (cmd == "hash") {
            out = "hash " + std::to_string(search.board.hash);
        } else if (cmd == "set") {
            std::string name, value;
            if (in >> name >> value) out = set_option(search, name, value, err) ? "ok" : "error " + err;
            else out = "error set needs: name value";
        } else if (cmd == "newgame") {
            search.new_game();
            out = "ok";
        } else if (cmd == "selftest") {
            int games = 200;
            uint64_t seed = 1;
            in >> games >> seed;
            out = selftest(games, seed);
        } else if (cmd == "isready") {
            out = "ok";
        } else {
            out = "error unknown command '" + cmd + "'";
        }
        std::cout << out << '\n' << std::flush;
    }
    return 0;
}
