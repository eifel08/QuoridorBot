"""C++ engine: rules parity with the Python engine, search correctness, GUI glue.

Skipped when the engine has not been built (``python build_engine.py``).
The heavy versions of the parity / TT checks are ``python -m tools.parity`` and
``python -m tools.verify_search``.
"""

import os
import random
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from quoridor import Board, BoardState, Game, Orientation, PawnMove, WallMove
from quoridor.bots import AlphaBetaBot, BotSpec, CppBot, CppEngine, EngineError, engine_available
from quoridor.bots import make_player, make_search_bot, parse_bot_spec
from quoridor.bots.cpp_engine import WIN, format_move, parse_move

from tests.test_web_engine import random_positions
from tools import parity, verify_search
from tools.positions import BENCH_POSITIONS, random_position

try:
    import pygame
except ImportError:  # pragma: no cover
    pygame = None

H, V = Orientation.HORIZONTAL, Orientation.VERTICAL
needs_engine = unittest.skipUnless(engine_available(), "C++ engine not built")


def all_moves(board: Board, player: int) -> set:
    return parity.python_moves(board, player)


class EngineCase(unittest.TestCase):
    engine: CppEngine

    @classmethod
    def setUpClass(cls):
        cls.engine = CppEngine()

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()

    def setUp(self):
        for name, value in {"futility": 1, "wall_filter": 1, "decided": 1, "tt": 1, "tt_exact_depth": 0,
                            "alphabeta": 1, "seed": 0}.items():
            self.engine.set_option(name, value)


@needs_engine
class TestProtocol(EngineCase):
    def test_move_text_round_trip(self):
        for move in (PawnMove(7, 4), WallMove(3, 4, H), WallMove(0, 7, V)):
            self.assertEqual(parse_move(format_move(move)), move)
        self.assertIsNone(parse_move("none"))

    def test_errors_are_reported(self):
        for bad in ("frobnicate", "position 0 8 4 8 4 10 10", "position 0 8 4 0 4 10 10 h88",
                    "set nonsense 1", "play h99"):
            with self.assertRaises(EngineError, msg=bad):
                self.engine.command(bad)
        self.engine.set_position(Board().snapshot(0))  # still usable afterwards
        self.assertEqual(len(self.engine.legal_moves()), 3 + 128)

    def test_illegal_play_rejected(self):
        self.engine.set_position(Board().snapshot(0))
        with self.assertRaises(EngineError):
            self.engine.play(PawnMove(6, 4))  # two squares away

    def test_internal_selftest(self):
        self.assertTrue(self.engine.command("selftest 40 3").startswith("ok"))


@needs_engine
class TestRulesParity(EngineCase):
    """The C++ rules must be exactly the Python rules (wall filtering off)."""

    def test_random_games(self):
        stats = parity.check_games(self.engine, games=8, seed=2024)
        self.assertGreater(stats["positions"], 300)

    def test_snapshots_match(self):
        for d in random_positions(4, seed=7):
            st = BoardState.from_dict(d)
            self.engine.set_position(st)
            want = {PawnMove(*m) for m in st.valid_pawn_moves} | {WallMove(*w) for w in st.valid_walls}
            self.assertEqual(set(self.engine.legal_moves()), want)
            self.assertEqual(self.engine.distances(), st.shortest_paths)

    def test_jump_rules(self):
        cases = [
            ([(5, 4), (4, 4)], []),                      # face to face: straight jump
            ([(5, 4), (4, 4)], [(3, 4, H)]),             # wall behind: both diagonals
            ([(5, 4), (4, 4)], [(3, 4, H), (4, 4, V)]),  # ... one diagonal walled off
            ([(1, 4), (0, 4)], []),                      # board edge behind: diagonals
            ([(4, 0), (4, 1)], [(4, 1, V)]),             # sideways, wall behind, at the edge
            ([(5, 4), (4, 4)], [(4, 4, H)]),             # wall between the pawns: no jump
            ([(8, 0), (7, 0)], [(6, 0, H)]),             # corner
            ([(4, 4), (4, 5)], [(3, 5, V), (3, 4, H)]),
        ]
        for pawns, walls in cases:
            for player in (0, 1):
                board = Board()
                board.pawns = list(pawns)
                board.walls = set(walls)
                self.engine.set_position(parity.board_command(board, player))
                self.assertEqual(set(self.engine.legal_moves()), all_moves(board, player), (pawns, walls, player))

    def test_no_walls_left(self):
        board = Board()
        board.walls_left = [0, 3]
        self.engine.set_position(parity.board_command(board, 0))
        self.assertEqual(set(self.engine.legal_moves()), all_moves(board, 0))
        self.assertEqual(len(self.engine.legal_moves()), 3)


@needs_engine
class TestWallFilter(EngineCase):
    def test_filtered_moves_are_legal_and_keep_pawn_moves(self):
        rng = random.Random(5)
        for _ in range(40):
            self.engine.set_option("wall_filter", 1)
            pos = random_position(self.engine, rng, rng.randint(2, 50))
            self.engine.set_position(pos)
            legal = set(self.engine.legal_moves())
            filtered = self.engine.search_order()
            self.assertEqual(len(filtered), len(set(filtered)))
            self.assertLessEqual(set(filtered), legal, pos)
            pawn = {m for m in legal if isinstance(m, PawnMove)}
            self.assertLessEqual(pawn, set(filtered), pos)
            self.engine.set_option("wall_filter", 2)  # attack walls only: a subset
            self.assertLessEqual(set(self.engine.search_order()), set(filtered), pos)
            self.engine.set_option("wall_filter", 0)  # off: every legal move
            self.engine.set_position(pos)
            self.assertEqual(set(self.engine.search_order()), legal, pos)

    def test_opening_filter(self):
        self.engine.set_position(Board().snapshot(0))
        moves = self.engine.search_order()
        self.assertEqual(moves[0], PawnMove(7, 4))  # the shortest-path pawn move comes first
        walls = {m for m in moves if isinstance(m, WallMove)}
        # Attack: horizontal walls across the opponent's column (every row).
        self.assertTrue({WallMove(r, c, H) for r in range(8) for c in (3, 4)} <= walls)
        # Defense: vertical walls crossing those slots, to keep my own column open.
        self.assertTrue({WallMove(r, c, V) for r in range(8) for c in (3, 4)} <= walls)
        self.assertNotIn(WallMove(0, 0, H), walls)
        self.assertLess(len(walls), 64)


@needs_engine
class TestSearch(EngineCase):
    def test_tt_and_alphabeta_do_not_change_the_value(self):
        stats = verify_search.run(self.engine, n_positions=10, seed=3, tt_depth=4, mm_depth=3,
                                  log=lambda *_: None)
        self.assertEqual(stats["ab_mismatch"], 0)
        self.assertEqual(stats["tt_exact_mismatch"], 0)
        self.assertLess(stats["nodes_ab"], stats["nodes_minimax"])

    def test_takes_immediate_win(self):
        board = Board()
        board.pawns = [(1, 2), (5, 5)]
        self.engine.set_position(board.snapshot(0))
        move, info = self.engine.go(depth=3)
        self.assertEqual(move, PawnMove(0, 2))
        self.assertEqual((info["forced"], info["value"]), ("win", WIN - 1))

    def test_prefers_the_faster_win(self):
        # Player 1 is 2 steps away and the opponent has no walls: win in 3 plies.
        board = Board()
        board.pawns = [(2, 2), (4, 6)]
        board.walls_left = [0, 0]
        self.engine.set_option("decided", 0)
        self.engine.set_position(board.snapshot(0))
        move, info = self.engine.go(depth=6)
        self.assertEqual(move, PawnMove(1, 2))
        self.assertEqual(info["value"], WIN - 3)
        # ... and the loser's best defence loses as late as possible.
        self.engine.set_position(board.snapshot(1))
        _, info = self.engine.go(depth=6)
        self.assertEqual(info["value"], -(WIN - 4))

    def test_blocks_opponent_about_to_win(self):
        board = Board()
        board.pawns = [(8, 0), (7, 4)]  # player 2 wins next move at (8, 4)
        self.engine.set_position(board.snapshot(0))
        move, _ = self.engine.go(depth=2)
        self.assertIsInstance(move, WallMove)
        board.apply_move(0, move)
        self.assertGreater(board.shortest_path_length(1), 1)

    def test_decided_race(self):
        board = Board()
        board.pawns = [(3, 0), (2, 8)]  # 3 vs 6 steps, pawns far apart, nobody has walls
        board.walls_left = [0, 0]
        self.engine.set_position(board.snapshot(1))
        _, info = self.engine.go(depth=1)
        self.assertEqual(info["forced"], "loss")
        self.engine.set_position(board.snapshot(0))
        _, info = self.engine.go(depth=1)
        self.assertEqual(info["forced"], "win")
        self.engine.set_option("decided", 0)
        _, info = self.engine.go(depth=1)
        self.assertIsNone(info["forced"])

    def test_moves_always_legal(self):
        rng = random.Random(11)
        configs = [{}, {"wall_filter": 0}, {"futility": 0, "decided": 0}, {"tt": 0}, {"seed": 7}]
        for i in range(60):
            for name, value in {"wall_filter": 1, "futility": 1, "decided": 1, "tt": 1, "seed": 0,
                                **configs[i % len(configs)]}.items():
                self.engine.set_option(name, value)
            pos = random_position(self.engine, rng, rng.randint(0, 80), wall_prob=0.5)
            self.engine.set_position(pos)
            legal = set(self.engine.legal_moves())
            move, info = self.engine.go(depth=1 + i % 4)
            self.assertIn(move, legal, pos)
            self.assertGreaterEqual(info["depth_reached"], 1)

    def test_time_limit_respected(self):
        self.engine.set_position(BENCH_POSITIONS["middlegame"])
        move, info = self.engine.go(depth=64, time_limit=0.3)
        self.assertIsNotNone(move)
        self.assertLess(info["seconds"], 0.6)
        self.assertLess(info["depth_reached"], 64)

    def test_fixed_depth_is_deterministic(self):
        self.engine.set_position(BENCH_POSITIONS["middlegame"])
        self.engine.new_game()
        first = self.engine.go(depth=5)
        self.engine.new_game()
        second = self.engine.go(depth=5)
        self.assertEqual(first[0], second[0])
        self.assertEqual((first[1]["value"], first[1]["nodes"]), (second[1]["value"], second[1]["nodes"]))
        self.assertEqual(first[1]["depth_reached"], 5)

    def test_bench_positions_are_valid(self):
        for name, pos in BENCH_POSITIONS.items():
            state = __import__("tools.bench", fromlist=["to_state"]).to_state(pos)
            board = state.to_board()
            self.assertIsNotNone(board.shortest_path_length(0), name)
            self.assertIsNotNone(board.shortest_path_length(1), name)
            for wall in board.walls:  # no wall overlaps or crosses another one
                others = [w for w in Board.conflicting_walls(wall) if w != wall]
                self.assertFalse(any(w in board.walls for w in others), (name, wall))
            self.assertEqual(len(board.walls), 20 - sum(board.walls_left), name)


@needs_engine
class TestCppBot(unittest.TestCase):
    def test_is_a_drop_in_alphabeta_bot(self):
        bot = make_search_bot("x", 3, 1.0)
        self.assertIsInstance(bot, CppBot)
        self.assertIsInstance(bot, AlphaBetaBot)
        self.assertEqual((bot.name, bot.depth, bot.time_limit), ("x", 3, 1.0))
        move = bot.get_move(Board().snapshot(0))
        self.assertIsNone(Board().move_error(0, move))
        for key in ("depth_reached", "depth_requested", "nodes", "seconds", "score", "forced"):
            self.assertIn(key, bot.last_info)  # what the GUI / CLI status line reads
        self.assertEqual(bot.last_info["depth_requested"], 3)
        bot.close()

    def test_python_fallback(self):
        bot = make_search_bot("x", 2, None, engine="py")
        self.assertNotIsInstance(bot, CppBot)
        self.assertIsInstance(bot, AlphaBetaBot)

    def test_bot_vs_bot_game_finishes(self):
        game = Game(CppBot("A", 3, 1), CppBot("B", 2, 1), max_invalid_attempts=1)
        self.assertIn(game.run(), (0, 1))
        for p in game.players:
            p.close()

    def test_plays_against_the_python_bot(self):
        # Not a strength test (see tools/match.py): both engines in one real game.
        game = Game(CppBot("cpp", 4, None), AlphaBetaBot("py", 2, None), max_invalid_attempts=1)
        self.assertIn(game.run(), (0, 1))
        game.players[0].close()

    def test_bot_specs(self):
        spec = parse_bot_spec("ab:depth=3,time=0.5,futility=0,filter=0,engine=cpp")
        self.assertEqual(spec, BotSpec(depth=3, time_limit=0.5, engine="cpp",
                                       options=(("wall_filter", 0), ("futility", 0))))
        self.assertEqual(parse_bot_spec(spec.to_string()), spec)
        bot = make_player(spec, "x")
        self.assertIsInstance(bot, CppBot)
        self.assertEqual(bot.options, {"wall_filter": 0, "futility": 0})
        self.assertNotIsInstance(make_player(parse_bot_spec("ab:depth=2,engine=py")), CppBot)
        for text, value in (("ab:walls=attack", 2), ("ab:walls=both", 1), ("ab:walls=all", 0), ("ab", 1)):
            self.assertEqual(parse_bot_spec(text).wall_filter, value, text)
        self.assertEqual(make_player(parse_bot_spec("ab:walls=all"), "x").options, {"wall_filter": 0})
        for bad in ("ab:engine=rust", "ab:futility=x", "ab:engine=py,futility=0", "ab:walls=some"):
            with self.assertRaises(ValueError, msg=bad):
                parse_bot_spec(bad)


@needs_engine
@unittest.skipIf(pygame is None, "pygame not installed")
class TestGuiEndToEnd(unittest.TestCase):
    """The existing animation, driven by the C++ engine, plays a whole game."""

    def test_bot_vs_bot_in_the_window(self):
        from quoridor.gui import MatchSettings, QuoridorGUI, build_players
        from quoridor.gui.player import GuiBotPlayer

        gui = QuoridorGUI()
        gui.on_game_over = lambda game, state: setattr(gui, "state", state)  # don't wait for a click
        settings = MatchSettings("bvb", bots=(BotSpec(depth=3), BotSpec(depth=4, time_limit=1.0)), move_delay=0.0)
        players = build_players(settings, gui, ("A", "B"))
        for p in players:
            self.assertIsInstance(p, GuiBotPlayer)
            self.assertIsInstance(p.bot, CppBot)
        game = Game(*players, observers=[gui], max_invalid_attempts=1)
        winner = game.run()
        self.assertIn(winner, (0, 1))
        self.assertEqual(gui.state.winner, winner)
        self.assertEqual(gui.bot_info["engine"], "cpp")  # shown in the side panel
        self.assertGreater(len(game.history), 10)

    def test_human_vs_bot_uses_the_engine(self):
        from quoridor.gui import MatchSettings, QuoridorGUI, build_players

        gui = QuoridorGUI()
        human, bot = build_players(MatchSettings("hvb", depth=6, time_limit=0.5, wall_filter=2), gui, ("A", "B"))
        self.assertIsInstance(bot.bot, CppBot)
        self.assertEqual(bot.bot.options, {"wall_filter": 2})  # the menu's wall choice reaches the engine
        state = Board().snapshot(1)
        move = bot.get_move(state)  # runs the engine in the GUI's worker thread
        self.assertIsNone(Board().move_error(1, move))
        self.assertGreaterEqual(gui.bot_info["depth_reached"], 1)


if __name__ == "__main__":
    unittest.main()
