"""Eval display: search scores, M/-M notation, background evaluator, GUI bars."""

import os
import threading
import time
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from quoridor import Board, Game, Orientation, PawnMove, WallMove
from quoridor.bots import AlphaBetaBot, BotSpec, CppBot, engine_available
from quoridor.evaluation import EVAL_TIME_LIMIT, BackgroundEvaluator, Eval, mate_from_plies

try:
    import pygame
except ImportError:  # pragma: no cover
    pygame = None

H = Orientation.HORIZONTAL
needs_engine = unittest.skipUnless(engine_available(), "C++ engine not built")


class TestNotation(unittest.TestCase):
    def test_text(self):
        self.assertEqual(Eval(0, steps=1.46).text(), "+1.5")
        self.assertEqual(Eval(0, steps=-0.8).text(), "-0.8")
        self.assertEqual(Eval(0, steps=0.04).text(), "0.0")
        self.assertEqual(Eval(0, mate=3).text(), "M3")
        self.assertEqual(Eval(0, mate=-5).text(), "-M5")
        self.assertEqual(Eval.game_won(1).text(), "Won")

    def test_perspective(self):
        ev = Eval(0, steps=2.0)
        self.assertEqual(ev.for_player(1).text(), "-2.0")
        self.assertEqual(ev.for_player(1).for_player(0), ev)
        self.assertEqual(Eval(1, mate=2).for_player(0).text(), "-M2")
        self.assertEqual(Eval(1, mate=2).winner, 1)
        self.assertEqual(Eval(1, mate=-2).winner, 0)
        self.assertEqual(Eval(1, steps=-1.0).leader(), 0)
        self.assertIsNone(Eval(1, steps=0.0).leader())

    def test_bar_share(self):
        self.assertEqual(Eval(0, steps=0.0).red_share(), 0.5)
        self.assertGreater(Eval(0, steps=2.0).red_share(), 0.5)
        self.assertLess(Eval(1, steps=2.0).red_share(), 0.5)  # good for Blue
        self.assertEqual(Eval(0, mate=4).red_share(), 1.0)
        self.assertEqual(Eval(0, mate=-4).red_share(), 0.0)
        self.assertGreater(Eval(0, steps=20).red_share(), Eval(0, steps=5).red_share())

    def test_mate_from_plies(self):
        # Ply 1 = the side to move's own move. A win on ply 1 is M1, on ply 3 is M2;
        # a loss on ply 2 (opponent's first move) is -M1, on ply 4 is -M2.
        self.assertEqual([mate_from_plies(True, p) for p in (1, 3, 5)], [1, 2, 3])
        self.assertEqual([mate_from_plies(False, p) for p in (2, 4, 6)], [-1, -2, -3])

    def test_from_info(self):
        self.assertIsNone(Eval.from_info(None, 0))
        self.assertIsNone(Eval.from_info({"nodes": 5}, 0))  # e.g. an ML agent: no score
        ev = Eval.from_info({"eval_steps": 1.5, "depth_reached": 7, "nodes": 99, "seconds": 0.2}, 1, "x")
        self.assertEqual((ev.player, ev.text(), ev.depth, ev.source), (1, "+1.5", 7, "x"))
        self.assertEqual(Eval.from_info({"forced": "loss", "plies": 6}, 0).text(), "-M3")


def race_board():
    """Red is 2 steps from its goal, Blue 4; nobody has walls: Red wins in 2 of its moves."""
    board = Board()
    board.pawns = [(2, 2), (4, 6)]
    board.walls_left = [0, 0]
    return board


def delay_board():
    """Blue is 1 step from winning, Red 8 steps away with 1 wall: Red is lost and can only delay.

    The best defence is a horizontal wall under Blue (it then needs 2 steps: -M2); every
    other move loses at once (-M1).
    """
    board = Board()
    board.pawns = [(8, 0), (7, 4)]
    board.walls_left = [1, 0]
    return board


class SearchScoreMixin:
    def bot(self, depth, **options):
        raise NotImplementedError

    def test_winner_takes_the_fastest_win(self):
        bot = self.bot(8)
        move = bot.get_move(race_board().snapshot(0))
        self.assertEqual(move, PawnMove(1, 2))
        self.assertEqual(Eval.from_info(bot.last_info, 0).text(), "M2")

    def test_loser_sees_the_loss(self):
        bot = self.bot(8)
        bot.get_move(race_board().snapshot(1))
        ev = Eval.from_info(bot.last_info, 1)
        self.assertEqual(ev.text(), "-M2")
        self.assertEqual(ev.for_player(0).text(), "M2")

    def test_loser_delays_as_long_as_possible(self):
        board = delay_board()
        bot = self.bot(6)
        move = bot.get_move(board.snapshot(0))
        self.assertIsInstance(move, WallMove)
        board.apply_move(0, move)
        self.assertEqual(board.shortest_path_length(1), 2)  # the longest delay available
        self.assertEqual(Eval.from_info(bot.last_info, 0).text(), "-M2")

    def test_score_is_backed_up_not_static(self):
        # In the delay position the static evaluation is a plain path difference, but the
        # search proves the loss.
        bot = self.bot(4)
        bot.get_move(delay_board().snapshot(0))
        self.assertEqual(bot.last_info["forced"], "loss")


@needs_engine
class TestCppSearchScore(SearchScoreMixin, unittest.TestCase):
    def bot(self, depth, **options):
        return CppBot("c", depth, None, options)

    def test_every_configuration_delays(self):
        for options in ({"wall_filter": 0}, {"wall_filter": 1}, {"wall_filter": 2}, {"decided": 0},
                        {"futility": 0}, {"tt": 0}):
            board = delay_board()
            bot = CppBot("c", 6, None, options)
            move = bot.get_move(board.snapshot(0))
            board.apply_move(0, move)
            self.assertEqual(board.shortest_path_length(1), 2, options)
            self.assertEqual(Eval.from_info(bot.last_info, 0).text(), "-M2", options)
            bot.close()

    def test_steps_scale(self):
        bot = CppBot("c", 4, None)
        bot.get_move(Board().snapshot(0))
        info = bot.last_info
        self.assertAlmostEqual(info["eval_steps"], info["value"] / 10)
        bot.close()


class TestPythonSearchScore(SearchScoreMixin, unittest.TestCase):
    def bot(self, depth, **options):
        return AlphaBetaBot("p", depth, None)


class TestBackgroundEvaluator(unittest.TestCase):
    def run_evaluator(self, engine):
        results = []
        done = threading.Event()

        def on_result(ev, move_number):
            results.append((ev, move_number))
            done.set()

        ev = BackgroundEvaluator(engine, on_result)
        try:
            board = Board()
            board.pawns = [(4, 4), (2, 6)]
            t0 = time.perf_counter()
            ev.submit(board.snapshot(0, move_number=7))
            self.assertTrue(done.wait(5))
            elapsed = time.perf_counter() - t0
        finally:
            ev.close()
        (result, move_number), = results
        self.assertEqual(move_number, 7)
        self.assertEqual(result.player, 0)
        self.assertLessEqual(result.seconds, EVAL_TIME_LIMIT + 0.05)  # strict 1 s search
        self.assertLess(elapsed, 2.5)
        self.assertGreaterEqual(result.depth, 1)
        return result, ev

    @needs_engine
    def test_cpp(self):
        result, ev = self.run_evaluator("attack")
        self.assertEqual(ev.label, "C++ attack only")
        self.assertEqual(result.source, "C++ attack only")

    def test_python(self):
        result, ev = self.run_evaluator("py")
        self.assertEqual(ev.label, "Python bot")

    def test_ignores_finished_games_and_closes(self):
        ev = BackgroundEvaluator("py", lambda *a: self.fail("no result expected"))
        board = Board()
        board.pawns = [(0, 4), (5, 5)]
        ev.submit(board.snapshot(0, winner=0))
        time.sleep(0.2)
        ev.close()
        ev._thread.join(3)
        self.assertFalse(ev._thread.is_alive())


@unittest.skipIf(pygame is None, "pygame not installed")
class TestGuiEvalBars(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from quoridor.gui import QuoridorGUI
        cls.gui = QuoridorGUI()

    def tearDown(self):
        self.gui.set_eval_display((False, False))

    def test_slots_per_mode(self):
        from quoridor.gui import MatchSettings, build_players
        gui = self.gui
        build_players(MatchSettings("hvb", human_side=0), gui, ("A", "B"))
        self.assertEqual(gui.eval_slots, [False, True])  # the bot is Blue: right of the board
        build_players(MatchSettings("hvb", human_side=1, show_eval=False), gui, ("A", "B"))
        self.assertEqual(gui.eval_slots, [False, False])
        build_players(MatchSettings("bvb", bot_evals=(True, True)), gui, ("A", "B"))
        self.assertEqual(gui.eval_slots, [True, True])  # both bots: opposite sides
        build_players(MatchSettings("bvb", bot_evals=(False, True)), gui, ("A", "B"))
        self.assertEqual(gui.eval_slots, [False, True])
        build_players(MatchSettings("hvh"), gui, ("A", "B"))
        self.assertEqual(gui.eval_slots, [False, False])
        self.assertIsNone(gui.evaluator)
        build_players(MatchSettings("hvh", hvh_eval=True, eval_engine="py"), gui, ("A", "B"))
        self.assertEqual(gui.eval_slots, [True, False])
        self.assertEqual(gui.evaluator.engine, "py")
        first = gui.evaluator
        build_players(MatchSettings("hvh"), gui, ("A", "B"))
        self.assertTrue(first._closed)  # a new game stops the old evaluator

    def test_bot_vs_bot_fills_both_bars(self):
        from quoridor.gui import MatchSettings, build_players
        gui = self.gui
        spec = BotSpec(depth=3)
        players = build_players(MatchSettings("bvb", bots=(spec, spec), move_delay=0.0), gui, ("A", "B"))
        game = Game(*players, observers=[gui])
        gui.on_game_start(game, game.state)
        game.play_turn()
        game.play_turn()
        for p in (0, 1):
            ev, owner = gui.eval_bars[p]
            self.assertEqual((ev.player, owner), (p, p))
        gui.draw()

    def test_human_vs_human_analysis(self):
        from quoridor.gui import MatchSettings, build_players
        gui = self.gui
        p1, p2 = build_players(MatchSettings("hvh", hvh_eval=True, eval_engine="py"), gui, ("A", "B"))
        game = Game(p1, p2, observers=[gui])
        gui.on_game_start(game, game.state)
        gui.on_turn_start(game, game.state)
        deadline = time.time() + 5
        while gui.eval_bars[0] is None and time.time() < deadline:
            time.sleep(0.05)
        ev, owner = gui.eval_bars[0]
        self.assertIsNone(owner)  # neutral analysis: drawn in the colour of the side it favours
        gui.draw()
        # A result for an old position is ignored.
        gui.eval_bars[0] = None
        gui._analysis_result(Eval(0, steps=1.0), move_number=99)
        self.assertIsNone(gui.eval_bars[0])

    def test_draws_every_kind_of_score(self):
        gui = self.gui
        gui.state = Board().snapshot(0)
        gui.eval_slots = [True, True]
        for bars in ([(Eval(0, mate=3), 0), (Eval(1, mate=-3), 1)], [(Eval(0, steps=-4.2), None), None],
                     [(Eval.game_won(1), None), (Eval(1, steps=0.0), 1)], [None, None]):
            gui.eval_bars = list(bars)
            gui.draw()

    def test_menu_toggles(self):
        from quoridor.gui import MatchSettings
        from quoridor.gui.menu import Menu
        menu = Menu(self.gui)
        s = MatchSettings("hvh")
        self.assertFalse(s.hvh_eval)
        s = menu._click(s, menu.hvh_eval_rects[1].center)
        self.assertTrue(s.hvh_eval)
        s = menu._click(s, menu.engine_rects[2].center)
        self.assertEqual(s.eval_engine, "all")
        s = menu._click(s, menu.hvh_eval_rects[0].center)
        self.assertFalse(s.hvh_eval)
        s = menu._click(MatchSettings("hvb"), menu.eval_rect.center)
        self.assertFalse(s.show_eval)
        s = menu._click(MatchSettings("bvb"), menu.bot_eval_rects[1].center)
        self.assertEqual(s.bot_evals, (True, False))
        for mode in ("hvh", "hvb", "bvb"):
            menu.draw(MatchSettings(mode, hvh_eval=True))


if __name__ == "__main__":
    unittest.main()
