"""Screen Assistant: screenshot parser, position tracking, notation and analysis.

The parser is tested on mock screenshots drawn by ``quoridor.vision.synthetic``
(the layout the assistant targets). Skipped without OpenCV / numpy.
"""

import random
import threading
import unittest
from dataclasses import replace

from quoridor import Board, Orientation, PawnMove, WallMove
from quoridor.bots import BotSpec, engine_available

try:
    import cv2  # noqa: F401
    import numpy as np
except ImportError:  # pragma: no cover
    cv2 = None

if cv2 is not None:
    from quoridor.vision.analysis import Analyzer
    from quoridor.vision.capture import ScreenWatcher
    from quoridor.vision.parser import BLUE, RED, Profile, ScreenParser, annotate, find_board
    from quoridor.vision.position import Position, PositionTracker, move_notation, mover_between, parse_notation
    from quoridor.vision.synthetic import DARK, LIGHT, render, render_state

needs_cv = unittest.skipUnless(cv2 is not None, "opencv-python / numpy not installed")
NO_OCR = dict(ocr=False)  # the tests check the tick bars; OCR is optional and slow


def random_position(rng: random.Random, moves: int):
    """A legal position reached by random play: (board, side to move)."""
    board, turn = Board(), 0
    for _ in range(moves):
        if any(board.has_won(p) for p in (0, 1)):
            break
        walls = board.valid_walls(turn) if board.walls_left[turn] else []
        if walls and rng.random() < 0.55:
            r, c, o = rng.choice(walls)
            board.apply_move(turn, WallMove(r, c, o))
        else:
            board.apply_move(turn, PawnMove(*rng.choice(board.valid_pawn_moves(turn))))
        turn = 1 - turn
    return board, turn


def walls_of(board: Board):
    return tuple(sorted((r, c, o.value) for r, c, o in board.walls))


@needs_cv
class TestParser(unittest.TestCase):
    def parse(self, image, **profile):
        return ScreenParser(Profile(**{**NO_OCR, **profile})).parse(image)

    def test_start_position(self):
        image, true_geometry = render(hints=[(8, 3), (7, 4), (8, 5)])
        res = self.parse(image)
        self.assertTrue(res.ok, res.reason)
        self.assertEqual(res.pawns, ((8, 4), (0, 4)))          # Red e1, Blue e9
        self.assertEqual(res.walls, ())
        self.assertEqual(res.walls_left, (10, 10))
        self.assertTrue(res.walls_consistent)
        self.assertEqual(res.border_turn, RED)
        self.assertEqual(res.hint_turn, RED)
        self.assertEqual(set(res.hints), {(8, 3), (7, 4), (8, 5)})  # outlines are not pawns or walls
        self.assertEqual(res.warnings, [])
        g = res.geometry
        self.assertAlmostEqual(g.x0, true_geometry.x0, delta=1.5)
        self.assertAlmostEqual(g.y0, true_geometry.y0, delta=1.5)
        self.assertAlmostEqual(g.pitch, true_geometry.pitch, delta=0.3)

    def test_random_positions_both_themes_and_sizes(self):
        rng = random.Random(7)
        layouts = [dict(cell=56, gap=12, origin=(90, 110), size=(1280, 800)),
                   dict(cell=40, gap=8, origin=(33, 71), size=(1024, 640)),
                   dict(cell=84, gap=14, origin=(301, 150), size=(1920, 1080)),
                   dict(cell=64, gap=6, origin=(150, 97), size=(1600, 900))]
        for i in range(24):
            board, turn = random_position(rng, rng.randrange(0, 70))
            theme = (DARK, LIGHT)[i % 2]
            state = board.snapshot(turn)
            image, _ = render_state(state, theme, **layouts[i % len(layouts)])
            res = self.parse(image)
            label = f"position {i}"
            self.assertTrue(res.ok, f"{label}: {res.reason}")
            self.assertEqual(res.pawns, state.pawns, label)
            self.assertEqual(res.walls, walls_of(board), label)
            self.assertEqual(res.walls_left, state.walls_left, label)
            self.assertEqual(res.border_turn, turn, label)
            if turn == RED and state.valid_pawn_moves:
                self.assertEqual(set(res.hints), set(state.valid_pawn_moves), label)
                self.assertIn(res.hint_turn, (RED, None), label)  # None when Blue could make the same moves
            else:
                self.assertEqual(res.hints, (), label)

    def test_board_full_of_walls(self):
        walls = [(r, c, "h") for r in (0, 2, 4, 6) for c in (0, 2, 5)] + [(r, 7, "v") for r in (0, 2, 4, 6)] \
            + [(1, 4, "v"), (3, 4, "v"), (5, 4, "v"), (7, 3, "h")]
        self.assertEqual(len(walls), 20)
        for theme in (DARK, LIGHT):
            image, _ = render(pawns=((5, 2), (3, 6)), walls=walls, walls_left=(0, 0), turn=BLUE, theme=theme)
            res = self.parse(image)
            self.assertTrue(res.ok, res.reason)
            self.assertEqual(res.walls, tuple(sorted(walls)))
            self.assertEqual(res.walls_left, (0, 0))
            self.assertEqual(res.border_turn, BLUE)

    def test_spent_ticks_in_a_dimmed_player_colour(self):
        theme = replace(DARK, dim_spent=True)
        image, _ = render(walls=[(3, 3, "h"), (4, 6, "v"), (6, 1, "h")], walls_left=(8, 9), theme=theme)
        res = self.parse(image)
        self.assertEqual((res.cards[RED].tick_count, res.cards[BLUE].tick_count), (8, 9))
        self.assertEqual(res.walls_left, (8, 9))

    def test_unreadable_card_is_filled_in_from_the_board(self):
        image, _ = render(walls=[(3, 3, "h"), (4, 6, "v"), (6, 1, "h")], walls_left=(8, 9))
        res = ScreenParser(Profile(**NO_OCR))
        parsed = res.parse(image)
        x1, y1, x2, y2 = parsed.cards[BLUE].ticks[0][0], parsed.cards[BLUE].ticks[0][1], \
            parsed.cards[BLUE].ticks[-1][2], parsed.cards[BLUE].ticks[-1][3]
        image[y1:y2, x1:x2] = image[y1 - 4, x1 - 4]  # paint over Blue's tick bar
        parsed = self.parse(image)
        self.assertEqual(parsed.walls_left, (8, 9))
        self.assertEqual(parsed.walls_source, "ticks+board")

    def test_wall_count_mismatch_is_reported(self):
        image, _ = render(walls=[(3, 3, "h")], walls_left=(10, 10))
        res = self.parse(image)
        self.assertTrue(res.ok)
        self.assertFalse(res.walls_consistent)
        self.assertTrue(any("walls used" in w for w in res.warnings))

    def test_no_border_no_turn(self):
        image, _ = render(turn=None)
        res = self.parse(image)
        self.assertTrue(res.ok)
        self.assertIsNone(res.turn)
        image, _ = render(turn=None, pawns=((6, 4), (0, 4)), hints=[(5, 4), (7, 4), (6, 3), (6, 5)])
        self.assertEqual(self.parse(image).turn, RED)  # legal-move outlines alone

    def test_no_board(self):
        image = np.full((600, 900, 3), 40, np.uint8)
        cv2.putText(image, "nothing here", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 200, 200), 3)
        self.assertIsNone(find_board(image))
        res = self.parse(image)
        self.assertFalse(res.ok)
        self.assertIn("no 9x9 board", res.reason)

    def test_half_placed_wall_is_not_a_position(self):
        image, g = render()
        x, y = (round(v) for v in g.center(3.5, 4))
        image[y - 3:y + 4, x - 20:x + 21] = (110, 190, 226)  # one groove segment only
        res = self.parse(image)
        self.assertFalse(res.ok)
        self.assertIn("pair up", res.reason)

    def test_board_is_tracked_and_found_again_after_moving(self):
        parser = ScreenParser(Profile(**NO_OCR))
        image, _ = render(origin=(90, 110))
        first = parser.parse(image).geometry
        self.assertIs(parser.parse(image).geometry, first)  # same object: no new search
        moved, _ = render(origin=(140, 95))
        second = parser.parse(moved).geometry
        self.assertAlmostEqual(second.x0 - first.x0, 50, delta=1.5)
        self.assertAlmostEqual(second.y0 - first.y0, -15, delta=1.5)

    def test_marker_outline_does_not_change_the_reading(self):
        """The strip overlay (outline around a wall slot or square) must not be read as a wall or pawn."""
        image, _ = render(walls=[(3, 3, "h")], walls_left=(9, 10))
        parser = ScreenParser(Profile(**NO_OCR))
        before = parser.parse(image)
        g = before.geometry
        for rect in (g.wall_rect(5, 2, "v", g.gap + 0.2 * g.side), g.wall_rect(6, 5, "h", g.gap + 0.2 * g.side),
                     g.cell_rect(7, 4)):
            cv2.rectangle(image, rect[:2], (rect[2] - 1, rect[3] - 1), (242, 61, 255), 3)
        after = parser.parse(image)
        self.assertTrue(after.ok, after.reason)
        self.assertEqual((after.pawns, after.walls, after.walls_left, after.border_turn),
                         (before.pawns, before.walls, before.walls_left, before.border_turn))

    def test_annotate_and_profile(self):
        image, _ = render(walls=[(3, 3, "h")], walls_left=(9, 10))
        res = self.parse(image)
        out = annotate(image, res)
        self.assertEqual(out.ndim, 3)
        self.assertLess(out.shape[1], image.shape[1])  # cropped to the game
        self.assertIn("red_hues", Profile().to_json())


@needs_cv
class TestNotation(unittest.TestCase):
    def test_squares_and_walls(self):
        self.assertEqual(move_notation(PawnMove(7, 4)), "e2")
        self.assertEqual(move_notation(PawnMove(0, 0)), "a9")
        self.assertEqual(move_notation(WallMove(5, 4, Orientation.HORIZONTAL)), "e3h")
        self.assertEqual(move_notation(WallMove(4, 3, Orientation.VERTICAL)), "d4v")
        for text in ("e2", "a9", "i1", "e3h", "d4v", "h8v", "a1h"):
            self.assertEqual(move_notation(parse_notation(text)), text)
        for bad in ("j2", "e0", "i3h", "e9v", "e3x", ""):
            with self.assertRaises(ValueError):
                parse_notation(bad)

    def test_wall_notation_matches_the_squares_it_blocks(self):
        board = Board()
        board.apply_move(0, parse_notation("e3h"))  # above e3 and f3
        self.assertTrue(board.is_blocked((6, 4), (5, 4)))
        self.assertTrue(board.is_blocked((6, 5), (5, 5)))
        self.assertFalse(board.is_blocked((6, 3), (5, 3)))
        board.apply_move(1, parse_notation("d4v"))  # right of d4 and d5
        self.assertTrue(board.is_blocked((5, 3), (5, 4)))
        self.assertTrue(board.is_blocked((4, 3), (4, 4)))


@needs_cv
class TestTracker(unittest.TestCase):
    def setUp(self):
        self.parser = ScreenParser(Profile(**NO_OCR))
        self.tracker = PositionTracker(stable_frames=2, unsure_frames=4)

    def feed(self, times=1, **kwargs):
        image, _ = render(**kwargs)
        out = [self.tracker.update(self.parser.parse(image)) for _ in range(times)]
        return [p for p in out if p is not None]

    def test_reports_each_position_once(self):
        self.assertEqual(self.feed(1), [])                        # not stable yet
        got = self.feed(3)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0], Position(((8, 4), (0, 4)), (), (10, 10), RED))
        self.assertEqual(self.feed(5), [])                        # identical frames: nothing new
        got = self.feed(3, pawns=((7, 4), (0, 4)), turn=BLUE)
        self.assertEqual([p.turn for p in got], [BLUE])
        self.assertEqual(self.tracker.turn_source, "border")

    def test_turn_from_history_when_no_border(self):
        self.feed(3)
        got = self.feed(3, pawns=((7, 4), (0, 4)), turn=None)     # Red moved, no border visible
        self.assertEqual([p.turn for p in got], [BLUE])
        self.assertEqual(self.tracker.turn_source, "history")
        got = self.feed(3, pawns=((7, 4), (0, 4)), walls=[(0, 3, "h")], walls_left=(10, 9), turn=None)
        self.assertEqual([p.turn for p in got], [RED])            # Blue placed a wall

    def test_stale_border_waits_longer(self):
        self.feed(3)
        # the pawn has moved but the border still marks Red: wait for it to switch
        self.assertEqual(self.feed(3, pawns=((7, 4), (0, 4)), turn=RED), [])
        got = self.feed(2, pawns=((7, 4), (0, 4)), turn=BLUE)
        self.assertEqual([p.turn for p in got], [BLUE])

    def test_forced_turn_and_rejected_positions(self):
        self.tracker.forced_turn = BLUE
        got = self.feed(3)
        self.assertEqual(got[0].turn, BLUE)
        self.assertEqual(self.tracker.turn_source, "manual")
        self.assertEqual(Position(((8, 4), (0, 4)), ((3, 3, "h"), (3, 4, "h")), (9, 9), RED).error(),
                         "detected walls overlap")
        self.assertIsNone(Position(((8, 4), (0, 4)), ((3, 3, "h"), (3, 5, "h")), (9, 9), RED).error())

    def test_mover_between(self):
        a = Position(((8, 4), (0, 4)), (), (10, 10), RED)
        self.assertEqual(mover_between(a, replace(a, pawns=((7, 4), (0, 4)))), RED)
        self.assertEqual(mover_between(a, replace(a, walls=((0, 3, "h"),), walls_left=(10, 9))), BLUE)
        self.assertIsNone(mover_between(a, replace(a, pawns=((7, 4), (1, 4)))))  # two moves at once

    def test_watcher_feeds_positions(self):
        frames = [render()[0]] * 3 + [render(pawns=((7, 4), (0, 4)), turn=BLUE)[0]] * 3
        seen, done = [], threading.Event()
        it = iter(frames)

        def grab():
            try:
                return next(it).copy(), 0, 0, 1.0
            except StopIteration:
                done.set()
                raise RuntimeError("no more frames")

        watcher = ScreenWatcher(self.parser, self.tracker, seen.append, fps=30, grabber=grab)
        watcher.start()
        self.assertTrue(done.wait(10))
        watcher.stop()
        watcher.join(5)
        self.assertEqual([p.turn for p in seen], [RED, BLUE])
        self.assertEqual(watcher.latest.to_screen(10, 20), (10, 20))


@needs_cv
class TestAnalyzer(unittest.TestCase):
    def analyse(self, position, spec, timeout=20):
        results, done = [], threading.Event()

        def on_result(analysis):
            results.append(analysis)
            if analysis.complete:
                done.set()

        analyzer = Analyzer(spec, on_result)
        try:
            analyzer.submit(position)
            self.assertTrue(done.wait(timeout), "analysis did not finish")
        finally:
            analyzer.close()
        return results[-1]

    def check_lines(self, spec):
        start = Position(((8, 4), (0, 4)), (), (10, 10), RED)
        a = self.analyse(start, spec)
        self.assertEqual(a.error, "")
        self.assertEqual(a.best.notation, "e2")
        self.assertIsNotNone(a.eval.steps)
        self.assertGreaterEqual(len(a.lines), 3)
        self.assertEqual(len({ln.notation for ln in a.lines}), len(a.lines))
        self.assertTrue(all(ln.reply for ln in a.lines))

        win = self.analyse(Position(((1, 4), (6, 4)), (), (0, 0), RED), spec)   # Red one step from the goal
        self.assertEqual(win.best.notation, "e9")
        self.assertEqual(win.eval.text(), "M1")
        self.assertEqual(win.eval.red_share(), 1.0)
        loss = self.analyse(Position(((4, 4), (7, 4)), (), (0, 0), RED), spec)  # Blue arrives first
        self.assertEqual(loss.eval.text(), "-M1")
        self.assertEqual(loss.eval.for_player(BLUE).text(), "M1")
        blue = self.analyse(Position(((4, 4), (7, 4)), (), (0, 0), BLUE), spec)
        self.assertEqual((blue.best.notation, blue.eval.text()), ("e1", "M1"))

    def test_python_search(self):
        self.check_lines(BotSpec(depth=4, engine="py"))

    @unittest.skipUnless(engine_available(), "C++ engine not built")
    def test_cpp_engine(self):
        self.check_lines(BotSpec(depth=30, time_limit=0.3))

    @unittest.skipUnless(engine_available(), "C++ engine not built")
    def test_new_position_cancels_the_running_search(self):
        results, done = [], threading.Event()
        second = Position(((7, 4), (0, 4)), (), (10, 10), BLUE)

        def on_result(analysis):
            results.append(analysis)
            if analysis.complete and analysis.position == second:
                done.set()

        analyzer = Analyzer(BotSpec(depth=30, time_limit=30.0), on_result)
        try:
            analyzer.submit(Position(((8, 4), (0, 4)), (), (10, 10), RED))
            threading.Event().wait(0.3)
            analyzer.set_spec(BotSpec(depth=30, time_limit=0.2))
            analyzer.submit(second)
            self.assertTrue(done.wait(15), "the 30 s search was not cancelled")
        finally:
            analyzer.close()
        self.assertTrue(all(a.position == second and not a.error for a in results), results)

    def test_game_over_and_unknown_turn_are_not_searched(self):
        results = []
        analyzer = Analyzer(BotSpec(depth=2, engine="py"), results.append)
        try:
            analyzer.submit(Position(((0, 4), (5, 4)), (), (10, 10), BLUE))   # Red already on its goal row
            analyzer.submit(Position(((8, 4), (0, 4)), (), (10, 10), None))
            threading.Event().wait(0.5)
        finally:
            analyzer.close()
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
