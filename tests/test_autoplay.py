"""Auto-Play: identity, move counting, turn signals, click points, stop switch.

Everything runs on mock screenshots with a recording fake mouse; nothing is
clicked for real. Skipped without OpenCV / numpy.
"""

import unittest

from quoridor import Orientation, PawnMove, WallMove

try:
    import cv2  # noqa: F401
except ImportError:  # pragma: no cover
    cv2 = None

if cv2 is not None:
    from quoridor.vision.analysis import Analysis, Line
    from quoridor.vision.autoplay import (AutoPlayer, ClickProfile, ClockWatcher, EmergencyStop, FakeMouse, Hand,
                                          StopSwitch, identify_bottom_player, legal_move_between, pawn_point,
                                          position_after, wall_points)
    from quoridor.vision.capture import Frame
    from quoridor.vision.parser import BLUE, RED, Profile, ScreenParser
    from quoridor.vision.position import Position, PositionTracker, move_notation, parse_notation
    from quoridor.vision.synthetic import render

needs_cv = unittest.skipUnless(cv2 is not None, "opencv-python / numpy not installed")


class Near:
    """Compares equal to a number within 2 (the detected board is within a pixel or two of the drawn one)."""

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return abs(other - self.value) <= 2

    def __repr__(self):
        return f"~{self.value}"


def near(x, y):
    return (Near(round(x)), Near(round(y)))
H, V = Orientation.HORIZONTAL, Orientation.VERTICAL


def fast_profile(**changes):
    return ClickProfile(move_seconds=(0.0, 0.0), hover_pause=0.0, press_seconds=0.0, confirm_timeout=2.0,
                        check_preview=False, **changes)


class Sim:
    """A mock game on a mock screen: renders frames, feeds parser, tracker and the auto-player."""

    def __init__(self, flipped=False, layout="timed", profile=None, join=False, **render_args):
        self.flipped = flipped
        self.bottom = BLUE if flipped else RED
        self.pos = Position(((0, 4), (8, 4)) if flipped else ((8, 4), (0, 4)), (), (10, 10), RED, flipped)
        self.render_args = dict(layout=layout, **render_args)
        self.clocks = ["5:00", "5:00"]
        self.parser = ScreenParser(Profile(ocr=False))
        self.tracker = PositionTracker()
        self.now = 100.0
        self.mouse = FakeMouse()
        self.stop = StopSwitch()
        self.player = AutoPlayer(self.mouse, profile or fast_profile(), self.stop, join_in_progress=join,
                                 clock=lambda: self.now, threaded=False)
        self.plan = {}          # notation the "engine" plays, by move count of the bot
        self.number = 0
        self.geometry = None

    def frame(self) -> "Frame":
        args = dict(pawns=self.pos.pawns, walls=self.pos.walls, walls_left=self.pos.walls_left, turn=self.pos.turn,
                    flipped=self.flipped, clocks=self.clocks)
        args.update(self.render_args)
        image, self.geometry = render(**args)
        res = self.parser.parse(image)
        self.tracker.update(res)
        self.number += 1
        return Frame(res, image, 0, 0, 1.0, self.number, 0.0, image.shape[1], image.shape[0])

    def analysis(self):
        cur = self.tracker.current
        if cur is None or cur.turn is None:
            return None
        move = parse_notation(self.plan.get(self.player.moves_made, "e2"))
        return Analysis(cur, [Line(move, move_notation(move))], complete=True)

    def step(self, times=1, dt=0.2):
        for _ in range(times):
            self.now += dt
            frame = self.frame()
            self.tracker.counted_turn = None if self.player.ply is None else self.player.ply % 2
            self.player.tick(frame, self.tracker.current, self.analysis())

    def play(self, player, notation):
        """A move appears on the mock screen (made by the site, i.e. by the opponent or by a registered click)."""
        self.pos = position_after(self.pos, player, parse_notation(notation))

    def cell_center(self, notation):
        m = parse_notation(notation)
        return near(*self.geometry.center(m.row, m.col))


@needs_cv
class TestIdentity(unittest.TestCase):
    def test_bottom_player_from_three_signs(self):
        for flipped, expected in ((False, RED), (True, BLUE)):
            for layout in ("classic", "timed"):
                image, _ = render(pawns=((0, 4), (8, 4)) if flipped else ((8, 4), (0, 4)), flipped=flipped,
                                  layout=layout)
                res = ScreenParser(Profile(ocr=False)).parse(image)
                identity, problem = identify_bottom_player(res)
                self.assertEqual(problem, "")
                self.assertEqual(identity.me, expected)
                self.assertEqual(identity.evidence, ["pawn on the bottom row", "bottom row tint", "bottom player card"])

    def test_needs_the_start_position_unless_joining(self):
        image, _ = render(pawns=((0, 4), (7, 4)), flipped=True, layout="timed")
        res = ScreenParser(Profile(ocr=False)).parse(image)
        identity, problem = identify_bottom_player(res)
        self.assertIsNone(identity)
        self.assertIn("start position", problem)
        identity, _ = identify_bottom_player(res, require_start=False)
        self.assertEqual((identity.me, identity.evidence), (BLUE, ["bottom row tint", "bottom player card"]))

    def test_disagreeing_signs_give_no_identity(self):
        image, _ = render(layout="timed")
        res = ScreenParser(Profile(ocr=False)).parse(image)
        res.flipped = True  # as if the bottom row were tinted blue under a red pawn
        identity, problem = identify_bottom_player(res)
        self.assertIsNone(identity)
        self.assertIn("disagree", problem)


@needs_cv
class TestClickPoints(unittest.TestCase):
    def setUp(self):
        _, self.g = render()

    def test_pawn_target_is_the_square_centre(self):
        self.assertEqual(pawn_point(self.g, PawnMove(7, 4)), self.g.center(7, 4))

    def test_wall_points_lie_in_the_groove(self):
        g = self.g
        h = wall_points(g, WallMove(5, 4, H))   # e3h: under... above e3 and f3, i.e. below row 5
        self.assertEqual(h["first"], g.center(5.5, 4))
        self.assertEqual(h["second"], g.center(5.5, 5))
        self.assertEqual(h["center"], g.center(5.5, 4.5))
        v = wall_points(g, WallMove(4, 3, V))   # d4v: right of column d, rows 4 and 5
        self.assertEqual(v["first"], g.center(4, 3.5))
        self.assertEqual(v["second"], g.center(5, 3.5))
        self.assertEqual(v["center"], g.center(4.5, 3.5))
        for x, y in (*h.values(), *v.values()):   # never inside a square
            for r in range(9):
                for c in range(9):
                    x1, y1, x2, y2 = g.cell_rect(r, c)
                    self.assertFalse(x1 <= x < x2 and y1 <= y < y2)

    def test_profile_file(self):
        import os
        import tempfile

        profile = ClickProfile(wall_click="second", wall_key_h="h")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "click.json")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(profile.to_json())
            loaded = ClickProfile.load(path)
            self.assertEqual((loaded.wall_click, loaded.wall_key_h, loaded.move_seconds), ("second", "h", (0.1, 0.3)))
            with open(path, "w", encoding="utf-8") as fh:
                fh.write('{"wall_click": "corner"}')
            with self.assertRaises(ValueError):
                ClickProfile.load(path)

    def test_legal_move_between(self):
        start = Position(((8, 4), (0, 4)), (), (10, 10))
        self.assertTrue(legal_move_between(start, position_after(start, RED, PawnMove(7, 4)), RED))
        self.assertFalse(legal_move_between(start, position_after(start, RED, PawnMove(7, 4)), BLUE))
        self.assertFalse(legal_move_between(start, position_after(start, RED, PawnMove(6, 4)), RED))  # two steps
        self.assertTrue(legal_move_between(start, position_after(start, BLUE, WallMove(7, 3, H)), BLUE))
        flipped = Position(((0, 4), (8, 4)), ((0, 4, "h"),), (10, 9), flipped=True)   # wall right below Red
        self.assertFalse(legal_move_between(flipped, position_after(flipped, RED, PawnMove(1, 4)), RED))
        self.assertTrue(legal_move_between(flipped, position_after(flipped, RED, PawnMove(0, 3)), RED))


@needs_cv
class TestAutoPlayer(unittest.TestCase):
    def test_plays_only_on_its_own_turn_and_only_once(self):
        sim = Sim()                      # Red at the bottom: the bot is Red and moves first
        sim.plan = {0: "e2", 1: "e3"}
        sim.step(4)
        self.assertEqual(sim.mouse.clicks, [])                       # not armed: never clicks
        sim.player.arm()
        sim.step(6)
        self.assertEqual(sim.player.me, RED)
        self.assertEqual(sim.mouse.clicks, [sim.cell_center("e2")])  # one click, on the target square
        self.assertEqual(sim.player.state, "confirming")
        self.assertNotIn(sim.mouse.position(), sim.mouse.clicks)     # cursor parked off the square again
        sim.play(RED, "e2")                                          # the site shows the move
        sim.step(4)
        self.assertEqual((sim.player.ply, sim.player.state), (1, "opponent"))
        self.assertEqual(sim.player.message, "Waiting for opponent...")
        sim.step(10)
        self.assertEqual(len(sim.mouse.clicks), 1)                   # opponent's turn: nothing
        sim.play(BLUE, "e8")
        sim.step(8)
        self.assertEqual(sim.player.last_move, "e3")
        self.assertEqual(sim.mouse.clicks, [sim.cell_center("e2"), sim.cell_center("e3")])

    def test_blue_at_the_bottom_waits_for_red(self):
        sim = Sim(flipped=True)
        sim.plan = {0: "e2"}
        sim.player.arm()
        sim.step(8)
        self.assertEqual((sim.player.me, sim.player.ply), (BLUE, 0))
        self.assertEqual(sim.mouse.clicks, [])                       # Red (top) moves first
        self.assertEqual(sim.player.message, "Waiting for opponent...")
        sim.play(RED, "e8")
        sim.step(8)
        self.assertEqual(sim.mouse.clicks, [sim.cell_center("e2")])

    def test_state_change_lock_retry_and_halt(self):
        sim = Sim()
        sim.player.arm()
        sim.step(6)
        self.assertEqual(len(sim.mouse.clicks), 1)
        sim.step(4)                                   # still inside the confirm time: no second click
        self.assertEqual((len(sim.mouse.clicks), sim.player.state), (1, "confirming"))
        sim.step(12)                                  # timeout: the board never changed -> one retry
        self.assertEqual(len(sim.mouse.clicks), 2)
        sim.step(25)                                  # still nothing: stop instead of clicking forever
        self.assertEqual(len(sim.mouse.clicks), 2)
        self.assertFalse(sim.player.armed)
        self.assertIn("did not change the board", sim.player.message)

    def test_opponent_answers_before_the_move_is_confirmed(self):
        sim = Sim()
        sim.plan = {0: "e2", 1: "e3"}
        sim.player.arm()
        sim.step(6)
        sim.play(RED, "e2")
        sim.play(BLUE, "e8")                          # instant reply: both moves appear together
        sim.step(4)
        self.assertEqual(sim.player.ply, 2)
        sim.step(6)
        self.assertEqual(len(sim.mouse.clicks), 2)

    def test_border_saying_opponent_blocks_the_click(self):
        sim = Sim(layout="classic")                   # green border on the active card
        sim.player.arm()
        sim.pos = Position(sim.pos.pawns, (), (10, 10), BLUE)   # the site marks Blue although no move was made
        sim.step(12)
        self.assertEqual(sim.mouse.clicks, [])
        self.assertIn("border", sim.player.message)
        sim.pos = Position(sim.pos.pawns, (), (10, 10), RED)
        sim.step(6)
        self.assertEqual(len(sim.mouse.clicks), 1)

    def test_unseen_moves(self):
        sim = Sim()
        sim.plan = {0: "e2", 1: "e3"}
        sim.player.arm()
        sim.step(6)
        sim.play(RED, "e2")
        sim.step(4)
        # two moves pass unseen (Blue, then Red by hand): the count stays exact
        sim.play(BLUE, "e8")
        sim.play(RED, "d2")
        clicks = len(sim.mouse.clicks)
        sim.step(6)
        self.assertEqual((sim.player.ply, len(sim.mouse.clicks)), (3, clicks))
        # something that cannot be explained by legal moves: stop
        sim.pos = Position(((5, 0), (4, 8)), (), (10, 10), RED)
        sim.step(6)
        self.assertFalse(sim.player.armed)
        self.assertIn("lost track", sim.player.message)
        self.assertEqual(len(sim.mouse.clicks), clicks)

    def test_joining_a_game_in_progress_needs_a_positive_sign(self):
        sim = Sim(join=True)
        sim.pos = Position(((6, 4), (2, 4)), (), (10, 10), RED)
        sim.player.arm()
        sim.step(8)
        self.assertEqual((sim.player.me, sim.player.ply), (RED, None))
        self.assertEqual(sim.mouse.clicks, [])        # no outlines, no clock: not known to be its turn
        sim.render_args["hints"] = [(5, 4), (7, 4), (6, 3), (6, 5)]
        sim.plan = {0: "e4"}
        sim.step(6)
        self.assertEqual(sim.mouse.clicks, [sim.cell_center("e4")])

    def test_new_game_relocks_the_colour(self):
        sim = Sim()
        sim.player.arm()
        sim.step(6)
        sim.play(RED, "e2")
        sim.step(4)
        self.assertEqual(sim.player.me, RED)
        # next game: the bot's side has the blue pieces now
        sim.flipped, sim.pos = True, Position(((0, 4), (8, 4)), (), (10, 10), RED, True)
        sim.parser.reset()
        sim.tracker.reset()
        sim.step(8)
        self.assertEqual((sim.player.me, sim.player.ply), (BLUE, 0))
        self.assertEqual(len(sim.mouse.clicks), 1)    # and it does not move: Red is to move


@needs_cv
class TestWallsAndMouse(unittest.TestCase):
    def wall_sim(self, **profile):
        sim = Sim(profile=fast_profile(**profile))
        sim.plan = {0: "e3h"}
        sim.player.arm()
        sim.step(6)
        return sim

    def test_wall_click_points(self):
        for where, col in (("first", 4), ("second", 5), ("center", 4.5)):
            sim = self.wall_sim(wall_click=where)
            x, y = sim.geometry.center(5.5, col)
            self.assertEqual(sim.mouse.clicks, [near(x, y)], where)
        sim.play(RED, "e3h")
        sim.step(4)
        self.assertEqual(sim.player.ply, 1)

    def test_wall_key_and_drag(self):
        sim = self.wall_sim(wall_key_h="h", wall_key_v="v")
        self.assertEqual(sim.mouse.events[0], ("key", "h"))
        sim = self.wall_sim(wall_gesture="drag")
        down, up = sim.mouse.events
        tick = sim.frame().result.cards[RED].ticks[-1]
        self.assertTrue(tick[0] <= down[1] <= tick[2] and tick[1] <= down[2] <= tick[3])  # from the reserve
        x, y = sim.geometry.center(5.5, 4)
        self.assertEqual(up, ("up", *near(x, y)))

    def test_pawn_gestures(self):
        sim = Sim(profile=fast_profile(pawn_click="pawn-then-target"))
        sim.player.arm()
        sim.step(6)
        self.assertEqual(sim.mouse.clicks, [sim.cell_center("e1"), sim.cell_center("e2")])
        sim = Sim(profile=fast_profile(pawn_click="drag"))
        sim.player.arm()
        sim.step(6)
        self.assertEqual(sim.mouse.events, [("down", *sim.cell_center("e1")), ("up", *sim.cell_center("e2"))])

    def test_stop_switch(self):
        sim = Sim()
        sim.player.arm()
        sim.stop.trigger("Esc pressed")
        sim.step(8)
        self.assertEqual(sim.mouse.clicks, [])
        self.assertFalse(sim.player.armed)
        self.assertIn("STOPPED (Esc pressed)", sim.player.message)
        sim.player.arm()                              # Start again clears the stop
        sim.step(6)
        self.assertEqual(len(sim.mouse.clicks), 1)

    def test_stop_interrupts_a_mouse_movement(self):
        stop = StopSwitch()
        mouse = FakeMouse((0, 0))
        steps = []

        def sleep(seconds):
            steps.append(seconds)
            if len(steps) == 3:
                stop.trigger("Esc pressed")

        hand = Hand(mouse, ClickProfile(move_seconds=(0.3, 0.3)), stop, sleep=sleep)
        with self.assertRaises(EmergencyStop):
            hand.click(900, 600)
        self.assertEqual(mouse.events, [])            # never pressed
        self.assertLess(mouse.position()[0], 900)     # and stopped on the way


@needs_cv
class TestClocks(unittest.TestCase):
    def frames(self, sim, seconds, ticking, start=300):
        """Advance the mock: ``ticking``'s clock counts down once per second."""
        for i in range(int(seconds / 0.2)):
            left = start - int(i * 0.2)
            sim.clocks = ["5:00", "5:00"]
            if ticking is not None:
                sim.clocks[ticking] = f"{left // 60}:{left % 60:02d}"
            sim.step(1)

    def test_ticking_clock_names_the_player(self):
        sim = Sim()
        self.frames(sim, 5, None)
        self.assertIsNone(sim.player.clocks.clock_turn(sim.now))      # both clocks stand still
        self.frames(sim, 5, BLUE)
        self.assertEqual(sim.player.clocks.clock_turn(sim.now), BLUE)
        self.frames(sim, 6, RED, start=200)
        self.assertEqual(sim.player.clocks.clock_turn(sim.now), RED)

    def test_opponents_clock_running_blocks_the_click(self):
        sim = Sim()                                   # the bot is Red; ply 0 says Red to move ...
        self.frames(sim, 5, BLUE)                     # ... but Blue's clock is the one running
        sim.player.arm()
        self.frames(sim, 3, BLUE, start=290)
        self.assertEqual(sim.mouse.clicks, [])
        self.assertIn("clock", sim.player.message)
        self.frames(sim, 1.6, RED, start=280)         # ours starts running: no longer contradicted
        self.assertEqual(len(sim.mouse.clicks), 1)

    def test_highlight_is_learned(self):
        watcher = ClockWatcher()
        parser = ScreenParser(Profile(ocr=False))
        now = 0.0

        def show(active):
            nonlocal now
            now += 0.2
            image, _ = render(layout="timed", active_clock=active)
            watcher.update(image, parser.parse(image), now)

        show(RED)
        self.assertIsNone(watcher.highlight_turn())   # nothing learned yet
        for turn in (RED, BLUE, RED, BLUE):
            show(turn)
            watcher.learn(turn)
        show(BLUE)
        self.assertEqual(watcher.highlight_turn(), BLUE)
        show(RED)
        self.assertEqual(watcher.highlight_turn(), RED)


if __name__ == "__main__":
    unittest.main()
