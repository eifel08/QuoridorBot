"""Headless GUI tests (skipped if pygame is not installed)."""

import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

try:
    import pygame
except ImportError:  # pragma: no cover
    pygame = None

from quoridor import Board, Orientation, PawnMove, WallMove

H, V = Orientation.HORIZONTAL, Orientation.VERTICAL


@unittest.skipIf(pygame is None, "pygame not installed")
class TestLayout(unittest.TestCase):
    def setUp(self):
        from quoridor.gui.layout import Layout
        self.L = Layout()

    def test_square_round_trip(self):
        for r in range(9):
            for c in range(9):
                self.assertEqual(self.L.square_at(*self.L.square_center(r, c)), (r, c))

    def test_horizontal_groove_extends_left(self):
        x, y, w, h = self.L.square_rect(2, 3)
        cands = self.L.wall_candidates(x + w / 2, y + h + self.L.gap / 2)
        self.assertEqual(cands[0], (2, 2, H))  # covers cols 2 and 3
        self.assertEqual(cands[1], (2, 3, H))

    def test_horizontal_groove_left_edge(self):
        x, y, w, h = self.L.square_rect(4, 0)
        self.assertEqual(self.L.wall_candidates(x + 5, y + h + 3)[0], (4, 0, H))

    def test_vertical_groove_extends_down(self):
        x, y, w, h = self.L.square_rect(5, 6)
        cands = self.L.wall_candidates(x + w + self.L.gap / 2, y + h / 2)
        self.assertEqual(cands[0], (5, 6, V))  # covers rows 5 and 6
        x, y, w, h = self.L.square_rect(8, 6)
        self.assertEqual(self.L.wall_candidates(x + w + 2, y + h / 2)[0], (7, 6, V))

    def test_off_board(self):
        self.assertEqual(self.L.wall_candidates(self.L.width - 5, 5), [])


@unittest.skipIf(pygame is None, "pygame not installed")
class TestGuiInteraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from quoridor.gui import QuoridorGUI
        cls.gui = QuoridorGUI()

    def setUp(self):
        self.board = Board()
        self.board.pawns = [(5, 4), (4, 4)]
        self.state = self.board.snapshot(0)
        self.gui.state = self.state
        self.gui.human_turn = True
        self.gui.anims.clear()
        self.gui._reset_interaction()
        self.L = self.gui.layout

    def ev(self, kind, pos, **kw):
        return self.gui.handle_event(pygame.event.Event(kind, pos=pos, **kw))

    def test_click_pawn_then_square(self):
        pc = self.L.square_center(5, 4)
        self.assertIsNone(self.ev(pygame.MOUSEBUTTONDOWN, pc, button=1))
        self.assertIsNone(self.ev(pygame.MOUSEBUTTONUP, pc, button=1))
        self.assertTrue(self.gui.selected)
        move = self.ev(pygame.MOUSEBUTTONDOWN, self.L.square_center(3, 4), button=1)  # jump
        self.assertEqual(move, PawnMove(3, 4))

    def test_direct_click_and_invalid_click(self):
        self.assertIsNone(self.ev(pygame.MOUSEBUTTONDOWN, self.L.square_center(0, 0), button=1))
        self.assertEqual(self.ev(pygame.MOUSEBUTTONDOWN, self.L.square_center(5, 3), button=1), PawnMove(5, 3))

    def test_drag_pawn(self):
        pc = self.L.square_center(5, 4)
        target = self.L.square_center(6, 4)
        self.ev(pygame.MOUSEBUTTONDOWN, pc, button=1)
        self.ev(pygame.MOUSEMOTION, target, rel=(0, 0), buttons=(1, 0, 0))
        self.assertTrue(self.gui.pawn_drag)
        self.assertEqual(self.ev(pygame.MOUSEBUTTONUP, target, button=1), PawnMove(6, 4))

    def test_drag_pawn_to_illegal_square_returns(self):
        pc = self.L.square_center(5, 4)
        bad = self.L.square_center(0, 0)
        self.ev(pygame.MOUSEBUTTONDOWN, pc, button=1)
        self.ev(pygame.MOUSEMOTION, bad, rel=(0, 0), buttons=(1, 0, 0))
        self.assertIsNone(self.ev(pygame.MOUSEBUTTONUP, bad, button=1))
        self.assertTrue(any(a.kind == "pawn" for a in self.gui.anims))

    def _drag_wall_to(self, pos):
        x, y, w, h = self.L.stub_rect(0, 9)
        self.ev(pygame.MOUSEBUTTONDOWN, (x + w / 2, y + h / 2), button=1)
        self.assertTrue(self.gui.wall_drag)
        self.ev(pygame.MOUSEMOTION, pos, rel=(0, 0), buttons=(1, 0, 0))
        return self.ev(pygame.MOUSEBUTTONUP, pos, button=1)

    def test_drag_wall(self):
        x, y, w, h = self.L.square_rect(2, 3)
        move = self._drag_wall_to((x + w / 2, y + h + self.L.gap / 2))
        self.assertEqual(move, WallMove(2, 2, H))

    def test_snaps_to_alternative_when_first_illegal(self):
        self.board.walls.add((2, 1, H))  # blocks (2,2,H) by overlap
        self.gui.state = self.board.snapshot(0)
        x, y, w, h = self.L.square_rect(2, 3)
        self.assertEqual(self._drag_wall_to((x + w / 2, y + h + self.L.gap / 2)), WallMove(2, 3, H))

    def test_illegal_wall_drop_rejected(self):
        self.board.walls.add((2, 2, V))  # both (2,2,V) and fallback (1,2,V) overlap it
        self.gui.state = self.board.snapshot(0)
        x, y, w, h = self.L.square_rect(2, 2)
        move = self._drag_wall_to((x + w + self.L.gap / 2, y + h / 2))
        self.assertIsNone(move)
        self.assertTrue(self.gui.message)
        self.assertTrue(any(a.kind == "wall_return" for a in self.gui.anims))

    def test_drawing_does_not_mutate_state(self):
        before = self.gui.state
        snapshot = before.to_dict()
        self.gui.selected = True
        self.gui.draw()
        self.assertIs(self.gui.state, before)
        self.assertEqual(before.to_dict(), snapshot)


if __name__ == "__main__":
    unittest.main()
