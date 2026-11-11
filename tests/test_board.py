import unittest

from quoridor import Board, Game, Orientation, PawnMove, Player, WallMove
from quoridor.player import parse_command
from quoridor.renderer import render_board

H, V = Orientation.HORIZONTAL, Orientation.VERTICAL


class TestPawnMoves(unittest.TestCase):
    def test_initial_moves(self):
        b = Board()
        self.assertEqual(sorted(b.valid_pawn_moves(0)), [(7, 4), (8, 3), (8, 5)])
        self.assertEqual(sorted(b.valid_pawn_moves(1)), [(0, 3), (0, 5), (1, 4)])

    def test_wall_blocks_step(self):
        b = Board()
        b.walls.add((7, 4, H))  # below (7,4)/(7,5): blocks (8,4)->(7,4)
        self.assertNotIn((7, 4), b.valid_pawn_moves(0))
        b2 = Board()
        b2.walls.add((7, 3, H))  # covers cols 3 and 4
        self.assertNotIn((7, 4), b2.valid_pawn_moves(0))

    def test_vertical_wall_blocks_side_step(self):
        b = Board()
        b.walls.add((7, 4, V))  # right of (7,4) and (8,4)
        self.assertNotIn((8, 5), b.valid_pawn_moves(0))

    def test_straight_jump(self):
        b = Board()
        b.pawns = [(5, 4), (4, 4)]
        moves = b.valid_pawn_moves(0)
        self.assertIn((3, 4), moves)
        self.assertNotIn((4, 4), moves)
        self.assertNotIn((4, 3), moves)

    def test_diagonal_jump_when_wall_behind(self):
        b = Board()
        b.pawns = [(5, 4), (4, 4)]
        b.walls.add((3, 4, H))  # wall above (4,4)
        moves = b.valid_pawn_moves(0)
        self.assertNotIn((3, 4), moves)
        self.assertIn((4, 3), moves)
        self.assertIn((4, 5), moves)

    def test_diagonal_jump_at_edge(self):
        b = Board()
        b.pawns = [(1, 4), (0, 4)]
        moves = b.valid_pawn_moves(0)
        self.assertIn((0, 3), moves)
        self.assertIn((0, 5), moves)

    def test_diagonal_jump_blocked_side(self):
        b = Board()
        b.pawns = [(5, 4), (4, 4)]
        b.walls.add((3, 4, H))
        b.walls.add((3, 3, V))  # left of (4,4)
        moves = b.valid_pawn_moves(0)
        self.assertNotIn((4, 3), moves)
        self.assertIn((4, 5), moves)

    def test_win(self):
        b = Board()
        b.pawns[0] = (1, 0)
        b.apply_move(0, PawnMove(0, 0))
        self.assertTrue(b.has_won(0))


class TestWalls(unittest.TestCase):
    def test_out_of_bounds(self):
        b = Board()
        for wall in [(8, 0, H), (0, 8, V), (-1, 0, H), (0, -1, V)]:
            self.assertFalse(b.is_valid_wall(0, wall), wall)

    def test_overlap_and_cross(self):
        b = Board()
        b.apply_move(0, WallMove(4, 4, H))
        self.assertFalse(b.is_valid_wall(1, (4, 4, H)))  # same
        self.assertFalse(b.is_valid_wall(1, (4, 3, H)))  # overlap left
        self.assertFalse(b.is_valid_wall(1, (4, 5, H)))  # overlap right
        self.assertFalse(b.is_valid_wall(1, (4, 4, V)))  # cross
        self.assertTrue(b.is_valid_wall(1, (4, 6, H)))   # adjacent ok
        self.assertTrue(b.is_valid_wall(1, (3, 4, V)))   # touching T ok
        self.assertTrue(b.is_valid_wall(1, (4, 3, V)))

    def test_wall_count(self):
        b = Board()
        b.walls_left[0] = 0
        self.assertIn("no walls", b.wall_error(0, (0, 0, H)))
        b2 = Board()
        b2.apply_move(0, WallMove(0, 0, H))
        self.assertEqual(b2.walls_left, [9, 10])

    def test_cannot_fully_block(self):
        b = Board()
        # Wall off row 7/8 boundary except the last two columns.
        for c in (0, 2, 4, 6):
            b.apply_move(0 if c % 4 == 0 else 1, WallMove(7, c, H))
        # Vertical wall right of col 7 in rows 7-8 would seal Player 1 in.
        err = b.wall_error(1, (7, 7, V))
        self.assertIsNotNone(err)
        self.assertIn("block", err)
        self.assertTrue(b.has_path(0))

    def test_path_length(self):
        b = Board()
        self.assertEqual(b.shortest_path_length(0), 8)
        self.assertEqual(b.shortest_path_length(1), 8)


class ScriptedPlayer(Player):
    def __init__(self, name, moves):
        super().__init__(name)
        self.moves = list(moves)
        self.rejected = []

    def get_move(self, board_state):
        return self.moves.pop(0)

    def notify_invalid_move(self, move, reason):
        self.rejected.append(move)


class TestGame(unittest.TestCase):
    def test_full_game(self):
        p1 = ScriptedPlayer("A", [PawnMove(0, 0)] + [PawnMove(r, 4) for r in range(7, -1, -1)])
        # B shuffles along the top row while A walks straight up.
        p2 = ScriptedPlayer("B", [PawnMove(0, c) for c in (3, 2, 1, 0, 1, 2, 3, 2)])
        game = Game(p1, p2)
        winner = game.run()
        self.assertEqual(winner, 0)
        self.assertEqual(p1.rejected, [PawnMove(0, 0)])


class TestParsingAndRendering(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_command("m 7 4"), ("move", PawnMove(7, 4)))
        self.assertEqual(parse_command("W 3,4 V"), ("move", WallMove(3, 4, V)))
        for bad in ["", "m 1", "m a b", "w 1 2 x", "jump"]:
            with self.assertRaises(ValueError):
                parse_command(bad)

    def test_render(self):
        b = Board()
        b.apply_move(0, WallMove(3, 4, H))
        b.apply_move(1, WallMove(5, 1, V))
        out = render_board(b)
        self.assertIn("1", out)
        self.assertIn("#", out)


if __name__ == "__main__":
    unittest.main()
