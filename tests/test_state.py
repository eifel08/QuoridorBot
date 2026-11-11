import json
import unittest

from quoridor import Board, BoardState, Game, Orientation, PawnMove, Player, WallMove

H, V = Orientation.HORIZONTAL, Orientation.VERTICAL


class TestBoardState(unittest.TestCase):
    def setUp(self):
        self.board = Board()
        self.board.apply_move(0, WallMove(3, 4, H))
        self.board.apply_move(1, WallMove(5, 1, V))
        self.state = self.board.snapshot(current_player=0, move_number=2)

    def test_contents(self):
        st = self.state
        self.assertEqual(st.pawns, ((8, 4), (0, 4)))
        self.assertEqual(st.goal_rows, (0, 8))
        self.assertEqual(st.walls_left, (9, 9))
        self.assertEqual(set(st.walls), {(3, 4, H), (5, 1, V)})
        self.assertEqual(set(st.valid_pawn_moves), set(self.board.valid_pawn_moves(0)))
        self.assertEqual(set(st.valid_walls), set(self.board.valid_walls(0)))
        self.assertTrue(st.is_valid_wall((0, 0, "h")))
        self.assertFalse(st.is_valid_wall((3, 4, V)))  # crosses existing wall
        self.assertEqual(st.shortest_paths, (9, 9))  # (3,4,H) forces a detour

    def test_immutable_and_detached(self):
        with self.assertRaises(Exception):
            self.state.pawns = ((0, 0), (1, 1))
        self.board.apply_move(0, PawnMove(7, 4))
        self.assertEqual(self.state.pawns[0], (8, 4))  # snapshot unaffected
        b2 = self.state.to_board()
        b2.apply_move(0, WallMove(0, 0, H))
        self.assertNotIn((0, 0, H), self.state.walls)

    def test_matrices(self):
        hm = self.state.wall_matrix(H)
        vm = self.state.wall_matrix(V)
        self.assertEqual(len(hm), 8)
        self.assertEqual(hm[3][4], 1)
        self.assertEqual(vm[5][1], 1)
        self.assertEqual(sum(map(sum, hm)) + sum(map(sum, vm)), 2)
        self.assertEqual(self.state.pawn_matrix(0)[8][4], 1)
        planes = self.state.to_planes()
        self.assertEqual((len(planes), len(planes[0]), len(planes[0][0])), (4, 9, 9))

    def test_round_trip(self):
        data = json.loads(self.state.to_json())
        self.assertEqual(data["walls"][0][2] in ("h", "v"), True)
        self.assertEqual(BoardState.from_dict(data), self.state)


class RecordingPlayer(Player):
    """Stateless bot-like player: decides from the snapshot alone."""

    def __init__(self, name):
        super().__init__(name)
        self.seen = []
        self.observed = 0

    def get_move(self, board_state):
        self.seen.append(board_state)
        # Greedy: step that minimises distance to the goal row.
        goal = board_state.goal_rows[board_state.current_player]
        return PawnMove(*min(board_state.valid_pawn_moves, key=lambda sq: abs(sq[0] - goal)))

    def observe(self, board_state):
        self.observed += 1


class TestEngineGivesSnapshots(unittest.TestCase):
    def test_players_get_states(self):
        a, b = RecordingPlayer("A"), RecordingPlayer("B")
        game = Game(a, b)
        winner = game.run()
        self.assertIsNotNone(winner)
        self.assertTrue(all(isinstance(s, BoardState) for s in a.seen + b.seen))
        self.assertTrue(all(s.current_player == 0 for s in a.seen))
        self.assertTrue(all(s.current_player == 1 for s in b.seen))
        self.assertEqual(a.observed, len(game.history))
        self.assertEqual(game.state.winner, winner)


if __name__ == "__main__":
    unittest.main()
