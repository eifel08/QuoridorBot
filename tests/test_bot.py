import json
import shutil
import subprocess
import unittest
from pathlib import Path

from quoridor import Board, BoardState, Game, PawnMove, WallMove
from quoridor.bots import AlphaBetaBot, search
from quoridor.bots.alphabeta import FastBoard, N, NW, evaluate

from tests.test_web_engine import random_positions

ROOT = Path(__file__).resolve().parent.parent
NODE = shutil.which("node")
STATES = [BoardState.from_dict(d) for d in random_positions(4, seed=99)]


class TestFastBoard(unittest.TestCase):
    """The bot's internal board must agree with the real rules engine."""

    def test_pawn_moves_and_distances(self):
        for st in STATES:
            fb = FastBoard(st)
            board = st.to_board()
            for p in (0, 1):
                got = sorted(divmod(s, N) for s in fb.pawn_moves(p))
                self.assertEqual(got, sorted(board.valid_pawn_moves(p)))
                self.assertEqual(fb.distance(p), board.shortest_path_length(p))
                self.assertEqual(len(fb.shortest_path(p)) - 1, board.shortest_path_length(p))

    def test_overlaps_match_engine(self):
        for st in STATES[::5]:
            fb = FastBoard(st)
            board = st.to_board()
            for o in "hv":
                for a in range(NW * NW):
                    r, c = divmod(a, NW)
                    engine_conflict = any(w in board.walls for w in board.conflicting_walls((r, c, o_enum(o))))
                    self.assertEqual(fb.overlaps(o, a), engine_conflict, (st, o, a))

    def test_make_unmake_restores_hash(self):
        st = STATES[10]
        fb = FastBoard(st)
        h0 = fb.hash
        from quoridor.bots.alphabeta import AlphaBetaSearch
        moves = AlphaBetaSearch(fb, None).ordered_moves(None)
        for m in moves:
            prev = fb.pawns[fb.turn]
            fb.make(m)
            fb.unmake(m, prev)
            self.assertEqual(fb.hash, h0)
        self.assertEqual(fb._full_hash(), h0)


def o_enum(o):
    from quoridor import Orientation
    return Orientation(o)


class TestSearch(unittest.TestCase):
    def test_heuristic_convention(self):
        board = Board()
        board.pawns[0] = (3, 4)  # player 0 needs 3 steps, player 1 needs 8
        st = board.snapshot(0)
        fb = FastBoard(st)
        self.assertEqual(evaluate(fb), 8 - 3)  # negamax value = -(Score)
        _, info = search(st, 1)
        self.assertLessEqual(info["score"], 3 - 8 + 1)  # Score = own - opponent, low is good

    def test_takes_immediate_win(self):
        board = Board()
        board.pawns = [(1, 2), (5, 5)]
        move, info = search(board.snapshot(0), 3)
        self.assertEqual(move, PawnMove(0, 2))
        self.assertEqual(info["forced"], "win")

    def test_blocks_opponent_about_to_win(self):
        board = Board()
        board.pawns = [(8, 0), (7, 4)]  # player 2 wins next move at (8,4)
        move, _ = search(board.snapshot(0), 2)
        self.assertIsInstance(move, WallMove)
        after = board.copy()
        after.apply_move(0, move)
        self.assertGreater(after.shortest_path_length(1), 1)

    def test_moves_always_legal(self):
        for st in STATES[::3]:
            if st.winner is not None:
                continue
            for depth in (1, 2, 3):
                move, _ = search(st, depth)
                self.assertIsNone(st.to_board().move_error(st.current_player, move), (st, move))

    def test_time_limit_respected(self):
        st = STATES[len(STATES) // 2]
        _, info = search(st, 30, time_limit=0.3)
        self.assertLess(info["seconds"], 1.0)
        self.assertLess(info["depth_reached"], 30)

    def test_bot_vs_bot_game_finishes(self):
        game = Game(AlphaBetaBot("A", 2, 2), AlphaBetaBot("B", 1, 2), max_invalid_attempts=1)
        self.assertIn(game.run(), (0, 1))

    def test_bot_is_stateless(self):
        bot = AlphaBetaBot("A", depth=2)
        st = STATES[3]
        self.assertEqual(bot.get_move(st), bot.get_move(st))
        self.assertEqual(set(vars(bot)), {"name", "index", "depth", "time_limit", "last_info"})


@unittest.skipIf(NODE is None, "node not installed")
class TestJsBotParity(unittest.TestCase):
    """web/bot.js must choose the same moves as the Python bot."""

    def test_same_moves(self):
        states = [s for s in STATES[::4] if s.winner is None][:15]
        depths = [1, 2, 3, 4]
        out = subprocess.run(
            [NODE, str(ROOT / "tests" / "js" / "bot_moves.mjs")],
            input=json.dumps({"states": [s.to_dict() for s in states], "depths": depths}),
            capture_output=True, text=True, check=True,
        )
        js = iter(json.loads(out.stdout))
        for st in states:
            for d in depths:
                move, info = search(st, d)
                j = next(js)
                jm = j["move"]
                js_move = PawnMove(jm["row"], jm["col"]) if jm["type"] == "pawn" else \
                    WallMove(jm["row"], jm["col"], o_enum(jm["orientation"]))
                self.assertEqual(js_move, move, (st.to_dict(), d))
                self.assertEqual(j["info"]["score"], info["score"])
                self.assertEqual(j["info"]["nodes"], info["nodes"])


if __name__ == "__main__":
    unittest.main()
