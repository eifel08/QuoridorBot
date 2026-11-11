"""Tests for the ML pipeline (skipped when PyTorch is not installed)."""

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from quoridor import BoardState
from quoridor.bots.alphabeta import NW, FastBoard
from quoridor.ml.arena import alphabeta_policy, match, random_opening, wilson_lower_bound
from quoridor.ml.encoding import (MIRROR_PERM, N_ACTIONS, action_to_move, encode, legal_mask,
                                  move_to_action)
from tests.test_web_engine import random_positions

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

STATES = [BoardState.from_dict(d) for d in random_positions(3, seed=42)]


class TestFastLegalMoves(unittest.TestCase):
    def test_legal_walls_match_engine(self):
        for st in STATES:
            got = sorted(FastBoard(st).legal_walls(st.current_player))
            exp = sorted((o.value, r * NW + c) for r, c, o in st.valid_walls)
            self.assertEqual(got, exp)

    def test_copy_is_independent(self):
        b = FastBoard.initial()
        c = b.copy()
        c.play(("h", 0))
        self.assertNotIn(0, b.h)
        self.assertTrue(b.open_[0][1])


class TestEncoding(unittest.TestCase):
    def test_action_round_trip(self):
        for flip in (False, True):
            for mirror in (False, True):
                for a in range(N_ACTIONS):
                    self.assertEqual(move_to_action(action_to_move(a, flip, mirror), flip, mirror), a)

    def test_perspective_is_symmetric(self):
        b = FastBoard.initial()
        x0 = encode(b)
        b.play(("p", 7 * 9 + 4))  # Red steps up; now Blue to move
        b.play(("p", 1 * 9 + 4))  # Blue steps down (mirror image); Red to move
        b.turn = 1  # pretend Blue is to move: position is symmetric, so planes match Red's view
        x1 = encode(b)
        b.turn = 0
        np.testing.assert_array_equal(encode(b), x1)
        self.assertEqual(x0.shape, (12, 9, 9))

    def test_mirror_batch_matches_mirrored_encoding(self):
        from quoridor.ml.encoding import mirror_batch
        for st in STATES[::7]:
            b = FastBoard(st)
            pi = legal_mask(b).astype(np.float32)
            planes, pis = mirror_batch(encode(b)[None], pi[None])
            np.testing.assert_allclose(planes[0], encode(b, mirror=True))
            # mirrored policy = legal mask of the mirrored position
            mirrored_moves = {move_to_action(m, b.turn == 1, True) for m in b.legal_moves()}
            self.assertEqual(set(np.flatnonzero(pis[0])), mirrored_moves)
            self.assertTrue((MIRROR_PERM[MIRROR_PERM] == np.arange(N_ACTIONS)).all())

    def test_wall_gain_planes(self):
        from quoridor import Board
        from quoridor.ml.encoding import wall_gains
        for st in STATES[::6]:
            b = FastBoard(st)
            board = st.to_board()
            for player in (0, 1):
                base = board.shortest_path_length(player)
                for (o, a), g in wall_gains(b, player).items():
                    r, c = divmod(a, NW)
                    tmp = board.copy()
                    tmp.walls.add((r, c, __import__("quoridor").Orientation(o)))
                    self.assertEqual(tmp.shortest_path_length(player) - base, g)
        # Start position: a horizontal wall right in front of the opponent adds a step.
        b = FastBoard.initial()
        x = encode(b)
        self.assertGreater(x[8].sum(), 0)
        self.assertTrue(np.all(x[8:, 8, :] == 0) and np.all(x[8:, :, 8] == 0))
        del Board

    def test_heuristic_value(self):
        from quoridor.ml.encoding import heuristic_value
        b = FastBoard.initial()
        self.assertAlmostEqual(heuristic_value(encode(b)), 0.0)
        b.pawns[0] = 2 * 9 + 4  # Red (to move) needs 2 steps, Blue 8
        self.assertGreater(heuristic_value(encode(b)), 0.9)
        b.turn = 1  # same position from Blue's side: clearly bad
        self.assertLess(heuristic_value(encode(b)), -0.9)

    def test_legal_mask(self):
        b = FastBoard.initial()
        self.assertEqual(int(legal_mask(b).sum()), 3 + 128)


class TestArena(unittest.TestCase):
    def test_openings_are_seeded(self):
        a = random_opening(random.Random(5)).hash
        self.assertEqual(a, random_opening(random.Random(5)).hash)

    def test_match_runs(self):
        r = match(alphabeta_policy(1), alphabeta_policy(2), 2, seed=3, opening_plies=2)
        self.assertEqual(r["wins"] + r["losses"] + r["draws"], 2)

    def test_wilson(self):
        self.assertAlmostEqual(wilson_lower_bound(95, 100), 0.888, places=2)
        self.assertEqual(wilson_lower_bound(0, 0), 0.0)


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestModelAndSearch(unittest.TestCase):
    def setUp(self):
        from quoridor.ml.model import Evaluator, ModelConfig, PolicyValueNet
        torch.manual_seed(0)
        self.net = PolicyValueNet(ModelConfig(16, 1))
        self.ev = Evaluator(self.net)

    def test_mcts_returns_legal_moves(self):
        from quoridor.ml.mcts import agent_move
        rng = np.random.default_rng(0)
        for st in STATES[::9]:
            if st.winner is not None:
                continue
            b = FastBoard(st)
            legal = set(b.legal_moves())
            for sims in (0, 8):
                move, pi = agent_move(self.ev, b, sims, 0.0, rng)
                self.assertIn(move, legal)
                self.assertAlmostEqual(float(pi.sum()), 1.0, places=5)

    def test_mcts_takes_winning_move(self):
        from quoridor.ml.mcts import agent_move
        b = FastBoard.initial()
        b.pawns[0] = 1 * 9 + 2
        move, _ = agent_move(self.ev, b, 16, 0.0, np.random.default_rng(0))
        self.assertEqual(move, ("p", 2))

    def test_ml_player_and_checkpoint(self):
        from quoridor import Board, Game
        from quoridor.bots import AlphaBetaBot
        from quoridor.ml.agent import MLPlayer
        from quoridor.ml.model import save_checkpoint
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.pt"
            save_checkpoint(path, self.net)
            agent = MLPlayer(str(path), sims=4, device="cpu")
            st = Board().snapshot(0)
            move = agent.get_move(st)
            self.assertIsNone(st.to_board().move_error(0, move))
            game = Game(agent, AlphaBetaBot("bot", 1, None), max_invalid_attempts=1)
            self.assertIn(game.run(), (0, 1))

    def test_training_smoke(self):
        from quoridor.ml.train import main
        with tempfile.TemporaryDirectory() as d:
            status = main(["--out", d, "--preset", "smoke", "--workers", "2", "--iterations", "1",
                           "--warmup-games", "2", "--eval-games", "2", "--confirm-games", "2",
                           "--target-win-rate", "0.0"])
            # target 0% => first evaluation reaches it, confirmation passes, milestone is written
            self.assertEqual(status, "target_reached")
            self.assertTrue((Path(d) / "milestone_0.pt").exists())
            self.assertTrue((Path(d) / "best.pt").exists())
            self.assertTrue((Path(d) / "metrics.jsonl").read_text().count('"kind": "milestone"'))


if __name__ == "__main__":
    unittest.main()
