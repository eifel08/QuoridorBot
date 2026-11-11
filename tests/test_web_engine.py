"""Checks the browser engine (web/engine.js) against the Python engine."""

import json
import random
import shutil
import subprocess
import unittest
from pathlib import Path

from quoridor import Board, PawnMove, WallMove

ROOT = Path(__file__).resolve().parent.parent
NODE = shutil.which("node")


def random_positions(n_games=12, seed=1234):
    rng = random.Random(seed)
    states = []
    for _ in range(n_games):
        board, player = Board(), 0
        for move_no in range(rng.randint(5, 60)):
            walls = board.valid_walls(player)
            if walls and rng.random() < 0.45:
                r, c, o = rng.choice(walls)
                board.apply_move(player, WallMove(r, c, o))
            else:
                board.apply_move(player, PawnMove(*rng.choice(board.valid_pawn_moves(player))))
            if board.has_won(player):
                break
            player = 1 - player
            states.append(board.snapshot(player, move_no + 1).to_dict())
    return states


def normalise(d):
    d = dict(d)
    for k in ("valid_pawn_moves",):
        d[k] = sorted(map(tuple, d[k]))
    for k in ("walls", "valid_walls"):
        d[k] = sorted(map(tuple, d[k]))
    return d


@unittest.skipIf(NODE is None, "node not installed")
class TestWebEngineParity(unittest.TestCase):
    def test_matches_python(self):
        states = random_positions()
        self.assertGreater(len(states), 150)
        out = subprocess.run(
            [NODE, str(ROOT / "tests" / "js" / "snapshots.mjs")],
            input=json.dumps(states), capture_output=True, text=True, check=True,
        )
        js_states = json.loads(out.stdout)
        self.assertEqual(len(js_states), len(states))
        for py, js in zip(states, js_states):
            self.assertEqual(normalise(js), normalise(py))


if __name__ == "__main__":
    unittest.main()
