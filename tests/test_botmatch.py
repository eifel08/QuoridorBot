"""Bot specs and headless Bot vs Bot matches."""

import json
import tempfile
import unittest
from pathlib import Path

from quoridor import Game
from quoridor.bots import AlphaBetaBot, BotSpec, make_player, parse_bot_spec
from quoridor.botmatch import main as botmatch_main
from quoridor.botmatch import run_match


class TestBotSpec(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_bot_spec("ab"), BotSpec())
        self.assertEqual(parse_bot_spec("ab:depth=2"), BotSpec(depth=2))
        self.assertEqual(parse_bot_spec("alphabeta:depth=8,time=2,random=1"),
                         BotSpec(depth=8, time_limit=2.0, random_ties=True))
        self.assertEqual(parse_bot_spec("ml:model=x.pt,sims=0"), BotSpec("ml", model="x.pt", sims=0))

    def test_round_trip(self):
        for text in ("ab:depth=3", "ab:depth=6,time=1.5,random=1", "ml:model=m.pt,sims=32"):
            self.assertEqual(parse_bot_spec(parse_bot_spec(text).to_string()), parse_bot_spec(text))

    def test_errors(self):
        for bad in ("xyz", "ab:depht=3", "ab:depth=0", "ab:depth=31", "ab:depth", "ml", "ml:sims=3"):
            with self.assertRaises(ValueError, msg=bad):
                parse_bot_spec(bad)

    def test_make_player(self):
        bot = make_player(BotSpec(depth=3, time_limit=1.0, random_ties=True), "x")
        self.assertIsInstance(bot, AlphaBetaBot)
        self.assertEqual((bot.name, bot.depth, bot.time_limit), ("x", 3, 1.0))
        self.assertIsNotNone(bot.rng)
        game = Game(make_player(BotSpec(depth=1)), make_player(BotSpec(depth=2)), max_invalid_attempts=1)
        self.assertIn(game.run(), (0, 1))


class TestBotMatch(unittest.TestCase):
    def test_run_match_counts(self):
        summary, records = run_match(BotSpec(depth=1), BotSpec(depth=2), games=6, seed=3,
                                     opening_plies=3, workers=2, progress=False)
        self.assertEqual(summary["games"], 6)
        self.assertEqual(summary["bot1"]["wins"] + summary["bot2"]["wins"] + summary["draws"], 6)
        self.assertEqual([r.bot1_side for r in records], [0, 1, 0, 1, 0, 1])  # colours alternate
        for r in records:
            self.assertIn(r.winner_bot, (1, 2, None))
            self.assertGreater(sum(r.moves), 0)

    def test_same_seed_same_games(self):
        a, ra = run_match(BotSpec(depth=1), BotSpec(depth=2), 4, seed=11, workers=1, progress=False)
        b, rb = run_match(BotSpec(depth=1), BotSpec(depth=2), 4, seed=11, workers=1, progress=False)
        self.assertEqual([(r.winner_bot, r.plies) for r in ra], [(r.winner_bot, r.plies) for r in rb])

    def test_no_swap(self):
        _, records = run_match(BotSpec(depth=1), BotSpec(depth=1), 3, swap=False, workers=1, progress=False)
        self.assertTrue(all(r.bot1_side == 0 for r in records))

    def test_cli_writes_files(self):
        with tempfile.TemporaryDirectory() as d:
            js, cs = Path(d) / "r.json", Path(d) / "r.csv"
            summary = botmatch_main(["--bot1", "ab:depth=1", "--bot2", "ab:depth=1", "--games", "2",
                                     "--workers", "1", "--json", str(js), "--csv", str(cs)])
            self.assertEqual(json.loads(js.read_text())["summary"]["games"], 2)
            self.assertEqual(len(cs.read_text().strip().splitlines()), 3)
            self.assertEqual(summary["bot1"]["spec"], "ab:depth=1")


if __name__ == "__main__":
    unittest.main()
