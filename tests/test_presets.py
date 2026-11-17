"""Bot presets offered by the Auto-Play bot selector."""

import tempfile
import unittest
from pathlib import Path

from quoridor.bots.factory import BotSpec, parse_bot_spec
from quoridor.bots.presets import BOT_PRESETS, find_checkpoints, label_for, preset_choices

DEFAULT_SPEC = BotSpec(depth=30, time_limit=1.0)  # quoridor.vision.analysis.DEFAULT_SPEC (needs OpenCV)


class TestPresets(unittest.TestCase):
    def test_every_preset_parses_and_round_trips(self):
        for label, text in BOT_PRESETS:
            spec = parse_bot_spec(text)
            self.assertEqual(parse_bot_spec(spec.to_string()), spec, label)

    def test_default_is_the_first_preset(self):
        choices = preset_choices(DEFAULT_SPEC, include_ml=False)
        self.assertEqual(choices[0][1], DEFAULT_SPEC)
        self.assertEqual(label_for(DEFAULT_SPEC, choices), BOT_PRESETS[0][0])

    def test_command_line_spec_is_offered_first(self):
        spec = BotSpec(depth=12, time_limit=2.0)
        choices = preset_choices(spec, include_ml=False)
        self.assertEqual(choices[0][1], spec)
        self.assertTrue(choices[0][0].startswith("From command line"))
        self.assertEqual(len(choices), len(BOT_PRESETS) + 1)

    def test_unknown_spec_gets_a_custom_label(self):
        spec = BotSpec(depth=7)
        self.assertTrue(label_for(spec, preset_choices(include_ml=False)).startswith("Custom"))

    def test_checkpoints_are_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "runs" / "exp1"
            run.mkdir(parents=True)
            (run / "best.pt").write_bytes(b"")
            self.assertEqual([p.name for p in find_checkpoints(root=Path(tmp))], ["best.pt"])


if __name__ == "__main__":
    unittest.main()
