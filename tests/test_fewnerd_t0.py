"""Checkpoint-selection precedence and frozen schedule boundaries."""

import json
import tempfile
import unittest
from pathlib import Path

from ncm_graphrag.ncm.fewnerd_t0 import FIXED_RECIPE, load_protocol, select_checkpoint


def checkpoint(step, f1, exact):
    return {"step": step, "tuning": {"entity_f1": f1, "target_exact": exact}}


class T0SelectionTests(unittest.TestCase):
    def test_f1_wins_then_earlier_checkpoint_not_target_exact(self):
        rows = [checkpoint(6000, 12, 100), checkpoint(1500, 10, 95), checkpoint(3000, 12, 20)]
        self.assertEqual(select_checkpoint(rows)["step"], 3000)
        rows[0]["tuning"]["entity_f1"] = 13
        self.assertEqual(select_checkpoint(rows)["step"], 6000)

    def test_zero_quality_still_selects_earliest_without_faking_a_pass(self):
        rows = [checkpoint(6000, 0, 0), checkpoint(3000, 0, 0), checkpoint(1500, 0, 0)]
        self.assertEqual(select_checkpoint(rows)["step"], 1500)
        self.assertEqual(select_checkpoint(rows)["tuning"]["entity_f1"], 0)

    def test_selection_rejects_missing_duplicate_or_unregistered_points(self):
        for steps in ((1500, 3000), (1500, 3000, 3000), (1500, 3000, 4500, 6000)):
            with self.subTest(steps=steps), self.assertRaisesRegex(ValueError, "predeclared"):
                select_checkpoint([checkpoint(step, 10, 10) for step in steps])

    def test_invalid_quality_scores_cannot_silently_win_or_lose(self):
        for invalid in (float("nan"), float("inf"), -0.1, 100.1):
            with self.subTest(score=invalid), self.assertRaisesRegex(ValueError, "Invalid tuning"):
                select_checkpoint([checkpoint(1500, 10, 10), checkpoint(3000, invalid, 10), checkpoint(6000, 20, 20)])

    def test_schedule_change_is_rejected_before_inputs_or_training(self):
        for field, value in (("steps", 6001), ("batch_size", 32), ("evaluation_steps", [1500, 3000, 4500, 6000])):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "protocol.json"
                path.write_text(json.dumps({**FIXED_RECIPE, field: value}))
                with self.assertRaisesRegex(ValueError, f"fixed protocol mismatch: {field}"):
                    load_protocol(path)


if __name__ == "__main__":
    unittest.main()
