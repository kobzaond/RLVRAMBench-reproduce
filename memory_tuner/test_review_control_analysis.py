import math
import unittest

from memory_tuner.review_control_analysis import PHASES, contrasts


class ControlContrastTests(unittest.TestCase):
    def rows(self):
        result = []
        for i, (condition, peak) in enumerate((
                ("external_b8", 1000), ("external_b16", 5000),
                ("full_b8", 1100), ("full_b16", 5200)), 1):
            result.append({
                "pair_id": "block", "model_family": "model", "dataset": "data",
                "training_seed": 121, "job_id": "123", "condition": condition,
                "period": i, "success": 1, "safe": 1,
                **{f"{phase}_peak_mib": peak for phase in PHASES},
            })
        return result

    def test_difference_of_differences(self):
        block = contrasts(self.rows())[0]
        self.assertEqual(block["actor_update_external_batch_effect_mib"], 4000)
        self.assertEqual(block["actor_update_full_batch_effect_mib"], 4100)
        self.assertEqual(block["actor_update_instrumentation_interaction_mib"], 100)

    def test_failure_is_retained_when_stage_observed(self):
        rows = self.rows()
        rows[-1]["success"] = rows[-1]["safe"] = 0
        block = contrasts(rows)[0]
        self.assertEqual(block["completed"], 3)
        self.assertEqual(block["actor_update_full_batch_effect_mib"], 4100)

    def test_missing_stage_is_not_imputed(self):
        rows = self.rows()
        rows[-1]["actor_update_peak_mib"] = math.nan
        self.assertTrue(math.isnan(contrasts(rows)[0]["actor_update_full_batch_effect_mib"]))

    def test_incomplete_block_rejected(self):
        with self.assertRaises(ValueError):
            contrasts(self.rows()[:-1])
