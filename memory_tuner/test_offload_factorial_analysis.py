"""Synthetic .45/.70 panel tests; no jobs, artifacts, or historical inputs."""

from collections import Counter
from copy import deepcopy
import csv
import io
from itertools import product
import json
import random
import unittest
from unittest.mock import patch

from memory_tuner.offload_factorial_analysis import analyse, EFFECTS, METRICS, PLANNED_SEEDS


METHODS = (
    "frozen_logistic", "component_regression", "offload_only", "strict_c1",
    "reservation_only", "offload_or_low_reservation",
)


def panel(models=("family-a-7b", "family-b-7b"), datasets=("math", "code", "gsm8k"),
          reservations=(.45, .7)):
    rows, slots, predictions = [], [], []
    for model, dataset in product(models, datasets):
        case = f"{model}/{dataset}"
        for seed in PLANNED_SEEDS:
            conditions = list(product((False, True), (4, 8), reservations))
            random.Random(seed).shuffle(conditions)
            for period, (offload, batch, reservation) in enumerate(conditions, 1):
                condition = f"offload-{offload}-batch-{batch}-reserve-{reservation}"
                configuration = f"{case}/{condition}"
                row = dict(
                    block_id=f"{case}/{seed}", experiment_id=f"{configuration}/{seed}",
                    configuration_id=configuration, case_id=case, model=model, dataset=dataset,
                    training_seed=str(seed), condition=condition, parameter_offload=str(offload),
                    actor_micro_batch=str(batch), vllm_gpu_memory_utilization=str(reservation), period=str(period),
                )
                rows.append(row)
                slots.append(dict(row, status="completed", completed_peak_mib=10000,
                                  stage_actor_peak_mib=9000, stage_generation_peak_mib=8000,
                                  stage_weight_sync_peak_mib=7000))
                if seed == PLANNED_SEEDS[0]:
                    for method in METHODS:
                        three = method in ("frozen_logistic", "component_regression")
                        approved = {
                            "frozen_logistic": True,
                            "component_regression": True,
                            "offload_only": offload,
                            "strict_c1": offload and batch == 4 and reservation == .45,
                            "reservation_only": reservation == .45,
                            "offload_or_low_reservation": offload or reservation == .45,
                        }[method]
                        predictions.append(dict(configuration_id=configuration, method=method,
                                                prediction_type="three_state" if three else "admission_only",
                                                approved=approved, predicted_state="within_margin" if three else None))
    return rows, slots, predictions


def score(result, method="frozen_logistic", stratum="overall", value="all"):
    return next(r for r in result["prediction_scores"]
                if r["method"] == method and r["stratum"] == stratum and r["stratum_value"] == value)


def config(result, configuration_id):
    return next(r for r in result["configurations"] if r["configuration_id"] == configuration_id)


def effect(result, metric="completed_peak_mib", name="offload"):
    return next(r for r in result["summary"]["block_effects"] if r["metric"] == metric and r["effect"] == name)


def select(slots, offload=True, batch=4, reservation=.45, case_id=None):
    case_id = case_id or slots[0]["case_id"]
    return sorted((s for s in slots if s["case_id"] == case_id
                   and s["parameter_offload"] == str(offload)
                   and s["actor_micro_batch"] == str(batch)
                   and float(s["vllm_gpu_memory_utilization"]) == reservation),
                  key=lambda s: s["training_seed"])


def noncompletion(slot, status):
    slot.update(status=status, completed_peak_mib=None)


def disagreement_panel():
    """Synthetic frozen preview: OR approves 36; classifier rejects seven of them."""
    rows, slots, predictions = panel()
    rejected = sorted({
        r["configuration_id"] for r in rows
        if r["parameter_offload"] == "True" and float(r["vllm_gpu_memory_utilization"]) == .7
        and (r["model"] == "family-a-7b" or (r["dataset"] == "gsm8k" and r["actor_micro_batch"] == "4"))
    })
    combined = {p["configuration_id"]: p["approved"] for p in predictions
                if p["method"] == "offload_or_low_reservation"}
    for prediction in predictions:
        if prediction["method"] == "frozen_logistic":
            approved = combined[prediction["configuration_id"]] and prediction["configuration_id"] not in rejected
            prediction.update(approved=approved, predicted_state="within_margin" if approved else "above_margin")
    return rows, slots, predictions, rejected


class OffloadFactorialAnalysisTests(unittest.TestCase):
    def test_full_panel_denominators_and_strata_are_configurations_not_seeds(self):
        result = analyse(*panel())
        summary = result["summary"]
        self.assertEqual(len(result["configurations"]), 48)
        self.assertEqual(summary["planned_slots"], 144)
        self.assertEqual(summary["planned_cases"], 6)
        self.assertEqual(summary["planned_blocks"], 18)
        self.assertTrue(summary["full_panel"])
        self.assertEqual(summary["panel_groups"], {"anchors": 6, "challenge": 42})
        self.assertEqual(len(result["offload_pairs"]), 72)
        self.assertEqual(len(result["block_contrasts"]), 18 * 7 * 5)
        self.assertEqual(len(result["prediction_scores"]), 6 * 7)
        self.assertEqual(summary["pair_denominators"]["both_eligible"], 72)
        self.assertEqual(summary["transition_counts"]["within_margin_to_within_margin"], 72)
        self.assertEqual(score(result)["useful_approvals"], 48)
        self.assertEqual(score(result)["resolved_configurations"], 48)
        self.assertEqual(score(result, stratum="model", value="family-a-7b")["planned_configurations"], 24)
        self.assertEqual(score(result, stratum="parameter_offload", value="True")["planned_configurations"], 24)
        self.assertEqual(score(result, stratum="panel_group", value="challenge")["planned_configurations"], 42)
        self.assertEqual(score(result, stratum="panel_group", value="anchors")["planned_configurations"], 6)

    def test_one_missing_execution_seed_remains_unresolved(self):
        rows, slots, predictions = panel()
        selected = select(slots)
        noncompletion(selected[2], "not_launched")
        result = analyse(rows, slots, predictions)
        observed = config(result, selected[0]["configuration_id"])
        self.assertEqual(observed["observed_state"], "unresolved")
        self.assertEqual(observed["eligible_seeds"], "241;242")
        self.assertEqual(observed["unresolved_seeds"], "243")
        self.assertEqual(observed["eligible_seed_count"], 2)
        self.assertFalse(observed["all_three_seeds_eligible"])
        self.assertIsNone(observed["completed_peak_mib"])
        self.assertEqual(observed["maximum_known_completed_peak_mib"], 10000)
        self.assertEqual(score(result)["resolved_configurations"], 47)
        self.assertEqual(score(result)["all_approved"], 48)
        self.assertEqual(score(result)["unresolved_approved"], 1)
        self.assertEqual(score(result)["unresolved_approvals_with_known_unsuitable"], 0)
        unresolved_pair = next(p for p in result["offload_pairs"] if p["on_experiment_id"] == selected[2]["experiment_id"])
        self.assertTrue(unresolved_pair["unresolved"])
        self.assertIsNone(unresolved_pair["categorical_transition"])
        self.assertIsNone(unresolved_pair["within_margin_indicator_on_minus_off"])
        self.assertEqual(result["summary"]["pair_denominators"]["planned"], 72)

    def test_missing_slot_or_missing_planned_seed_is_rejected(self):
        rows, slots, predictions = panel()
        with self.assertRaisesRegex(ValueError, "one slot"):
            analyse(rows, slots[:-1], predictions)
        missing_seed = rows[-1]["experiment_id"]
        with self.assertRaisesRegex(ValueError, "three distinct planned seeds"):
            analyse([r for r in rows if r["experiment_id"] != missing_seed],
                    [s for s in slots if s["experiment_id"] != missing_seed], predictions)

    def test_maximum_seed_peak_and_inclusive_margin_boundary(self):
        rows, slots, predictions = panel()
        selected = select(slots)
        selected[0]["completed_peak_mib"] = 38912
        result = analyse(rows, slots, predictions)
        observed = config(result, selected[0]["configuration_id"])
        self.assertEqual(observed["observed_state"], "within_margin")
        self.assertEqual(observed["completed_peak_mib"], 38912)
        selected[1]["completed_peak_mib"] = 38912.1
        result = analyse(rows, slots, predictions)
        self.assertEqual(config(result, selected[0]["configuration_id"])["observed_state"], "above_margin")
        self.assertEqual(score(result)["above_margin_approvals"], 1)
        self.assertEqual(score(result)["useful_approvals"], 47)
        self.assertEqual(score(result)["memory_failure_approvals"], 0)

    def test_unresolved_known_above_is_not_scored_as_safe_or_fully_observed(self):
        rows, slots, predictions = panel()
        selected = select(slots)
        selected[0]["completed_peak_mib"] = 39000
        noncompletion(selected[2], "infrastructure_invalid")
        result = analyse(rows, slots, predictions)
        observed = config(result, selected[0]["configuration_id"])
        self.assertEqual(observed["observed_state"], "unresolved")
        self.assertTrue(observed["known_margin_exceedance"])
        self.assertFalse(observed["known_memory_failure"])
        self.assertTrue(observed["unresolved_known_unsuitable"])
        self.assertIsNone(observed["completed_peak_mib"])
        self.assertEqual(observed["maximum_known_completed_peak_mib"], 39000)
        metrics = score(result)
        self.assertEqual(metrics["above_margin_approvals"], 0)
        self.assertEqual(metrics["unresolved_approvals_with_known_margin_exceedance"], 1)
        self.assertEqual(metrics["unresolved_approvals_with_known_unsuitable"], 1)
        self.assertEqual(metrics["three_state_scored"], 47)

    def test_failure_is_categorical_and_never_gets_a_fabricated_peak(self):
        rows, slots, predictions = panel()
        selected = select(slots)
        noncompletion(selected[0], "memory_failure")
        selected[0]["stage_actor_peak_mib"] = 39500  # Partial telemetry cannot supply a completed peak.
        selected[1]["completed_peak_mib"] = 39000
        result = analyse(rows, slots, predictions)
        observed = config(result, selected[0]["configuration_id"])
        self.assertEqual(observed["observed_state"], "memory_failure")
        self.assertTrue(observed["known_memory_failure"])
        self.assertTrue(observed["known_margin_exceedance"])
        self.assertIsNone(observed["completed_peak_mib"])
        self.assertEqual(observed["maximum_known_completed_peak_mib"], 39000)
        self.assertEqual(score(result)["memory_failure_approvals"], 1)
        self.assertEqual(score(result)["above_margin_approvals"], 0)
        pair = next(p for p in result["offload_pairs"] if p["on_experiment_id"] == selected[0]["experiment_id"])
        self.assertEqual(pair["categorical_transition"], "within_margin_to_memory_failure")
        self.assertEqual(pair["within_margin_indicator_on_minus_off"], -1)
        self.assertTrue(pair["both_eligible"])
        self.assertFalse(pair["both_completed"])
        for metric in METRICS[1:]:
            self.assertIsNone(pair[f"{metric}_on_minus_off"])
        block = [r for r in result["block_contrasts"] if r["block_id"] == selected[0]["block_id"]]
        self.assertEqual(sum(r["contrast"] is not None for r in block), 7)
        self.assertTrue(all(r["contrast"] is not None for r in block if r["metric"] == "within_margin_indicator"))

    def test_partial_failure_and_margin_flags_do_not_double_count_unsuitable_approvals(self):
        rows, slots, predictions = panel()
        selected = select(slots)
        noncompletion(selected[0], "memory_failure")
        selected[1]["completed_peak_mib"] = 39000
        noncompletion(selected[2], "not_launched")
        result = analyse(rows, slots, predictions)
        self.assertEqual(config(result, selected[0]["configuration_id"])["observed_state"], "unresolved")
        metrics = score(result)
        self.assertEqual(metrics["unresolved_approvals_with_known_memory_failure"], 1)
        self.assertEqual(metrics["unresolved_approvals_with_known_margin_exceedance"], 1)
        self.assertEqual(metrics["unresolved_approvals_with_known_unsuitable"], 1)
        self.assertEqual(metrics["memory_failure_approvals"], 0)
        self.assertEqual(metrics["resolved_approved"], 47)

    def test_offload_only_and_strict_c1_disagree_on_novel_conditions(self):
        rows, slots, predictions = panel()
        result = analyse(rows, slots, predictions)
        selected = select(slots, batch=8)
        observed = config(result, selected[0]["configuration_id"])
        self.assertEqual(observed["panel_group"], "challenge")
        self.assertIsNone(observed["anchor_condition"])
        self.assertTrue(observed["offload_only_approved"])
        self.assertFalse(observed["strict_c1_approved"])
        self.assertEqual(score(result, "offload_only")["useful_approvals"], 24)
        self.assertEqual(score(result, "strict_c1")["useful_approvals"], 6)
        self.assertEqual(score(result, "strict_c1")["within_margin_rejected"], 42)
        self.assertEqual(score(result, "reservation_only")["useful_approvals"], 24)
        self.assertEqual(score(result, "offload_or_low_reservation")["useful_approvals"], 36)
        self.assertIsNone(score(result, "offload_only")["three_state_accuracy"])
        self.assertIsNone(score(result, "offload_only")["correct_three_state"])
        self.assertEqual(score(result, "offload_only")["three_state_scored"], 0)
        # Classification of anchors depends on exact settings, not condition naming.
        anchors = [r for r in result["configurations"] if r["panel_group"] == "anchors"]
        self.assertEqual(Counter(r["anchor_condition"] for r in anchors), {"c1": 6})

    def test_sixth_or_rule_is_scored_from_input_with_strict_unresolved_accounting(self):
        rows, slots, predictions = panel()
        # OR approves both on/high and off/low; the one-factor rules disagree.
        high_on = select(slots, batch=8, reservation=.7)
        low_off = select(slots, offload=False)
        noncompletion(high_on[0], "memory_failure")
        noncompletion(high_on[2], "not_launched")
        low_off[0]["completed_peak_mib"] = 39000
        result = analyse(rows, slots, predictions)
        metrics = score(result, "offload_or_low_reservation")
        self.assertEqual(metrics["all_approved"], 36)
        self.assertEqual(metrics["unresolved_approved"], 1)
        self.assertEqual(metrics["resolved_approved"], 35)
        self.assertEqual(metrics["unresolved_approvals_with_known_memory_failure"], 1)
        self.assertEqual(metrics["memory_failure_approvals"], 0)
        self.assertEqual(metrics["above_margin_approvals"], 1)
        self.assertEqual(metrics["useful_approvals"], 34)
        self.assertEqual(score(result, "strict_c1", "panel_group", "challenge")["all_approved"], 0)
        self.assertEqual(score(result, "strict_c1")["all_approved"], 6)

    def test_predictions_remain_supplied_decisions_when_targets_change(self):
        rows, slots, predictions = panel()
        before = deepcopy(predictions)
        first = analyse(rows, slots, predictions)
        for slot in slots:
            noncompletion(slot, "memory_failure")
        second = analyse(rows, slots, predictions)
        self.assertEqual(predictions, before)
        for method in METHODS:
            self.assertEqual(score(first, method)["all_approved"], score(second, method)["all_approved"])
            self.assertEqual(score(second, method)["useful_approvals"], 0)
            self.assertEqual(score(second, method)["memory_failure_approvals"], score(second, method)["all_approved"])
        self.assertEqual(second["summary"]["transition_counts"]["memory_failure_to_memory_failure"], 72)
        self.assertTrue(all(p["within_margin_indicator_on_minus_off"] == 0 for p in second["offload_pairs"]))

    def test_three_state_confusion_and_binary_decisions_use_resolved_configs(self):
        rows, slots, predictions = panel()
        selected = select(slots)
        cid = selected[0]["configuration_id"]
        selected[0]["completed_peak_mib"] = 39000
        for prediction in predictions:
            if prediction["configuration_id"] == cid and prediction["method"] == "frozen_logistic":
                prediction.update(predicted_state="memory_failure", approved=False)
        result = analyse(rows, slots, predictions)
        metrics = score(result)
        self.assertEqual(metrics["predicted_memory_failure_observed_above_margin"], 1)
        self.assertEqual(metrics["correct_three_state"], 47)
        self.assertEqual(metrics["three_state_accuracy"], 47 / 48)
        self.assertEqual(metrics["admission_accuracy"], 1)

    def test_pairs_match_block_and_settings_despite_period_and_input_order(self):
        rows, slots, predictions = panel()
        for slot in slots:
            offset = 100 * (int(slot["training_seed"]) - 241) + int(slot["actor_micro_batch"])
            slot["completed_peak_mib"] = 10000 + offset + (17 if slot["parameter_offload"] == "True" else 0)
        result = analyse(rows, slots, predictions)
        self.assertTrue(all(p["completed_peak_mib_on_minus_off"] == 17 for p in result["offload_pairs"]))
        self.assertTrue(any(p["on_period"] < p["off_period"] for p in result["offload_pairs"]))
        self.assertTrue(any(p["on_period"] > p["off_period"] for p in result["offload_pairs"]))
        random.Random(123).shuffle(rows)
        random.Random(456).shuffle(slots)
        random.Random(789).shuffle(predictions)
        self.assertEqual(analyse(rows, slots, predictions), result)

    def test_all_seven_factorial_contrasts_have_documented_scaling(self):
        rows, slots, predictions = panel(models=("family-a-7b",), datasets=("math",))
        for slot in slots:
            o = slot["parameter_offload"] == "True"
            b = slot["actor_micro_batch"] == "8"
            r = float(slot["vllm_gpu_memory_utilization"]) == .7
            peak = 10000 + 3 * o + 10 * b + 100 * r + 20 * o * b + 40 * o * r + 60 * b * r + 80 * o * b * r
            slot.update(completed_peak_mib=peak, stage_actor_peak_mib=peak - 1000,
                        stage_generation_peak_mib=peak - 2000, stage_weight_sync_peak_mib=peak - 3000)
        result = analyse(rows, slots, predictions)
        expected = dict(
            offload=53, actor_micro_batch=70, reservation=170,
            offload_x_actor_micro_batch=60, offload_x_reservation=80,
            actor_micro_batch_x_reservation=100, offload_x_actor_micro_batch_x_reservation=80,
        )
        for row in result["block_contrasts"]:
            with self.subTest(metric=row["metric"], effect=row["effect"]):
                self.assertEqual(row["contrast"], 0 if row["metric"] == "within_margin_indicator" else expected[row["effect"]])
                self.assertEqual(row["contrast_divisor"], 2 ** (3 - row["interaction_order"]))
                self.assertEqual(row["available_measurements"], 8)
        self.assertEqual(effect(result)["equal_case_mean"], 53)
        self.assertEqual(effect(result)["case_contributions"][0]["within_case_min"], 53)
        self.assertEqual(effect(result)["case_contributions"][0]["within_case_max"], 53)

    def test_reservation_levels_are_derived_from_rows_and_anchors_remain_exact(self):
        for high in (.6, .7):
            with self.subTest(high=high):
                rows, slots, predictions = panel(reservations=(.45, high))
                for slot in slots:
                    slot["completed_peak_mib"] = 10000 + (
                        100 if float(slot["vllm_gpu_memory_utilization"]) == high else 0)
                result = analyse(rows, slots, predictions)
                self.assertEqual(result["summary"]["factor_levels"]["reservation"], [.45, high])
                self.assertEqual(effect(result, name="reservation")["equal_case_mean"], 100)
                self.assertEqual(len(result["offload_pairs"]), 72)
                self.assertEqual(result["summary"]["panel_groups"],
                                 {"anchors": 12, "challenge": 36} if high == .6 else
                                 {"anchors": 6, "challenge": 42})
                self.assertTrue(any(p["vllm_gpu_memory_utilization"] == high for p in result["offload_pairs"]))

    def test_stage_missingness_only_suppresses_relevant_completed_contrasts(self):
        rows, slots, predictions = panel()
        selected = select(slots)
        selected[0]["stage_actor_peak_mib"] = None
        result = analyse(rows, slots, predictions)
        block = [r for r in result["block_contrasts"] if r["block_id"] == selected[0]["block_id"]]
        actor = [r for r in block if r["metric"] == "stage_actor_peak_mib"]
        self.assertEqual(len(actor), 7)
        self.assertTrue(all(r["contrast"] is None and r["available_measurements"] == 7 for r in actor))
        self.assertTrue(all(r["contrast"] is not None for r in block if r["metric"] != "stage_actor_peak_mib"))
        self.assertTrue(all(r["all_eight_completed"] for r in block))
        pair = next(p for p in result["offload_pairs"] if p["on_experiment_id"] == selected[0]["experiment_id"])
        self.assertIsNone(pair["stage_actor_peak_mib_on_minus_off"])
        self.assertEqual(pair["completed_peak_mib_on_minus_off"], 0)
        self.assertEqual(result["summary"]["pair_denominators"]["available_measurements"]["stage_actor_peak_mib"], 71)
        self.assertEqual(effect(result, "stage_actor_peak_mib")["contributing_blocks"], 17)
        self.assertEqual(effect(result)["contributing_blocks"], 18)

    def test_equal_case_means_do_not_weight_cases_by_completed_block_counts(self):
        rows, slots, predictions = panel(models=("family-a-7b",), datasets=("math", "code"))
        for slot in slots:
            slot["completed_peak_mib"] = 10000 + ((10 if slot["dataset"] == "math" else 30)
                                                if slot["parameter_offload"] == "True" else 0)
            if slot["dataset"] == "code" and slot["training_seed"] != "241":
                noncompletion(slot, "infrastructure_invalid")
        result = analyse(rows, slots, predictions)
        summary = effect(result)
        self.assertEqual(summary["equal_case_mean"], 20)
        self.assertEqual(summary["available_case_mean"], 20)
        self.assertEqual(summary["contributing_blocks"], 4)
        self.assertEqual(summary["planned_blocks"], 6)
        self.assertTrue(summary["every_case_contributes"])
        self.assertFalse(summary["all_planned_measurements_available"])
        contributions = {r["dataset"]: r for r in summary["case_contributions"]}
        self.assertEqual(contributions["math"]["contributing_blocks"], 3)
        self.assertEqual(contributions["code"]["contributing_blocks"], 1)
        paired = next(r for r in result["summary"]["offload_effects"]
                      if r["stratum"] == "overall" and r["metric"] == "completed_peak_mib")
        self.assertEqual(paired["equal_case_mean"], 20)
        self.assertEqual(paired["available_measurements"], 16)
        self.assertTrue(paired["conditional_on_completion"])
        # An entirely unobserved case cannot disappear from the primary mean.
        for slot in slots:
            if slot["dataset"] == "code":
                noncompletion(slot, "not_launched")
        summary = effect(analyse(rows, slots, predictions))
        self.assertIsNone(summary["equal_case_mean"])
        self.assertEqual(summary["available_case_mean"], 10)
        self.assertFalse(summary["every_case_contributes"])
        self.assertEqual(len(summary["case_contributions"]), 2)
        missing = next(r for r in summary["case_contributions"] if r["dataset"] == "code")
        self.assertIsNone(missing["mean"])
        self.assertIsNone(missing["within_case_min"])
        self.assertEqual(missing["contributing_blocks"], 0)
        self.assertEqual(missing["planned_blocks"], 3)

    def test_within_case_ranges_reflect_the_observed_seed_effects(self):
        rows, slots, predictions = panel(models=("family-a-7b",), datasets=("math",))
        for slot in slots:
            slot["completed_peak_mib"] = 10000 + (
                10 * (int(slot["training_seed"]) - 240) if slot["parameter_offload"] == "True" else 0)
        summary = effect(analyse(rows, slots, predictions))
        self.assertEqual(summary["equal_case_mean"], 20)
        contribution = summary["case_contributions"][0]
        self.assertEqual(contribution["within_case_min"], 10)
        self.assertEqual(contribution["within_case_max"], 30)
        self.assertEqual(contribution["contributing_blocks"], 3)

    def test_pooled_continuous_effect_can_have_every_case_but_no_high_reservation_strata(self):
        rows, slots, predictions = panel()
        for slot in slots:
            if float(slot["vllm_gpu_memory_utilization"]) == .7:
                noncompletion(slot, "memory_failure")
            else:
                difference = 10 if slot["actor_micro_batch"] == "4" else 20
                slot["completed_peak_mib"] = 10000 + (difference if slot["parameter_offload"] == "True" else 0)
        result = analyse(rows, slots, predictions)
        effects = [r for r in result["summary"]["offload_effects"] if r["metric"] == "completed_peak_mib"]
        pooled = next(r for r in effects if r["stratum"] == "overall")
        self.assertTrue(pooled["every_case_contributes"])
        self.assertEqual(pooled["contributing_cases"], 6)
        self.assertEqual(pooled["planned_measurements"], 72)
        self.assertEqual(pooled["available_measurements"], 36)
        self.assertFalse(pooled["all_planned_measurements_available"])
        self.assertEqual(pooled["equal_case_mean"], 15)
        self.assertEqual(pooled["comparison_role"], "descriptive_pooled")
        self.assertEqual(pooled["aggregation_scope"], "observed_matched_completions")
        self.assertIn("not a full-grid effect", pooled["scope_note"])
        for stratum in (r for r in effects if r["stratum"] == "batch_reservation"):
            self.assertEqual(stratum["comparison_role"], "primary_fixed_batch_reservation")
            if stratum["vllm_gpu_memory_utilization"] == .7:
                self.assertEqual(stratum["planned_measurements"], 18)
                self.assertEqual(stratum["available_measurements"], 0)
                self.assertIsNone(stratum["equal_case_mean"])
                self.assertIsNone(stratum["available_case_mean"])
                self.assertTrue(all(r["mean"] is None for r in stratum["case_contributions"]))
            else:
                self.assertEqual(stratum["equal_case_mean"], 10 if stratum["actor_micro_batch"] == 4 else 20)
        self.assertIsNone(effect(result)["equal_case_mean"])  # No block has eight completions.
        self.assertEqual(result["summary"]["pair_denominators"]["both_eligible"], 72)

    def test_within_margin_indicator_is_completion_and_margin_compliance(self):
        rows, slots, predictions = panel()
        selected = select(slots)
        selected[0]["completed_peak_mib"] = 39000
        result = analyse(rows, slots, predictions)
        pair = next(p for p in result["offload_pairs"] if p["on_experiment_id"] == selected[0]["experiment_id"])
        self.assertTrue(pair["both_completed"])
        self.assertEqual(pair["within_margin_indicator_on_minus_off"], -1)
        self.assertEqual(result["summary"]["metric_definitions"]["within_margin_indicator"]["event"],
                         "completion_and_margin_compliance")
        indicator = next(r for r in result["summary"]["offload_effects"]
                         if r["stratum"] == "overall" and r["metric"] == "within_margin_indicator")
        self.assertEqual(indicator["aggregation_scope"], "observed_eligible_pairs")
        self.assertFalse(indicator["conditional_on_completion"])

    def test_zero_resolved_denominator_preserves_all_planned_rows(self):
        rows, slots, predictions = panel()
        for slot in slots:
            noncompletion(slot, "not_launched")
        result = analyse(rows, slots, predictions)
        self.assertEqual(result["summary"]["unresolved_configurations"], 48)
        self.assertEqual(result["summary"]["pair_denominators"]["unresolved"], 72)
        self.assertTrue(all(r["contrast"] is None for r in result["block_contrasts"]))
        metrics = score(result)
        self.assertEqual(metrics["all_approved"], 48)
        self.assertEqual(metrics["unresolved_approved"], 48)
        self.assertEqual(metrics["resolved_approved"], 0)
        self.assertIsNone(metrics["useful_approval_fraction"])
        self.assertIsNone(metrics["within_margin_rejection_fraction"])
        self.assertIsNone(metrics["three_state_accuracy"])
        self.assertIsNone(metrics["admission_accuracy"])
        self.assertEqual(metrics["correct_three_state"], 0)
        self.assertIsNone(effect(result)["equal_case_mean"])
        self.assertIsNone(effect(result)["available_case_mean"])
        self.assertEqual(len(effect(result)["case_contributions"]), 6)

    def test_empty_panel_has_explicit_zero_denominators(self):
        result = analyse([], [], [])
        self.assertEqual(result["configurations"], [])
        self.assertEqual(result["offload_pairs"], [])
        self.assertEqual(result["block_contrasts"], [])
        self.assertEqual(result["summary"]["planned_slots"], 0)
        self.assertFalse(result["summary"]["full_panel"])
        self.assertEqual(result["prediction_scores"], [])
        self.assertEqual(effect(result)["case_contributions"], [])
        self.assertIsNone(effect(result)["equal_case_mean"])
        self.assertEqual(len(result["summary"]["block_effects"]), len(METRICS) * len(EFFECTS))

    def test_costs_keep_partial_known_totals_even_for_invalid_or_failed_outcomes(self):
        rows, slots, predictions = panel()
        noncompletion(slots[0], "memory_failure")
        noncompletion(slots[1], "infrastructure_invalid")
        slots[0].update(task_gpu_seconds=20, invocation_elapsed_seconds=6)
        slots[1].update(task_gpu_seconds=8, invocation_elapsed_seconds=3)
        costs = analyse(rows, slots, predictions)["summary"]["costs"]
        self.assertEqual(costs["task_gpu_seconds"]["known_sum"], 28)
        self.assertEqual(costs["task_gpu_seconds"]["known_slots"], 2)
        self.assertEqual(costs["task_gpu_seconds"]["unknown_slots"], 142)
        self.assertIsNone(costs["task_gpu_seconds"]["complete_total"])
        for slot in slots:
            slot["invocation_elapsed_seconds"] = 0
        cost = analyse(rows, slots, predictions)["summary"]["costs"]["invocation_elapsed_seconds"]
        self.assertEqual(cost["complete_total"], 0)
        self.assertEqual(cost["known_slots"], 144)

    def test_prediction_coverage_duplicate_and_extra_methods_are_rejected(self):
        rows, slots, predictions = panel()
        mutations = {
            "missing_one": predictions[:-1],
            "missing_all": [],
            "duplicate": [*predictions[:-1], dict(predictions[0])],
            "extra_method": [dict(predictions[0], method="donor_copy"), *predictions[1:]],
            "extra_configuration": [dict(predictions[0], configuration_id="unplanned"), *predictions[1:]],
        }
        for name, changed in mutations.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                analyse(rows, slots, changed)

    def test_method_inventory_is_input_driven_but_each_method_needs_full_coverage(self):
        rows, slots, predictions = panel()
        subset = [p for p in predictions if p["method"] == "offload_only"]
        result = analyse(rows, slots, subset)
        self.assertEqual({r["method"] for r in result["prediction_scores"]}, {"offload_only"})
        self.assertFalse(any("strict_c1" in key for key in result["configurations"][0]))
        renamed = [dict(p, method="new_frozen_rule") for p in subset]
        self.assertEqual(score(analyse(rows, slots, renamed), "new_frozen_rule")["all_approved"], 24)
        with self.assertRaisesRegex(ValueError, "coverage"):
            analyse(rows, slots, [*predictions, renamed[0]])
        result = analyse(rows, slots, [*predictions, *renamed])
        self.assertEqual(len({r["method"] for r in result["prediction_scores"]}), 7)

    def test_prediction_types_states_and_approval_boolean_are_validated(self):
        rows, slots, predictions = panel()
        for mutation in (
            {"approved": "True"}, {"approved": 1}, {"predicted_state": "unresolved"},
            {"prediction_type": "binary"}, {"approved": False}, {"predicted_state": None},
            {"prediction_type": "admission_only", "predicted_state": None},
        ):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                analyse(rows, slots, [dict(predictions[0], **mutation), *predictions[1:]])
        admission_index = next(i for i, p in enumerate(predictions) if p["method"] == "offload_only")
        bad = deepcopy(predictions)
        bad[admission_index]["predicted_state"] = "above_margin"
        with self.assertRaisesRegex(ValueError, "admission_only"):
            analyse(rows, slots, bad)
        bad = deepcopy(predictions)
        del bad[0]["predicted_state"]
        with self.assertRaisesRegex(ValueError, "missing predicted_state"):
            analyse(rows, slots, bad)

    def test_duplicate_rows_slots_and_extra_slot_are_rejected(self):
        rows, slots, predictions = panel()
        with self.assertRaisesRegex(ValueError, "Duplicate matrix"):
            analyse([*rows[:-1], dict(rows[0])], slots, predictions)
        with self.assertRaisesRegex(ValueError, "Duplicate slot"):
            analyse(rows, [*slots[:-1], dict(slots[0])], predictions)
        with self.assertRaisesRegex(ValueError, "Extra slot"):
            analyse(rows, [dict(slots[0], experiment_id="unplanned"), *slots[1:]], predictions)
        # Two different experiment IDs cannot masquerade as distinct repetitions.
        selected = select(slots)
        target = selected[1]["experiment_id"]
        changed = [dict(r, training_seed="241") if r["experiment_id"] == target else r for r in rows]
        with self.assertRaisesRegex(ValueError, "three distinct planned seeds"):
            analyse(changed, slots, predictions)

    def test_every_required_slot_identity_or_setting_must_match(self):
        rows, slots, predictions = panel()
        changes = dict(
            block_id="wrong", configuration_id="wrong", case_id="wrong", model="wrong", dataset="wrong",
            training_seed="242", condition="wrong", parameter_offload=str(slots[0]["parameter_offload"] != "True"),
            actor_micro_batch="8" if slots[0]["actor_micro_batch"] == "4" else "4",
            vllm_gpu_memory_utilization=".7" if float(slots[0]["vllm_gpu_memory_utilization"]) == .45 else ".45",
            period="2" if slots[0]["period"] == "1" else "1",
        )
        for field, value in changes.items():
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "slot/matrix mismatch"):
                analyse(rows, [dict(slots[0], **{field: value}), *slots[1:]], predictions)
        bad = dict(slots[0])
        del bad["condition"]
        with self.assertRaisesRegex(ValueError, "missing fields"):
            analyse(rows, [bad, *slots[1:]], predictions)

    def test_duplicate_period_and_changed_config_settings_are_rejected(self):
        rows, slots, predictions = panel()
        same_block = [r for r in rows if r["block_id"] == rows[0]["block_id"]]
        target = same_block[1]["experiment_id"]
        changed = [dict(r, period=same_block[0]["period"]) if r["experiment_id"] == target else r for r in rows]
        with self.assertRaisesRegex(ValueError, "eight distinct periods"):
            analyse(changed, slots, predictions)
        changed = [dict(rows[0], condition="changed"), *rows[1:]]
        with self.assertRaisesRegex(ValueError, "configuration settings"):
            analyse(changed, slots, predictions)

    def test_out_of_range_matrix_values_and_panel_size_are_rejected(self):
        rows, slots, predictions = panel()
        changes = (
            ("training_seed", "240"), ("training_seed", "244"), ("training_seed", True),
            ("actor_micro_batch", "16"), ("actor_micro_batch", "4.5"),
            ("period", "0"), ("period", "9"), ("period", 1.0),
            ("parameter_offload", "true"), ("parameter_offload", 1),
            ("vllm_gpu_memory_utilization", "0"), ("vllm_gpu_memory_utilization", "1"),
            ("vllm_gpu_memory_utilization", ".5"), ("vllm_gpu_memory_utilization", "nan"),
            ("vllm_gpu_memory_utilization", float("inf")), ("vllm_gpu_memory_utilization", True),
            ("model", ""), ("case_id", None),
        )
        for field, value in changes:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                analyse([dict(rows[0], **{field: value}), *rows[1:]], slots, predictions)
        with self.assertRaisesRegex(ValueError, "144-slot"):
            analyse([*rows, rows[0]], slots, predictions)
        with self.assertRaisesRegex(ValueError, "two-model"):
            analyse(*panel(models=("a", "b", "c"), datasets=("math",)))

    def test_invalid_measurements_and_statuses_are_rejected(self):
        rows, slots, predictions = panel()
        for field in ("completed_peak_mib", "stage_actor_peak_mib", "stage_generation_peak_mib",
                      "stage_weight_sync_peak_mib", "task_gpu_seconds", "invocation_elapsed_seconds"):
            for value in (-1, float("nan"), float("inf"), True, "10000", 10 ** 1000):
                with self.subTest(field=field, value=str(value)[:20]), self.assertRaises(ValueError):
                    analyse(rows, [dict(slots[0], **{field: value}), *slots[1:]], predictions)
        for slot in (
            dict(slots[0], completed_peak_mib=None),
            dict(slots[0], status="timeout"),
            dict(slots[0], status="memory_failure"),
            dict(slots[0], status="infrastructure_invalid"),
            dict(slots[0], status="not_launched"),
        ):
            with self.subTest(status=slot["status"]), self.assertRaises(ValueError):
                analyse(rows, [slot, *slots[1:]], predictions)
        for limit in (0, -1, float("nan"), float("inf"), True, "38912"):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                analyse(rows, slots, predictions, limit_mib=limit)

    def test_equivalent_matrix_and_slot_scalar_formats_are_supported(self):
        rows, slots, predictions = panel()
        for slot in slots:
            slot.update(training_seed=int(slot["training_seed"]), period=int(slot["period"]),
                        actor_micro_batch=int(slot["actor_micro_batch"]),
                        parameter_offload=slot["parameter_offload"] == "True",
                        vllm_gpu_memory_utilization=float(slot["vllm_gpu_memory_utilization"]))
        self.assertEqual(analyse(rows, slots, predictions)["summary"]["resolved_configurations"], 48)

    def test_custom_limit_only_changes_labels_not_frozen_predictions(self):
        rows, slots, predictions = panel()
        selected = select(slots)
        selected[0]["completed_peak_mib"] = 10500
        result = analyse(rows, slots, predictions, limit_mib=10240)
        self.assertEqual(result["summary"]["limit_mib"], 10240)
        self.assertEqual(config(result, selected[0]["configuration_id"])["observed_state"], "above_margin")
        self.assertTrue(config(result, selected[0]["configuration_id"])["frozen_logistic_approved"])

    def test_output_is_deterministic_scalar_csv_and_json_ready_without_io_or_input_mutation(self):
        inputs = panel()
        before = deepcopy(inputs)
        with patch("builtins.open", side_effect=AssertionError("file I/O forbidden")), \
                patch("io.open", side_effect=AssertionError("file I/O forbidden")):
            result = analyse(*inputs)
            self.assertEqual(analyse(*inputs), result)
        self.assertEqual(inputs, before)
        for name in ("configurations", "prediction_scores", "offload_pairs", "block_contrasts"):
            values = result[name]
            for row in values:
                self.assertEqual(set(row), set(values[0]))
                self.assertTrue(all(value is None or type(value) in (str, int, float, bool) for value in row.values()))
            stream = io.StringIO()
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
            self.assertEqual(len(list(csv.DictReader(io.StringIO(stream.getvalue())))), len(values))
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)


class ClassifierCombinedShortcutTests(unittest.TestCase):
    def test_all_within_margin_disagreements_are_usable_lost_not_errors_avoided(self):
        rows, slots, predictions, rejected = disagreement_panel()
        result = analyse(rows, slots, predictions)
        comparison = result["summary"]["classifier_vs_combined_shortcut"]
        self.assertEqual(set(result), {
            "configurations", "prediction_scores", "offload_pairs", "block_contrasts", "summary",
        })
        self.assertTrue(comparison["selected_from_predictions_before_outcomes"])
        self.assertEqual(comparison["comparison_unit"], "configuration")
        self.assertEqual(comparison["required_seeds_per_configuration"], 3)
        self.assertEqual(comparison["planned_configurations"], 48)
        self.assertEqual(comparison["resolved_configurations"], 48)
        self.assertEqual(comparison["unresolved_configurations"], 0)
        self.assertEqual(comparison["classifier_approved_configurations"], 29)
        self.assertEqual(comparison["combined_shortcut_approved_configurations"], 36)
        self.assertEqual(comparison["planned_disagreements"], 7)
        self.assertEqual(comparison["resolved_disagreements"], 7)
        self.assertEqual(comparison["disagreement_resolved_fraction"], 1)
        self.assertEqual(comparison["avoided_memory_failure"], 0)
        self.assertEqual(comparison["avoided_above_margin"], 0)
        self.assertEqual(comparison["usable_rejected"], 7)
        self.assertEqual(comparison["unresolved_rejected_with_known_unsuitable"], 0)
        self.assertEqual(comparison["unresolved_rejected_still_unknown"], 0)
        self.assertEqual([r["configuration_id"] for r in comparison["disagreements"]], rejected)
        for row in comparison["disagreements"]:
            self.assertEqual(row["direction"], "combined_approved_classifier_rejected")
            self.assertEqual(row["panel_group"], "challenge")
            self.assertEqual(row["observed_state"], "within_margin")
            self.assertTrue(row["parameter_offload"])
            self.assertEqual(row["vllm_gpu_memory_utilization"], .7)
            self.assertFalse(row["frozen_logistic_approved"])
            self.assertTrue(row["offload_or_low_reservation_approved"])
            self.assertFalse(row["unresolved"])
            self.assertFalse(row["known_unsuitable"])
            self.assertEqual(row["eligible_seed_count"], 3)
            self.assertIn(row["case_id"], {r["case_id"] for r in rows})
            self.assertIn(row["actor_micro_batch"], (4, 8))
        self.assertEqual(comparison["by_model"]["family-a-7b"]["usable_rejected"], 6)
        self.assertEqual(comparison["by_model"]["family-b-7b"]["usable_rejected"], 1)
        mistral_like = next(r for r in comparison["disagreements"] if r["model"] == "family-b-7b")
        self.assertEqual(mistral_like["dataset"], "gsm8k")
        self.assertEqual(comparison["by_model"]["family-a-7b"]["planned_configurations"], 24)
        reverse = comparison["directions"]["classifier_approved_combined_rejected"]
        self.assertEqual(reverse["planned_configurations"], 0)
        self.assertIsNone(reverse["resolved_fraction"])

    def test_all_failure_disagreements_are_errors_avoided_without_peak_imputation(self):
        rows, slots, predictions, rejected = disagreement_panel()
        for slot in slots:
            noncompletion(slot, "memory_failure")
        comparison = analyse(rows, slots, predictions)["summary"]["classifier_vs_combined_shortcut"]
        self.assertEqual(comparison["avoided_memory_failure"], 7)
        self.assertEqual(comparison["avoided_above_margin"], 0)
        self.assertEqual(comparison["usable_rejected"], 0)
        self.assertEqual(comparison["unresolved_disagreements"], 0)
        self.assertEqual([r["configuration_id"] for r in comparison["disagreements"]], rejected)
        self.assertTrue(all(r["observed_state"] == "memory_failure" and not r["unresolved"]
                            and r["known_memory_failure"] for r in comparison["disagreements"]))
        self.assertEqual(comparison["by_model"]["family-a-7b"]["avoided_memory_failure"], 6)
        self.assertEqual(comparison["by_model"]["family-b-7b"]["avoided_memory_failure"], 1)

    def test_mixed_unresolved_partial_flags_stay_separate_from_avoided_errors(self):
        rows, slots, predictions, rejected = disagreement_panel()
        before = deepcopy(predictions)
        selected = {cid: sorted((s for s in slots if s["configuration_id"] == cid),
                                key=lambda s: s["training_seed"]) for cid in rejected}
        selected[rejected[0]][0]["completed_peak_mib"] = 39000
        noncompletion(selected[rejected[1]][0], "memory_failure")
        selected[rejected[1]][1]["completed_peak_mib"] = 39000
        # Both partial failure and above-margin evidence count as one known-unsuitable configuration.
        noncompletion(selected[rejected[3]][0], "memory_failure")
        selected[rejected[3]][1]["completed_peak_mib"] = 39000
        noncompletion(selected[rejected[3]][2], "not_launched")
        selected[rejected[4]][0]["completed_peak_mib"] = 39000
        noncompletion(selected[rejected[4]][2], "infrastructure_invalid")
        for slot in selected[rejected[5]]:
            noncompletion(slot, "not_launched")
        noncompletion(selected[rejected[6]][2], "infrastructure_invalid")
        # An unresolved agreement belongs to panel coverage but not disagreement coverage.
        unselected = next(s["configuration_id"] for s in slots if s["configuration_id"] not in rejected)
        noncompletion(next(s for s in slots if s["configuration_id"] == unselected), "not_launched")
        comparison = analyse(rows, slots, predictions)["summary"]["classifier_vs_combined_shortcut"]
        self.assertEqual(predictions, before)
        self.assertEqual(comparison["planned_disagreements"], 7)
        self.assertEqual(comparison["resolved_disagreements"], 3)
        self.assertEqual(comparison["unresolved_disagreements"], 4)
        self.assertEqual(comparison["disagreement_resolved_fraction"], 3 / 7)
        self.assertEqual(comparison["unresolved_configurations"], 5)
        self.assertEqual(comparison["resolved_fraction"], 43 / 48)
        self.assertEqual(comparison["avoided_memory_failure"], 1)
        self.assertEqual(comparison["avoided_above_margin"], 1)
        self.assertEqual(comparison["usable_rejected"], 1)
        self.assertEqual(comparison["unresolved_rejected_with_known_unsuitable"], 2)
        self.assertEqual(comparison["unresolved_rejected_still_unknown"], 2)
        forward = comparison["directions"]["combined_approved_classifier_rejected"]
        self.assertEqual(forward["unresolved_with_known_memory_failure"], 1)
        self.assertEqual(forward["unresolved_with_known_margin_exceedance"], 2)
        self.assertEqual(forward["unresolved_with_known_unsuitable"], 2)
        self.assertEqual(forward["unresolved_still_unknown"], 2)
        partial = next(r for r in comparison["disagreements"] if r["configuration_id"] == rejected[3])
        self.assertEqual(partial["observed_state"], "unresolved")
        self.assertTrue(partial["known_memory_failure"])
        self.assertTrue(partial["known_margin_exceedance"])
        self.assertTrue(partial["unresolved_known_unsuitable"])
        self.assertEqual(partial["unresolved_seeds"], "243")
        self.assertEqual(comparison["by_model"]["family-a-7b"]["resolved_disagreements"], 3)
        self.assertEqual(comparison["by_model"]["family-a-7b"]["unresolved_disagreements"], 3)
        self.assertEqual(comparison["by_model"]["family-b-7b"]["unresolved_disagreements"], 1)

    def test_reverse_disagreements_are_reported_without_assuming_a_subset(self):
        rows, slots, predictions, rejected = disagreement_panel()
        reverse_ids = sorted(p["configuration_id"] for p in predictions
                             if p["method"] == "offload_or_low_reservation" and not p["approved"])[:4]
        for prediction in predictions:
            if prediction["method"] == "frozen_logistic" and prediction["configuration_id"] in reverse_ids:
                prediction.update(approved=True, predicted_state="within_margin")
        noncompletion(next(s for s in slots if s["configuration_id"] == reverse_ids[0]), "memory_failure")
        next(s for s in slots if s["configuration_id"] == reverse_ids[1])["completed_peak_mib"] = 39000
        noncompletion(next(s for s in slots if s["configuration_id"] == reverse_ids[3]), "not_launched")
        comparison = analyse(rows, slots, predictions)["summary"]["classifier_vs_combined_shortcut"]
        self.assertEqual(comparison["planned_disagreements"], 11)
        self.assertEqual(comparison["resolved_disagreements"], 10)
        self.assertEqual(comparison["unresolved_disagreements"], 1)
        self.assertEqual(comparison["classifier_approved_configurations"], 33)
        self.assertEqual(comparison["avoided_memory_failure"], 0)
        self.assertEqual(comparison["avoided_above_margin"], 0)
        self.assertEqual(comparison["usable_rejected"], 7)
        self.assertEqual(comparison["unresolved_rejected_with_known_unsuitable"], 0)
        self.assertEqual(comparison["unresolved_rejected_still_unknown"], 0)
        reverse = comparison["directions"]["classifier_approved_combined_rejected"]
        self.assertEqual(reverse["planned_configurations"], 4)
        self.assertEqual(reverse["resolved_configurations"], 3)
        self.assertEqual(reverse["observed_states"],
                         {"within_margin": 1, "above_margin": 1, "memory_failure": 1, "unresolved": 1})
        self.assertEqual({r["configuration_id"] for r in comparison["disagreements"]},
                         set(rejected) | set(reverse_ids))
        self.assertTrue(all(r["frozen_logistic_approved"] and not r["offload_or_low_reservation_approved"]
                            for r in comparison["disagreements"]
                            if r["direction"] == "classifier_approved_combined_rejected"))

    def test_selection_is_unchanged_by_outcomes_or_unresolved_coverage(self):
        rows, slots, predictions, rejected = disagreement_panel()
        before = analyse(rows, slots, predictions)["summary"]["classifier_vs_combined_shortcut"]
        for slot in slots:
            noncompletion(slot, "not_launched")
        after = analyse(rows, slots, predictions)["summary"]["classifier_vs_combined_shortcut"]
        self.assertEqual([(r["configuration_id"], r["direction"]) for r in before["disagreements"]],
                         [(r["configuration_id"], r["direction"]) for r in after["disagreements"]])
        self.assertEqual(after["planned_disagreements"], len(rejected))
        self.assertEqual(after["unresolved_disagreements"], 7)
        self.assertEqual(after["unresolved_configurations"], 48)
        self.assertEqual(after["avoided_memory_failure"], 0)
        self.assertEqual(after["avoided_above_margin"], 0)
        self.assertEqual(after["usable_rejected"], 0)
        self.assertEqual(after["unresolved_rejected_with_known_unsuitable"], 0)
        self.assertEqual(after["unresolved_rejected_still_unknown"], 7)
        self.assertEqual(after["disagreement_resolved_fraction"], 0)

    def test_absent_methods_or_empty_panel_return_none_without_changing_generic_coverage(self):
        self.assertIsNone(analyse([], [], [])["summary"]["classifier_vs_combined_shortcut"])
        rows, slots, predictions, _ = disagreement_panel()
        for missing in ("frozen_logistic", "offload_or_low_reservation"):
            with self.subTest(missing=missing):
                selected = [p for p in predictions if p["method"] != missing]
                self.assertIsNone(analyse(rows, slots, selected)["summary"]["classifier_vs_combined_shortcut"])
        unrelated = [p for p in predictions if p["method"] == "reservation_only"]
        self.assertIsNone(analyse(rows, slots, unrelated)["summary"]["classifier_vs_combined_shortcut"])
        with self.assertRaisesRegex(ValueError, "coverage"):
            analyse(rows, slots, [])
        only_compared = [p for p in predictions if p["method"] in ("frozen_logistic", "offload_or_low_reservation")]
        self.assertEqual(analyse(rows, slots, only_compared)["summary"]["classifier_vs_combined_shortcut"]
                         ["planned_disagreements"], 7)

    def test_agreement_only_is_available_with_zero_disagreement_denominator(self):
        rows, slots, predictions, _ = disagreement_panel()
        combined = {p["configuration_id"]: p["approved"] for p in predictions
                    if p["method"] == "offload_or_low_reservation"}
        for prediction in predictions:
            if prediction["method"] == "frozen_logistic":
                approved = combined[prediction["configuration_id"]]
                prediction.update(approved=approved, predicted_state="within_margin" if approved else "above_margin")
        comparison = analyse(rows, slots, predictions)["summary"]["classifier_vs_combined_shortcut"]
        self.assertEqual(comparison["planned_configurations"], 48)
        self.assertEqual(comparison["planned_disagreements"], 0)
        self.assertEqual(comparison["disagreements"], [])
        self.assertIsNone(comparison["disagreement_resolved_fraction"])
        self.assertEqual(comparison["avoided_memory_failure"], 0)
        self.assertEqual(comparison["usable_rejected"], 0)
        self.assertEqual(set(comparison["by_model"]), {"family-a-7b", "family-b-7b"})
        for model in comparison["by_model"].values():
            self.assertEqual(model["planned_configurations"], 24)
            self.assertEqual(model["planned_disagreements"], 0)
            self.assertIsNone(model["disagreement_resolved_fraction"])


if __name__ == "__main__":
    unittest.main()
