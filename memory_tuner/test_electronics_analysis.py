import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from memory_tuner.collect_llmem_native import collect
from memory_tuner.device_contract import digest
from memory_tuner.electronics_followup_analysis import (
    adaptation_offload_contrasts, metrics, matched_contrasts, summarize_cases, score_predictions,
)
from memory_tuner.electronics_resource_accounting import allocation_cost, allocation_rows


class NativeEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.protocol = self.root/"protocol"
        self.output = self.root/"output"
        self.protocol.mkdir()
        row = dict(experiment_id="native-case", block_index="0", model="opt",
                   batch="1", sequence="256", seed="1")
        self.row = row
        with (self.protocol/"matrix.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        (self.protocol/"protocol.json").write_text(json.dumps(dict(
            matrix_sha256=digest(self.protocol/"matrix.csv"),
            upstream_estimator_sha256="estimator", models={"opt": {"path": "/model"}})))
        self.anchor = digest(self.protocol/"protocol.json")
        self.pair = self.output/"block-0"/row["experiment_id"]
        self.pair.mkdir(parents=True)
        (self.pair.parent/"claim.json").write_text(json.dumps(dict(
            source_commit="source", protocol_sha256=self.anchor, rows=[row],
            gpu_uuid="GPU-physical", job_id="100")))
        executions = {}
        for mode, offset in (("estimate", 0), ("measure", 100)):
            directory = self.pair/mode
            directory.mkdir()
            record = dict(mode=mode, gpu_uuid="GPU-physical", model="/model",
                          batch=1, sequence=256, seed=1, source_sha256="estimator",
                          torch="2.0.1+cu117", transformers="4.29.2", colossalai="0.3.0",
                          started_ns=offset+1, finished_ns=offset+9,
                          observed_process_peak_mib=3000, baseline_mib=500, pid=offset+10)
            if mode == "estimate":
                record.update(status="estimated", predicted_peak_mib=3100,
                              predicted_batch=1, m_pbase_mib=1000)
            else:
                record.update(status="completed", measurement_start_ns=offset+3,
                              training_sampled_peak_mib=3000, completed_steps=3,
                              allocator_peak_allocated_mib=1800, allocator_peak_reserved_mib=2000,
                              losses=[1.0, .9, .8],
                              stage_observations=[{"step": s} for s in (1, 2, 3)])
            (directory/"result.json").write_text(json.dumps(record))
            (directory/"nvml.csv").write_text(
                f"timestamp_ns,used_mib\n{offset+2},1000\n{offset+4},3000\n{offset+8},2000\n")
            executions[mode] = dict(exit_code=0, timeout=False, started_ns=offset,
                                    finished_ns=offset+10)
        self.terminal = dict(row=row, executions=executions)
        self.receipt()

    def tearDown(self):
        self.temp.cleanup()

    def receipt(self):
        self.terminal["files_sha256"] = {
            str(p.relative_to(self.pair)): digest(p)
            for p in self.pair.rglob("*") if p.is_file() and p.name != "terminal.json"}
        (self.pair/"terminal.json").write_text(json.dumps(self.terminal))

    def run_audit(self):
        return collect(self.protocol, self.output, "source", self.anchor)

    def test_independent_completed_prediction_and_measurement(self):
        summary, rows, _ = self.run_audit()
        self.assertEqual(rows[0]["status"], "completed_pair")
        self.assertEqual(summary["equal_case_mae_mib"], 100)

    def test_modified_measurement_invalidates_pair_instead_of_changing_error(self):
        with (self.pair/"measure/nvml.csv").open("a") as handle:
            handle.write("109,3500\n")
        summary, rows, _ = self.run_audit()
        self.assertEqual(rows[0]["status"], "unresolved")
        self.assertIsNone(summary["equal_case_mae_mib"])

    def test_hashed_receipt_cannot_hide_wrong_reported_peak(self):
        path = self.pair/"measure/result.json"
        record = json.loads(path.read_text())
        record["training_sampled_peak_mib"] = 2500
        path.write_text(json.dumps(record))
        self.receipt()
        _, rows, _ = self.run_audit()
        self.assertIn("Training window differs", rows[0]["reason"])

    def test_prediction_must_precede_measurement(self):
        path = self.pair/"estimate/result.json"
        record = json.loads(path.read_text())
        record["finished_ns"] = 102
        path.write_text(json.dumps(record))
        self.terminal["executions"]["estimate"]["finished_ns"] = 103
        self.receipt()
        _, rows, _ = self.run_audit()
        self.assertIn("Prediction followed measured outcome", rows[0]["reason"])

    def test_missing_pair_remains_in_denominator(self):
        (self.pair/"terminal.json").unlink()
        summary, rows, _ = self.run_audit()
        self.assertEqual(summary["planned_pairs"], 1)
        self.assertEqual(rows[0]["status"], "unresolved")
        self.assertEqual(summary["complete_cases"], 0)

    def test_invalid_training_window_cannot_choose_a_later_lower_peak(self):
        path = self.pair/"measure/result.json"
        record = json.loads(path.read_text())
        record["measurement_start_ns"] = 110
        path.write_text(json.dumps(record))
        self.receipt()
        _, rows, _ = self.run_audit()
        self.assertIn("Invalid training-window", rows[0]["reason"])


class FollowupAnalysisTests(unittest.TestCase):
    def row(self, rank, status):
        return dict(block_id="block", configuration_id="rank"+str(rank),
                    model="model", dataset="gsm8k", lora_rank=str(rank), use_kl_loss="True",
                    parameter_offload="False", vllm_gpu_memory_utilization=".65",
                    free_cache_engine="True", gpu_count="4", condition="rank"+str(rank),
                    training_seed="10", experiment_id="mode"+str(rank), status=status)

    def test_failed_partial_peak_is_not_a_completed_demand_contrast(self):
        a, b = self.row(64, "within_margin"), self.row(0, "memory_failure")
        a["completed_peak_mib"] = 30000
        b["observed_peak_mib"] = 39000
        contrast, = matched_contrasts([a, b], "adaptation")
        self.assertTrue(contrast["resolved"])
        self.assertFalse(contrast["both_completed"])
        self.assertNotIn("completed_peak_mib_difference", contrast)

    def test_missing_repeat_cannot_be_declared_within_margin(self):
        a, b = self.row(64, "within_margin"), self.row(64, "not_launched")
        case, = summarize_cases([a, b])
        self.assertEqual(case["planned"], 2)
        self.assertEqual(case["configuration_label"], "unresolved")

    def test_offload_pairs_cannot_mix_adaptation_modes_or_objectives(self):
        rows = []
        for rank in (0, 64):
            for kl in ("True", "False"):
                for offload in ("True", "False"):
                    row = self.row(rank, "within_margin")
                    row.update(use_kl_loss=kl, parameter_offload=offload,
                               completed_peak_mib=20000+(100 if offload=="True" else 0))
                    rows.append(row)
        contrasts = adaptation_offload_contrasts(rows)
        self.assertEqual(len(contrasts), 4)
        self.assertEqual({(r["lora_rank"],r["use_kl_loss"]) for r in contrasts},
                         {("0","True"),("0","False"),("64","True"),("64","False")})
        self.assertTrue(all(r["completed_peak_mib_difference"]==100 for r in contrasts))

    def test_metric_text_must_match_identity_and_hash(self):
        text = "step:2 - training/global_step:2 - timing_s/step:4"
        record = dict(console_line=text, console_line_sha256=hashlib.sha256(text.encode()).hexdigest(), step=2)
        self.assertEqual(metrics(record)["timing_s/step"], 4)
        record["step"] = 3
        with self.assertRaisesRegex(ValueError, "step identity"):
            metrics(record)

    def test_unsupported_external_target_is_not_a_correct_rejection(self):
        case = self.row(64, "within_margin")
        case.update(configuration_label="within_margin", completed_peak_mib_max=30000)
        prediction = dict(configuration_id=case["configuration_id"], method="external",
                          prediction_type="unsupported", approved=None, predicted_state=None,
                          predicted_peak_mib=None)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"predictions.json").write_text(json.dumps([prediction]))
            records, summaries = score_predictions(root, [case])
        self.assertFalse(records[0]["scored"])
        overall = next(r for r in summaries if r["scope"]=="all")
        self.assertEqual(overall["supported"], 0)
        self.assertEqual(overall["within_rejected"], 0)
        self.assertIsNone(overall["within_recall"])
        self.assertIsNone(overall["mae_mib"])
        stratum = next(r for r in summaries if r["scope"] == "all" and r["model"] == "model")
        self.assertEqual(stratum["actor_offload"], "False")
        self.assertEqual(stratum["planned"], 1)
        self.assertEqual(stratum["scored"], 0)


class AllocationCostTests(unittest.TestCase):
    def test_slurm_steps_do_not_triple_allocation_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"accounting.psv"
            path.write_text(
                "JobID|JobIDRaw|State|ExitCode|ElapsedRaw|AllocTRES|NodeList|Start|End\n"
                "100_0|101|COMPLETED|0:0|3600|gres/gpu=4|node|start|end\n"
                "100_0.batch|101.batch|COMPLETED|0:0|3600|gres/gpu=4|node|start|end\n"
                "100_0.extern|101.extern|COMPLETED|0:0|3600|gres/gpu=4|node|start|end\n")
            records = allocation_rows(path)
        self.assertEqual(len(records), 1)
        self.assertEqual(allocation_cost(records["100_0"])["allocated_gpu_hours"], 4)

    def test_running_allocation_is_not_final_cost(self):
        with self.assertRaisesRegex(ValueError, "not terminal"):
            allocation_cost(dict(JobID="100_0", State="RUNNING"))


if __name__ == "__main__":
    unittest.main()
