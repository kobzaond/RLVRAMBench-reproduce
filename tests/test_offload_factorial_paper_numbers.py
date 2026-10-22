"""CPU-only check of the independent offload factorial numbers reported in the paper.

Reads only the curated evidence under ``benchmark/offload_factorial`` and
``benchmark/revision_20260917``. Outcome labels and paired contrasts are
recomputed from ``results/slots.csv``; approval decisions come from the
pre-execution ``predictions.json`` sealed by ``freeze.json``. The derived
tables (``configurations.csv``, ``prediction_scores.csv``, ``summary.json``)
are used only as cross-checks. Re-deriving ``slots.csv`` itself from the raw
per-process records requires the release evidence archive and is not done here
(see README, "Independent offload factorial").
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

import pytest

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "benchmark/offload_factorial"
RESULTS = PANEL / "results"
REVISION = ROOT / "benchmark/revision_20260917"

MODELS = {"Qwen/Qwen2.5-7B-Instruct": "qwen", "mistralai/Mistral-7B-Instruct-v0.3": "mistral"}

# Paper Table tab:offload-prediction: (useful approved, failure approved, useful rejected)
PAPER_PREDICTION_TABLE = {
    "offload_or_low_reservation": (36, 0, 0),
    "frozen_logistic": (29, 0, 7),
    "component_regression": (36, 6, 0),
    "offload_only": (24, 0, 12),
    "reservation_only": (24, 0, 12),
    "strict_c1": (6, 0, 30),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(name: str) -> list[dict]:
    with (RESULTS / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def truthy(value: str) -> bool:
    assert value in {"True", "False"}, value
    return value == "True"


@pytest.fixture(scope="module")
def protocol():
    return json.loads((PANEL / "protocol.json").read_text())


@pytest.fixture(scope="module")
def slots():
    return rows("slots.csv")


@pytest.fixture(scope="module")
def labels(slots, protocol):
    """Configuration label recomputed from the three per-seed slot outcomes."""
    limit = float(protocol["memory_margin_limit_mib"])
    by_config = defaultdict(list)
    for slot in slots:
        assert truthy(slot["eligible"])
        if slot["status"] == "completed":
            state = "within_margin" if float(slot["completed_peak_mib"]) <= limit else "above_margin"
        else:
            assert slot["status"] == "memory_failure"
            state = "memory_failure"
        by_config[slot["configuration_id"]].append((int(slot["training_seed"]), state, slot))
    result = {}
    for config, entries in by_config.items():
        assert sorted(seed for seed, _, _ in entries) == [241, 242, 243]
        states = {state for _, state, _ in entries}
        assert len(states) == 1, f"{config}: repetitions disagree"
        slot = entries[0][2]
        result[config] = {
            "state": states.pop(),
            "model": MODELS[slot["model"]],
            "offload": truthy(slot["parameter_offload"]),
            "batch": int(slot["actor_micro_batch"]),
            "reservation": float(slot["vllm_gpu_memory_utilization"]),
        }
    return result


@pytest.fixture(scope="module")
def decisions():
    out = defaultdict(dict)
    for record in json.loads((PANEL / "predictions.json").read_text()):
        assert record["configuration_id"] not in out[record["method"]]
        out[record["method"]][record["configuration_id"]] = record["approved"]
    return out


def test_frozen_inputs_and_curated_results_match_recorded_hashes():
    freeze = json.loads((PANEL / "freeze.json").read_text())
    assert freeze["status"] == "sealed_before_offload_factorial_execution"
    for name, digest in freeze["files_sha256"].items():
        assert sha256(PANEL / name) == digest, name
    report = json.loads((REVISION / "offload-independent-reconstruction.json").read_text())
    assert report["status"] == "passed"
    assert report["freeze_sha256"] == sha256(PANEL / "freeze.json")
    for name, digest in report["result_sha256"].items():
        assert sha256(RESULTS / name) == digest, name
    summary = json.loads((RESULTS / "summary.json").read_text())
    assert summary["freeze_sha256"] == report["freeze_sha256"]
    assert summary["execution_source_commit"] == report["source_commit"]


def test_slots_cover_the_frozen_matrix_without_retry(slots):
    with (PANEL / "matrix.csv").open(newline="") as handle:
        planned = [row["experiment_id"] for row in csv.DictReader(handle)]
    assert len(planned) == len(set(planned)) == 144
    assert sorted(slot["experiment_id"] for slot in slots) == sorted(planned)
    assert Counter(slot["status"] for slot in slots) == {"completed": 108, "memory_failure": 36}
    # All failures are generation-engine startup rejections.
    assert {slot["failure_kind"] for slot in slots if slot["status"] == "memory_failure"} == {
        "rollout_init_memory"}
    assert len({slot["block_id"] for slot in slots}) == 18


def test_outcome_table(labels):
    assert len(labels) == 48
    assert Counter(entry["state"] for entry in labels.values()) == {
        "within_margin": 36, "memory_failure": 12}
    cells = defaultdict(Counter)
    for entry in labels.values():
        cells[(entry["offload"], entry["reservation"])][entry["state"]] += 1
    assert cells == {
        (False, 0.45): {"within_margin": 12},
        (False, 0.70): {"memory_failure": 12},
        (True, 0.45): {"within_margin": 12},
        (True, 0.70): {"within_margin": 12},
    }
    # Cross-check against the derived configuration table.
    derived = {row["configuration_id"]: row["observed_state"] for row in rows("configurations.csv")}
    assert derived == {config: entry["state"] for config, entry in labels.items()}


def score(labels, approvals):
    assert set(approvals) == set(labels)
    useful = [labels[c]["state"] == "within_margin" for c in labels]
    approved = [approvals[c] for c in labels]
    return (sum(a and u for a, u in zip(approved, useful)),
            sum(a and not u for a, u in zip(approved, useful)),
            sum(u and not a for a, u in zip(approved, useful)))


@pytest.mark.parametrize("method", sorted(PAPER_PREDICTION_TABLE))
def test_prediction_table(labels, decisions, method):
    assert score(labels, decisions[method]) == PAPER_PREDICTION_TABLE[method]
    overall = [row for row in rows("prediction_scores.csv")
               if row["method"] == method and row["stratum"] == "overall"]
    assert len(overall) == 1
    row = overall[0]
    assert (int(row["useful_approvals"]), int(row["memory_failure_approvals"]),
            int(row["within_margin_rejected"])) == PAPER_PREDICTION_TABLE[method]


def test_prediction_text_details(labels, decisions):
    combined, logistic, regression = (decisions[m] for m in (
        "offload_or_low_reservation", "frozen_logistic", "component_regression"))
    # Classifier approvals are a subset of the combined rule; all seven
    # disagreements complete within margin (six Qwen, one Mistral).
    assert all(combined[c] for c in labels if logistic[c])
    disagreements = [c for c in labels if combined[c] and not logistic[c]]
    assert all(labels[c]["state"] == "within_margin" for c in disagreements)
    assert Counter(labels[c]["model"] for c in disagreements) == {"qwen": 6, "mistral": 1}
    wrong = [c for c in labels if regression[c] and labels[c]["state"] != "within_margin"]
    assert len(wrong) == 6 and {labels[c]["model"] for c in wrong} == {"mistral"}
    # 42 newly crossed configurations, 30 within margin, all recovered by the combined rule.
    anchors = {row["configuration_id"] for row in rows("configurations.csv")
               if row["panel_group"] == "anchors"}
    assert len(anchors) == 6
    assert all(labels[c]["offload"] and labels[c]["batch"] == 4
               and labels[c]["reservation"] == 0.45 for c in anchors)
    challenge_useful = [c for c in labels if c not in anchors
                        and labels[c]["state"] == "within_margin"]
    assert len(challenge_useful) == 30 and all(combined[c] for c in challenge_useful)


def test_offload_pairs_and_peak_reductions(slots):
    pairs = defaultdict(dict)
    for slot in slots:
        key = (slot["case_id"], slot["block_id"], int(slot["actor_micro_batch"]),
               float(slot["vllm_gpu_memory_utilization"]))
        pairs[key][truthy(slot["parameter_offload"])] = slot
    assert len(pairs) == 72 and all(set(pair) == {False, True} for pair in pairs.values())

    transitions = Counter((key[3], pair[False]["status"], pair[True]["status"])
                          for key, pair in pairs.items())
    assert transitions == {(0.45, "completed", "completed"): 36,
                           (0.70, "memory_failure", "completed"): 36}

    def equal_case_mean(batch, metric):
        per_case = defaultdict(list)
        for (case, _, b, reservation), pair in pairs.items():
            if b == batch and reservation == 0.45:
                per_case[case].append(float(pair[True][metric]) - float(pair[False][metric]))
        assert len(per_case) == 6 and all(len(v) == 3 for v in per_case.values())
        return mean(mean(v) for v in per_case.values()), [x for v in per_case.values() for x in v]

    peak4, values4 = equal_case_mean(4, "completed_peak_mib")
    peak8, values8 = equal_case_mean(8, "completed_peak_mib")
    assert round(-peak4) == 3226 and round(-peak8) == 3223
    reductions = [-x for x in values4 + values8]
    assert all(x > 0 for x in reductions)
    assert (min(reductions), max(reductions)) == (1868, 4296)
    for batch in (4, 8):
        generation, _ = equal_case_mean(batch, "stage_generation_peak_mib")
        assert abs(-generation - 12750) < 25

    # With offload at reservation 0.45 the whole-run peak is the weight-sync stage.
    offloaded = [pair[True] for key, pair in pairs.items() if key[3] == 0.45]
    assert len(offloaded) == 36
    for slot in offloaded:
        stages = {name: float(slot[f"stage_{name}_peak_mib"])
                  for name in ("actor", "generation", "weight_sync")}
        assert max(stages, key=stages.get) == "weight_sync"
        assert float(slot["completed_peak_mib"]) == stages["weight_sync"]

    # Cross-check against the derived summary.
    summary = json.loads((RESULTS / "summary.json").read_text())
    reported = {e["actor_micro_batch"]: e["equal_case_mean"] for e in summary["offload_effects"]
                if e["metric"] == "completed_peak_mib"
                and e["comparison_role"] == "primary_fixed_batch_reservation"
                and e["equal_case_mean"] is not None}
    assert reported == pytest.approx({4: peak4, 8: peak8})
    assert summary["statuses"] == {"completed": 108, "infrastructure_invalid": 0,
                                   "memory_failure": 36, "not_launched": 0}
    assert summary["transition_counts"]["memory_failure_to_within_margin"] == 36
    assert summary["transition_counts"]["within_margin_to_within_margin"] == 36
