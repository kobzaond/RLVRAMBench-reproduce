"""Freeze one separate host-resource follow-up; never fit or revise predictors."""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import random
import re
from types import SimpleNamespace

from benchmark import write_csv
from memory_tuner.device_contract import digest, write_new_json

GROUP = "estimation_host_capacity"
DIRECTORY = "benchmark/host_capacity"
SEEDS = [161, 162, 163]
DATASETS = ("gsm8k", "math", "code_heavy_tail")
ORIGINAL_PROTOCOL_SHA256 = "1694e447e842600612604891e7b3d036ff0791fa84c4bdbeba54ed093feefdeb"
ORIGINAL_SEAL_SHA256 = "5ed714caff922a5109346a4242545d19f2824636d0f52bb0bbd2a09b40ed4666"
IMPLEMENTATIONS = (
    "memory_tuner/host_capacity_protocol.py", "memory_tuner/run_host_capacity.py",
    "memory_tuner/run_matched_gpu.py", "run_host_capacity.slurm")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def original_anchor(root):
    directory = root / "benchmark/estimation"
    require(digest(directory / "protocol.json") == ORIGINAL_PROTOCOL_SHA256
            and digest(directory / "prediction_freeze.json") == ORIGINAL_SEAL_SHA256,
            "Original externally anchored protocol/seal changed")
    seal = json.loads((directory / "prediction_freeze.json").read_text())
    for name, expected in seal["files_sha256"].items():
        require(name in {"fitted_model.json", "predictions.json", "retrospective.json"}
                and digest(directory / name) == expected, "Changed original sealed predictor")
    for name in ("matrix.csv", "targets.csv"):
        require(digest(directory / name) ==
                seal["effective_input_sha256"]["benchmark/estimation/" + name],
                "Changed original sealed settings")


def planned(root):
    """Deterministic new IDs/seeds/order; every model/workload setting is inherited."""
    old = root / "benchmark/estimation"
    original_rows = read_csv(old / "matrix.csv")
    original_targets = read_csv(old / "targets.csv")
    original_predictions = json.loads((old / "predictions.json").read_text())
    schedule, targets, origins = [], [], {}
    subsets = list(itertools.combinations(range(4), 2)) + [(0, 1), (2, 3), (0, 2)]
    random.Random(2026091512).shuffle(subsets)
    order_rng = random.Random(2026091511)
    for cell, dataset in enumerate(DATASETS):
        first_counts = [2, 4, 2] if cell != 1 else [4, 2, 4]
        order_rng.shuffle(first_counts)
        for count in (2, 4):
            original_id = f"est-qwen25-7b-{dataset}-c2-{count}gpu"
            cid = f"host-qwen25-7b-{dataset}-c2-{count}gpu"
            target = dict(next(t for t in original_targets
                               if t["configuration_id"] == original_id))
            target.update(configuration_id=cid, study=GROUP,
                          case_id=f"host-qwen25-7b-{dataset}-{count}gpu")
            targets.append(target)
            origins[cid] = original_id
        for seed_index, (seed, first) in enumerate(zip(SEEDS, first_counts)):
            pair = f"host-qwen25-7b-{dataset}-c2-s{seed}"
            subset = subsets[cell * 3 + seed_index]
            for period, count in enumerate((first, 6 - first), 1):
                row = dict(next(r for r in original_rows if
                                r["dataset"] == dataset and r["configuration_level"] == "c2"
                                and r["gpu_count"] == str(count)))
                row.update(
                    experiment_id=f"{pair}-{count}gpu", pair_id=pair,
                    configuration_id=f"host-qwen25-7b-{dataset}-c2-{count}gpu",
                    case_id=f"host-qwen25-7b-{dataset}-{count}gpu",
                    training_seed=str(seed), period=str(period), condition=f"{count}gpu",
                    run_group=GROUP, study_stage=GROUP, revision_study=GROUP,
                    gpu_subset_ordinals=";".join(map(str, subset if count == 2 else range(4))),
                )
                schedule.append(row)
    pair_order = list(dict.fromkeys(row["pair_id"] for row in schedule))
    random.Random(2026091513).shuffle(pair_order)
    schedule.sort(key=lambda row: (pair_order.index(row["pair_id"]), int(row["period"])))
    predictions = []
    for cid, origin in origins.items():
        for original in original_predictions:
            if original["configuration_id"] == origin:
                pred = copy.deepcopy(original)
                pred["configuration_id"] = cid
                predictions.append(pred)
    require(len(schedule) == 18 and len(targets) == 6 and len(predictions) == 18,
            "Incomplete follow-up inventory")
    return schedule, targets, predictions, origins


def validate_matrix(rows, root=None):
    root = Path(root or Path(__file__).resolve().parents[1])
    expected, _, _, _ = planned(root)
    require(rows == expected, "Follow-up rows differ from the fixed inherited schedule")
    pairs = {}
    for row in rows:
        pairs.setdefault(row["pair_id"], []).append(row)
    return pairs


def validate_allocation(allocation):
    """The new condition fixes host requests as well as task-device assignment."""
    record = allocation["scheduler_record"]
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z_/]+)=([^\s]+)", record))
    tres = dict(item.split("=", 1) for item in fields.get("AllocTRES", "").split(",") if "=" in item)
    require(tres.get("gres/gpu") == "4" and tres.get("cpu") == "64"
            and tres.get("mem") in {"384G", "393216M"}
            and fields.get("CPUs/Task") == "64",
            "Follow-up requires four allocated GPUs, 64 CPUs/task and 384 GiB host memory")


def load_design(root, directory=None, protocol_sha256=None):
    """Verify design and prediction inheritance without opening any target outcomes."""
    root = Path(root).resolve()
    directory = Path(directory or root / DIRECTORY).resolve()
    original = root / "benchmark/estimation"
    original_anchor(root)
    protocol_path = directory / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol_sha256:
        require(digest(protocol_path) == protocol_sha256, "Follow-up protocol anchor changed")
    fixed = {
        "protocol_version": "host-capacity-1.0", "group": GROUP,
        "planned_pairs": 9, "planned_processes": 18, "target_configurations": 6,
        "evaluation_seeds": SEEDS, "allocation_gpus": 4, "cpus_per_task": 64,
        "host_memory_mib": 393216, "memory_margin_limit_mib": 38912,
        "device_capacity_mib": 40960,
        "original_protocol_sha256": ORIGINAL_PROTOCOL_SHA256,
        "original_prediction_seal_sha256": ORIGINAL_SEAL_SHA256,
    }
    require(all(protocol.get(k) == v for k, v in fixed.items()), "Changed follow-up design")
    require(protocol["resource_cap"] == {
        "initial_allocations": 9, "max_concurrent_allocations": 3,
        "gpus_per_allocation": 4, "wall_minutes_per_allocation": 55,
        "payload_timeout_seconds": 1500, "idle_timeout_seconds": 180,
        "maximum_reserved_gpu_hours": 33, "retries": 0, "requeue": False,
    }, "Changed follow-up resource ceiling")
    require(set(protocol["implementation_sha256"]) == set(IMPLEMENTATIONS),
            "Incomplete frozen execution implementation")
    for name, expected_hash in protocol["implementation_sha256"].items():
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts, "Unsafe implementation path")
        require(digest(root / path) == expected_hash, f"Changed frozen implementation: {name}")
    expected_rows, targets, predictions, origins = planned(root)
    require(read_csv(directory / "matrix.csv") == expected_rows
            and read_csv(directory / "targets.csv") == targets
            and json.loads((directory / "predictions.json").read_text()) == predictions
            and protocol["prediction_origins"] == origins,
            "New design/predictions differ from exact historical-predictor inheritance")
    old_protocol = json.loads((original / "protocol.json").read_text())
    require(protocol["data_sha256"] == old_protocol["data_sha256"]
            and protocol["model"] == old_protocol["model"]
            and protocol["model_revision"] == old_protocol["model_revision"],
            "Follow-up changes data or model")
    freeze = json.loads((directory / "study_freeze.json").read_text())
    require(freeze["protocol_sha256"] == digest(protocol_path)
            and freeze["original_prediction_seal_sha256"] == ORIGINAL_SEAL_SHA256
            and freeze["status"] == "frozen_after_host_censoring_before_followup_execution"
            and set(freeze["files_sha256"]) == {"matrix.csv", "targets.csv", "predictions.json"},
            "Follow-up freeze does not bind the complete design")
    for name, expected_hash in freeze["files_sha256"].items():
        require(digest(directory / name) == expected_hash, f"Changed sealed follow-up file: {name}")
    when = datetime.fromisoformat(freeze["frozen_at_utc"])
    require(when.tzinfo is not None and when <= datetime.now(timezone.utc)
            and when >= datetime.fromisoformat(protocol["specified_at_utc"]),
            "Invalid follow-up freeze chronology")
    return protocol, freeze, expected_rows, targets, predictions


def verify_protocol(source, args, rows):
    from memory_tuner import run_matched_gpu as runner
    source = Path(source).resolve()
    require(args.protocol.resolve().parent == source / DIRECTORY
            and args.matrix.resolve() == source / DIRECTORY / "matrix.csv",
            "Unexpected follow-up source paths")
    protocol, _, expected, _, _ = load_design(source, protocol_sha256=args.protocol_sha256)
    require(rows == expected and digest(args.matrix) == args.matrix_sha256,
            "Follow-up matrix anchor differs")
    # Reuse the original seal validator, including all source and data hashes.
    old = source / "benchmark/estimation"
    original_args = SimpleNamespace(
        protocol=old / "protocol.json", protocol_sha256=ORIGINAL_PROTOCOL_SHA256,
        matrix=old / "matrix.csv", matrix_sha256=digest(old / "matrix.csv"))
    runner.verify_protocol(source, original_args, read_csv(old / "matrix.csv"))
    return protocol


def freeze(root):
    root = Path(root).resolve()
    directory = root / DIRECTORY
    require(not directory.exists(), "Never overwrite a frozen follow-up")
    original = root / "benchmark/estimation"
    original_anchor(root)
    rows, targets, predictions, origins = planned(root)
    old_protocol = json.loads((original / "protocol.json").read_text())
    protocol = dict(
        protocol_version="host-capacity-1.0", specified_at_utc=datetime.now(timezone.utc).isoformat(),
        group=GROUP, planned_pairs=9, planned_processes=18, target_configurations=6,
        evaluation_seeds=SEEDS, allocation_gpus=4, cpus_per_task=64, host_memory_mib=393216,
        device_capacity_mib=40960, memory_margin_limit_mib=38912,
        model=old_protocol["model"], model_revision=old_protocol["model_revision"],
        data_sha256=old_protocol["data_sha256"], prediction_origins=origins,
        original_protocol_sha256=ORIGINAL_PROTOCOL_SHA256,
        original_prediction_seal_sha256=ORIGINAL_SEAL_SHA256,
        rationale="Original four-GPU c2 runs were censored by the 192 GiB host limit. "
                  "This one follow-up tests all three original workloads under a separately "
                  "declared expanded host-resource condition; it does not replace original slots.",
        estimators="Exact inherited predictions, donor choices and numerical peaks. No fitting, "
                   "threshold change, calibration, or update from either panel's outcomes.",
        scope="Known Qwen family and A100 stack. New seeds and host-resource condition; "
              "not blind method development, external validation, or cross-hardware transfer.",
        comparison="Two versus four task GPUs within the same four-GPU/64-CPU/384-GiB allocation. "
                   "Matched settings and prompt seed, not necessarily identical generated responses. "
                   "No causal effect of host capacity is estimated against the original panel.",
        order_seed=2026091511, subset_seed=2026091512, acquisition_seed=2026091513,
        randomization="Five two-GPU-first and four four-GPU-first pairs; each workload has both "
                      "orders. All six two-of-four subsets occur, with three appearing twice; "
                      "each allocation ordinal appears four or five times.",
        outcome_rule="All three eligible seeds required for a repeated label. Any validated GPU "
                     "memory failure implies memory_failure; otherwise any completed peak above "
                     "38912 MiB implies above_margin; otherwise within_margin. Other slots are "
                     "unresolved with partial eligible evidence retained.",
        failure_policy="No retries, replacements, requeues or further host-memory escalation. "
                       "Each pair is claimed once. Continue period two only after verified cleanup "
                       "and all-device idle checks. Preserve nonlaunches and incomplete pairs.",
        stopping="Exactly nine planned allocation claims, each capped at 55 minutes. A systemic "
                 "safety or infrastructure problem may stop remaining work but cannot authorize "
                 "replacement slots. Stopping must not depend on prediction correctness or "
                 "favorability. Diagnosed GPU-memory failures are expected study outcomes, not "
                 "by themselves infrastructure-abort grounds. Any safety or infrastructure stop "
                 "is documented, and all unstarted slots remain in the inventory.",
        primary_reporting="Separate six-configuration inventory, all eighteen seed slots, each "
                          "method's useful/failing/above-margin approvals and unscored approvals. "
                          "No pooled denominator with the original panel and no population interval.",
        numerical_reporting="Peak error uses only configurations with three completed invocations, "
                            "comparing the inherited prediction with their maximum external completed "
                            "peak. Donor-copy numerical values are inherited source measurements, "
                            "not refitted target estimates. Host-kill and GPU-failure peaks are "
                            "not imputed or scored as completed peaks. Paired peak and runtime "
                            "differences require both invocations to complete.",
        cost_scope=old_protocol["cost_scope"],
        resource_cap=dict(initial_allocations=9, max_concurrent_allocations=3,
                          gpus_per_allocation=4, wall_minutes_per_allocation=55,
                          payload_timeout_seconds=1500, idle_timeout_seconds=180,
                          maximum_reserved_gpu_hours=33, retries=0, requeue=False),
        implementation_sha256={name: digest(root / name) for name in IMPLEMENTATIONS},
    )
    directory.mkdir(parents=True)
    write_csv(directory / "matrix.csv", rows)
    write_csv(directory / "targets.csv", targets)
    write_new_json(directory / "predictions.json", predictions)
    write_new_json(directory / "protocol.json", protocol)
    write_new_json(directory / "study_freeze.json", dict(
        status="frozen_after_host_censoring_before_followup_execution",
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=digest(directory / "protocol.json"),
        original_prediction_seal_sha256=ORIGINAL_SEAL_SHA256,
        files_sha256={name: digest(directory / name) for name in
                      ("matrix.csv", "targets.csv", "predictions.json")},
    ))
    load_design(root)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    directory = freeze(args.root)
    print(json.dumps({"directory": str(directory),
                      "protocol_sha256": digest(directory / "protocol.json"),
                      "matrix_sha256": digest(directory / "matrix.csv"),
                      "study_freeze_sha256": digest(directory / "study_freeze.json")}))


if __name__ == "__main__":
    main()
