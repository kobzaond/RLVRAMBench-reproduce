"""CPU-only independent reconstruction of the prospective paired GPU study.

Usage: python -m memory_tuner.collect_estimation_study --root ARTIFACT
--source-root SOURCE --protocol-dir PROTOCOL_DIR --output NEW_DIRECTORY
--source-commit COMMIT. Source defaults to ARTIFACT; protocol directory defaults
to SOURCE/benchmark/estimation. No fitting, GPU queries, launches,
or raw-file rewriting. Recorded absolute paths remap inside ARTIFACT only.
Alternatively --execution-commits accepts an externally supplied JSON object
pair_id -> exact 40-character execution commit, with exactly all 18 planned IDs.

Optional --allocation-accounting: JSON list of scheduler records with job_id,
allocated_gpu_count, started_at_epoch, ended_at_epoch. A manifest's scheduled
EndTime is a deadline, NOT actual job end. Without accounting, whole-allocation
cost remains unknown. GPU time is provisioned/reserved, not utilization weighted.

Hashes detect inconsistency, not replacement of an entire archive. Optional
--protocol-sha256 and --prediction-seal-sha256 accept externally retained anchors.
The execution runner verifies the seal but does not record its hash per period.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import re

from memory_tuner import estimation_baselines as eb
from memory_tuner import run_matched_gpu as runner
from memory_tuner import capture_allocation_accounting as scheduler_accounting
from memory_tuner.artifact_paths import PATH_FIELDS, recorded_path
from memory_tuner.benchmark_v2_env import FIELD_TO_ENV, OPTIONAL_FIELD_TO_ENV
from memory_tuner.device_contract import digest, write_new_json
from memory_tuner.grpo_raw_evidence import MATCH_FIELDS, normalized, select_attempt
from memory_tuner.scientific_revision_analysis import validate_fixed_stack

LIMIT = 38912.
SEEDS = [151, 152, 153]
DATA_ROOTS = {"gsm8k": "data/gsm8k", "math": "data/math",
              "code_heavy_tail": "data/codecontests/heavy_tail"}
EVIDENCE_ERRORS = (ValueError, KeyError, TypeError, OSError, IndexError)
DERIVED_PERIOD_FIELDS = {
    "status", "identity_validated", "observed_worker_uuids", "worker_coverage_complete",
    "parent_cuda_inventory_validated", "probe_context_released", "observed_worker_records",
    "worker_initialization", "task_peak_memory_mib", "task_trace_samples",
    "task_trace_first_timestamp_ns", "task_trace_last_timestamp_ns",
    "failure_kind", "failure_stage", "validation_error",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, name):
    require(not isinstance(value, bool), f"Invalid {name}")
    number = float(value)
    require(math.isfinite(number) and number >= 0, f"Invalid {name}")
    return number


def epoch(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, "Provenance time lacks timezone")
    return parsed.timestamp()


def inside(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe artifact path")
    path = root / relative
    require(path.resolve().is_relative_to(root.resolve()), "Evidence escapes supplied root")
    return path


def remap(value, root, recorded_source=None):
    path = Path(str(value))
    require(".." not in path.parts and str(value), "Unsafe recorded path")
    if path.is_absolute():
        if recorded_source and path.is_relative_to(Path(recorded_source)):
            path = path.relative_to(Path(recorded_source))
        elif path.is_relative_to(root):
            path = path.relative_to(root)
        else:
            anchors = [i for i, part in enumerate(path.parts)
                       if part in {"output", "data", "profiles", "benchmark", "memory_tuner"}]
            require(bool(anchors), "Unmapped recorded path")
            path = Path(*path.parts[anchors[0]:])
    return inside(root, path)


class Evidence:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.hashes = {}

    def touch(self, path):
        path = inside(self.root, Path(path).relative_to(self.root))
        value = {"sha256": digest(path), "size_bytes": path.stat().st_size}
        name = path.relative_to(self.root).as_posix()
        require(name not in self.hashes or self.hashes[name] == value,
                f"Evidence changed during collection: {name}")
        self.hashes[name] = value
        return path

    def json(self, path):
        return json.loads(self.touch(path).read_text())

    def error(self, error):
        # Missingness is part of the result. Its diagnostics must not depend on
        # the temporary extraction directory used by an isolated replay.
        return f"{type(error).__name__}: {error}".replace(str(self.root), "<artifact>")

    def journal(self, path):
        rows = [json.loads(line) for line in self.touch(path).read_text().splitlines()]
        require(bool(rows), "Empty durable journal")
        for i, row in enumerate(rows):
            require(row["sequence"] == i and row["timestamp_ns"] > 0
                    and row["monotonic_ns"] >= 0, "Broken journal sequence/time")
            if i:
                require(row["monotonic_ns"] >= rows[i - 1]["monotonic_ns"]
                        and row["timestamp_ns"] >= rows[i - 1]["timestamp_ns"],
                        "Journal clock moves backwards")
        return rows

    def verify_unchanged(self):
        for name, value in self.hashes.items():
            require(digest(inside(self.root, name)) == value["sha256"],
                    f"Evidence changed during collection: {name}")


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_frozen(evidence, protocol_sha256=None, prediction_seal_sha256=None,
                protocol_evidence=None):
    root = evidence.root
    protocol_evidence = protocol_evidence or Evidence(inside(root, "benchmark/estimation"))
    directory = protocol_evidence.root
    base_path = directory / "protocol.json"
    base = protocol_evidence.json(base_path)
    base_hash = digest(base_path)
    if protocol_sha256:
        require(base_hash == protocol_sha256, "Externally anchored protocol hash differs")
    fixed = {"protocol_version": "estimation-1.0", "allocation_gpus": 4,
             "planned_pairs": 18, "planned_processes": 36,
             "model": runner.MODEL, "model_revision": runner.MODEL_REVISION,
             "evaluation_seeds": SEEDS, "memory_margin_limit_mib": LIMIT,
             "device_capacity_mib": 40960}
    require(all(base.get(k) == v for k, v in fixed.items()), "Wrong prospective protocol")
    require(all(base["resource_cap"].get(k) == v for k, v in {
        "payload_timeout_seconds": 1500, "idle_timeout_seconds": 180,
        "wall_minutes_per_allocation": 55, "retries": 0, "requeue": False,
    }.items()), "Changed frozen execution limits")
    effective = {**base, "input_sha256": dict(base["input_sha256"])}
    amendments, last_time = {}, epoch(base["frozen_at_utc"])
    for i, path in enumerate(sorted(directory.glob("amendment-*.json")), 1):
        amendment = protocol_evidence.json(path)
        require(path.name == f"amendment-{i:02d}.json"
                and amendment["base_protocol_sha256"] == base_hash
                and amendment["before_new_target_execution"] is True,
                "Invalid ordered amendment overlay")
        require(set(amendment["updates"]) <= {
            "features", "feature_scope", "component_regression", "scoring"},
            "Amendment changes experimental design")
        for name in amendment["implementation_sha256"]:
            inside(root, name)
            require(name == "benchmark.py" or
                    (name.startswith("memory_tuner/") and name.endswith(".py")),
                    "Amendment changes non-implementation input")
        amended_time = epoch(amendment["amended_at_utc"])
        require(amended_time >= last_time, "Amendments not chronological")
        last_time = amended_time
        effective.update(amendment["updates"])
        effective["input_sha256"].update(amendment["implementation_sha256"])
        for name, expected in amendment.get("unchanged_inputs_verified_sha256", {}).items():
            require(effective["input_sha256"].get(name) == expected,
                    "Amendment unchanged-input claim differs")
        amendments[path.name] = digest(path)
    require(effective["features"] == list(eb.FEATURES), "Scorer feature order differs")
    require({"benchmark/estimation/" + name for name in
             ("matrix.csv", "targets.csv", "model_metadata.json")} <= set(effective["input_sha256"]),
            "Protocol does not seal all settings/metadata inputs")
    for name, expected in effective["input_sha256"].items():
        require(digest(evidence.touch(inside(root, name))) == expected,
                f"Effective frozen input differs: {name}")
    # Explicit alternate protocol directories must contain the SAME sealed
    # settings/metadata, not merely refer to correctly hashed files in SOURCE.
    for name in ("matrix.csv", "targets.csv", "model_metadata.json"):
        require(digest(protocol_evidence.touch(directory / name)) ==
                effective["input_sha256"]["benchmark/estimation/" + name],
                f"Actual protocol-directory input differs: {name}")
    seal_path = directory / "prediction_freeze.json"
    seal = protocol_evidence.json(seal_path)
    if prediction_seal_sha256:
        require(digest(seal_path) == prediction_seal_sha256, "Prediction seal anchor differs")
    require(seal["protocol_sha256"] == base_hash
            and seal["amendments_sha256"] == amendments
            and seal["effective_input_sha256"] == effective["input_sha256"]
            and seal["target_configurations"] == 12 and seal["predictions"] == 36
            and seal["status"] == "sealed_before_new_standard_grpo_execution"
            and epoch(seal["frozen_at_utc"]) >= last_time,
            "Prediction seal does not bind the effective protocol")
    require(seal["implementation_sha256"] == digest(Path(eb.__file__))
            == effective["input_sha256"]["memory_tuner/estimation_baselines.py"],
            "Active scoring implementation differs from frozen implementation")
    require(set(seal["files_sha256"]) ==
            {"retrospective.json", "fitted_model.json", "predictions.json"},
            "Incomplete prediction freeze")
    for name, expected in seal["files_sha256"].items():
        require(digest(protocol_evidence.touch(directory / name)) == expected,
                f"Frozen prediction output differs: {name}")
    rows = read_csv(protocol_evidence.touch(directory / "matrix.csv"))
    pairs = runner.validate_matrix(rows)
    targets = read_csv(protocol_evidence.touch(directory / "targets.csv"))
    ids = {r["configuration_id"] for r in targets}
    require(len(targets) == len(ids) == 12 and ids == set(base["target_configuration_ids"])
            == {r["configuration_id"] for r in rows}, "Target configuration coverage differs")
    for target in targets:
        selected = [r for r in rows if r["configuration_id"] == target["configuration_id"]]
        require(sorted(int(r["training_seed"]) for r in selected) == SEEDS,
                "Target does not have exactly three distinct planned seeds")
        for row in selected:
            for field in MATCH_FIELDS:
                if field != "training_seed":
                    target_field = "model_id" if field == "model" else field
                    require(normalized(target[target_field]) == normalized(row[field]),
                            f"Target settings differ from matrix: {field}")
            require(target["legacy_dataset_id"] == row["dataset"]
                    and target["configuration_level"] == row["configuration_level"],
                    "Target workload/level differs")
    predictions = protocol_evidence.json(directory / "predictions.json")
    require(Counter((p["configuration_id"], p["method"]) for p in predictions) ==
            Counter((cid, method) for cid in ids for method in eb.METHODS),
            "Frozen prediction coverage differs")
    for pred in predictions:
        require(pred["predicted_state"] in eb.STATES, "Invalid predicted state")
        if pred["predicted_peak_mib"] is not None:
            finite(pred["predicted_peak_mib"], "predicted peak")
    return dict(protocol=effective, protocol_sha256=base_hash, amendments_sha256=amendments,
                prediction_seal_sha256=digest(seal_path), seal=seal, rows=rows, pairs=pairs,
                targets=targets, predictions=predictions,
                matrix_sha256=digest(directory / "matrix.csv"))


def validate_payload(evidence, attempt, recorded):
    require(isinstance(recorded, dict) and recorded, "Missing payload hash manifest")
    actual = {}
    for path in sorted(attempt.rglob("*")):
        require(not path.is_symlink(), "Symlink in immutable attempt payload")
        if path.is_file():
            evidence.touch(path)
            actual[path.relative_to(attempt).as_posix()] = evidence.hashes[
                path.relative_to(evidence.root).as_posix()]
    require(actual == recorded, "Attempt payload files/hash/size mismatch")


def one_event(events, name, period=None):
    selected = [e for e in events if e["event"] == name
                and (period is None or e.get("period") == period)]
    require(len(selected) == 1, f"Expected one {name} event for period {period}")
    return selected[0]


def validate_sample(sample, inventory):
    runner.validate_allocation_inventory(sample["gpus"])
    by_uuid = {g["uuid"]: g for g in inventory}
    require({g["uuid"] for g in sample["gpus"]} == set(by_uuid),
            "Allocation trace does not cover the allocated four UUIDs")
    for gpu in sample["gpus"]:
        require(all(gpu[k] == by_uuid[gpu["uuid"]][k]
                    for k in ("index", "pci_bus_id", "name", "total_mib", "mig_mode")),
                "Allocation device identity changed")
        require(finite(gpu["used_mib"], "allocation memory") <= 40960
                and finite(gpu["utilization_pct"], "allocation utilization") <= 100,
                "Invalid allocation sample")
    for process in sample["processes"]:
        require(process["uuid"] in by_uuid and isinstance(process["pid"], int)
                and process["pid"] > 0, "Invalid allocation process evidence")


def validate_pair(evidence, frozen, pair_id, source_commit):
    group = frozen.get("group", runner.GROUP)
    protocol_relative_dir = Path(frozen.get("protocol_relative_dir", "benchmark/estimation"))
    directory = inside(evidence.root, Path("output") / group / "pairs" / pair_id)
    manifest = evidence.json(directory / "pair.json")
    events = evidence.journal(directory / "ledger.jsonl")
    selected = sorted(frozen["pairs"][pair_id], key=lambda r: int(r["period"]))
    require(manifest["pair_id"] == pair_id and manifest["rows"] == selected
            and manifest["matrix_sha256"] == frozen["matrix_sha256"]
            and manifest["protocol_sha256"] == frozen["protocol_sha256"]
            and manifest["no_retry"] is True, "Pair manifest differs from frozen design")
    for field, name in (("matrix_path", "matrix.csv"), ("protocol_path", "protocol.json")):
        recorded = Path(manifest[field])
        expected = protocol_relative_dir / name
        require(recorded.parts[-len(expected.parts):] == expected.parts,
                "Unexpected recorded protocol/settings location")
        recorded_source = manifest.get("source_root") or str(recorded.parents[len(expected.parts) - 1])
        require(remap(manifest[field], evidence.root, recorded_source) ==
                evidence.root / expected, "Pair frozen path differs")
    claimed = one_event(events, "pair_claimed")
    require(claimed["sequence"] == 0 and claimed["pair_id"] == pair_id
            and claimed["matrix_sha256"] == frozen["matrix_sha256"]
            and claimed["protocol_sha256"] == frozen["protocol_sha256"]
            and claimed["source_commit_requested"] == source_commit,
            "Pair claim has wrong provenance")
    freeze = frozen.get("study_freeze", frozen["seal"])
    require(claimed["timestamp_ns"] / 1e9 > epoch(freeze["frozen_at_utc"]),
            "Pair claimed before follow-up study freeze" if "study_freeze" in frozen
            else "Pair claimed before prediction freeze")
    for event in events:
        require(event["event"] in {"pair_claimed", "pair_started", "dispatched",
                                  "spawned", "terminal", "pair_terminal"},
                "Unknown/retry journal event")
        if event["event"] in {"dispatched", "spawned", "terminal"}:
            require(event.get("period") in (1, 2), "Unexpected period in ledger")
            require(sum(e["event"] == event["event"] and e.get("period") == event["period"]
                        for e in events) == 1, "Duplicate dispatch/spawn/terminal")
    allocation = manifest["allocation"]
    if allocation is None:
        require(not any(e["event"] in {"dispatched", "spawned"} for e in events),
                "Launch without allocation provenance")
        return dict(directory=directory, manifest=manifest, events=events, trace=[])
    require(manifest["source_commit"] == source_commit, "Wrong frozen execution source")
    require(str(manifest["job_id"]) == str(allocation["job_id"])
            and allocation["gpu_count"] == 4, "Wrong allocation job/count")
    if frozen.get("allocation_validator"):
        frozen["allocation_validator"](allocation)
    runner.validate_allocation_inventory(allocation["gpus"])
    require(allocation["gpus"] == sorted(allocation["gpus"],
            key=lambda g: (g["pci_bus_id"], g["uuid"])), "Unfrozen physical GPU ordering")
    require(0 < allocation["ends_at_epoch"] - allocation["started_at_epoch"] <= 3301,
            "Allocation deadline differs from cap")
    started = one_event(events, "pair_started")
    require(started["pair_id"] == pair_id and started["allocation_gpu_count"] == 4,
            "Wrong pair-start provenance")
    trace = evidence.journal(directory / "allocation-trace.jsonl")
    for sample in trace:
        require(sample["event"] == "allocation_sample" and sample["period"] in (1, 2)
                and sample["phase"] in {"pre_idle", "payload", "post_idle"},
                "Unexpected allocation trace event")
        validate_sample(sample, allocation["gpus"])
        require(allocation["started_at_epoch"] <= sample["timestamp_ns"] / 1e9
                <= allocation["ends_at_epoch"] + 30, "Allocation trace outside job interval")
    return dict(directory=directory, manifest=manifest, events=events, trace=trace)


def idle_gate(samples):
    return len(samples) >= 2 and all(runner.is_idle(s) for s in samples[-2:])


def validate_period_provenance(evidence, frozen, pair, row, record):
    manifest, events, trace = pair["manifest"], pair["events"], pair["trace"]
    period, job = int(row["period"]), str(manifest["job_id"])
    invocation = row["experiment_id"] + "-j" + job
    require(record["pair_id"] == row["pair_id"] and record["period"] == period
            and record["experiment_id"] == row["experiment_id"]
            and record["task_gpu_count"] == int(row["gpu_count"]),
            "Period record belongs to another slot")
    terminal = one_event(events, "terminal", period)
    for field in (record.keys() | terminal.keys()) - DERIVED_PERIOD_FIELDS - {
            "event", "sequence", "timestamp_ns", "monotonic_ns"}:
        require(record.get(field) == terminal.get(field), f"Period/ledger differ: {field}")
    spawn = [e for e in events if e["event"] == "spawned" and e["period"] == period]
    require(record["launched"] is bool(spawn), "Stored launch flag disagrees with spawn ledger")
    if not spawn:
        require(not record.get("evidence_files"), "Unspawned period has unexplained payload")
        return None
    allocation = manifest["allocation"]
    require(allocation is not None, "Launched period lacks allocation")
    expected = [allocation["gpus"][i]["uuid"] for i in runner.subset_ordinals(row)]
    require(record["requested_uuids"] == expected and record["invocation_id"] == invocation
            and str(record["job_id"]) == job and record["allocation_gpu_count"] == 4,
            "Period UUID/job/invocation mismatch")
    group = frozen.get("group", runner.GROUP)
    attempt = inside(evidence.root, Path("output") / group / row["experiment_id"]
                     / ("attempt-" + job))
    require(list(attempt.parent.glob("attempt-*")) == [attempt]
            and not list(attempt.parent.glob("trial-*.json")), "Extra/retried attempt in frozen slot")
    paths = {"attempt_path": attempt, "trial_path": attempt / f"trial-{job}.json",
             "run_log": attempt / f"training-{job}.log",
             "environment_json": attempt / f"environment-{job}.json",
             "phase_memory_csv": attempt / f"phase-memory-{job}.csv",
             "gpu_telemetry_csv": attempt / f"gpu-telemetry-{job}.csv",
             "device_evidence_dir": attempt / "device-evidence",
             "allocation_trace_path": pair["directory"] / "allocation-trace.jsonl"}
    for field, path in paths.items():
        require(remap(record[field], evidence.root) == path, f"Wrong recorded {field}")
    validate_payload(evidence, attempt, record["evidence_files"])
    dispatched = one_event(events, "dispatched", period)
    require(dispatched["experiment_id"] == row["experiment_id"]
            and dispatched["invocation_id"] == invocation
            and dispatched["requested_uuids"] == expected
            and remap(dispatched["attempt_path"], evidence.root) == attempt,
            "Wrong dispatch provenance")
    require(dispatched["sequence"] < spawn[0]["sequence"] < terminal["sequence"],
            "Dispatch/spawn/terminal order differs")
    dispatch = evidence.json(attempt / "dispatch.json")
    require(dispatch["row"] == row and dispatch["invocation_id"] == invocation
            and dispatch["requested_uuids"] == expected
            and dispatch["matrix_sha256"] == frozen["matrix_sha256"]
            and dispatch["protocol_sha256"] == frozen["protocol_sha256"],
            "Raw dispatch differs from frozen slot")
    env = dispatch["environment"]
    for field, name in {**FIELD_TO_ENV, **OPTIONAL_FIELD_TO_ENV}.items():
        if field == "custom_reward_function_path" or not row.get(field):
            continue
        require(normalized(env.get(name)) == normalized(row[field]),
                f"Dispatched setting differs: {field}")
    require(env["RLVRAM_GPU_UUIDS"] == ",".join(expected)
            and env["RLVRAM_ALLOCATION_GPU_UUIDS"] ==
                ",".join(g["uuid"] for g in allocation["gpus"])
            and env["RLVRAM_INVOCATION_ID"] == invocation
            and env["RLVRAM_PERIOD"] == str(period)
            and env["BENCHMARK_SOURCE_ROOT"] == manifest["source_root"]
            and env["RUN_NAME"] == f"{group}/{row['experiment_id']}/attempt-{job}"
            and Path(env["RLVRAM_MODEL_SNAPSHOT"]).name == runner.MODEL_REVISION,
            "Dispatch runtime/device/source identity differs")
    if row["dataset"] == "code_heavy_tail":
        require(Path(env["CUSTOM_REWARD_FUNCTION_PATH"]) ==
                Path(manifest["source_root"]) / "memory_tuner/code_shape_reward.py",
                "Code reward not rebound to immutable execution source")
    else:
        require(env["CUSTOM_REWARD_FUNCTION_PATH"] == "null", "Unexpected custom reward")
    samples = [s for s in trace if s["period"] == period]
    phases = {phase: [s for s in samples if s["phase"] == phase]
              for phase in ("pre_idle", "payload", "post_idle")}
    require(idle_gate(phases["pre_idle"]), "Missing all-four pre-period idle evidence")
    require(phases["pre_idle"][-1]["monotonic_ns"] < dispatched["monotonic_ns"],
            "Idle gate follows dispatch")
    require(phases["payload"] and record["allocation_samples"] == len(phases["payload"]),
            "Missing/incomplete allocation-wide payload sampling")
    require(all(spawn[0]["monotonic_ns"] <= s["monotonic_ns"] < terminal["monotonic_ns"]
                for s in phases["payload"]), "Payload sample belongs to another invocation")
    if phases["post_idle"]:
        require(phases["payload"][-1]["monotonic_ns"] < phases["post_idle"][0]["monotonic_ns"]
                and phases["post_idle"][-1]["monotonic_ns"] < terminal["monotonic_ns"],
                "Cleanup sample order differs")
    if period == 2:
        previous = one_event(events, "terminal", 1)
        previous_post = [s for s in trace if s["period"] == 1 and s["phase"] == "post_idle"]
        require(idle_gate(previous_post)
                and previous["monotonic_ns"] < phases["pre_idle"][0]["monotonic_ns"]
                and previous_post[-1]["monotonic_ns"] < previous["monotonic_ns"],
                "Second period lacks preceding all-four cleanup gate")
    contaminated = False
    for sample in phases["payload"]:
        excluded = [p for p in sample["processes"] if p["uuid"] not in expected]
        require(sample["excluded_device_processes"] == excluded,
                "Stored excluded-process list disagrees with raw allocation trace")
        require(all(p in sample["processes"] for p in sample["foreign_processes"]),
                "Foreign process annotation absent from process inventory")
        contaminated |= bool(excluded or sample["foreign_processes"])
    require(record["contamination_detected"] is contaminated,
            "Stored contamination flag disagrees with allocation trace")
    duration = finite(record["invocation_elapsed_seconds"], "invocation duration")
    payload = finite(record["payload_elapsed_seconds"], "payload duration")
    available = (terminal["monotonic_ns"] - dispatched["monotonic_ns"]) / 1e9
    require(0 < payload <= duration <= available + .1, "Invocation duration differs from ledger")
    require(payload <= 1501 or record["timed_out"] or record["error"], "Unreported payload timeout")
    for field, cost in (("task_gpu_seconds", duration * len(expected)),
                        ("allocated_gpu_seconds_during_invocation", duration * 4)):
        require(math.isclose(finite(record[field], field), cost, abs_tol=.01),
                "Stored GPU cost disagrees with duration/count")
    return dict(attempt=attempt, expected=expected, invocation=invocation, job=job,
                dispatched=dispatched, spawned=spawn[0], terminal=terminal, contaminated=contaminated,
                post_idle=idle_gate(phases["post_idle"]), invocation_seconds=duration)


def validate_raw_environment(evidence, frozen, pair, row, context):
    attempt, job, expected = context["attempt"], context["job"], context["expected"]
    trial = evidence.json(attempt / f"trial-{job}.json")
    allocation = pair["manifest"]["allocation"]
    require(str(trial["job_id"]) == job, "Raw trial belongs to another job")
    require(all(trial.get(k) for k in (
        "run_log", "environment_json", "phase_memory_csv", "gpu_telemetry_csv", "data_dir")),
        "Raw trial lacks designated evidence/data paths")
    for field in PATH_FIELDS:
        if not trial.get(field):
            continue
        mapped = remap(trial[field], evidence.root)
        require(recorded_path(str(trial[field]), evidence.root) == mapped,
                "Raw parser and collector path mappings differ")
        if field == "data_dir":
            require(mapped == evidence.root / DATA_ROOTS[row["dataset"]], "Wrong raw data path")
        else:
            suffixes = {"run_log": f"training-{job}.log",
                        "phase_memory_csv": f"phase-memory-{job}.csv",
                        "gpu_telemetry_csv": f"gpu-telemetry-{job}.csv",
                        "environment_json": f"environment-{job}.json",
                        "allocator_trace_dir": "allocator-traces",
                        "response_length_trace": f"response-lengths-{job}.jsonl"}
            require(mapped == attempt / suffixes[field], f"Wrong raw evidence path: {field}")
    environment = evidence.json(attempt / f"environment-{job}.json")
    validate_fixed_stack(environment)
    require(environment["host"]["hostname"] == allocation["hostname"]
            and str(environment["slurm"]["SLURM_JOB_ID"]) == job
            and environment["model"]["requested_model"] == runner.MODEL
            and environment["gpu_device_contract"]["task_gpu_count"] == len(expected),
            "Raw environment host/job/model/task count differs")
    require(context["dispatched"]["timestamp_ns"] / 1e9
            <= epoch(environment["captured_at_utc"]) <= context["terminal"]["timestamp_ns"] / 1e9,
            "Environment metadata outside actual period")
    gpus = list(csv.reader(environment["gpus_csv"].splitlines(), skipinitialspace=True))
    require(len(gpus) == len(expected) and {g[2].strip() for g in gpus} == set(expected),
            "Environment GPU inventory differs")
    for gpu in gpus:
        original = next(g for g in allocation["gpus"] if g["uuid"] == gpu[2].strip())
        require(len(gpu) == 5 and int(gpu[0]) == original["index"]
                and gpu[1].strip() == original["name"]
                and gpu[3].strip() == original["driver_version"]
                and float(gpu[4].split()[0]) == 40960, "Environment physical device differs")
    for name in ("train.parquet", "test.parquet"):
        item = environment["dataset"][name]
        relative = DATA_ROOTS[row["dataset"]] + "/" + name
        require(item["sha256"] == frozen["protocol"]["data_sha256"][relative]
                and finite(item["size_bytes"], "dataset size") > 0
                and remap(item["path"], evidence.root) == evidence.root / relative,
                "Dataset provenance differs from frozen hashes")
    directory = attempt / "device-evidence"
    for path in directory.glob("*.json"):
        item = evidence.json(path)
        require(item["invocation_id"] == context["invocation"]
                and item["hostname"] == allocation["hostname"]
                and context["dispatched"]["timestamp_ns"] <= item["timestamp_ns"]
                <= context["terminal"]["timestamp_ns"], "Stale/wrong-period device identity")
    gate = evidence.json(directory / "probe-exit-idle.json")
    require(gate["allocation_uuids"] == [g["uuid"] for g in allocation["gpus"]],
            "Probe idle gate belongs to another allocation")
    for sample in gate["observations"]:
        validate_sample(sample, allocation["gpus"])
    mapping = evidence.json(attempt / "gpu-device-map.json")
    require(mapping["scope"] == "task_devices"
            and mapping["columns"] == [
                {"gpu_index": i, "memory_csv_column": i + 1, "uuid": uuid}
                for i, uuid in enumerate(expected)], "Wrong task trace column mapping")
    with (attempt / f"gpu-memory-{job}.csv").open() as handle:
        memory = [(int(r[0]), list(map(float, r[1:]))) for r in csv.reader(handle)]
    with (attempt / f"phase-memory-{job}.csv").open() as handle:
        phases = list(csv.reader(handle))
    require(len(phases) == len(memory), "Phase/memory sampling lengths differ")
    telemetry = read_csv(attempt / f"gpu-telemetry-{job}.csv")
    require(len(telemetry) == len(memory) * len(expected), "Incomplete per-device telemetry")
    for sample_index, ((timestamp, values), phase) in enumerate(zip(memory, phases)):
        require(context["dispatched"]["timestamp_ns"] <= timestamp
                <= context["terminal"]["timestamp_ns"], "Task trace outside actual period")
        require(len(phase) == len(expected) + 3 and int(phase[0]) == timestamp
                and list(map(float, phase[3:])) == values and int(phase[1]) in (0, 1),
                "Phase/memory samples differ or wrong horizon")
        observed = telemetry[sample_index * len(expected):(sample_index + 1) * len(expected)]
        require({int(t["gpu_index"]) for t in observed} == set(range(len(expected))),
                "Duplicate/missing device in telemetry sample")
        for item in observed:
            require(int(item["timestamp_ns"]) == timestamp
                    and int(item["step"]) == int(phase[1]) and item["phase"] == phase[2]
                    and float(item["memory_used_mib"]) == values[int(item["gpu_index"])],
                    "Per-device telemetry does not match memory/phase sample")
    return trial


def validate_completed_allocator(evidence, context, trial):
    """Require observed instrumentation, not merely its requested launch flags.

    The hook has no invocation ID: bind its PID/rank/device and wall-clock
    interval to this period's validated worker identities. One actor-update
    event suffices; this is not a per-rank or full-phase coverage requirement.
    """
    attempt = context["attempt"]
    directory = attempt / "allocator-traces"
    require(trial.get("allocator_trace_dir")
            and remap(trial["allocator_trace_dir"], evidence.root) == directory,
            "Completed run lacks its designated allocator trace directory")
    paths = sorted(directory.glob("*.jsonl"))
    require(bool(paths), "Completed run lacks allocator events")
    workers = [evidence.json(path) for path in
               sorted((attempt / "device-evidence").glob("worker-*.json"))]
    trainer = evidence.json(attempt / "device-evidence/trainer-terminal.json")
    actor_updates = 0
    for path in paths:
        lines = evidence.touch(path).read_text().splitlines()
        require(bool(lines), f"Empty allocator trace: {path.name}")
        for line_number, line in enumerate(lines, 1):
            label = f"Allocator event {path.name}:{line_number}"
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{label}: malformed JSON") from error
            require(isinstance(event, dict), f"{label}: expected an object")
            for field in ("pid", "rank", "device", "started_ns", "finished_ns"):
                require(type(event.get(field)) is int
                        and event[field] >= (1 if field in
                                            {"pid", "started_ns", "finished_ns"} else 0),
                        f"{label}: invalid {field}")
            require(path.name == f"allocator-{event['pid']}.jsonl",
                    f"{label}: filename/PID mismatch")
            require(isinstance(event.get("phase"), str) and event["phase"].strip(),
                    f"{label}: missing phase")
            require(event.get("error_type") == "", f"{label}: missing/nonempty error_type")
            for field in ("allocated_mib", "reserved_mib", "max_allocated_mib", "max_reserved_mib"):
                require(type(event.get(field)) in (int, float), f"{label}: invalid {field}")
                finite(event[field], f"{label} {field}")
            require(context["spawned"]["timestamp_ns"] <= event["started_ns"]
                    <= event["finished_ns"] <= context["terminal"]["timestamp_ns"],
                    f"{label}: outside actual period or reversed interval")
            associated = [
                worker for worker in workers
                if type(worker["pid"]) is int and type(worker["rank"]) is int
                and worker["pid"] == event["pid"] and worker["rank"] == event["rank"]
                and worker["timestamp_ns"] <= event["started_ns"]
                and any(type(device["cuda_index"]) is int
                        and device["cuda_index"] == event["device"]
                        and device["uuid"] == worker["active_uuid"]
                        for device in worker["cuda_devices"])
            ]
            require(bool(associated), f"{label}: no matching initialized worker PID/rank/device")
            if event["phase"] == "actor_update":
                require(any("actor" in worker["role"].split("_") for worker in associated),
                        f"{label}: actor update belongs to a non-actor worker")
                require(event["finished_ns"] <= trainer["timestamp_ns"],
                        f"{label}: actor update follows trainer termination")
                actor_updates += 1
    require(actor_updates > 0, "Completed run lacks an allocator actor_update event")


def invocation_cost(record, pair, row):
    """Timing eligibility is separate from eligibility of a memory outcome."""
    period = int(row["period"])
    events = pair["events"]
    terminal = one_event(events, "terminal", period)
    fields = ("pair_id", "experiment_id", "period", "task_gpu_count", "launched",
              "invocation_elapsed_seconds", "task_gpu_seconds",
              "allocated_gpu_seconds_during_invocation")
    require(all(record.get(k) == terminal.get(k) for k in fields),
            "Invocation timing/identity differs from durable terminal ledger")
    require(record["pair_id"] == row["pair_id"] and record["experiment_id"] == row["experiment_id"]
            and record["period"] == period and record["task_gpu_count"] == int(row["gpu_count"]),
            "Invocation cost belongs to another slot")
    spawned = [e for e in events if e["event"] == "spawned" and e["period"] == period]
    require(record["launched"] is bool(spawned), "Invocation launch count differs")
    duration = finite(record["invocation_elapsed_seconds"], "invocation duration")
    dispatched = [e for e in events if e["event"] == "dispatched" and e["period"] == period]
    if dispatched:
        require(dispatched[0]["experiment_id"] == row["experiment_id"]
                and duration <= (terminal["monotonic_ns"] - dispatched[0]["monotonic_ns"]) / 1e9 + .1,
                "Invocation duration exceeds ledger interval")
        if spawned:
            require(dispatched[0]["sequence"] < spawned[0]["sequence"] < terminal["sequence"]
                    and duration > 0, "Invocation timing order differs")
    else:
        require(not spawned and duration == 0, "Nonzero invocation cost without dispatch")
    task, reservation = duration * int(row["gpu_count"]), duration * 4
    require(math.isclose(record["task_gpu_seconds"], task, abs_tol=.01)
            and math.isclose(record["allocated_gpu_seconds_during_invocation"], reservation, abs_tol=.01),
            "Invocation GPU cost differs from duration/count")
    return dict(launched=bool(spawned), task_gpu_seconds=task,
                invocation_reservation_gpu_seconds=reservation, invocation_elapsed_seconds=duration)


def process_slot(evidence, frozen, pair, row, pair_error=None):
    result = {k: row[k] for k in ("pair_id", "experiment_id", "configuration_id", "dataset",
                                "configuration_level")}
    result.update(period=int(row["period"]), training_seed=int(row["training_seed"]),
                  task_gpu_count=int(row["gpu_count"]), allocation_gpu_count=4,
                  state="unresolved", eligible=False, completed=False, launched=None,
                  dispatched=None, durable_terminal_record_present=False,
                  online_status=None, failure_kind=None, failure_stage=None,
                  known_memory_failure=False, known_margin_exceedance=False,
                  completed_peak_mib=None, observed_task_peak_mib=None,
                  task_gpu_seconds=None, invocation_reservation_gpu_seconds=None,
                  task_runtime_seconds=None, invocation_elapsed_seconds=None,
                  worker_initialization=None, validation_errors=[])
    try:
        require(pair_error is None, pair_error)
        period_events = [e for e in pair["events"] if e.get("period") == int(row["period"])]
        result.update(launched=any(e["event"] == "spawned" for e in period_events),
                      dispatched=any(e["event"] == "dispatched" for e in period_events),
                      durable_terminal_record_present=any(e["event"] == "terminal" for e in period_events))
        record = evidence.json(pair["directory"] / f"period-{row['period']}.json")
        result["online_status"] = record.get("status")
        result.update(invocation_cost(record, pair, row))
        context = validate_period_provenance(evidence, frozen, pair, row, record)
        if context is None:
            result.update(launched=False, task_gpu_seconds=0.,
                          invocation_reservation_gpu_seconds=0., invocation_elapsed_seconds=0.,
                          failure_stage="not_started", reason=record.get("reason"))
            return result
        result.update(launched=True, job_id=context["job"], invocation_id=context["invocation"],
                      invocation_elapsed_seconds=context["invocation_seconds"],
                      task_gpu_seconds=context["invocation_seconds"] * int(row["gpu_count"]),
                      invocation_reservation_gpu_seconds=context["invocation_seconds"] * 4,
                      artifact_path=str(context["attempt"].relative_to(evidence.root)))
        execution = {k: record[k] for k in
                     ("wrapper_exit_code", "error", "sampling_valid", "timed_out", "cleanup")}
        execution["contamination_detected"] = context["contaminated"]
        require(not record["cleanup"].get("remaining_pids"), "Remaining payload processes")
        execution["cleanup"] = {"complete": record["cleanup"]["complete"] and context["post_idle"]}
        assessed = runner.assess_period(context["attempt"], row, execution, context["expected"],
                                        context["invocation"], context["job"],
                                        pair["manifest"]["source_commit"])
        result.update(failure_kind=assessed.get("failure_kind"),
                      failure_stage=assessed.get("failure_stage"),
                      worker_initialization=assessed.get("worker_initialization"))
        require(assessed["status"] in {"completed", "memory_failure"},
                assessed.get("validation_error", "Raw period assessment unresolved"))
        trial = validate_raw_environment(evidence, frozen, pair, row, context)
        completed = assessed["status"] == "completed"
        if completed:
            validate_completed_allocator(evidence, context, trial)
        # Prompt profiles enrich historical descriptive rows but are neither
        # needed nor part of this study's hashed outcome evidence.
        raw = select_attempt(evidence.root, row, exclusions={}, annotations={},
                             trial_path=context["attempt"] / f"trial-{context['job']}.json",
                             include_prompt_profile=False)
        require(bool(raw["success"]) == completed and str(raw["job_id"]) == context["job"]
                and raw["peak_gpu_memory_mib"] == assessed["task_peak_memory_mib"],
                "Independent raw attempt reconstruction disagrees")
        duration = finite(trial["elapsed_seconds"], "task runtime")
        require(duration <= context["invocation_seconds"] + 1.1,
                "Task runtime exceeds enclosing invocation")
        peak = assessed["task_peak_memory_mib"]
        state = ("above_margin" if peak > LIMIT else "within_margin") if completed else "memory_failure"
        result.update(eligible=True, completed=completed, state=state,
                      observed_task_peak_mib=peak, completed_peak_mib=peak if completed else None,
                      known_memory_failure=not completed,
                      known_margin_exceedance=completed and peak > LIMIT,
                      task_runtime_seconds=duration, failure_kind=raw["failure_kind"],
                      worker_initialization=assessed["worker_initialization"])
    except EVIDENCE_ERRORS as error:
        if result["failure_stage"] is None:
            result["failure_stage"] = "evidence_missing" if isinstance(error, FileNotFoundError) \
                else "evidence_invalid"
        result["validation_errors"].append(evidence.error(error))
    return result


def configurations(processes, targets, evaluation_seeds=None):
    seeds = list(SEEDS if evaluation_seeds is None else evaluation_seeds)
    grouped = defaultdict(list)
    for row in processes:
        grouped[row["configuration_id"]].append(row)
    result = []
    for settings in targets:
        rows = grouped[settings["configuration_id"]]
        eligible = [r for r in rows if r["eligible"]]
        resolved = (len(eligible) == len(seeds)
                    and sorted(r["training_seed"] for r in eligible) == sorted(seeds))
        known_failure = any(r["known_memory_failure"] for r in eligible)
        known_margin = any(r["known_margin_exceedance"] for r in eligible)
        state = ("memory_failure" if known_failure else
                 "above_margin" if known_margin else "within_margin") if resolved else "unresolved"
        peaks = [r["completed_peak_mib"] for r in eligible if r["completed"]]
        result.append(dict(
            configuration_id=settings["configuration_id"], settings=settings, state=state,
            planned_seeds=seeds, eligible_seeds=sorted(r["training_seed"] for r in eligible),
            unresolved_seeds=sorted(r["training_seed"] for r in rows if not r["eligible"]),
            eligible_processes=len(eligible), completed_processes=len(peaks),
            known_memory_failure=known_failure, known_margin_exceedance=known_margin,
            completed_peak_mib=max(peaks) if resolved and len(peaks) == len(seeds) else None,
            maximum_known_completed_peak_mib=max(peaks) if peaks else None))
    return result


def flat_configurations(configs, predictions):
    """Scalar columns suitable for a public dataset viewer; no target refitting."""
    by_prediction = {(p["configuration_id"], p["method"]): p for p in predictions}
    rows = []
    for config in configs:
        settings = config["settings"]
        row = dict(configuration_id=config["configuration_id"],
                   model=settings["model_id"], workload=settings["legacy_dataset_id"],
                   configuration_level=settings["configuration_level"],
                   gpu_count=int(settings["gpu_count"]), required_seeds=len(config["planned_seeds"]),
                   eligible_seed_count=len(config["eligible_seeds"]),
                   eligible_seeds=";".join(map(str, config["eligible_seeds"])),
                   unresolved_seeds=";".join(map(str, config["unresolved_seeds"])),
                   all_three_seeds_eligible=len(config["eligible_seeds"]) == 3,
                   observed_state=config["state"], unresolved=config["state"] == "unresolved",
                   known_memory_failure=config["known_memory_failure"],
                   known_margin_exceedance=config["known_margin_exceedance"],
                   completed_processes=config["completed_processes"],
                   completed_peak_mib=config["completed_peak_mib"],
                   maximum_known_completed_peak_mib=config["maximum_known_completed_peak_mib"])
        for method in eb.METHODS:
            prediction = by_prediction[(config["configuration_id"], method)]
            row[method + "_predicted_state"] = prediction["predicted_state"]
            row[method + "_predicted_peak_mib"] = prediction["predicted_peak_mib"]
            row[method + "_unscored_approval"] = (
                row["unresolved"] and prediction["predicted_state"] == "within_margin")
        rows.append(row)
    return rows


def evaluate(predictions, configs):
    resolved = [c for c in configs if c["state"] != "unresolved"]
    ids = {c["configuration_id"] for c in resolved}
    unscored = {c["configuration_id"]: c for c in configs if c["state"] == "unresolved"}
    metrics = eb.score([p for p in predictions if p["configuration_id"] in ids], resolved) if resolved else None
    approvals = {}
    for method in eb.METHODS:
        approved = {p["configuration_id"] for p in predictions
                    if p["method"] == method and p["predicted_state"] == "within_margin"}
        unknown = approved & set(unscored)
        approvals[method] = dict(
            full_panel_approved_configurations=len(approved), full_panel_approved_ids=sorted(approved),
            scored_approved_configurations=len(approved & ids),
            unscored_approved_configurations=len(unknown), unscored_approved_ids=sorted(unknown),
            unscored_approvals_with_known_failure=sorted(
                cid for cid in unknown if unscored[cid]["known_memory_failure"]),
            unscored_approvals_with_known_margin_exceedance=sorted(
                cid for cid in unknown if unscored[cid]["known_margin_exceedance"]))
    return dict(metric_scope="resolved_subset_only", planned_configurations=len(configs),
                resolved_configurations=len(resolved), unresolved_configurations=len(unscored),
                resolved_ids=sorted(ids), unresolved_ids=sorted(unscored),
                full_panel_coverage=len(resolved) / len(configs),
                resolved_subset_metrics=metrics, approvals=approvals,
                interpretation="Unresolved configurations are unscored, not correct predictions or safe labels.")


def paired_summaries(processes):
    groups = defaultdict(list)
    for row in processes:
        groups[row["pair_id"]].append(row)
    result = []
    for pair_id, rows in groups.items():
        by_gpu = {r["task_gpu_count"]: r for r in rows}
        a, b = by_gpu[2], by_gpu[4]
        joint = a["completed"] and b["completed"]
        result.append(dict(pair_id=pair_id, dataset=a["dataset"],
                           configuration_level=a["configuration_level"], training_seed=a["training_seed"],
                           first_task_gpu_count=next(r["task_gpu_count"] for r in rows if r["period"] == 1),
                           two_gpu_state=a["state"], four_gpu_state=b["state"],
                           both_eligible=a["eligible"] and b["eligible"], both_completed=joint,
                           peak_difference_4_minus_2_mib=(
                               b["completed_peak_mib"] - a["completed_peak_mib"] if joint else None),
                           runtime_difference_4_minus_2_seconds=(
                               b["task_runtime_seconds"] - a["task_runtime_seconds"] if joint else None)))
    return result


def allocation_cost(evidence, pair, accounting):
    manifest, events = pair["manifest"], pair["events"]
    result = dict(pair_id=manifest["pair_id"], job_id=manifest.get("job_id"),
                  allocation_gpu_count=4, observed_allocation_gpu_seconds=None,
                  whole_allocation_gpu_seconds=None, accounting_valid=False, errors=[])
    allocation = manifest["allocation"]
    job = str(manifest.get("job_id"))
    # Scheduler accounting can establish whole-job cost even when cancellation
    # prevented a pair-summary or the last period's terminal record.
    if job in accounting:
        try:
            item = accounting[job]
            if "accounting_schema_version" in item:
                scheduler_accounting.validate_record(item)
                require(item["pair_id"] == manifest["pair_id"]
                        and item["pair_manifest_sha256"] ==
                        digest(evidence.touch(pair["directory"] / "pair.json"))
                        and item["pair_ledger_sha256"] ==
                        digest(evidence.touch(pair["directory"] / "ledger.jsonl")),
                        "Scheduler accounting has wrong raw pair-claim identity")
                start, end = item["started_at_epoch"], item["ended_at_epoch"]
                precision = item["timestamp_precision_seconds"]
                batch_start, batch_end = item["batch_started_at_epoch"], item["batch_ended_at_epoch"]
                last = events[-1]["timestamp_ns"] / 1e9
                # Precision describes a half-open interval, not an extra billed
                # second. Only explicit cancellation plus the matching batch
                # record can explain a ledger that outlives the outer job.
                bound = max(end, batch_end) if batch_end is not None else end
                require(item["allocated_gpu_count"] == 4
                        and start <= events[0]["timestamp_ns"] / 1e9
                        and last < bound + precision, "Scheduler accounting interval/count differs")
                if batch_end is not None:
                    require(batch_start <= events[0]["timestamp_ns"] / 1e9
                            and last < batch_end + precision,
                            "Durable pair ledger lies outside its batch step")
                if allocation is not None:
                    require(abs(start - allocation["started_at_epoch"]) <= precision,
                            "Scheduler allocation start differs")
                    require(item["scheduler_node_list"] == allocation["hostname"]
                            or item["scheduler_node_list"] == allocation["hostname"].split(".")[0],
                            "Scheduler allocation node differs")
                else:
                    require(re.fullmatch(r"[1-9]\d*", job)
                            and not any(e["event"] in {"dispatched", "spawned", "pair_started"}
                                        for e in events),
                            "Nonlaunch accounting lacks a valid unstarted pair claim")
                result.update(
                    scheduler_scope=item["scheduler_scope"], scheduler_state=item["scheduler_state"],
                    timestamp_precision_seconds=precision, timestamp_interval=item["timestamp_interval"],
                    scheduler_started_at_epoch=start, scheduler_ended_at_epoch=end,
                    batch_started_at_epoch=batch_start, batch_ended_at_epoch=batch_end,
                    batch_state=item["batch_state"], ledger_ended_at_epoch=last,
                    ledger_minus_outer_end_seconds=last - end,
                    batch_minus_outer_end_seconds=batch_end - end if batch_end is not None else None,
                    batch_step_gpu_seconds=4 * (batch_end - batch_start)
                    if batch_end is not None else None,
                    raw_record_provenance=item["raw_record_provenance"],
                    pair_manifest_sha256=item["pair_manifest_sha256"],
                    pair_ledger_sha256=item["pair_ledger_sha256"])
            else:
                # Preserve the old exact-end contract when precision and
                # claim-bound scheduler evidence have not been supplied.
                if allocation is None:
                    return result
                require(item["allocated_gpu_count"] == 4
                        and abs(item["started_at_epoch"] - allocation["started_at_epoch"]) <= 1
                        and item["ended_at_epoch"] >= events[-1]["timestamp_ns"] / 1e9,
                        "Scheduler accounting interval/count differs")
            result.update(whole_allocation_gpu_seconds=4 * (
                item["ended_at_epoch"] - item["started_at_epoch"]), accounting_valid=True)
        except EVIDENCE_ERRORS as error:
            result["errors"].append(evidence.error(error))
    if allocation is None:
        return result
    try:
        summary = evidence.json(pair["directory"] / "pair-summary.json")
        terminal = one_event(events, "pair_terminal")
        require(all(terminal.get(k) == v for k, v in summary.items()),
                "Pair summary/terminal ledger mismatch")
        require(summary["pair_id"] == manifest["pair_id"] and str(summary["job_id"]) == job
                and summary["allocation_started_at_epoch"] == allocation["started_at_epoch"],
                "Allocation cost has wrong provenance")
        period_terminals = [e for e in events if e["event"] == "terminal"]
        require(summary["planned_invocations"] == 2
                and summary["launched_invocations"] == sum(e["event"] == "spawned" for e in events)
                and len(period_terminals) == 2
                and summary["periods"] == [{"period": e["period"], "status": e["status"]}
                                           for e in period_terminals]
                and summary["validated_outcomes"] == sum(
                    e["status"] in {"completed", "memory_failure"} for e in period_terminals),
                "Pair summary counts disagree with durable period ledger")
        for field in ("task_gpu_seconds", "allocated_gpu_seconds_during_invocations"):
            period_field = field[:-1] if field.endswith("invocations") else field
            require(math.isclose(summary[field], sum(e[period_field] for e in period_terminals),
                                 abs_tol=.01), "Pair GPU cost sum differs from period ledger")
        end = finite(summary["observed_until_epoch"], "observed allocation end")
        require(events[-1] == terminal
                and allocation["started_at_epoch"] <= end <= terminal["timestamp_ns"] / 1e9
                and terminal["timestamp_ns"] / 1e9 - end < 30,
                "Observed allocation end differs from durable finalization")
        observed = 4 * (end - allocation["started_at_epoch"])
        require(math.isclose(observed, summary["allocation_gpu_seconds_observed"], abs_tol=.01),
                "Recorded allocation cost mismatch")
        result["observed_allocation_gpu_seconds"] = observed
    except EVIDENCE_ERRORS as error:
        result["errors"].append(evidence.error(error))
    return result


def cost_summary(processes, allocations):
    def total(rows, field):
        known = [r[field] for r in rows if r.get(field) is not None]
        return {"known_gpu_seconds": sum(known), "known_records": len(known),
                "unknown_records": len(rows) - len(known),
                "complete_total_gpu_seconds": sum(known) if len(known) == len(rows) else None}
    result = dict(
        task_provisioned_time=total(processes, "task_gpu_seconds"),
        invocation_reservation_time=total(processes, "invocation_reservation_gpu_seconds"),
        observed_allocation_time=total(allocations, "observed_allocation_gpu_seconds"),
        whole_allocation_time=total(allocations, "whole_allocation_gpu_seconds"),
        allocations=allocations,
        scope="Task/invocation time includes failed and unresolved work when timing is validated. "
              "Observed allocation time ends at runner finalization; whole-job time requires "
              "scheduler accounting. The 55-minute limit is not actual elapsed time. "
              "Missing records are unknown, not zero cost. No utilization weighting.")
    if any("timestamp_precision_seconds" in allocation for allocation in allocations):
        result["batch_step_time"] = total(allocations, "batch_step_gpu_seconds")
        result["scheduler_scope_note"] = (
            "Whole-allocation totals use reported outer Start/End only. Batch-step totals "
            "are separate, overlapping spans and must never be added to outer totals. "
            "Timestamp precision validates ledger containment but adds no billed seconds.")
    return result


def collect(root, *, source_commit=None, execution_commits=None, source_root=None,
            protocol_dir=None, accounting=None, protocol_sha256=None, prediction_seal_sha256=None,
            study_frozen_loader=None):
    """Return reconstructed data in memory. Never modify the supplied artifact."""
    require((source_commit is None) != (execution_commits is None),
            "Supply exactly one execution-commit anchor: default commit or complete pair map")
    evidence = Evidence(root)
    source_evidence = Evidence(source_root or root)
    protocol_evidence = Evidence(protocol_dir or source_evidence.root / "benchmark/estimation")
    frozen = (study_frozen_loader or load_frozen)(
        source_evidence, protocol_sha256, prediction_seal_sha256,
        protocol_evidence=protocol_evidence)
    if source_commit is not None:
        execution_commits = {pair_id: source_commit for pair_id in frozen["pairs"]}
    require(isinstance(execution_commits, dict) and set(execution_commits) == set(frozen["pairs"]),
            f"Execution-commit map must cover exactly all {len(frozen['pairs'])} planned pairs")
    require(all(isinstance(commit, str) and re.fullmatch(r"[a-f0-9]{40}", commit)
                for commit in execution_commits.values()), "Every execution commit must be exact 40-hex")
    accounting_by_job = {}
    for row in accounting or []:
        job = str(row["job_id"])
        require(job not in accounting_by_job, "Duplicate scheduler accounting job")
        require(finite(row["started_at_epoch"], "job start")
                <= finite(row["ended_at_epoch"], "job end"), "Invalid scheduler interval")
        accounting_by_job[job] = row
    processes, allocations, pair_errors = [], [], {}
    observed_jobs = {}
    expected_pairs = set(frozen["pairs"])
    pair_root = inside(evidence.root, Path("output") / frozen.get("group", runner.GROUP) / "pairs")
    extras = [p.name for p in pair_root.iterdir() if p.is_dir() and p.name not in expected_pairs] \
        if pair_root.exists() else []
    require(not extras, f"Unexpected extra study pair directories: {extras}")
    for pair_id, rows in frozen["pairs"].items():
        pair, error = None, None
        try:
            pair = validate_pair(evidence, frozen, pair_id, execution_commits[pair_id])
            if pair["manifest"].get("job_id") is not None:
                job = str(pair["manifest"]["job_id"])
                require(job not in observed_jobs, "One job reused for multiple pairs")
                observed_jobs[job] = pair_id
            allocations.append(allocation_cost(evidence, pair, accounting_by_job))
        except EVIDENCE_ERRORS as exc:
            error = evidence.error(exc)
            pair_errors[pair_id] = error
            allocations.append(dict(pair_id=pair_id, job_id=None,
                                    observed_allocation_gpu_seconds=None,
                                    whole_allocation_gpu_seconds=None, errors=[error]))
        for row in sorted(rows, key=lambda r: int(r["period"])):
            reconstructed = process_slot(evidence, frozen, pair, row, error)
            reconstructed["expected_execution_commit"] = execution_commits[pair_id]
            processes.append(reconstructed)
    require(set(accounting_by_job) <= set(observed_jobs),
            "Accounting contains jobs not identifiable in this panel")
    configs = configurations(processes, frozen["targets"], frozen.get("evaluation_seeds", SEEDS))
    evaluation = evaluate(frozen["predictions"], configs)
    results = dict(
        processes=processes, configurations=configs, pairs=paired_summaries(processes),
        configurations_flat=flat_configurations(configs, frozen["predictions"]),
        evaluation=evaluation, costs=cost_summary(processes, allocations),
        summary=dict(schema_version=1, planned_processes=len(frozen["rows"]),
                     planned_pairs=len(frozen["pairs"]), planned_configurations=len(frozen["targets"]),
                     eligible_processes=sum(p["eligible"] for p in processes),
                     completed_processes=sum(p["completed"] for p in processes),
                     unresolved_processes=sum(not p["eligible"] for p in processes),
                     known_launched_processes=sum(p["launched"] is True for p in processes),
                     unknown_launch_status_processes=sum(p["launched"] is None for p in processes),
                     resolved_configurations=evaluation["resolved_configurations"],
                     pair_errors=pair_errors),
        provenance=dict(
            source_commit_expected=source_commit, execution_commits_expected=dict(execution_commits),
            protocol_sha256=frozen["protocol_sha256"],
            amendments_sha256=frozen["amendments_sha256"],
            prediction_seal_sha256=frozen["prediction_seal_sha256"],
            prediction_frozen_at_utc=frozen["seal"]["frozen_at_utc"],
            prediction_seal_binding="Effective input/output hashes and pre-claim chronology; "
                                    "runner verifies seal but does not store its hash per period.",
            data_verification_scope="Environment-captured dataset file hashes compared with frozen protocol; "
                                    "no external dataset/cache paths are opened.",
            contamination_scope="All-four allocation samples, excluded-device process membership, and "
                                "recorded live-child ownership annotations. This is not continuous "
                                "monitoring or an independent offline reconstruction of PID ancestry.",
            raw_statuses_used_as_labels=False, analysis_implementation_sha256=digest(Path(__file__)),
            runner_validation_implementation_sha256=digest(Path(runner.__file__)),
            input_evidence=evidence.hashes, source_input_evidence=source_evidence.hashes,
            protocol_input_evidence=protocol_evidence.hashes))
    results["provenance"].update(frozen.get("study_provenance", {}))
    if any("accounting_schema_version" in row for row in accounting or []):
        results["provenance"]["accounting_validation_implementation_sha256"] = digest(
            Path(scheduler_accounting.__file__))
    evidence.verify_unchanged()
    source_evidence.verify_unchanged()
    protocol_evidence.verify_unchanged()
    return results


def write_results(results, output):
    """Exclusive output directory: an earlier report is never overwritten."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, value in results.items():
        write_new_json(output / f"{name}.json", value)
    for name in ("processes", "configurations", "configurations_flat", "pairs"):
        rows = results[name]
        fields = sorted({k for row in rows for k in row})
        with (output / f"{name}.csv").open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows({k: json.dumps(v, sort_keys=True, allow_nan=False)
                              if isinstance(v, (dict, list)) else v for k, v in row.items()}
                             for row in rows)
    write_new_json(output / "output_manifest.json", {
        path.name: {"sha256": digest(path), "size_bytes": path.stat().st_size}
        for path in sorted(output.iterdir()) if path.is_file()})


def load_execution_commits(path):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate execution-map key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def main(*, study_frozen_loader=None, protocol_relative_dir="benchmark/estimation"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--protocol-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    commits = parser.add_mutually_exclusive_group(required=True)
    commits.add_argument("--source-commit")
    commits.add_argument("--execution-commits", type=Path)
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--prediction-seal-sha256")
    parser.add_argument("--allocation-accounting", type=Path)
    args = parser.parse_args()
    # Outputs must not be nested in the evidence or frozen input directories.
    root, output = args.root.resolve(), args.output.resolve()
    source_root = (args.source_root or root).resolve()
    default_protocol_dir = (inside(source_root, protocol_relative_dir)
                            if study_frozen_loader is not None and args.protocol_dir is None
                            else source_root / protocol_relative_dir)
    protocol_dir = (args.protocol_dir or default_protocol_dir).resolve()
    require(not output.is_relative_to(root / "output")
            and not output.is_relative_to(root / "benchmark")
            and not output.is_relative_to(source_root / "benchmark")
            and not output.is_relative_to(protocol_dir)
            and not root.is_relative_to(output), "Report output overlaps immutable inputs")
    accounting_hash = digest(args.allocation_accounting) if args.allocation_accounting else None
    accounting = json.loads(args.allocation_accounting.read_text()) if args.allocation_accounting else None
    execution_map_hash = digest(args.execution_commits) if args.execution_commits else None
    execution_map = load_execution_commits(args.execution_commits) if args.execution_commits else None
    result = collect(root, source_commit=args.source_commit, source_root=source_root,
                     execution_commits=execution_map,
                     protocol_dir=protocol_dir, accounting=accounting,
                     protocol_sha256=args.protocol_sha256,
                     prediction_seal_sha256=args.prediction_seal_sha256,
                     study_frozen_loader=study_frozen_loader)
    if args.allocation_accounting:
        require(digest(args.allocation_accounting) == accounting_hash,
                "Allocation accounting changed during collection")
        result["provenance"]["allocation_accounting_sha256"] = accounting_hash
    if args.execution_commits:
        require(digest(args.execution_commits) == execution_map_hash,
                "Execution-commit map changed during collection")
        result["provenance"]["execution_commits_file_sha256"] = execution_map_hash
    write_results(result, output)
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
