"""CPU-only raw reconstruction of all 108 new slots; four compact output files.

Neither online status nor an exit-only/log-free completion is an endpoint.
No model cache, GPU, scheduler query, training, refitting or raw rewriting occurs.
Optional allocation accounting is a JSON list of actual scheduler job intervals.
--root may be an extracted archive. Recorded absolute output paths are mapped to
the planned local slot only; they are never opened as fallback paths. Model
weights and the original live data tree are not needed for CPU reconstruction.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import re
import time

from memory_tuner import collect_estimation_study as core
from memory_tuner import completion_replication_protocol as design
from memory_tuner import run_matched_gpu as base
from memory_tuner.benchmark_v2_env import FIELD_TO_ENV, OPTIONAL_FIELD_TO_ENV
from memory_tuner.device_contract import digest, write_new_json
from memory_tuner.durable_metrics import validate_completed_step
from memory_tuner.grpo_raw_evidence import MATCH_FIELDS, normalized
from memory_tuner.log_parser import classify_failure_detailed
from memory_tuner.scientific_revision_analysis import validate_fixed_stack

require = design.require
ERRORS = (ValueError, KeyError, OSError, TypeError, IndexError)
LIMIT = 38912


def relocate_evidence(value, root, relative):
    """Lexical recorded-path check followed by explicit local-root confinement."""
    require(isinstance(value, str) and value, "Missing recorded evidence path")
    recorded, relative = Path(value), Path(relative)
    require(".." not in recorded.parts and not relative.is_absolute()
            and ".." not in relative.parts
            and recorded.parts[-len(relative.parts):] == relative.parts,
            "Recorded evidence path does not identify the planned archive slot")
    return core.inside(root, relative)


def aligned_traces(attempt, job, expected, invocation, earliest, latest):
    observed = base.validate_trace(attempt, expected, invocation, job)
    mapping = json.loads((attempt / "gpu-device-map.json").read_text())
    require(mapping["scope"] == "task_devices" and mapping["columns"] == [
        {"gpu_index": i, "memory_csv_column": i + 1, "uuid": uid}
        for i, uid in enumerate(expected)], "Task trace physical column mapping differs")
    with (attempt / f"gpu-memory-{job}.csv").open() as handle:
        memory = list(csv.reader(handle))
    with (attempt / f"phase-memory-{job}.csv").open() as handle:
        phases = list(csv.reader(handle))
    telemetry = design.eb.read_csv(attempt / f"gpu-telemetry-{job}.csv")
    require(len(memory) == len(phases) and len(telemetry) == len(memory) * len(expected),
            "Incomplete phase/per-device trace coverage")
    for index, (sample, phase) in enumerate(zip(memory, phases)):
        stamp, values = int(sample[0]), list(map(float, sample[1:]))
        require(earliest <= stamp <= latest and len(phase) == len(expected) + 3
                and int(phase[0]) == stamp and int(phase[1]) in (0, 1)
                and list(map(float, phase[3:])) == values and phase[2],
                "Trace time, requested horizon or phase/memory alignment differs")
        require(phase[2] not in ("validation", "checkpoint"), "Disabled lifecycle phase executed")
        device_rows = telemetry[index * len(expected):(index + 1) * len(expected)]
        require({int(r["gpu_index"]) for r in device_rows} == set(range(len(expected))),
                "Duplicate or missing device telemetry")
        for row in device_rows:
            require(int(row["timestamp_ns"]) == stamp and int(row["step"]) == int(phase[1])
                    and row["phase"] == phase[2]
                    and float(row["memory_used_mib"]) == values[int(row["gpu_index"])]
                    and 0 <= float(row["gpu_utilization_pct"]) <= 100,
                    "Per-device telemetry differs from raw trace")
    return observed


def assess_period(attempt, row, execution, allocation, protocol, source_commit):
    """Reevaluate raw outcome, used online and independently by offline collection."""
    job = str(allocation["job_id"])
    invocation = row["experiment_id"] + "-j" + job
    expected = [allocation["gpus"][i]["uuid"] for i in base.subset_ordinals(row)]
    log = attempt / f"training-{job}.log"
    text = log.read_text(errors="replace") if log.is_file() else ""
    code = execution.get("wrapper_exit_code")
    kind = classify_failure_detailed(text, code if code is not None else 1)
    result = dict(status="infrastructure_invalid", failure_kind=kind, completed=False,
                  identity_validated=False, completed_peak_mib=None,
                  observed_task_peak_mib=None, validation_error=None)
    try:
        require(execution["spawned"] and not execution["error"] and execution["sampling_valid"]
                and not execution["contamination_detected"] and execution["cleanup"]["complete"]
                and not execution["cleanup"]["remaining_pids"], "Invalid execution/cleanup evidence")
        require(not execution["timed_out"] and not execution.get("interrupted"),
                "Timeout/interruption is unresolved, never a validated memory failure")
        require(type(code) is int, "Missing process exit")
        require(text.strip(), "Missing or empty immutable raw console")
        earliest, latest = execution["started_ns"], execution["finished_ns"]
        require(type(earliest) is int and type(latest) is int and 0 < earliest < latest,
                "Invalid execution lifetime")
        trial = json.loads((attempt / f"trial-{job}.json").read_text())
        require(all(normalized(trial.get(k)) == normalized(row.get(k)) for k in MATCH_FIELDS),
                "Trial configuration differs from frozen row")
        require(str(trial["job_id"]) == job and trial["exit_code"] == code
                and trial["model"] == row["model"], "Trial job/model/exit differs")
        for field, name in {
            "run_log": f"training-{job}.log", "environment_json": f"environment-{job}.json",
            "phase_memory_csv": f"phase-memory-{job}.csv",
            "gpu_telemetry_csv": f"gpu-telemetry-{job}.csv",
        }.items():
            # Archive relocation is permitted, but the full original slot suffix is fixed.
            path = Path(trial[field])
            require(path.parts[-5:] == ("output", design.GROUP, row["experiment_id"],
                                        "attempt-" + job, name), "Raw evidence path belongs to another slot")
        require(Path(trial["data_dir"]) == Path(row["data_dir"]), "Trial workload data differs")
        runtime = core.finite(trial["elapsed_seconds"], "task runtime")
        require(runtime <= core.finite(execution["invocation_elapsed_seconds"], "invocation") + 1.1,
                "Task runtime exceeds invocation")
        successful_exit = code == 0
        require(successful_exit or kind in base.MEMORY_FAILURES, "Non-memory process failure")
        result.update(base.validate_identity(attempt / "device-evidence", expected, invocation,
                                            success=successful_exit))
        observed = aligned_traces(attempt, job, expected, invocation, earliest, latest)
        result["observed_task_peak_mib"] = observed["task_peak_memory_mib"]
        require(abs(float(trial["peak_gpu_memory_mib"]) - observed["task_peak_memory_mib"]) <= .01,
                "Trial peak differs from raw physical-device trace")
        parent = json.loads((attempt / "device-evidence/trainer-cuda.json").read_text())
        trainer = json.loads((attempt / "device-evidence/trainer-exec.json").read_text())
        gate = json.loads((attempt / "device-evidence/probe-exit-idle.json").read_text())
        for path in (attempt / "device-evidence").glob("*.json"):
            item = json.loads(path.read_text())
            require(item["invocation_id"] == invocation and item["hostname"] == allocation["hostname"]
                    and earliest <= item["timestamp_ns"] <= latest, "Stale or foreign device evidence")
        require(observed["task_trace_first_timestamp_ns"] <= parent["timestamp_ns"]
                <= observed["task_trace_last_timestamp_ns"]
                and parent["timestamp_ns"] <= gate["timestamp_ns"] <= trainer["timestamp_ns"],
                "CUDA probe/trainer chronology or trace coverage differs")
        require(gate["allocation_uuids"] == [g["uuid"] for g in allocation["gpus"]]
                and len(gate["observations"]) >= 2, "Probe did not check this full allocation")
        for sample in gate["observations"]:
            core.validate_sample(sample, allocation["gpus"])
        require(all(base.is_idle(s) for s in gate["observations"][-2:]), "Probe contexts were not released")
        env = json.loads((attempt / f"environment-{job}.json").read_text())
        validate_fixed_stack(env)
        require(env["host"]["hostname"] == allocation["hostname"]
                and str(env["slurm"]["SLURM_JOB_ID"]) == job
                and env["repository"]["head"] == source_commit and not env["repository"]["dirty"]
                and env["model"]["requested_model"] == row["model"]
                and env["model"]["resolved_revision"] == design.MODELS[row["model"]]["revision"]
                and env["model"]["actual_snapshot_path"] == protocol["model_cache"][row["model"]]["path"]
                and env["gpu_device_contract"]["invocation_id"] == invocation
                and env["gpu_device_contract"]["requested_uuids"] == expected
                and env["gpu_device_contract"]["task_gpu_count"] == len(expected)
                and {g["uuid"] for g in env["software"]["cuda_devices"]} == set(expected),
                "Environment/source/model/physical-device provenance differs")
        require(earliest <= core.epoch(env["captured_at_utc"]) * 1e9 <= latest,
                "Environment capture outside execution lifetime")
        gpus = list(csv.reader(env["gpus_csv"].splitlines(), skipinitialspace=True))
        require(len(gpus) == len(expected) and {g[2].strip() for g in gpus} == set(expected),
                "Environment physical inventory differs")
        for gpu in gpus:
            original = next(g for g in allocation["gpus"] if g["uuid"] == gpu[2].strip())
            require(len(gpu) == 5 and int(gpu[0]) == original["index"]
                    and gpu[1].strip() == original["name"]
                    and gpu[3].strip() == original["driver_version"]
                    and float(gpu[4].split()[0]) == 40960, "Environment GPU identity changed")
        for split in ("train.parquet", "test.parquet"):
            item = env["dataset"][split]
            relative = design.DATA_ROOTS[row["dataset"]] + "/" + split
            require(item["sha256"] == protocol["data_sha256"][relative]
                    and core.finite(item["size_bytes"], "data size") > 0
                    and Path(item["path"]) == Path(row["data_dir"]) / split,
                    "Frozen dataset provenance differs")
        # A real printed config may be present; if so it must not contradict the horizon.
        configured = {int(v) for v in re.findall(r"['\"]total_training_steps['\"]:\s*(\d+)", text)}
        require(not configured or configured == {1}, "Console configuration changed training horizon")
        logged = {int(v) for v in re.findall(r"training/global_step:\s*(\d+)", text)}
        require(logged <= {1}, "Raw console contains a different training horizon")
        terminal_path = attempt / "device-evidence/trainer-terminal.json"
        if terminal_path.is_file():
            terminal = json.loads(terminal_path.read_text())
            require(terminal["pid"] == trainer["pid"] and terminal["timestamp_ns"] >= trainer["timestamp_ns"]
                    and observed["task_trace_last_timestamp_ns"] >= terminal["timestamp_ns"] - 1_000_000_000,
                    "Trainer terminal identity/trace lifetime differs")
        elif successful_exit:
            raise ValueError("Successful process lacks trainer terminal evidence")
        if successful_exit:
            durable = validate_completed_step(
                attempt / "trainer-metrics", invocation, requested_steps=1,
                earliest_ns=trainer["timestamp_ns"], latest_ns=terminal["timestamp_ns"])
            require(durable["logger_hostname"] == allocation["hostname"], "Foreign durable metric logger")
            evidence = core.Evidence(attempt)
            context = dict(attempt=attempt, spawned={"timestamp_ns": earliest},
                           terminal={"timestamp_ns": latest})
            # Rebind only in memory for CPU archive relocation; never rewrite the trial.
            rebound_trial = dict(trial, allocator_trace_dir=str(attempt / "allocator-traces"))
            require(Path(trial["allocator_trace_dir"]).parts[-5:] ==
                    ("output", design.GROUP, row["experiment_id"], "attempt-" + job, "allocator-traces"),
                    "Allocator trace belongs to another slot")
            core.validate_completed_allocator(evidence, context, rebound_trial)
            evidence.verify_unchanged()
            result.update(durable, status="completed", completed=True, failure_kind=None,
                          completed_peak_mib=observed["task_peak_memory_mib"])
        else:
            result["status"] = "memory_failure"
        result["task_runtime_seconds"] = runtime
    except ERRORS as error:
        result["validation_error"] = f"{type(error).__name__}: {error}"
    return result


def period_context(evidence, frozen, pair, row, record):
    """Reconstruct dispatch, process start, allocation sampling, cleanup and raw manifest."""
    manifest, events, trace = pair["manifest"], pair["events"], pair["trace"]
    period, job = int(row["period"]), str(manifest["job_id"])
    terminal = core.one_event(events, "terminal", period)
    require({k: v for k, v in terminal.items()
             if k not in ("event", "sequence", "timestamp_ns", "monotonic_ns")} == record,
            "Period record and terminal journal differ")
    spawned = [e for e in events if e["event"] == "spawned" and e.get("period") == period]
    require(record["launched"] is bool(spawned), "Launch flag differs from raw spawn ledger")
    attempt = core.inside(evidence.root, Path("output") / design.GROUP / row["experiment_id"]
                          / ("attempt-" + job))
    attempts = list(attempt.parent.glob("attempt-*"))
    require(not list(attempt.parent.glob("trial-*.json"))
            and (attempts == [attempt] or not attempts), "Extra or replacement attempt")
    if attempt.exists():
        core.validate_payload(evidence, attempt, record["evidence_files"])
    else:
        require(not record["evidence_files"] and not spawned, "Missing dispatched raw attempt")
    if not spawned:
        require(not (attempt.exists() and list(attempt.glob("trial-*.json"))),
                "Unspawned slot contains an unexplained trial")
        return None
    expected = [manifest["allocation"]["gpus"][i]["uuid"] for i in base.subset_ordinals(row)]
    invocation = row["experiment_id"] + "-j" + job
    require(record["requested_uuids"] == expected and record["invocation_id"] == invocation
            and str(record["job_id"]) == job and record["allocation_gpu_count"] == 4,
            "Period device/invocation identity differs")
    relative_attempt = attempt.relative_to(evidence.root)
    require(relocate_evidence(record["attempt_path"], evidence.root, relative_attempt) == attempt,
            "Wrong recorded attempt")
    for field, name in {
        "trial_path": f"trial-{job}.json", "run_log": f"training-{job}.log",
        "environment_json": f"environment-{job}.json",
        "phase_memory_csv": f"phase-memory-{job}.csv",
        "gpu_telemetry_csv": f"gpu-telemetry-{job}.csv",
        "device_evidence_dir": "device-evidence",
    }.items():
        relocate_evidence(record[field], evidence.root, relative_attempt / name)
    dispatched = core.one_event(events, "dispatched", period)
    require(dispatched["sequence"] < spawned[0]["sequence"] < terminal["sequence"]
            and dispatched["experiment_id"] == row["experiment_id"]
            and dispatched["invocation_id"] == invocation
            and dispatched["requested_uuids"] == expected
            and relocate_evidence(dispatched["attempt_path"], evidence.root, relative_attempt) == attempt,
            "Dispatch/spawn/terminal identity differs")
    dispatch = evidence.json(attempt / "dispatch.json")
    require(dispatch["row"] == row and dispatch["invocation_id"] == invocation
            and dispatch["requested_uuids"] == expected
            and dispatch["freeze_sha256"] == frozen["freeze_sha256"]
            and dispatch["matrix_sha256"] == frozen["matrix_sha256"]
            and dispatch["protocol_sha256"] == frozen["protocol_sha256"],
            "Raw dispatch differs from frozen plan")
    env = dispatch["environment"]
    for field, name in {**FIELD_TO_ENV, **OPTIONAL_FIELD_TO_ENV}.items():
        if field != "custom_reward_function_path" and row.get(field):
            require(normalized(env.get(name)) == normalized(row[field]), f"Wrong dispatch: {field}")
    recorded_source = Path(manifest["source_root"])  # Compare only; never open this live path.
    require(env["BENCHMARK_SOURCE_ROOT"] == manifest["source_root"]
            and env["RUN_NAME"] == f"{design.GROUP}/{row['experiment_id']}/attempt-{job}"
            and env["RLVRAM_MODEL_SNAPSHOT"] == frozen["protocol"]["model_cache"][row["model"]]["path"]
            and env["RLVRAM_GPU_UUIDS"] == ",".join(expected)
            and env["RLVRAM_ALLOCATION_GPU_UUIDS"] ==
            ",".join(g["uuid"] for g in manifest["allocation"]["gpus"])
            and env["RLVRAM_INVOCATION_ID"] == invocation and env["RLVRAM_PERIOD"] == str(period)
            and env["TRAINER_LOGGER"] == '["console"]'
            and relocate_evidence(env["RLVRAM_DURABLE_METRICS_DIR"], evidence.root,
                                  relative_attempt / "trainer-metrics") == attempt / "trainer-metrics",
            "Dispatch source/model/metric/device contract differs")
    reward = str(recorded_source / "memory_tuner/code_shape_reward.py") \
        if row["dataset"] == "code_heavy_tail" else "null"
    require(env["CUSTOM_REWARD_FUNCTION_PATH"] == reward, "Wrong frozen reward")
    execution = evidence.json(attempt / "execution.json")
    require(all(record.get(k) == v for k, v in execution.items()), "Stored execution/terminal disagree")
    require(dispatched["timestamp_ns"] <= execution["started_ns"] <= spawned[0]["timestamp_ns"]
            <= execution["finished_ns"] <= terminal["timestamp_ns"], "Execution lifetime differs from journal")
    phases = {p: [s for s in trace if s["period"] == period and s["phase"] == p]
              for p in ("pre_idle", "payload", "post_idle")}
    require(core.idle_gate(phases["pre_idle"]) and core.idle_gate(phases["post_idle"]),
            "Missing full-allocation pre/post idle evidence")
    require(phases["pre_idle"][-1]["monotonic_ns"] < dispatched["monotonic_ns"]
            and execution["finished_ns"] <= phases["post_idle"][0]["timestamp_ns"]
            and phases["post_idle"][-1]["monotonic_ns"] < terminal["monotonic_ns"],
            "Idle/cleanup chronology differs")
    require(len(phases["payload"]) == execution["allocation_samples"] > 0,
            "Missing/incomplete allocation-wide payload sampling")
    contaminated = False
    for sample in phases["payload"]:
        require(spawned[0]["timestamp_ns"] <= sample["timestamp_ns"] <= execution["finished_ns"],
                "Allocation sample outside invocation")
        excluded = [p for p in sample["processes"] if p["uuid"] not in expected]
        require(sample["excluded_device_processes"] == excluded
                and all(p in sample["processes"] for p in sample["foreign_processes"]),
                "Contamination annotations disagree with process inventory")
        contaminated |= bool(excluded or sample["foreign_processes"])
    require(execution["contamination_detected"] is contaminated, "Contamination flag differs")
    if period == 2:
        previous = core.one_event(events, "terminal", 1)
        previous_post = [s for s in trace if s["period"] == 1 and s["phase"] == "post_idle"]
        require(core.idle_gate(previous_post)
                and previous_post[-1]["monotonic_ns"] < previous["monotonic_ns"]
                < phases["pre_idle"][0]["monotonic_ns"], "Period two lacks prior cleanup")
    duration = core.finite(execution["invocation_elapsed_seconds"], "invocation time")
    payload = core.finite(execution["payload_elapsed_seconds"], "payload time")
    require(0 < payload <= duration <= (execution["finished_ns"] - execution["started_ns"]) / 1e9 + .1
            and (payload <= 1501 or execution["timed_out"] or execution["error"]),
            "Execution timing/cap differs")
    return attempt, execution


def slot(evidence, frozen, pair, row, pair_error=None):
    result = {k: row[k] for k in ("pair_id", "experiment_id", "configuration_id",
                                  "model", "dataset", "configuration_level")}
    result.update(period=int(row["period"]), training_seed=int(row["training_seed"]),
                  task_gpu_count=int(row["gpu_count"]), status="missing", state="unresolved",
                  eligible=False, completed=False, launched=None, online_status=None,
                  known_memory_failure=False, known_margin_exceedance=False,
                  completed_peak_mib=None, observed_task_peak_mib=None,
                  task_gpu_seconds=None, invocation_reservation_gpu_seconds=None,
                  invocation_elapsed_seconds=None, task_runtime_seconds=None, validation_errors=[])
    try:
        require(pair_error is None, pair_error)
        events = [e for e in pair["events"] if e.get("period") == result["period"]]
        result["launched"] = any(e["event"] == "spawned" for e in events)
        record = evidence.json(pair["directory"] / f"period-{row['period']}.json")
        result["online_status"] = record.get("status")
        # Cost evidence is independently eligible even when completion evidence fails.
        result.update(core.invocation_cost(record, pair, row))
        context = period_context(evidence, frozen, pair, row, record)
        if context is None:
            result.update(status="not_launched", reason=record.get("reason"))
            return result
        attempt, execution = context
        assessed = assess_period(attempt, row, execution, pair["manifest"]["allocation"],
                                 frozen["protocol"], pair["manifest"]["source_commit"])
        result.update(assessed)
        require(assessed["status"] in ("completed", "memory_failure"), assessed["validation_error"])
        complete = assessed["status"] == "completed"
        peak = assessed["observed_task_peak_mib"]
        result.update(eligible=True, state=("above_margin" if peak > LIMIT else "within_margin")
                      if complete else "memory_failure", known_memory_failure=not complete,
                      known_margin_exceedance=complete and peak > LIMIT)
    except ERRORS as error:
        result["validation_errors"].append(evidence.error(error))
        if result["launched"]:
            result["status"] = "infrastructure_invalid"
    return result


def read_accounting(path):
    if path is None:
        return {}
    rows = json.loads(path.read_text())
    require(isinstance(rows, list), "Accounting must be a JSON list")
    result = {}
    for row in rows:
        job = str(row["job_id"])
        require(re.fullmatch(r"[1-9]\d*", job) and job not in result, "Duplicate/invalid accounting job")
        require(row["allocated_gpu_count"] == 4
                and 0 <= core.finite(row["started_at_epoch"], "start")
                < core.finite(row["ended_at_epoch"], "end"), "Invalid actual allocation interval")
        result[job] = row
    return result


def stratified_evaluation(predictions, configs):
    """Keep offloaded c1 separate from the previously difficult resident settings."""
    result = {}
    for model in design.MODELS:
        result[model] = {}
        for name, levels in design.LEVEL_GROUPS.items():
            selected = [c for c in configs if c["settings"]["model_id"] == model
                        and c["settings"]["configuration_level"] in levels]
            ids = {c["configuration_id"] for c in selected}
            result[model][name] = core.evaluate(
                [p for p in predictions if p["configuration_id"] in ids], selected)
    return result


def collect(root, source, directory, freeze_sha256, source_commit, accounting=None,
            runtime_amendment=None, amendment_sha256=None, original_source_root=None):
    started = time.monotonic()
    require(re.fullmatch(r"[a-f0-9]{40}", source_commit or ""), "Exact source commit required")
    frozen = design.selected_runtime(source, directory, root, freeze_sha256,
                                     runtime_amendment, amendment_sha256, original_source_root)
    amendment = frozen.get("runtime_amendment")
    if amendment:
        require(source_commit == amendment["amended_source_commit"],
                "--source-commit must identify the amended execution source")
    evidence = core.Evidence(root)
    group = core.inside(root, Path("output") / design.GROUP)
    require(not group.is_symlink(), "Linked raw study root is forbidden")
    processes, pairs, allocation_costs, errors = [], [], [], []
    used_jobs = set()
    expected_slots = {r["experiment_id"] for r in frozen["rows"]}
    unplanned = sorted(p.name for p in group.iterdir()
                       if p.name not in expected_slots | {"pairs"}
                       and not re.fullmatch(r"concurrency-[0-2]\.lock", p.name)) if group.exists() else []
    extra_pairs = sorted(p.name for p in (group / "pairs").iterdir()
                         if p.name not in frozen["pairs"]) if (group / "pairs").exists() else []
    for index, (pair_id, rows) in enumerate(frozen["pairs"].items()):
        effective = design.pair_runtime(frozen, index)
        expected_commit = (design.EXECUTION_SOURCE_COMMIT if index == 0 else
                           amendment["amended_source_commit"]) if amendment else source_commit
        provenance = dict(pair_index=index, expected_execution_source_commit=expected_commit,
                          authorized_runtime_amendment_sha256=amendment_sha256 if amendment and index else None)
        pair, error = None, None
        try:
            pair = core.validate_pair(evidence, effective, pair_id, expected_commit)
            design.validate_runtime_binding(evidence, effective, pair, index)
            require(pair["manifest"]["freeze_sha256"] == freeze_sha256
                    and core.one_event(pair["events"], "pair_claimed")["freeze_sha256"] == freeze_sha256,
                    "Pair claim bound to another freeze")
            job = str(pair["manifest"]["job_id"])
            require(job not in used_jobs, "One allocation cannot replace another pair")
            used_jobs.add(job)
            allocation_costs.append(core.allocation_cost(evidence, pair, accounting or {}))
        except ERRORS as exc:
            error = evidence.error(exc)
            errors.append(dict(pair_id=pair_id, error=error))
        provenance.update(execution_source_commit=pair["manifest"].get("source_commit") if pair else None,
                          runtime_amendment_sha256=pair["manifest"].get("runtime_amendment_sha256")
                          if pair else None)
        selected = [dict(slot(evidence, effective, pair, row, error), **provenance) for row in rows]
        processes.extend(selected)
        by_gpu = {r["task_gpu_count"]: r for r in selected}
        both = all(r["completed"] and r["eligible"] for r in selected)
        pairs.append(dict(provenance, pair_id=pair_id, model=rows[0]["model"], dataset=rows[0]["dataset"],
                          configuration_level=rows[0]["configuration_level"],
                          training_seed=int(rows[0]["training_seed"]),
                          first_gpu_count=int(rows[0]["gpu_count"]),
                          two_gpu_status=by_gpu[2]["status"], four_gpu_status=by_gpu[4]["status"],
                          both_completed=both, both_eligible=all(r["eligible"] for r in selected),
                          peak_difference_4_minus_2_mib=by_gpu[4]["completed_peak_mib"]
                          - by_gpu[2]["completed_peak_mib"] if both else None,
                          runtime_difference_4_minus_2_seconds=by_gpu[4]["task_runtime_seconds"]
                          - by_gpu[2]["task_runtime_seconds"] if both else None))
    # Preserve every actual attempt path, including orphan/extra attempts; never select a survivor.
    actual_attempts = sorted(str(p.relative_to(root)) for p in group.glob("*/attempt-*"))
    configs = core.configurations(processes, frozen["targets"], design.SEEDS)
    costs = {}
    for field in ("task_gpu_seconds", "invocation_reservation_gpu_seconds"):
        values = [r[field] for r in processes if r[field] is not None]
        costs[field] = dict(known_sum=sum(values), known_slots=len(values), planned_slots=108,
                            complete_total=sum(values) if len(values) == 108 else None)
    for field in ("observed_allocation_gpu_seconds", "whole_allocation_gpu_seconds"):
        values = [r[field] for r in allocation_costs if r[field] is not None]
        costs[field] = dict(known_sum=sum(values), known_pairs=len(values), planned_pairs=54,
                            complete_total=sum(values) if len(values) == 54 else None)
    costs.update(reserved_gpu_hours_cap=198, historical_source=frozen["protocol"]["source_cost"],
                 incremental_source_collection_gpu_seconds=0,
                 prediction_cpu_wall_seconds=frozen["protocol"]["prediction_cpu_wall_seconds"],
                 interpretation=frozen["protocol"]["cost_scope"])
    summary = dict(
        schema_version=design.VERSION, freeze_sha256=freeze_sha256, source_commit=source_commit,
        interpretation=frozen["protocol"]["interpretation"], planned_slots=108, planned_pairs=54,
        planned_configurations=36, statuses=dict(Counter(p["status"] for p in processes)),
        actual_attempt_paths=actual_attempts, unplanned_entries=unplanned, unplanned_pairs=extra_pairs,
        evaluation=core.evaluate(frozen["predictions"], configs),
        evaluation_by_model={m: core.evaluate(
            [p for p in frozen["predictions"] if p["configuration_id"] in
             {c["configuration_id"] for c in configs if c["settings"]["model_id"] == m}],
            [c for c in configs if c["settings"]["model_id"] == m]) for m in design.MODELS},
        evaluation_by_model_and_level_group=stratified_evaluation(frozen["predictions"], configs),
        donor_gpu_count_transfer_ids=sorted(
            p["configuration_id"] for p in frozen["predictions"]
            if p["method"] == "donor_copy" and not p["donor_gpu_count_matches"]),
        costs=costs, allocation_costs=allocation_costs, pair_errors=errors,
        predictions=frozen["predictions"], evidence_files=evidence.hashes,
        collection_cpu_wall_seconds=time.monotonic() - started,
        offline_limit="Hashes detect alteration against the supplied freeze and archive, not "
        "wholesale replacement. PID ancestry uses recorded runtime ownership annotations; "
        "physical allocation samples are periodic. No hardware-confinement claim from masks.")
    if amendment:
        summary.update(
            source_commit=None, runtime_amendment_sha256=amendment_sha256,
            runtime_amendment=amendment,
            authorized_source_by_index={str(i): (design.EXECUTION_SOURCE_COMMIT if i == 0 else source_commit)
                                        for i in range(54)},
            source_seal_interpretation=design.AMENDMENT_SCOPE)
    evidence.verify_unchanged()
    # Recheck code, plan, predictions and inherited inputs after reading raw outcomes.
    design.selected_runtime(source, directory, root, freeze_sha256,
                            runtime_amendment, amendment_sha256, original_source_root)
    flat = core.flat_configurations(configs, frozen["predictions"])
    donors = {p["configuration_id"]: p for p in frozen["predictions"]
              if p["method"] == "donor_copy"}
    for row in flat:
        donor = donors[row["configuration_id"]]
        row.update(donor_configuration_id=donor["donor_configuration_id"],
                   donor_gpu_count_matches=donor["donor_gpu_count_matches"])
    return dict(summary=summary, slots=processes,
                configurations=flat, pairs=pairs)


def write_results(results, output):
    output.mkdir(parents=True, exist_ok=False)
    write_new_json(output / "summary.json", results["summary"])
    for name in ("slots", "configurations", "pairs"):
        rows = results[name]
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (output / f"{name}.csv").open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list))
                                 else v for k, v in row.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=design.CANONICAL_ROOT)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--protocol-dir", type=Path)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--allocation-accounting", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    design.amendment_arguments(parser)
    args = parser.parse_args(argv)
    source = args.source_root.resolve()
    directory = (args.protocol_dir or source / design.DIRECTORY).resolve()
    output = args.output.resolve()
    require(not output.is_relative_to(args.root.resolve() / "output")
            and not output.is_relative_to(directory), "Collector output must be outside raw/frozen inputs")
    result = collect(args.root.resolve(), source, directory, args.freeze_sha256, args.source_commit,
                     read_accounting(args.allocation_accounting), args.runtime_amendment,
                     args.amendment_sha256, args.original_source_root)
    write_results(result, output)
    print(json.dumps(dict(output=str(output), planned_slots=108,
                          statuses=result["summary"]["statuses"],
                          resolved_configurations=result["summary"]["evaluation"]["resolved_configurations"]),
                     indent=2))


if __name__ == "__main__":
    main()
