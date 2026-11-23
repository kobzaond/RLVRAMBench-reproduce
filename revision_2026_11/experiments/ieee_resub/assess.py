"""CPU-only raw-evidence assessment of one invocation (E1-E3).

Derived from memory_tuner/electronics_experiment.py (assess). Additional
checks: two-GPU allocations, a step range for resumed runs, every scheduled
validation/checkpoint, the batch-shape trace (E1), fixed response lengths, and
the read-only restore source. Labels follow the paper: within margin if every
sampled task-GPU value is <= 38,912 MiB after a complete eligible run; memory
failure only with an explicit GPU-memory diagnostic; everything else unresolved.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import re

from ieee_resub import common as C

C.bootstrap()

TRIAL_FIELDS = (
    "algorithm", "model", "gpu_count", "rollout_tp_size", "actor_micro_batch",
    "rollout_logprob_micro_batch", "ref_logprob_micro_batch", "vllm_gpu_memory_utilization",
    "parameter_offload", "optimizer_offload", "free_cache_engine", "max_prompt_length",
    "max_response_length", "max_model_len", "max_num_seqs", "rollout_n", "train_batch_size",
    "training_seed", "gpu_monitor_interval_ms", "total_training_steps", "save_freq", "test_freq",
    "val_before_train", "total_epochs", "allocator_trace_enabled", "fixed_response_length",
    "canonical_batch_order", "resume_mode",
)


def normalized(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value).lower()


# --------------------------------------------------------------------------- identity
def validate_identity(directory, expected, invocation, *, success, allocation_gpus, is_idle):
    """run_matched_gpu.validate_identity with the allocation size as a parameter."""
    parent = json.loads((directory / "trainer-cuda.json").read_text())
    C.require(parent["invocation_id"] == invocation and parent["requested_uuids"] == expected
              and len(parent["cuda_devices"]) == len(expected)
              and {d["uuid"] for d in parent["cuda_devices"]} == set(expected),
              "Parent actual CUDA identity mismatch")
    gate = json.loads((directory / "probe-exit-idle.json").read_text())
    trainer = json.loads((directory / "trainer-exec.json").read_text())
    from memory_tuner.device_contract import digest
    C.require(parent.get("probe_kind") == "exited_subprocess_same_container"
              and gate["invocation_id"] == invocation and trainer["invocation_id"] == invocation
              and gate["probe_sha256"] == digest(directory / "trainer-cuda.json")
              and gate["probe_pid"] == parent["pid"] and parent["pid"] != trainer["pid"]
              and parent["parent_pid"] == trainer["pid"] and gate["pid"] == trainer["pid"]
              and trainer["no_probe_context_in_trainer"]
              and trainer["cuda_visible_devices"] == ",".join(expected)
              and gate["all_allocation_devices_idle"]
              and len(set(gate["allocation_uuids"])) == allocation_gpus
              and set(expected).issubset(gate["allocation_uuids"])
              and gate["observations"] and is_idle(gate["observations"][-1]),
              "Probe context release/fresh-trainer evidence is invalid")
    coverage, roles, records = set(), {}, []
    for path in sorted(directory.glob("worker-*.json")):
        record = json.loads(path.read_text())
        C.require("validation_error" not in record, "Observed worker device gate failed")
        devices = [d["uuid"] for d in record["cuda_devices"]]
        C.require(record["invocation_id"] == invocation and record["hostname"] == parent["hostname"]
                  and record["world_size"] == len(expected) and 0 <= record["rank"] < len(expected)
                  and devices and len(set(devices)) == len(devices) and set(devices).issubset(expected)
                  and record["active_uuid"] in devices, "Observed worker identity is invalid")
        ranks = roles.setdefault(record["role"], {})
        C.require(record["rank"] not in ranks, "Duplicate worker rank/role identity")
        ranks[record["rank"]] = record["active_uuid"]
        coverage.add(record["active_uuid"])
        records.append(record)
    complete = coverage == set(expected) and bool(roles) and all(
        set(r) == set(range(len(expected))) and set(r.values()) == set(expected) for r in roles.values())
    ray_path = directory / "ray-resources.json"
    if ray_path.exists():
        ray = json.loads(ray_path.read_text())
        C.require(ray["invocation_id"] == invocation and ray["resources"].get("GPU") == len(expected)
                  and len(ray["nodes"]) == 1 and ray["nodes"][0].get("Alive"), "Ray resource identity mismatch")
    else:
        C.require(not success, "Successful process lacks Ray resource evidence")
    C.require(not success or complete, "Successful process lacks full worker/device coverage")
    return dict(identity_validated=True, observed_worker_uuids=sorted(coverage),
                observed_worker_records=len(records), worker_coverage_complete=complete)


# --------------------------------------------------------------------------- traces
def phase_trace(attempt: Path, job: str, expected_devices: int, max_step: int) -> dict:
    """Stage peaks, per-step peaks and (step, stage) coverage from the raw traces."""
    memory_rows = list(csv.reader((attempt / f"gpu-memory-{job}.csv").open()))
    phase_rows = list(csv.reader((attempt / f"phase-memory-{job}.csv").open()))
    C.require(len(memory_rows) == len(phase_rows) and phase_rows, "Phase/raw sample inventory differs")
    stage_peaks, durations, step_peaks, pairs = {}, {}, {}, {}
    previous = None
    for sample, memory_sample in zip(phase_rows, memory_rows):
        C.require(len(sample) == expected_devices + 3, "Phase trace shape differs")
        step = int(sample[1])
        C.require(0 <= step <= max_step, "Phase step outside the requested horizon")
        C.require(re.fullmatch(r"[A-Za-z0-9_]+", sample[2]) is not None, "Malformed phase label")
        C.require(sample[:1] + sample[3:] == memory_sample, "Phase/raw observations differ")
        C.require(previous is None or int(sample[0]) > int(previous[0]), "Nonmonotonic phase clock")
        value = max(map(float, sample[3:]))
        C.require(math.isfinite(value) and 0 <= value <= C.CAPACITY_MIB, "Invalid memory observation")
        stage = sample[2]
        stage_peaks[stage] = max(stage_peaks.get(stage, 0.0), value)
        step_peaks[step] = max(step_peaks.get(step, 0.0), value)
        key = f"{step}:{stage}"
        pairs[key] = max(pairs.get(key, 0.0), value)
        if previous is not None and stage == previous[2]:
            durations[stage] = durations.get(stage, 0.0) + (int(sample[0]) - int(previous[0])) / 1e9
        previous = sample
    return dict(stage_peaks_mib=stage_peaks, stage_sampled_seconds=durations,
                step_peaks_mib={str(k): v for k, v in sorted(step_peaks.items())}, step_stage_peaks_mib=pairs)


def growth(step_peaks: dict, first_step: int) -> dict:
    peaks = {int(k): float(v) for k, v in step_peaks.items()}
    if first_step not in peaks:
        return dict(first_step_peak_mib=None, maximum_later_step_increase_mib=None, step_of_maximum_later_increase=None)
    later = [(peaks[s] - peaks[first_step], s) for s in sorted(peaks) if s > first_step]
    best = max(later) if later else (None, None)
    return dict(first_step_peak_mib=peaks[first_step], maximum_later_step_increase_mib=best[0],
                step_of_maximum_later_increase=best[1])


def validate_step_range(directory: Path, invocation: str, first: int, last: int, *, earliest_ns=None, latest_ns=None):
    """memory_tuner.durable_metrics.validate_completed_step for steps first..last."""
    C.require(1 <= first <= last, "Invalid durable step range")
    C.require(C.SAFE_ID.fullmatch(invocation) is not None, "Invalid invocation identity")
    C.require(directory.is_dir() and not directory.is_symlink(), "Durable trainer-metric directory is absent")
    expected = {f"step-{s}.json" for s in range(first, last + 1)}
    C.require({p.name for p in directory.iterdir()} == expected, "Durable trainer-metric inventory differs")
    records, hashes = [], {}
    for step in range(first, last + 1):
        path = directory / f"step-{step}.json"
        raw = path.read_bytes()
        record = json.loads(raw)
        line = record.get("console_line")
        C.require(record.get("schema_version") == 1 and record.get("event") == "trainer_metric_logged"
                  and record.get("invocation_id") == invocation and record.get("step") == step
                  and record.get("training_global_step") == step and isinstance(line, str) and "\n" not in line
                  and record.get("console_line_sha256") == hashlib.sha256(line.encode()).hexdigest(),
                  "Durable trainer-metric identity or content is invalid")
        fields = line.split(" - ")
        metric = [f.removeprefix("training/global_step:") for f in fields[1:] if f.startswith("training/global_step:")]
        C.require(fields[0] == f"step:{step}" and len(metric) == 1 and float(metric[0]) == step,
                  "Captured metric line does not contain the exact training step")
        stamp = record["timestamp_ns"]
        C.require((earliest_ns is None or stamp >= earliest_ns) and (latest_ns is None or stamp <= latest_ns),
                  "Durable metric lies outside the measured trainer lifetime")
        C.require(not records or (stamp > records[-1]["timestamp_ns"] and record["pid"] == records[0]["pid"]),
                  "Training-step chronology or logger identity changed")
        records.append(record)
        hashes[path.name] = hashlib.sha256(raw).hexdigest()
    return dict(durable_completed_step_validated=True, first_completed_step=first, completed_steps=last - first + 1,
                logger_pid=records[0]["pid"], logger_hostname=records[0]["hostname"],
                durable_record_sha256=hashes, validation_metric_steps=[
                    r["step"] for r in records if " - val-" in r["console_line"]])


def scheduled_steps(row: dict, first: int, last: int):
    save, test = int(row["save_freq"]), int(row["test_freq"])
    checkpoints = sorted({s for s in range(first, last + 1) if save > 0 and (s % save == 0 or s == last)})
    validations = sorted({s for s in range(first, last + 1) if test > 0 and (s % test == 0 or s == last)})
    if row["val_before_train"] == "True" and test > 0:
        validations = [first - 1] + validations
    return checkpoints, validations


def checkpoint_inventory(folder: Path) -> dict:
    """Written by the runner right after the invocation, so that later
    re-assessment does not depend on scratch-resident checkpoints."""
    if not folder.is_dir():
        return dict(folder=str(folder), exists=False, steps={}, latest=None)
    steps = {}
    for step_dir in sorted(folder.glob("global_step_*")):
        actor = step_dir / "actor"
        steps[step_dir.name.removeprefix("global_step_")] = dict(
            actor_present=actor.is_dir(), data_present=(step_dir / "data.pt").is_file(),
            actor_files={p.name: p.stat().st_size for p in sorted(actor.glob("*")) if p.is_file()} if actor.is_dir() else {})
    tracker = folder / "latest_checkpointed_iteration.txt"
    return dict(folder=str(folder), exists=True, steps=steps,
                latest=tracker.read_text().strip() if tracker.is_file() else None)


def lifecycle_evidence(row, trace, first, last, inventory: dict | None):
    checkpoints, validations = scheduled_steps(row, first, last)
    pairs = trace["step_stage_peaks_mib"]
    for step in checkpoints:
        C.require(f"{step}:checkpoint" in pairs, f"Scheduled checkpoint at step {step} not observed")
    for step in validations:
        C.require(f"{step}:validation" in pairs, f"Scheduled validation at step {step} not observed")
    retained = []
    if checkpoints:
        C.require(inventory is not None and inventory["exists"], "Checkpoint directory missing")
        keep = int(row["max_actor_ckpt_to_keep"])
        retained = checkpoints[-keep:]
        for step in retained:
            entry = inventory["steps"].get(str(step))
            C.require(entry and entry["actor_present"] and entry["data_present"] and entry["actor_files"],
                      f"Retained checkpoint {step} missing")
        C.require(inventory["latest"] == str(last), "Latest checkpoint tracker differs from the final step")
    return dict(scheduled_checkpoint_steps=checkpoints, scheduled_validation_steps=validations,
                retained_checkpoint_steps=retained,
                validation_peak_mib=max([pairs[f"{s}:validation"] for s in validations], default=None),
                checkpoint_peak_mib=max([pairs[f"{s}:checkpoint"] for s in checkpoints], default=None))


def load_shape_module():
    import importlib.util
    path = C.EXPERIMENT_DIR / "instrumented_python_packages/verl/utils/rlvram_shape_trace.py"
    spec = importlib.util.spec_from_file_location("ieee_resub._rlvram_shape_trace", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def shape_trace_evidence(attempt: Path, job: str, row: dict, invocation: str, first: int, last: int):
    shapes = load_shape_module()
    records = shapes.read_records(attempt / f"shape-trace-{job}.jsonl")
    C.require([r["step"] for r in records] == list(range(first, last + 1)), "Shape trace step inventory differs")
    expected_sequences = int(row["train_batch_size"]) * int(row["rollout_n"])
    for record in records:
        C.require(record["invocation_id"] == invocation, "Shape trace invocation differs")
        C.require(record["canonical_order"] == (row["canonical_batch_order"] == "True"), "Canonical-order flag differs")
        C.require(record["sequences"] == expected_sequences and record["padding_entries"] == 0,
                  "Unexpected sequence count or padding")
        C.require(record["actor_micro_batch"] == int(row["actor_micro_batch"]), "Actor micro-batch differs in trace")
        if row["fixed_response_length"] == "True":
            C.require(set(record["dispatch_response_lengths"]) == {int(row["max_response_length"])},
                      "A response did not reach the fixed response length")
        C.require(max(record["dispatch_prompt_lengths"]) <= int(row["max_prompt_length"]), "Prompt exceeds cap")
    final = records[-1]
    signature = shapes.shape_signature(final)
    return dict(shape_signature=list(signature), prompt_multiset_sha256=final["prompt_multiset_sha256"],
                length_multiset_sha256=final["length_multiset_sha256"],
                response_tokens_sha256=final["response_tokens_sha256"], total_tokens=final["total_tokens"],
                actor_micro_batch_tokens=final["actor_micro_batch_tokens"],
                max_actor_micro_batch_tokens=max(max(r) for r in final["actor_micro_batch_tokens"]),
                response_lengths=sorted(set(final["dispatch_response_lengths"])))


def restore_evidence(attempt: Path, row: dict, text: str, trace: dict):
    step = int(row["resume_step"])
    C.require(re.search(rf"RLVRAM_RESUME global_step_folder=\S+/global_step_{step} global_steps={step}\b", text),
              "Trainer did not report the requested resume step")
    C.require(f"{step}:checkpoint_load" in trace["step_stage_peaks_mib"], "Checkpoint-load interval not observed")
    before = json.loads((attempt / "restore-source.json").read_text())
    after = json.loads((attempt / "restore-source-post.json").read_text())
    C.require(before["folder"] == after["folder"] and before["manifest"] == after["manifest"],
              "Restore source checkpoint changed")
    return dict(restore_source=before["folder"], restore_source_files=len(before["manifest"]),
                checkpoint_load_peak_mib=trace["step_stage_peaks_mib"][f"{step}:checkpoint_load"])


# --------------------------------------------------------------------------- printed configuration
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
TASK_RUNNER_RECORD = re.compile(r"^\(TaskRunnerV1 pid=(\d+)\) (.*)$")
MISSING_CONFIGURATION = "Missing printed actor configuration"


def rejoin_split_configuration_start(text: str) -> str | None:
    """e3-amendment-01, defect 2: Ray can emit the opening brace of the printed
    Hydra configuration as its own record, "(TaskRunnerV1 pid=N) {", followed
    by the record "(TaskRunnerV1 pid=N) 'actor_rollout_ref': {...". Return the
    console text with exactly that brace joined onto the next record of the
    same process (no other change), or None if no such split start exists."""
    lines = ANSI_ESCAPE.sub("", text).splitlines()
    for i, line in enumerate(lines):
        brace = TASK_RUNNER_RECORD.match(line)
        if not brace or brace[2].strip() != "{":
            continue
        for j in range(i + 1, len(lines)):
            following = TASK_RUNNER_RECORD.match(lines[j])
            if following and following[1] == brace[1]:
                if following[2].lstrip().startswith("'actor_rollout_ref':"):
                    joined = f"(TaskRunnerV1 pid={brace[1]}) {{{following[2].lstrip()}"
                    return "\n".join(lines[:i] + lines[i + 1:j] + [joined] + lines[j + 1:])
                break  # the next record of this process is something else
    return None


def applied_actor_configuration(text):
    """memory_tuner.electronics_experiment.applied_actor_configuration, unchanged
    for every console it accepts. Only when it finds no start record is the
    split opening brace (defect 2) rejoined and the same parser applied again;
    no field is inferred from the planned configuration."""
    from memory_tuner.electronics_experiment import applied_actor_configuration as frozen
    try:
        return frozen(text)
    except ValueError as error:
        if str(error) != MISSING_CONFIGURATION:
            raise
        rejoined = rejoin_split_configuration_start(text)
        if rejoined is None:
            raise
    return frozen(rejoined)


# --------------------------------------------------------------------------- assessment
def assess(attempt, row, execution, allocation, protocol, source_commit, base):
    from memory_tuner.electronics_experiment import parameter_evidence
    from memory_tuner.electronics_failure_evidence import has_gpu_memory_evidence
    from memory_tuner.log_parser import classify_failure_detailed
    from ieee_resub.env import checkpoint_dir

    result = dict(status="unresolved", failure_kind=None, completed_peak_mib=None,
                  observed_peak_mib=None, validation_error=None)
    job = str(allocation["job_id"])
    invocation = row["experiment_id"] + "-j" + job
    expected = [allocation["gpus"][i]["uuid"] for i in base.subset_ordinals(row)]
    restore = row["role"] == "restore"
    first = int(row["resume_step"]) + 1 if restore else 1
    last = int(row["total_training_steps"])
    try:
        C.require(execution["spawned"] and execution["sampling_valid"] and not execution["contamination_detected"]
                  and not execution["error"] and execution["cleanup"]["complete"]
                  and not execution["cleanup"]["remaining_pids"], "Invalid process/measurement/cleanup evidence")
        C.require(not execution["timed_out"] and not execution.get("interrupted"), "Timed out or interrupted; unresolved")
        code = execution["wrapper_exit_code"]
        text = (attempt / f"training-{job}.log").read_text(errors="replace")
        C.require(text.strip(), "Missing training console")
        config_pid, actual = applied_actor_configuration(text)
        C.require(actual["model"]["lora_rank"] == int(row["lora_rank"])
                  and actual["model"]["lora_alpha"] == int(row["lora_alpha"])
                  and actual["actor"]["use_kl_loss"] is (row["use_kl_loss"] == "True")
                  and actual["actor"]["ppo_micro_batch_size_per_gpu"] == int(row["actor_micro_batch"])
                  and actual["rollout"]["fixed_response_length"] is (row["fixed_response_length"] == "True"),
                  "Applied actor/rollout settings differ from the frozen matrix")
        trial = json.loads((attempt / f"trial-{job}.json").read_text())
        for field in TRIAL_FIELDS:
            C.require(normalized(trial.get(field)) == normalized(row.get(field)), f"Trial field differs: {field}")
        C.require(trial["exit_code"] == code and str(trial["job_id"]) == job, "Exit/job mismatch")
        env = json.loads((attempt / f"environment-{job}.json").read_text())
        C.require(env["repository"]["head"] == source_commit and not env["repository"]["dirty"], "Unfrozen source")
        C.require(env["container"]["sha256"] == C.CONTAINER_SHA256, "Container differs")
        C.require(env["software"]["packages"]["torch"] == C.TORCH_VERSION
                  and env["software"]["packages"]["vllm"] == C.VLLM_VERSION, "Runtime versions differ")
        C.require(env["model"]["resolved_revision"] == row["model_revision"]
                  and env["model"]["requested_model"] == row["model"], "Model identity differs")
        for split in ("train.parquet", "test.parquet"):
            relative = f"data/{C.DATA_SUBDIRS[row['dataset']]}/{split}"
            C.require(env["dataset"][split]["sha256"] == protocol["data_sha256"][relative], "Dataset changed")
        result.update(base.validate_identity(attempt / "device-evidence", expected, invocation, success=code == 0))
        result.update(base.validate_trace(attempt, expected, invocation, job))
        result["observed_peak_mib"] = result["task_peak_memory_mib"]
        C.require(abs(float(trial["peak_gpu_memory_mib"]) - result["observed_peak_mib"]) < .01, "Reported/raw peak differ")
        trace = phase_trace(attempt, job, len(expected), last)
        result.update(trace, elapsed_seconds=trial["elapsed_seconds"])
        kind = classify_failure_detailed(text, code)
        if code != 0:
            C.require(kind in C.MEMORY_FAILURES, "Non-memory or undiagnosed failure: " + str(kind))
            C.require(has_gpu_memory_evidence(text), "No explicit GPU-memory diagnostic")
            result.update(status="memory_failure", failure_kind=kind)
            return result
        terminal = json.loads((attempt / "device-evidence/trainer-terminal.json").read_text())
        trainer = json.loads((attempt / "device-evidence/trainer-exec.json").read_text())
        C.require(terminal["pid"] == trainer["pid"] and terminal["timestamp_ns"] > trainer["timestamp_ns"],
                  "Successful exit lacks trainer identity/lifetime")
        result.update(validate_step_range(attempt / "trainer-metrics", invocation, first, last,
                                          earliest_ns=trainer["timestamp_ns"], latest_ns=terminal["timestamp_ns"]))
        C.require(result["logger_pid"] == config_pid, "Configuration and completed-work producer differ")
        result.update(parameter_evidence(attempt, row))
        saved = attempt / "checkpoint-inventory.json"
        if saved.is_file():
            inventory = json.loads(saved.read_text())
        else:
            folder = checkpoint_dir(row, job) if row["checkpoint_to_scratch"] == "True" else attempt
            inventory = checkpoint_inventory(folder)
        result.update(lifecycle_evidence(row, trace, first, last, inventory))
        result.update(growth(trace["step_peaks_mib"], first))
        if row["shape_trace"] == "True":
            result.update(shape_trace_evidence(attempt, job, row, invocation, first, last))
        if restore:
            result.update(restore_evidence(attempt, row, text, trace))
        within = result["observed_peak_mib"] <= C.MARGIN_MIB
        result.update(status="within_margin" if within else "above_margin", completed_peak_mib=result["observed_peak_mib"])
    except (ValueError, KeyError, OSError, TypeError, IndexError) as error:
        result.update(status="unresolved", validation_error=f"{type(error).__name__}: {error}")
    return result
