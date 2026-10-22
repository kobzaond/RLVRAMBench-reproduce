"""Frozen Electronics revision experiments and independently checked raw outcomes.

`prepare-pilot` creates excluded development runs. `run` executes one permanently
claimed block within an existing Slurm allocation; it never submits jobs.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import time

from memory_tuner.benchmark_v2_env import environment
from memory_tuner.device_contract import Journal, digest, write_new_json
from memory_tuner.durable_metrics import validate_completed_step
from memory_tuner.grpo_raw_evidence import MATCH_FIELDS, normalized
from memory_tuner.log_parser import classify_failure_detailed
from memory_tuner.resolve_hf_snapshot import validate_snapshot
from memory_tuner.run_offload_factorial import InterruptState, clean_source, enable_subreaper, inventory
from memory_tuner.host_memory_monitor import HostMemorySampler
from memory_tuner.electronics_failure_evidence import has_gpu_memory_evidence

ROOT = Path("/mnt/proj3/open-35-44/verl/rl")
SOURCE = Path(__file__).resolve().parents[1]
DIRECTORY = Path("benchmark/electronics_revision_20261016")
CACHE = ROOT.parent / "runtime/kobzaond/cache/huggingface/hub"
REVISIONS = {
    "Qwen/Qwen2.5-1.5B-Instruct": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    "Qwen/Qwen2.5-3B-Instruct": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
    "Qwen/Qwen2.5-7B-Instruct": "a09a35458c702b33eeacc393d103063234e8bc28",
    "mistralai/Mistral-7B-Instruct-v0.3": "c170c708c41dac9275d15a8fff4eca08d52bab71",
}
MEMORY_FAILURES = {"cuda_oom", "rollout_init_memory", "weight_sync_oom",
                   "actor_update_oom", "rollout_oom"}
CONTAINER_SHA256 = "3dc581ca780cd19d565445b8a02e5daa98b350b3d2116d4d64476a64a2614912"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def runtime(caps):
    path = SOURCE / "memory_tuner/run_matched_gpu.py"
    spec = importlib.util.spec_from_file_location("memory_tuner._electronics_runtime", path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    base.ALLOCATION_CAP = caps["wall_seconds"]
    base.PAYLOAD_TIMEOUT = caps["payload_timeout_seconds"]
    base.CLEANUP_RESERVE = 60
    base.IDLE_TIMEOUT = 180
    return base


def freeze(directory, rows, *, study, caps, development=False, predictions=None):
    require(not directory.exists(), "A protocol directory may not be overwritten")
    require(rows and len({r["experiment_id"] for r in rows}) == len(rows), "Duplicate slots")
    directory.mkdir(parents=True)
    write_rows(directory / "matrix.csv", rows)
    model_cache = {}
    for model in sorted({r["model"] for r in rows}):
        revision = next(r["model_revision"] for r in rows if r["model"] == model)
        snapshot = validate_snapshot(CACHE / ("models--" + model.replace("/", "--")) /
                                     "snapshots" / revision)
        model_cache[model] = dict(path=str(snapshot), revision=revision,
                                 files_sha256={p.name: digest(p) for p in sorted(snapshot.iterdir())
                                               if p.is_file()})
    data_sha256 = {str(Path(r["data_dir"]).relative_to(ROOT) / split):
                   digest(Path(r["data_dir"]) / split)
                   for r in rows for split in ("train.parquet", "test.parquet")}
    protocol = dict(
        schema_version=1, study=study, development=development,
        endpoint_scope="Development compatibility only; excluded from confirmatory estimates."
        if development else "Frozen selected-grid resource outcomes, not deployment probabilities.",
        frozen_at_utc=datetime.now(timezone.utc).isoformat(), seed_role="fresh process repeats",
        no_automatic_retries=True, group=rows[0]["run_group"], caps=caps,
        matrix_sha256=digest(directory / "matrix.csv"), model_cache=model_cache,
        data_sha256=data_sha256, capacity_mib=40960, margin_mib=38912,
        planned_invocations=len(rows), blocks=len({r["block_id"] for r in rows}),
        allocation_gpus=4,
        measurement="Per-task-device NVML, 100 ms; host stage markers are not allocation ownership.",
        failure_policy="Only diagnosed GPU-memory errors are memory outcomes; timeouts and host OOM remain unresolved.",
        non_lora_policy="rank=0; native KL retains a separate CPU-offloaded reference; KL-disabled matched controls remove it in both training modes.",
        success_policy="All requested durable step metrics, successful trainer exit, physical-device identity, raw traces and parameter metadata.",
    )
    if predictions is not None:
        write_new_json(directory / "predictions.json", predictions)
        protocol["predictions_sha256"] = digest(directory / "predictions.json")
    write_new_json(directory / "protocol.json", protocol)
    print(json.dumps(dict(directory=str(directory), protocol_sha256=digest(directory / "protocol.json"),
                          planned_invocations=len(rows))))


def prepare_pilot(directory):
    template = read_rows(ROOT / "benchmark/offload_factorial/matrix.csv")[0]
    rows = []
    for index, (model, rank, kl) in enumerate([
        ("Qwen/Qwen2.5-1.5B-Instruct", 0, "True"),
        ("Qwen/Qwen2.5-1.5B-Instruct", 64, "True"),
        ("Qwen/Qwen2.5-3B-Instruct", 0, "True"),
    ]):
        identity = f"electronics-pilot-{index}-s{301+index}"
        row = dict(template)
        row.update(experiment_id=identity, run_group="electronics_pilot_20261016",
                   study_stage="development", revision_study="electronics_pilot",
                   model=model, model_revision=REVISIONS[model],
                   block_id=identity, pair_id=identity, block_index=str(index),
                   period="1", condition=f"rank{rank}-kl{kl.lower()}",
                   configuration_id=identity, case_id=f"pilot-{index}", sequence_id="0",
                   training_seed=str(301+index), lora_rank=str(rank), lora_alpha="32",
                   use_kl_loss=kl, actor_micro_batch="4",
                   vllm_gpu_memory_utilization="0.45", parameter_offload="True",
                   optimizer_offload="False", total_training_steps="2",
                   train_max_samples="64", capture_response_lengths="True",
                   allocator_trace_enabled="False", allocator_trace_sync="0",
                   per_process_timeout_seconds="1500", novelty="development")
        rows.append(row)
    freeze(directory, rows, study="full_parameter_compatibility",
           development=True, caps=dict(wall_seconds=3300, payload_timeout_seconds=1500,
                                      max_concurrent_allocations=3))


def adaptation_rows():
    """Native mode comparison, plus a separate matched no-reference control."""
    templates = read_rows(ROOT / "benchmark/offload_factorial/matrix.csv")
    rows = []
    block_index = 0
    rng = random.Random(202610163)
    for model in ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"):
        for dataset in ("gsm8k", "code_heavy_tail"):
            template = next(r for r in templates if r["dataset"] == dataset)
            for seed in (311, 312, 313):
                short = "q15" if "1.5B" in model else "q3"
                block = f"adapt-{short}-{dataset}-s{seed}"
                conditions = [(rank, offload, reservation, "True")
                              for rank in (0, 64) for offload in ("True", "False")
                              for reservation in ("0.45", "0.65")]
                rng.shuffle(conditions)
                for period, (rank, offload, reservation, kl) in enumerate(conditions, 1):
                    condition = f"rank{rank}-off{int(offload == 'True')}-r{reservation[2:]}-kl1"
                    row = adaptation_row(template, model, dataset, seed, block_index,
                                         block, period, condition, rank, offload, reservation, kl)
                    rows.append(row)
                block_index += 1
    # Removing the reference in both modes controls its presence within these
    # comparisons. This changes the objective and is not pooled with native KL.
    model, dataset = "Qwen/Qwen2.5-3B-Instruct", "gsm8k"
    template = next(r for r in templates if r["dataset"] == dataset)
    for seed in (321, 322, 323):
        block = f"adapt-nokl-q3-gsm8k-s{seed}"
        conditions = [(rank, offload) for rank in (0, 64) for offload in ("True", "False")]
        rng.shuffle(conditions)
        for period, (rank, offload) in enumerate(conditions, 1):
            condition = f"rank{rank}-off{int(offload == 'True')}-r65-kl0"
            rows.append(adaptation_row(template, model, dataset, seed, block_index,
                                       block, period, condition, rank, offload, "0.65", "False"))
        block_index += 1
    return rows


def adaptation_row(template, model, dataset, seed, index, block, period, condition,
                   rank, offload, reservation, kl):
    row = dict(template)
    row.update(experiment_id=f"{block}-{condition}", run_group="electronics_adaptation_20261016",
               study_stage="confirmatory", revision_study="electronics_adaptation",
               model=model, model_family="qwen25", model_revision=REVISIONS[model],
               block_id=block, pair_id=block, block_index=str(index), period=str(period),
               condition=condition, configuration_id=f"{model}-{dataset}-{condition}",
               configuration_level="independent-adaptation", case_id=f"{model}-{dataset}-kl{kl}",
               sequence_id=str(index), training_seed=str(seed), lora_rank=str(rank),
               lora_alpha="32", use_kl_loss=kl, actor_micro_batch="4",
               vllm_gpu_memory_utilization=reservation, parameter_offload=offload,
               optimizer_offload="False", total_training_steps="5", train_max_samples="160",
               capture_response_lengths="True", allocator_trace_enabled="False",
               allocator_trace_sync="0", per_process_timeout_seconds="900",
               novelty="fresh native-adaptation comparison" if kl == "True" else "separate no-reference control")
    return row


def prepare_adaptation(directory):
    rows = adaptation_rows()
    freeze(directory, rows, study="lora_and_full_parameter_training",
           caps=dict(wall_seconds=10800, payload_timeout_seconds=900,
                     max_concurrent_allocations=4))


def load(directory, expected_hash):
    require(re.fullmatch("[a-f0-9]{64}", expected_hash or ""), "Invalid protocol anchor")
    require(digest(directory / "protocol.json") == expected_hash, "Protocol changed")
    protocol = json.loads((directory / "protocol.json").read_text())
    require(digest(directory / "matrix.csv") == protocol["matrix_sha256"], "Matrix changed")
    rows = read_rows(directory / "matrix.csv")
    require(len(rows) == protocol["planned_invocations"], "Changed slot inventory")
    if "predictions_sha256" in protocol:
        require(digest(directory / "predictions.json") == protocol["predictions_sha256"],
                "Frozen predictions changed")
    require(len({r["experiment_id"] for r in rows}) == len(rows), "Duplicate invocation")
    for row in rows:
        require(re.fullmatch("[a-z0-9][a-z0-9_-]*", row["experiment_id"]), "Unsafe slot")
        require(re.fullmatch("[a-z0-9][a-z0-9_-]*", row["block_id"]), "Unsafe block")
        require(row["run_group"] == protocol["group"], "Group mismatch")
        require(int(row["lora_rank"]) in (0, 64) and row["use_kl_loss"] in ("True", "False"),
                "Undeclared adaptation/reference condition")
    return protocol, rows


def payload_command():
    path = SOURCE / "run_grpo_instrumented.slurm"
    script = path.read_text()
    old = ('    [[ "$RLVRAM_PERIOD" =~ ^[12]$ ]]\n'
           '    PORT_SEED=$((SLURM_JOB_ID * 3 + RLVRAM_PERIOD))')
    new = ('    [[ "$RLVRAM_PERIOD" =~ ^[1-9][0-9]?$ ]]\n'
           '    PORT_SEED=$((SLURM_JOB_ID * 100 + RLVRAM_PERIOD))')
    require(script.count(old) == 1, "Unexpected legacy period selector")
    return ["bash", "-c", script.replace(old, new, 1), str(path)]


def parameter_evidence(attempt, row):
    records = [json.loads(p.read_text()) for p in (attempt / "parameter-counts").glob("*.json")]
    actors = [r for r in records if r["training_role"]]
    require(len(actors) == 1, "Expected one rank-zero actor parameter record")
    actor = actors[0]
    require(actor["lora_rank"] == int(row["lora_rank"]), "Actual adaptation differs")
    total, trainable = actor["total_elements"], actor["requires_grad_elements"]
    require(0 < trainable <= total, "Invalid trainable parameter inventory")
    if int(row["lora_rank"]) == 0:
        require(trainable == total and actor.get("adapter_elements", 0) == 0,
                "Rank-zero run did not train all parameters")
    else:
        require(trainable == actor["adapter_elements"] and trainable < total,
                "LoRA run trainable parameters are not adapters only")
    references = [r for r in records if not r["training_role"]]
    require(len(references) == int(row["lora_rank"] == "0" and row["use_kl_loss"] == "True"),
            "Actual reference model layout differs")
    return dict(parameter_counts=actor, separate_reference_models=len(references))


def applied_actor_configuration(text):
    """Read the actual printed Hydra actor section, not launcher intentions.

The legacy trial JSON does not contain the new adaptation flags. Ray can
interleave stderr later in the printed configuration, so parse its complete
actor section ending at the next top-level algorithm key.
"""
    lines = re.sub(r"\x1b\[[0-9;]*m", "", text).splitlines()
    pid, parts = None, []
    for line in lines:
        match = re.match(r"^\(TaskRunnerV1 pid=(\d+)\) (.*)$", line)
        if not match:
            continue
        content = match[2]
        if pid is None and content.startswith("{'actor_rollout_ref':"):
            pid = int(match[1])
        if pid != int(match[1]):
            continue
        # Ray may strip leading spaces or split the next dictionary key
        # across records. The preceding actor dictionary must still parse
        # completely; no fields are inferred from the planned configuration.
        if content.lstrip().startswith("'algorithm':") or content.strip() == "'algorithm'":
            try:
                parsed = ast.literal_eval("\n".join(parts).rstrip().rstrip(",")+"}")
            except (SyntaxError, ValueError) as error:
                raise ValueError("Printed actor configuration is incomplete") from error
            return pid, parsed["actor_rollout_ref"]
        parts.append(content)
    raise ValueError("Missing printed actor configuration")


def assess(attempt, row, execution, allocation, protocol, source_commit, base):
    """CPU-only raw assessment, reusable after relocation to an extracted archive."""
    result = dict(status="unresolved", failure_kind=None, completed_peak_mib=None,
                  observed_peak_mib=None, validation_error=None)
    job = str(allocation["job_id"])
    invocation = row["experiment_id"] + "-j" + job
    expected = [allocation["gpus"][i]["uuid"] for i in base.subset_ordinals(row)]
    try:
        require(execution["spawned"] and execution["sampling_valid"]
                and not execution["contamination_detected"] and not execution["error"]
                and execution["cleanup"]["complete"] and not execution["cleanup"]["remaining_pids"],
                "Invalid process/measurement/cleanup evidence")
        require(not execution["timed_out"] and not execution.get("interrupted"),
                "Timed out or interrupted; outcome unresolved")
        code = execution["wrapper_exit_code"]
        text = (attempt / f"training-{job}.log").read_text(errors="replace")
        require(text.strip(), "Missing training console")
        config_pid, actual = applied_actor_configuration(text)
        require(actual["model"]["lora_rank"] == int(row["lora_rank"])
                and actual["model"]["lora_alpha"] == int(row["lora_alpha"])
                and actual["actor"]["use_kl_loss"] is (row["use_kl_loss"] == "True"),
                "Applied adaptation/KL settings differ from frozen matrix")
        result["applied_adaptation"] = dict(
            lora_rank=actual["model"]["lora_rank"], lora_alpha=actual["model"]["lora_alpha"],
            use_kl_loss=actual["actor"]["use_kl_loss"], configuration_pid=config_pid)
        trial = json.loads((attempt / f"trial-{job}.json").read_text())
        require(all(normalized(trial.get(k)) == normalized(row.get(k)) for k in MATCH_FIELDS),
                "Observed trial differs from frozen settings")
        require(trial["exit_code"] == code and str(trial["job_id"]) == job, "Exit/job mismatch")
        env = json.loads((attempt / f"environment-{job}.json").read_text())
        require(env["repository"]["head"] == source_commit and not env["repository"]["dirty"],
                "Unfrozen executing source")
        require(env["container"]["sha256"] == CONTAINER_SHA256, "Container differs from the documented stack")
        require(env["software"]["packages"]["torch"] == "2.10.0+cu129"
                and env["software"]["packages"]["vllm"] == "0.18.0",
                "Core runtime versions differ from the documented stack")
        require(env["model"]["resolved_revision"] == row["model_revision"]
                and env["model"]["requested_model"] == row["model"], "Model identity differs")
        for split in ("train.parquet", "test.parquet"):
            relative = str(Path(row["data_dir"]).relative_to(ROOT) / split)
            require(env["dataset"][split]["sha256"] == protocol["data_sha256"][relative],
                    "Dataset changed")
        result.update(base.validate_identity(attempt / "device-evidence", expected, invocation,
                                             success=code == 0))
        result.update(base.validate_trace(attempt, expected, invocation, job))
        result["observed_peak_mib"] = result["task_peak_memory_mib"]
        require(abs(float(trial["peak_gpu_memory_mib"]) - result["observed_peak_mib"]) < .01,
                "Reported and raw peaks differ")
        phases, durations = {}, {}
        previous = None
        steps = int(row["total_training_steps"])
        memory_rows = list(csv.reader((attempt / f"gpu-memory-{job}.csv").open()))
        phase_rows = list(csv.reader((attempt / f"phase-memory-{job}.csv").open()))
        require(len(memory_rows) == len(phase_rows) and bool(phase_rows), "Phase/raw sample inventory differs")
        for sample, memory_sample in zip(phase_rows, memory_rows):
            require(len(sample) == len(expected) + 3 and 0 <= int(sample[1]) <= steps,
                    "Phase trace shape/horizon differs")
            require(sample[:1] + sample[3:] == memory_sample, "Phase/raw observations differ")
            require(previous is None or int(sample[0]) > int(previous[0]), "Nonmonotonic phase clock")
            value = max(map(float, sample[3:]))
            require(math.isfinite(value) and 0 <= value <= 40960, "Invalid memory observation")
            phases[sample[2]] = max(phases.get(sample[2], 0), value)
            if previous is not None and sample[2] == previous[2]:
                durations[sample[2]] = durations.get(sample[2], 0) + (int(sample[0])-int(previous[0]))/1e9
            previous = sample
        result.update(stage_peaks_mib=phases, stage_sampled_seconds=durations,
                      elapsed_seconds=trial["elapsed_seconds"])
        kind = classify_failure_detailed(text, code)
        if code != 0:
            require(kind in MEMORY_FAILURES, "Non-memory or undiagnosed failure: " + str(kind))
            require(has_gpu_memory_evidence(text), "No explicit GPU-memory diagnostic")
            result.update(status="memory_failure", failure_kind=kind)
        else:
            terminal = json.loads((attempt / "device-evidence/trainer-terminal.json").read_text())
            trainer = json.loads((attempt / "device-evidence/trainer-exec.json").read_text())
            require(terminal["pid"] == trainer["pid"] and terminal["timestamp_ns"] > trainer["timestamp_ns"],
                    "Successful exit lacks trainer identity/lifetime")
            result.update(validate_completed_step(
                attempt / "trainer-metrics", invocation, requested_steps=steps,
                earliest_ns=trainer["timestamp_ns"], latest_ns=terminal["timestamp_ns"]))
            require(result["logger_pid"] == config_pid, "Configuration and completed-work producer differ")
            result.update(parameter_evidence(attempt, row))
            for phase, setting in [("checkpoint", "save_freq"), ("validation", "test_freq")]:
                if int(row[setting]) > 0:
                    require(phase in phases, "Required lifecycle phase missing: " + phase)
            result.update(status="within_margin" if result["observed_peak_mib"] <= 38912 else "above_margin",
                          completed_peak_mib=result["observed_peak_mib"])
    except (ValueError, KeyError, OSError, TypeError, IndexError) as error:
        result.update(status="unresolved", validation_error=f"{type(error).__name__}: {error}")
    return result


def run(args):
    directory = SOURCE / DIRECTORY / args.panel
    protocol, rows = load(directory, args.protocol_sha256)
    selected = [r for r in rows if int(r["block_index"]) == args.index]
    require(selected, "Unknown block index")
    require(len({r["block_id"] for r in selected}) == 1, "Multiple block identities")
    selected.sort(key=lambda r: int(r["period"]))
    require([int(r["period"]) for r in selected] == list(range(1, len(selected)+1)),
            "Missing/duplicate periods")
    group = ROOT / "output" / protocol["group"]
    block_dir = group / "blocks" / selected[0]["block_id"]
    base = runtime(protocol["caps"])
    ledger = base.claim_pair(block_dir)
    trace = Journal(block_dir / "allocation-trace.jsonl")
    interrupt = InterruptState()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupt.signal)
    allocation, results, fatal = None, [], None
    try:
        head = clean_source(SOURCE, args.source_commit)
        require(os.environ.get("SLURM_RESTART_COUNT", "0") == "0", "Requeued job forbidden")
        allocation = base.allocation_context()
        base.validate_allocation_inventory(allocation["gpus"])
        require(os.environ.get("SLURM_ARRAY_TASK_ID") == str(args.index), "Array slot differs")
        ledger.emit("block_claimed", source_commit=head, protocol_sha256=args.protocol_sha256,
                    block_id=selected[0]["block_id"], rows=selected, allocation=allocation)
        write_new_json(block_dir / "block.json", dict(source_commit=head, protocol=protocol,
                       protocol_sha256=args.protocol_sha256, allocation=allocation, rows=selected))
        for name, expected in protocol["data_sha256"].items():
            require(digest(ROOT / name) == expected, "Frozen dataset changed")
        cache = protocol["model_cache"][selected[0]["model"]]
        snapshot = validate_snapshot(Path(cache["path"]))
        for name, expected in cache["files_sha256"].items():
            require(digest(snapshot / name) == expected, "Frozen model changed: " + name)
        allocated = [g["uuid"] for g in allocation["gpus"]]
        deadline = time.monotonic() + allocation["ends_at_epoch"] - time.time() - 10
        enable_subreaper()
        for row in selected:
            period = int(row["period"])
            result = dict(experiment_id=row["experiment_id"], status="not_launched", launched=False)
            attempt = None
            try:
                require(fatal is None and not interrupt.requested, fatal or "Interrupted")
                base.PAYLOAD_TIMEOUT = int(row["per_process_timeout_seconds"])
                require(deadline-time.monotonic() >= base.PAYLOAD_TIMEOUT + 60, "Insufficient remaining allocation")
                require(interrupt.call(base.wait_idle, allocated, trace, period, "pre_idle", deadline),
                        "Allocation is not idle")
                expected = [allocated[i] for i in base.subset_ordinals(row)]
                job = allocation["job_id"]
                invocation = row["experiment_id"] + "-j" + job
                attempt = group / row["experiment_id"] / ("attempt-" + job)
                require(not attempt.parent.exists(), "An invocation may not be replaced")
                attempt.mkdir(parents=True)
                values = environment(directory/"matrix.csv", rows.index(row),
                                     expected_gpu_count=int(row["gpu_count"]))
                values.update(BENCHMARK_SOURCE_ROOT=str(SOURCE), BENCHMARK_OUTPUT_ROOT=str(ROOT/"output"),
                    RUN_NAME=f"{protocol['group']}/{row['experiment_id']}/attempt-{job}",
                    PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(SOURCE),
                    RLVRAM_GPU_UUIDS=",".join(expected), RLVRAM_ALLOCATION_GPU_UUIDS=",".join(allocated),
                    RLVRAM_INVOCATION_ID=invocation, RLVRAM_PERIOD=str(period),
                    RLVRAM_CONTRACT_ALLOCATION_GPUS="4", RLVRAM_MODEL_SNAPSHOT=str(snapshot),
                    RLVRAM_DURABLE_METRICS_DIR=str(attempt/"trainer-metrics"),
                    CUSTOM_REWARD_FUNCTION_PATH=str(SOURCE/"memory_tuner/code_shape_reward.py")
                    if row["dataset"].startswith("code") else "null",
                    FATAL_WATCHDOG_GRACE_SECONDS="60")
                write_new_json(attempt/"dispatch.json", dict(row=row, environment=values,
                    protocol_sha256=args.protocol_sha256, source_commit=head, invocation_id=invocation))
                ledger.emit("dispatched", period=period, experiment_id=row["experiment_id"], invocation_id=invocation)
                started_ns = time.time_ns()
                with HostMemorySampler(attempt, job):
                    execution = interrupt.payload(base.run_payload, payload_command(), base.clean_environment(values),
                        attempt/"launcher.log", allocated, expected, ledger, trace, period, deadline)
                execution.update(started_ns=started_ns, finished_ns=time.time_ns())
                write_new_json(attempt/"execution.json", execution)
                result.update(launched=execution["spawned"], attempt_path=str(attempt),
                              invocation_gpu_seconds=execution["invocation_elapsed_seconds"]*int(row["gpu_count"]))
                clean = interrupt.call(base.wait_idle, allocated, trace, period, "post_idle", deadline,
                                       allow_pending=True)
                result.update(assess(attempt, row, execution, allocation, protocol, head, base))
                if not clean or not execution["cleanup"]["complete"]:
                    fatal = "Post-invocation cleanup/idle gate failed"
                    result.update(status="unresolved", validation_error=fatal)
            except (Exception, KeyboardInterrupt) as error:
                fatal = f"{type(error).__name__}: {error}"
                result.update(reason=fatal)
            finally:
                result["evidence_files"] = inventory(attempt) if attempt else {}
                write_new_json(block_dir/f"period-{period}.json", result)
                ledger.emit("terminal", period=period, **result)
                results.append(result)
        write_new_json(block_dir/"summary.json", dict(results=results, fatal=fatal))
    except (Exception, KeyboardInterrupt) as error:
        write_new_json(block_dir/"setup-error.json", dict(error=f"{type(error).__name__}: {error}",
                       source_commit=args.source_commit, allocation=allocation))
        raise
    finally:
        ledger.close()
        trace.close()
    return int(any(r["status"] not in ("within_margin", "above_margin", "memory_failure") for r in results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-pilot")
    prepare.add_argument("--directory", type=Path, default=SOURCE/DIRECTORY/"pilot")
    adaptation = sub.add_parser("prepare-adaptation")
    adaptation.add_argument("--directory", type=Path, default=SOURCE/DIRECTORY/"adaptation")
    execute = sub.add_parser("run")
    execute.add_argument("--panel", required=True)
    execute.add_argument("--protocol-sha256", required=True)
    execute.add_argument("--source-commit", required=True)
    execute.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    if args.command == "prepare-pilot":
        prepare_pilot(args.directory)
        return 0
    if args.command == "prepare-adaptation":
        prepare_adaptation(args.directory)
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
