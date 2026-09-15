"""Validate the frozen admission panel and reconstruct its independent replay."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile

import decision_benchmark as db
from memory_tuner.artifact_paths import ORIGINAL_ROOT
from memory_tuner.grpo_raw_evidence import EvidenceValidationError, select_attempt
from memory_tuner.log_parser import classify_failure_detailed
from memory_tuner.review_attempt_workload_audit import extract_workload_log
from memory_tuner.review_evidence_analysis import clean_json, trace_summary
from memory_tuner.scientific_revision_analysis import validate_allocation

MEMORY_KINDS = {"cuda_oom", "rollout_init_memory", "weight_sync_oom"}


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, empty_fields):
    fields = list(dict.fromkeys(k for row in rows for k in row)) or empty_fields
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def collect(root, protocol_dir, output):
    if output.exists():
        raise FileExistsError(f"Use a new output directory; preserving existing results: {output}")
    protocol = json.loads((protocol_dir / "protocol.json").read_text())
    matrix = protocol_dir / "matrix.csv"
    if hashlib.sha256(matrix.read_bytes()).hexdigest() != protocol["matrix_sha256"]:
        raise ValueError("Matrix differs from pre-outcome freeze")
    specs = read_csv(matrix)
    if len(specs) != protocol["planned_processes"]:
        raise ValueError("Wrong number of planned slots")
    repairs_path = protocol_dir / "repairs.json"
    repairs = json.loads(repairs_path.read_text()) if repairs_path.is_file() else {}
    attempts, validated, phases, work, allocation_rows, unavailable = [], [], [], [], [], []
    for spec in specs:
        slot = spec["experiment_id"]
        directory = root / "output/prospective_decision"
        manifests = sorted((directory / "pairs" / slot).glob("pair-*.json"),
                           key=lambda p: int(p.stem.split("-")[-1]))
        jobs = {str(json.loads(p.read_text())["job_id"]) for p in manifests}
        slot_directory = directory / slot
        for attempt_directory in slot_directory.glob("attempt-*"):
            job = attempt_directory.name.removeprefix("attempt-")
            if job not in jobs and any(attempt_directory.iterdir()):
                raise RuntimeError(f"{slot}: orphan payload evidence for job {job}; reconcile invocation ledger")
        for trial_link in slot_directory.glob("trial-*.json"):
            if trial_link.stem.removeprefix("trial-") not in jobs:
                raise RuntimeError(f"{slot}: trial without allocation record; reconcile invocation ledger")
        if not manifests:
            unavailable.append(slot)
            continue
        for index, path in enumerate(manifests, 1):
            manifest = json.loads(path.read_text())
            if manifest["pair_id"] != slot:
                raise ValueError("Allocation slot differs from frozen specification")
            job = str(manifest["job_id"])
            periods = manifest["periods"]
            if not periods:
                attempt_directory = slot_directory / f"attempt-{job}"
                if ((attempt_directory.is_dir() and any(attempt_directory.iterdir()))
                        or (slot_directory / f"trial-{job}.json").exists()):
                    raise RuntimeError(f"{slot}: empty periods contradict payload evidence for job {job}; reconcile invocation ledger")
                allocation_rows.append({"slot_id": slot, "job_id": job,
                                        "status": manifest["pair_status"],
                                        "process_started": False,
                                        "error": manifest.get("error", "")})
                continue
            if len(periods) != 1 or periods[0]["experiment_id"] != slot:
                raise ValueError("Expected one frozen process per allocation")
            period = periods[0]
            if not (math.isfinite(float(period["elapsed_seconds"]))
                    and float(period["elapsed_seconds"]) >= 0):
                raise ValueError(f"{slot}: invalid recorded invocation duration")
            trial = root / period["trial"]
            log_path = trial.parent / f"training-{job}.log"
            log = log_path.read_text(errors="replace") if log_path.is_file() else ""
            failure = classify_failure_detailed(log, int(period["wrapper_exit_code"]))
            known_oom = failure in MEMORY_KINDS or bool(re.search(
                r"CUDA (?:error: )?out of memory|torch\.(?:cuda\.)?OutOfMemoryError", log))
            attempt_id = f"{slot}-job{job}"
            prior = [a for a in attempts if a["slot_id"] == slot]
            attempt = {
                "attempt_id": attempt_id, "attempt_index": len(prior) + 1,
                "slot_id": slot, "candidate_id": spec["candidate_id"],
                "case_id": spec["case_id"], "role": spec["seed_role"],
                "training_seed": int(spec["training_seed"]), "job_id": job,
                "state": "unresolved", "known_memory_failure": known_oom,
                "peak_mib": None, "completed_final_operations": False,
                "wrapper_exit_code": int(period["wrapper_exit_code"]),
                "elapsed_seconds": period["elapsed_seconds"],
                "validation_error": "", "repair_reason": repairs.get(attempt_id, {}).get("reason", ""),
                "allocation_manifest": str(path.relative_to(root)),
                "attempt_directory": str(trial.parent.relative_to(root)),
                "source_head": manifest["source_head"],
                "last_completed_step": max(
                    (int(n) for n in re.findall(r"training/global_step:(\d+)", log)), default=0),
            }
            allocation_rows.append({"slot_id": slot, "job_id": job,
                                    "status": manifest["pair_status"],
                                    "process_started": True,
                                    "error": manifest.get("error", "")})
            other_jobs = {str(json.loads(p.read_text())["job_id"])
                          for p in manifests if p != path}
            try:
                if manifest["pair_status"] != "complete":
                    raise EvidenceValidationError("Allocation did not complete its validation")
                row = select_attempt(root, spec, exclusions=other_jobs,
                                     annotations={}, trial_path=trial)
                if str(row["job_id"]) != job or Path(row["artifact_path"]).resolve() != trial.resolve():
                    raise EvidenceValidationError("Selected trial does not belong to the current allocation")
                provenance = validate_allocation(root, [row])
                if not provenance["source_head"].startswith("1271267"):
                    raise EvidenceValidationError("Execution source differs from frozen checkout")
            except (EvidenceValidationError, json.JSONDecodeError, FileNotFoundError) as error:
                message = str(error)
                for prefix in sorted({str(root), str(ORIGINAL_ROOT)}, key=len, reverse=True):
                    message = message.replace(prefix, "<ARTIFACT_ROOT>")
                attempt["validation_error"] = f"{type(error).__name__}: {message}"
            else:
                # Unexpected parser or schema defects must fail collection.
                # Nothing is committed until all mandatory derivations pass.
                state = ("within_margin" if row["safe"] else
                         "above_margin" if row["success"] else "memory_failure")
                peak_by_phase, duration = trace_summary(row["phase_memory_csv"])
                attempt_phases = []
                for phase, peak in peak_by_phase.items():
                    attempt_phases.append({"attempt_id": attempt_id, "slot_id": slot,
                                   "candidate_id": spec["candidate_id"], "role": spec["seed_role"],
                                   "phase": phase, "observed_peak_mib": peak,
                                   "sampled_duration_seconds": duration.get(phase),
                                   "completed_process": row["success"],
                                   "failure_kind": row["failure_kind"]})
                attempt_work = {"attempt_id": attempt_id, "slot_id": slot,
                             "candidate_id": spec["candidate_id"], "role": spec["seed_role"],
                             **extract_workload_log(log, row)}
                # Store portable evidence paths. Never include model/checkpoint
                # payloads or user credentials in the benchmark tables.
                for field in ("artifact_path", "phase_memory_csv", "gpu_telemetry_csv",
                              "environment_json", "run_log", "allocator_trace_dir"):
                    if row.get(field):
                        row[field] = str(Path(row[field]).relative_to(root))
                validated.append(row)
                phases.extend(attempt_phases)
                work.append(attempt_work)
                attempt.update(state=state, peak_mib=float(row["peak_gpu_memory_mib"])
                               if row["success"] else None,
                               known_memory_failure=state == "memory_failure",
                               completed_final_operations=bool(row["success"]))
            attempts.append(attempt)
    if unavailable:
        raise ValueError(f"{len(unavailable)} slots lack finalized allocation records; collection still incomplete")
    # Validation precedes publication; incomplete resource outcomes remain
    # explicit and are analyzed with the frozen sensitivity rules.
    scores = db.replay(protocol, attempts)
    candidate_rows = []
    for case in protocol["cases"]:
        for candidate in case["candidates"]:
            cid = candidate["candidate_id"]
            truth = scores["candidate_evaluation"][cid]
            screen_records = [a for a in attempts if a["candidate_id"] == cid and a["role"] == "screen"]
            eval_records = [a for a in attempts if a["candidate_id"] == cid and a["role"] == "evaluation"]
            specification = next(s for s in specs if s["candidate_id"] == cid)
            candidate_rows.append({
                "model_name": {"qwen25_3b": "Qwen2.5-3B-Instruct",
                               "phi4_mini": "Phi-4-mini-instruct"}[case["model_family"]],
                "workload_name": {"gsm8k": "GSM8K", "math": "MATH"}[case["dataset"]],
                "configuration_level": candidate["configuration_level"],
                "evaluation_state": truth["state"],
                "screen_state": screen_records[-1]["state"] if screen_records else "not_started",
                "source_state": case["donors"][cid]["observed_state"],
                "actor_micro_batch": candidate["actor_micro_batch"],
                "reservation": candidate["reservation"],
                "gpu_count": 2, "training_steps": 5,
                "prompt_cap": int(specification["max_prompt_length"]),
                "response_cap": int(specification["max_response_length"]),
                "responses_per_prompt": int(specification["rollout_n"]),
                "global_prompt_batch": int(specification["train_batch_size"]),
                "evaluation_known_memory_failure": truth["known_memory_failure"],
                "evaluation_known_above_margin": truth["known_above_margin"],
                "screen_peak_mib": screen_records[-1]["peak_mib"] if screen_records else None,
                "evaluation_max_completed_peak_mib": max(
                    (a["peak_mib"] for a in eval_records if a["peak_mib"] is not None), default=None),
                "recorded_screen_attempts": len(screen_records),
                "recorded_evaluation_attempts": len(eval_records),
                "candidate_id": cid, "case_id": case["case_id"],
                "model_family": case["model_family"], "dataset": case["dataset"]})
    inventory = {
        "planned_slots": len(specs), "recorded_attempts": len(attempts),
        "validated_memory_outcomes": len(validated),
        "attempt_states": dict(Counter(a["state"] for a in attempts)),
        "hidden_evaluation_states": dict(Counter(
            t["state"] for t in scores["candidate_evaluation"].values())),
        "source_heads": sorted({a["source_head"] for a in attempts}),
        "physical_recorded_gpu_seconds": sum(a["elapsed_seconds"] * 2 for a in attempts),
        "gpu_seconds_scope": "Runner invocation elapsed times times allocated GPUs; not queue time or total allocation wall time.",
        "matrix_sha256": protocol["matrix_sha256"],
        "protocol_sha256": hashlib.sha256((protocol_dir / "protocol.json").read_bytes()).hexdigest(),
        "slot_inventory": [
            {"slot_id": spec["experiment_id"],
             "recorded_attempts": sum(a["slot_id"] == spec["experiment_id"] for a in attempts),
             "allocation_records": sum(a["slot_id"] == spec["experiment_id"] for a in allocation_rows),
             "status": "has_invocations" if any(a["slot_id"] == spec["experiment_id"] for a in attempts)
                       else "not_started"}
            for spec in specs],
    }
    json_outputs = {"attempts.json": attempts, "scores.json": scores,
                    "collection_summary.json": inventory}
    csv_outputs = {
        "attempts.csv": (attempts, ["attempt_id", "slot_id", "state"]),
        "validated_runs.csv": (validated, list(specs[0]) + ["job_id", "success", "safe"]),
        "stage_measurements.csv": (phases, ["attempt_id", "slot_id", "candidate_id", "role",
                                          "phase", "observed_peak_mib", "sampled_duration_seconds",
                                          "completed_process", "failure_kind"]),
        "realized_work.csv": (work, ["attempt_id", "slot_id", "candidate_id", "role"]),
        "allocations.csv": (allocation_rows, ["slot_id", "job_id", "status", "process_started", "error"]),
        "summary.csv": (scores["summary"], ["rule", "budget_target_attempts"]),
        "candidate_results.csv": (candidate_rows, ["candidate_id", "evaluation_state"]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staged = Path(temporary) / "complete"
        staged.mkdir()
        for name, value in json_outputs.items():
            db.save(staged / name, clean_json(value))
        for name, (rows, empty_fields) in csv_outputs.items():
            write_csv(staged / name, clean_json(rows), empty_fields)
        names = sorted(set(json_outputs) | set(csv_outputs))
        db.save(staged / "manifest.json", {
            "algorithm": "sha256",
            "files": {name: hashlib.sha256((staged / name).read_bytes()).hexdigest()
                      for name in names}})
        # Atomic publication of one complete allowlisted set. Existing results
        # are never reused, cleared, or mixed with a new reconstruction.
        if output.exists():
            raise FileExistsError(f"Output appeared during collection: {output}")
        staged.rename(output)
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.root.resolve(), args.protocol_dir.resolve(),
                             args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
