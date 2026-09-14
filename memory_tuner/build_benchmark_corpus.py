#!/usr/bin/env python3
"""Build the canonical RLVRAMBench corpus and validity manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

from memory_tuner.log_parser import classify_failure_detailed
from memory_tuner.screening_analysis import phase_peaks


DEFAULT_FAILURE_ANNOTATIONS = Path("memory_tuner/failure_annotations.csv")
DEFAULT_ATTEMPT_EXCLUSIONS = Path("memory_tuner/attempt_exclusions.csv")
DEFAULT_METHOD_ARTIFACTS = Path(
    "profiles/benchmark/prospective-method-artifacts.csv"
)
DEFAULT_METHOD_CORPUS = Path(
    "profiles/benchmark/prospective-method-corpus.csv"
)
DEFAULT_METHOD_CORPUS_DIGEST = Path(
    "profiles/benchmark/prospective-method-corpus.sha256"
)
DEFAULT_METHOD_CORPUS_DRIFT = Path(
    "profiles/benchmark/prospective-method-corpus-derivation-drift.csv"
)

VALID_ROOTS = (
    "phase_screening",
    "phase_confirmatory",
    "phase_mitigation",
    "math_transfer",
    "instrumentation_smoke",
    "benchmark_3b",
    "benchmark_7b_4gpu",
    "grpo_smoke",
    "benchmark_code",
    "benchmark_code_expansion",
    "benchmark_grpo_math_1p5b",
    "benchmark_grpo_3b_gsm",
    "benchmark_shape_short",
    "benchmark_shape_short_3b",
    "benchmark_repetitions",
    "benchmark_sleep_ablation",
    "benchmark_calibration_wave2_oracle_2gpu",
    "benchmark_calibration_wave2_grpo_2gpu",
    "benchmark_calibration_wave2_oracle_4gpu",
    "benchmark_case_expansion_v3_grpo_2gpu",
)

MODEL_METADATA = {
    "Qwen/Qwen2.5-1.5B-Instruct": ("qwen25_1p5b", 1.5),
    "Qwen/Qwen2.5-3B-Instruct": ("qwen25_3b", 3.0),
    "Qwen/Qwen2.5-7B-Instruct": ("qwen25_7b", 7.0),
}

INVALID_REASON_PATTERNS = (
    (
        "qgpu-exp-gpu-visibility",
        "scheduler_gpu_visibility_infrastructure_failure",
    ),
    ("hydra_crlf", "hydra_crlf_infrastructure_failure"),
    ("missing-transferqueue", "missing_transferqueue_dependency"),
    ("missing-v1-phase-markers", "missing_v1_phase_instrumentation"),
    ("math-empty-dataloader", "math_prompt_limit_empty_dataloader"),
    ("7b-unbounded-model-len", "unbounded_model_context_initialization_failure"),
)

NON_SCIENTIFIC_FAILURE_REASONS = {
    "instrumentation_missing_phase_hook": "missing_phase_hook_instrumentation_failure",
    "model_cache_vocabulary_load": "model_cache_vocabulary_load_failure",
    "hf_rate_limit": "huggingface_rate_limit_infrastructure_failure",
    "empty_dataloader": "empty_dataloader_configuration_failure",
    "distributed_port_collision": "distributed_port_collision_infrastructure_failure",
    "insufficient_allocated_gpus": "scheduler_gpu_visibility_infrastructure_failure",
    "launcher_mutation_race": "launcher_mutation_infrastructure_failure",
    "configuration": "configuration_failure",
    "environment": "environment_failure",
    "runtime": "runtime_failure_unclassified",
}

SCIENTIFIC_FAILURE_KINDS = {
    "cuda_oom",
    "rollout_init_memory",
    "weight_sync_oom",
}


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_method_artifacts(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"missing prospective method-artifact lock: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "artifact_path",
        "job_id",
        "experiment_id",
        "source_root",
        "sha256",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(
            f"invalid prospective method-artifact lock columns: {path}"
        )
    paths = [row["artifact_path"] for row in rows]
    job_ids = [row["job_id"] for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError(f"duplicate artifact path in {path}")
    if len(job_ids) != len(set(job_ids)):
        raise ValueError(f"duplicate job id in {path}")
    return rows


def select_locked_method_rows(
    valid_rows: list[dict],
    locked_artifacts: list[dict[str, str]],
    *,
    repo_root: Path = Path("."),
) -> list[dict]:
    """Select and verify the exact pre-repetition method corpus."""
    by_artifact = {
        str(row["artifact_path"]): row
        for row in valid_rows
    }
    selected = []
    errors = []
    for locked in locked_artifacts:
        artifact = str(locked["artifact_path"])
        row = by_artifact.get(artifact)
        if row is None:
            errors.append(f"locked artifact is not scientifically valid: {artifact}")
            continue
        for field in ("job_id", "experiment_id", "source_root"):
            if str(row.get(field, "")) != str(locked[field]):
                errors.append(
                    f"{artifact}: {field} mismatch "
                    f"({row.get(field)!r} != {locked[field]!r})"
                )
        path = Path(artifact)
        if not path.is_absolute():
            path = repo_root / path
        if not path.is_file():
            errors.append(f"locked artifact is missing: {path}")
        elif file_sha256(path) != locked["sha256"]:
            errors.append(f"locked artifact SHA-256 mismatch: {path}")
        selected.append(row)
    if errors:
        raise ValueError("\n".join(errors))
    return selected


def infer_metadata(experiment_id: str, trial: dict) -> dict:
    model = trial.get("model")
    if not model:
        if "1p5b" in experiment_id:
            model = "Qwen/Qwen2.5-1.5B-Instruct"
        elif "7b" in experiment_id:
            model = "Qwen/Qwen2.5-7B-Instruct"
        elif "3b" in experiment_id:
            model = "Qwen/Qwen2.5-3B-Instruct"
    model_tag, parameters = MODEL_METADATA.get(model, ("unknown", float("nan")))
    data_dir = trial.get("data_dir", "")
    dataset_hint = f"{data_dir} {experiment_id}".lower()
    if "codecontests" in dataset_hint or "code-" in dataset_hint:
        dataset = (
            "code_heavy_tail"
            if "heavy_tail" in dataset_hint
            else "code_standard"
        )
    elif "math" in dataset_hint:
        dataset = "math"
    else:
        dataset = "gsm8k"
    algorithm = trial.get("algorithm", "oracle_sppo")
    gpu_count = int(trial.get("gpu_count", 4 if "4gpu" in experiment_id else 2))
    return {
        "model": model or "unknown",
        "model_tag": model_tag,
        "model_parameters_b": parameters,
        "dataset": dataset,
        "algorithm": algorithm,
        "gpu_count": gpu_count,
        "case_id": f"{algorithm}|{model_tag}|{dataset}|{gpu_count}gpu",
        "parameter_offload": trial.get(
            "parameter_offload", "True" if "-po" in experiment_id else "False"
        ),
        "optimizer_offload": trial.get("optimizer_offload", "False"),
        "free_cache_engine": trial.get("free_cache_engine", "True"),
    }


def invalid_reason(path: Path) -> str:
    lowered = "/".join(path.parts).lower()
    for pattern, reason in INVALID_REASON_PATTERNS:
        if pattern in lowered:
            return reason
    return "excluded_infrastructure_or_instrumentation_failure"


def scientific_validity(row: dict) -> tuple[bool, str]:
    """Return whether a trial is a usable scientific benchmark outcome."""
    if int(row.get("success", 0)):
        return True, ""
    failure_kind = str(row.get("failure_kind", "") or "")
    reason = NON_SCIENTIFIC_FAILURE_REASONS.get(failure_kind)
    if reason:
        return False, reason
    if not failure_kind:
        return False, "unclassified_failure_without_evidence"
    if failure_kind not in SCIENTIFIC_FAILURE_KINDS:
        return False, f"non_memory_failure:{failure_kind}"
    return True, ""


def load_failure_annotations(path: Path) -> dict[str, dict[str, str]]:
    """Load explicit classifications for legacy trials lacking embedded logs."""
    if not path.is_file():
        return {}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    annotations = {}
    for row in rows:
        experiment_id = str(row.get("experiment_id", "") or "")
        failure_kind = str(row.get("failure_kind", "") or "")
        if not experiment_id or not failure_kind:
            raise ValueError(f"invalid failure annotation in {path}: {row}")
        if experiment_id in annotations:
            raise ValueError(
                f"duplicate failure annotation for {experiment_id} in {path}"
            )
        annotations[experiment_id] = row
    return annotations


def load_attempt_exclusions(path: Path) -> dict[str, dict[str, str]]:
    """Load result-blind exclusions for redundant or invalid attempts."""
    if not path.is_file():
        return {}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    exclusions = {}
    for row in rows:
        job_id = str(row.get("job_id", "") or "")
        reason = str(row.get("reason", "") or "")
        if not job_id or not reason:
            raise ValueError(f"invalid attempt exclusion in {path}: {row}")
        if job_id in exclusions:
            raise ValueError(f"duplicate attempt exclusion for job {job_id}")
        exclusions[job_id] = row
    return exclusions


def prompt_profile(metadata: dict, trial: dict) -> dict:
    explicit_data_dir = trial.get("data_dir")
    candidates = []
    if explicit_data_dir:
        candidates.append(Path(explicit_data_dir) / "profile.json")
    dataset = metadata["dataset"]
    if dataset == "gsm8k":
        candidates.append(Path("data/gsm8k/profile.json"))
    elif dataset == "math":
        candidates.append(Path("data/math/profile.json"))
    for path in candidates:
        if not path.exists():
            continue
        profile = json.loads(path.read_text())
        combined = profile.get("combined", {})
        return {
            "prompt_p50_tokens": combined.get("prompt_tokens_p50", ""),
            "prompt_p95_tokens": combined.get("prompt_tokens_p95", ""),
            "prompt_p99_tokens": combined.get("prompt_tokens_p99", ""),
            "prompt_max_tokens": combined.get("prompt_tokens_max", ""),
            "dataset_profile_path": str(path),
        }
    return {}


def trial_row(
    trial_path: Path,
    failure_annotations: dict[str, dict[str, str]] | None = None,
) -> dict:
    from memory_tuner.artifact_paths import load_trial_record
    trial = load_trial_record(trial_path)
    experiment_id = trial_path.parent.name
    metadata = infer_metadata(experiment_id, trial)
    row = {
        "experiment_id": experiment_id,
        "source_root": trial_path.parents[1].name,
        "artifact_path": str(trial_path),
        "scientific_valid": 1,
        "validity_reason": "",
        **metadata,
        **prompt_profile(metadata, trial),
        **trial,
        "success": int(int(trial.get("exit_code", 1)) == 0),
    }
    phase_path = Path(trial.get("phase_memory_csv", ""))
    if phase_path.is_file():
        for phase, peak in phase_peaks(phase_path).items():
            row[f"phase_peak_{phase}_mib"] = peak
    row.setdefault("max_prompt_length", 512)
    row.setdefault("max_response_length", 1024)
    row.setdefault("rollout_n", 5)
    row.setdefault("rollout_logprob_micro_batch", 8)
    row.setdefault("ref_logprob_micro_batch", 8)
    row.setdefault("max_model_len", 32_768)
    row.setdefault("max_num_seqs", 256)
    row.setdefault("train_batch_size", 128)
    row.setdefault("rollout_tp_size", 1)
    row["workload_signature"] = (
        f"rollout{row['rollout_n']}|prompt{row['max_prompt_length']}|"
        f"response{row['max_response_length']}|batch{row['train_batch_size']}"
    )
    row["case_id"] = (
        f"{row['algorithm']}|{row['model_tag']}|{row['dataset']}|"
        f"{row['gpu_count']}gpu|{row['workload_signature']}"
    )
    run_log_value = str(row.get("run_log", "") or "")
    if run_log_value and not row.get("failure_kind"):
        run_log = Path(run_log_value)
        if run_log.is_file():
            row["failure_kind"] = classify_failure_detailed(
                run_log.read_text(errors="replace"),
                int(row.get("exit_code", 1)),
            ) or ""
    annotation = (failure_annotations or {}).get(experiment_id)
    if (
        int(row.get("exit_code", 1)) != 0
        and not row.get("failure_kind")
        and annotation
    ):
        row["failure_kind"] = annotation["failure_kind"]
        row["failure_evidence"] = annotation.get("evidence_path", "")
        row["failure_evidence_note"] = annotation.get("evidence_note", "")
    return row


def csv_payload(rows: list[dict]) -> bytes:
    fields = sorted({field for row in rows for field in row})
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8")


def verify_method_corpus_digest(
    rows: list[dict], digest_path: Path
) -> str:
    if not digest_path.is_file():
        raise ValueError(f"missing prospective method-corpus digest: {digest_path}")
    tokens = digest_path.read_text(encoding="utf-8").split()
    if not tokens:
        raise ValueError(f"empty prospective method-corpus digest: {digest_path}")
    expected = tokens[0]
    actual = hashlib.sha256(csv_payload(rows)).hexdigest()
    if actual != expected:
        raise ValueError(
            "prospective method corpus SHA-256 mismatch: "
            f"{actual} != {expected}"
        )
    return actual


def load_locked_method_corpus(
    path: Path,
    digest_path: Path,
) -> list[dict[str, str]]:
    """Load and verify the immutable, pre-repetition method-analysis CSV."""
    if not path.is_file():
        raise ValueError(f"missing prospective method corpus: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty prospective method corpus: {path}")
    verify_method_corpus_digest(rows, digest_path)
    return rows


def audit_locked_method_derivation(
    locked_rows: list[dict],
    derived_rows: list[dict],
) -> list[dict[str, str]]:
    """Permit only subtype-only taxonomy refinements after the corpus lock."""
    key_fields = (
        "artifact_path",
        "job_id",
        "experiment_id",
        "source_root",
    )

    def key(row: dict) -> tuple[str, ...]:
        return tuple(str(row.get(field, "")) for field in key_fields)

    locked_by_key = {key(row): row for row in locked_rows}
    derived_by_key = {key(row): row for row in derived_rows}
    if len(locked_by_key) != len(locked_rows):
        raise ValueError("duplicate identity in prospective method corpus")
    if len(derived_by_key) != len(derived_rows):
        raise ValueError("duplicate identity in derived method corpus")
    if set(locked_by_key) != set(derived_by_key):
        missing = sorted(set(locked_by_key) - set(derived_by_key))
        extra = sorted(set(derived_by_key) - set(locked_by_key))
        raise ValueError(
            "prospective method derivation identity mismatch: "
            f"missing={missing}; extra={extra}"
        )

    drift_rows = []
    errors = []
    for row_key in sorted(locked_by_key):
        locked = locked_by_key[row_key]
        derived = derived_by_key[row_key]
        differing_fields = {
            field
            for field in set(locked) | set(derived)
            if str(locked.get(field, "")) != str(derived.get(field, ""))
        }
        if not differing_fields:
            continue
        if differing_fields != {"failure_kind"}:
            errors.append(
                f"{row_key[0]}: non-taxonomy derivation drift "
                f"{sorted(differing_fields)}"
            )
            continue
        locked_kind = str(locked.get("failure_kind", ""))
        derived_kind = str(derived.get("failure_kind", ""))
        if (
            str(locked.get("success", "")) != "0"
            or str(derived.get("success", "")) != "0"
            or locked_kind not in SCIENTIFIC_FAILURE_KINDS
            or derived_kind not in SCIENTIFIC_FAILURE_KINDS
        ):
            errors.append(
                f"{row_key[0]}: invalid failure-subtype drift "
                f"{locked_kind!r} -> {derived_kind!r}"
            )
            continue
        drift_rows.append(
            {
                "artifact_path": row_key[0],
                "job_id": row_key[1],
                "experiment_id": row_key[2],
                "source_root": row_key[3],
                "locked_failure_kind": locked_kind,
                "current_failure_kind": derived_kind,
                "binary_outcome_unchanged": "1",
                "reason": "post_lock_terminal_stack_taxonomy_refinement",
            }
        )
    if errors:
        raise ValueError("\n".join(errors))
    return drift_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(csv_payload(rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("profiles/benchmark/benchmark-corpus.csv"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("profiles/benchmark/artifact-manifest.csv"),
    )
    parser.add_argument(
        "--method-artifacts",
        type=Path,
        default=DEFAULT_METHOD_ARTIFACTS,
    )
    parser.add_argument(
        "--method-corpus",
        type=Path,
        default=DEFAULT_METHOD_CORPUS,
    )
    parser.add_argument(
        "--method-corpus-digest",
        type=Path,
        default=DEFAULT_METHOD_CORPUS_DIGEST,
    )
    parser.add_argument(
        "--method-corpus-drift",
        type=Path,
        default=DEFAULT_METHOD_CORPUS_DRIFT,
    )
    parser.add_argument(
        "--failure-annotations",
        type=Path,
        default=DEFAULT_FAILURE_ANNOTATIONS,
    )
    parser.add_argument(
        "--attempt-exclusions",
        type=Path,
        default=DEFAULT_ATTEMPT_EXCLUSIONS,
    )
    args = parser.parse_args()

    failure_annotations = load_failure_annotations(args.failure_annotations)
    attempt_exclusions = load_attempt_exclusions(args.attempt_exclusions)
    candidate_rows = []
    for root_name in VALID_ROOTS:
        root = args.output_root / root_name
        for trial_path in sorted(root.glob("*/trial-*.json")):
            candidate_rows.append(trial_row(trial_path, failure_annotations))

    valid_rows = []
    manifest_rows = []
    for row in candidate_rows:
        explicit_exclusion = attempt_exclusions.get(str(row.get("job_id", "")))
        if explicit_exclusion:
            is_valid = False
            reason = explicit_exclusion["reason"]
            row["attempt_exclusion_note"] = explicit_exclusion.get(
                "evidence_note", ""
            )
        else:
            is_valid, reason = scientific_validity(row)
        row["scientific_valid"] = int(is_valid)
        row["validity_reason"] = reason
        manifest_rows.append(
            {
            "experiment_id": row["experiment_id"],
            "job_id": row.get("job_id", ""),
            "artifact_path": row["artifact_path"],
            "scientific_valid": int(is_valid),
            "validity_reason": reason,
            "attempt_exclusion_note": row.get("attempt_exclusion_note", ""),
            "exit_code": row.get("exit_code", ""),
            }
        )
        if is_valid:
            valid_rows.append(row)
    invalid_root = args.output_root / "invalid"
    for trial_path in sorted(invalid_root.glob("**/trial-*.json")):
        trial = json.loads(trial_path.read_text())
        manifest_rows.append(
            {
                "experiment_id": trial_path.parent.name,
                "job_id": trial.get("job_id", ""),
                "artifact_path": str(trial_path),
                "scientific_valid": 0,
                "validity_reason": invalid_reason(trial_path),
                "attempt_exclusion_note": "",
                "exit_code": trial.get("exit_code", ""),
            }
        )

    if not valid_rows:
        raise SystemExit("no valid benchmark trials found")
    try:
        derived_method_rows = select_locked_method_rows(
            valid_rows,
            load_method_artifacts(args.method_artifacts),
            repo_root=Path.cwd(),
        )
        locked_method_rows = load_locked_method_corpus(
            args.method_corpus,
            args.method_corpus_digest,
        )
        method_drift_rows = audit_locked_method_derivation(
            locked_method_rows,
            derived_method_rows,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    write_csv(args.corpus, valid_rows)
    write_csv(args.method_corpus_drift, method_drift_rows)
    write_csv(args.manifest, manifest_rows)
    print(
        f"wrote {len(valid_rows)} valid corpus rows and "
        f"verified {len(locked_method_rows)} locked method rows; "
        f"{len(method_drift_rows)} subtype-only derivation drifts; "
        f"{sum(int(row['scientific_valid']) == 0 for row in manifest_rows)} "
        "invalid manifest rows"
    )


if __name__ == "__main__":
    main()
