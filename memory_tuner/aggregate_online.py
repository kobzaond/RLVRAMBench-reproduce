#!/usr/bin/env python3
"""Validate and aggregate the targeted online vLLM screening trials."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from memory_tuner.trial_attempts import select_trial_attempts
from memory_tuner.validate_trial_matrix import validate_environment_metadata


INFRASTRUCTURE_FAILURE_MARKERS = (
    "address already in use",
    "eaddrinuse",
    "no space left on device",
    "stale file handle",
)
MEMORY_FAILURE_MARKERS = (
    "cuda out of memory",
    "outofmemoryerror",
    "no available memory for the cache blocks",
    "free memory on device",
)
RETAINABLE_FAILURE_KINDS = {
    "serving_memory_failure",
    "load_execution_failure",
}
READINESS_MARKERS = (
    "application startup complete",
    'get /health http/1.1" 200 ok',
)


def classify_failed_attempt(attempt: dict) -> str:
    """Separate analyzable serving failures from invalid infrastructure runs."""

    trial = attempt["trial"]
    trial_dir = attempt["trial_path"].parent
    diagnostics = []
    for filename in ("server.err", "server.out"):
        path = trial_dir / filename
        if path.is_file():
            diagnostics.append(path.read_text(errors="replace"))
    text = "\n".join(diagnostics).lower()
    if any(marker in text for marker in INFRASTRUCTURE_FAILURE_MARKERS):
        return "infrastructure_failure"
    if any(marker in text for marker in MEMORY_FAILURE_MARKERS):
        return "serving_memory_failure"
    readiness_observed = any(marker in text for marker in READINESS_MARKERS)
    # run_vllm_online.slurm starts elapsed_seconds only after a successful
    # health check. Requiring both the launcher field and an independent
    # server-log marker prevents an unrelated pre-readiness crash from being
    # admitted as a scientific load-execution failure.
    if (
        float(trial.get("elapsed_seconds", 0) or 0) > 0
        and readiness_observed
    ):
        return "load_execution_failure"
    return "unclassified_readiness_failure"


def aggregate(
    matrix: Path,
    root: Path,
    *,
    require_metadata: bool = False,
    retain_scientific_failures: bool = False,
) -> tuple[list[dict], list[str], int]:
    with matrix.open() as handle:
        expected = {
            row["experiment_id"]: row for row in csv.DictReader(handle)
        }
    attempts = select_trial_attempts(sorted(root.glob("*/trial.json")))
    observed = {
        experiment_id: attempt
        for experiment_id, attempt in attempts.items()
        if experiment_id in expected
    }

    rows = []
    errors = []
    for experiment_id, config in expected.items():
        if experiment_id not in observed:
            errors.append(f"missing online trial: {experiment_id}")
            continue
        attempt = observed[experiment_id]
        trial = attempt["trial"]
        result = attempt["result"]
        if require_metadata:
            if not trial.get("environment_json"):
                errors.append(
                    f"{experiment_id}: missing environment_json declaration"
                )
            errors.extend(
                f"{experiment_id}: {error}"
                for error in validate_environment_metadata(
                    trial, require_dataset=False
                )
            )
            for field in ("phase_memory_csv", "gpu_telemetry_csv"):
                path = Path(str(trial.get(field, "")))
                if not path.is_file() or path.stat().st_size == 0:
                    errors.append(
                        f"{experiment_id}: missing nonempty {field}: {path}"
                    )
        success = attempt["successful"]
        failure_kind = "" if success else classify_failed_attempt(attempt)
        if not success:
            if (
                not retain_scientific_failures
                or failure_kind not in RETAINABLE_FAILURE_KINDS
            ):
                errors.append(
                    f"failed online trial: {experiment_id} "
                    f"({failure_kind})"
                )
        row = {
            **config,
            **trial,
            "success": int(success),
            "failure_kind": failure_kind,
            "attempt_count": attempt["attempt_count"],
            "failed_attempt_count": attempt["failed_attempt_count"],
            "selected_trial_path": str(attempt["trial_path"]),
        }
        for key, value in result.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                row[f"result_{key}"] = value
        rows.append(row)
    return rows, errors, len(expected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("memory_tuner/rlvram_online_screening.csv"),
    )
    parser.add_argument(
        "--root", type=Path, default=Path("profiles/vllm-online")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("profiles/screening/online-screening.csv"),
    )
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--require-metadata", action="store_true")
    parser.add_argument(
        "--retain-scientific-failures",
        action="store_true",
        help=(
            "Retain observed serving-memory and post-readiness load failures "
            "as unsafe outcomes while still rejecting missing, infrastructure, "
            "and unclassified readiness failures."
        ),
    )
    args = parser.parse_args()

    try:
        rows, errors, expected_count = aggregate(
            args.matrix,
            args.root,
            require_metadata=args.require_metadata,
            retain_scientific_failures=args.retain_scientific_failures,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"validated {len(rows)}/{expected_count} online trials; "
        f"issues={len(errors)}"
    )
    if args.require_complete and errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
