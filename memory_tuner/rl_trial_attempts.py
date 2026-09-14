"""Attempt selection for retryable colocated-RL trial directories."""

from __future__ import annotations

from pathlib import Path

from memory_tuner.build_benchmark_corpus import (
    DEFAULT_ATTEMPT_EXCLUSIONS,
    DEFAULT_FAILURE_ANNOTATIONS,
    load_attempt_exclusions,
    load_failure_annotations,
    scientific_validity,
    trial_row,
)


def select_scientific_rl_attempt(
    trial_paths: list[Path],
) -> tuple[dict | None, list[tuple[Path, str]]]:
    """Return the sole scientific outcome and excluded retry attempts."""
    annotations = load_failure_annotations(DEFAULT_FAILURE_ANNOTATIONS)
    attempt_exclusions = load_attempt_exclusions(DEFAULT_ATTEMPT_EXCLUSIONS)
    valid = []
    excluded = []
    for path in sorted(trial_paths):
        row = trial_row(path, annotations)
        explicit_exclusion = attempt_exclusions.get(str(row.get("job_id", "")))
        if explicit_exclusion:
            excluded.append((path, explicit_exclusion["reason"]))
            continue
        is_valid, reason = scientific_validity(row)
        if is_valid:
            valid.append(row)
        else:
            excluded.append((path, reason))
    if len(valid) > 1:
        paths = ", ".join(str(row["artifact_path"]) for row in valid)
        raise ValueError(f"multiple scientific attempts: {paths}")
    if not valid:
        return None, excluded
    selected = dict(valid[0])
    selected["attempt_count"] = len(trial_paths)
    selected["excluded_attempt_count"] = len(excluded)
    return selected, excluded
