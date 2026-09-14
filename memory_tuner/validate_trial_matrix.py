#!/usr/bin/env python3
"""Validate one immutable trial artifact per benchmark-matrix row."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

from memory_tuner.rl_trial_attempts import select_scientific_rl_attempt


def observed_phase_labels(path: Path) -> set[str]:
    """Return the exact phase labels retained by the external monitor."""
    labels = set()
    with path.open(newline="") as handle:
        for fields in csv.reader(handle):
            if len(fields) >= 3:
                labels.add(str(fields[2]))
    return labels


def observed_phase_steps(path: Path) -> set[tuple[str, int]]:
    """Return exact (phase, global-step) pairs from the external monitor."""
    observations = set()
    with path.open(newline="") as handle:
        for fields in csv.reader(handle):
            if len(fields) < 3:
                continue
            try:
                step = int(fields[1])
            except ValueError:
                continue
            observations.add((str(fields[2]), step))
    return observations


def validate_environment_metadata(
    trial: dict,
    *,
    require_dataset: bool = True,
) -> list[str]:
    value = trial.get("environment_json")
    if not value:
        return []
    path = Path(value)
    if not path.is_file():
        return [f"missing environment metadata: {path}"]
    try:
        metadata = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return [f"invalid environment metadata {path}: {error}"]

    errors = []
    required = {
        "host.hostname": metadata.get("host", {}).get("hostname"),
        "slurm.SLURM_JOB_ID": metadata.get("slurm", {}).get("SLURM_JOB_ID"),
        "container.sha256": metadata.get("container", {}).get("sha256"),
        "repository.head": metadata.get("repository", {}).get("head"),
        "model.resolved_revision": metadata.get("model", {}).get(
            "resolved_revision"
        ),
        "gpus_csv": metadata.get("gpus_csv"),
    }
    if require_dataset:
        required.update(
            {
                "dataset.train.parquet.sha256": metadata.get("dataset", {})
                .get("train.parquet", {})
                .get("sha256"),
                "dataset.test.parquet.sha256": metadata.get("dataset", {})
                .get("test.parquet", {})
                .get("sha256"),
            }
        )
    packages = metadata.get("software", {}).get("packages", {})
    for package in ("torch", "vllm", "transformers", "ray"):
        required[f"software.packages.{package}"] = packages.get(package)
    for field, field_value in required.items():
        if not field_value:
            errors.append(f"{path}: missing required field {field}")
    digest = str(required["container.sha256"] or "")
    if digest and (
        len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        errors.append(f"{path}: container.sha256 is not a lowercase SHA-256")
    return errors


def validate(
    matrix: Path | None,
    root: Path,
    expected_count: int | None = None,
    require_telemetry: bool = False,
    require_allocator: bool = False,
    required_phases: Sequence[str] = (),
    required_phase_steps: Sequence[tuple[str, int]] = (),
    required_first_positive_step: int | None = None,
    maximum_observed_step: int | None = None,
    allow_extra: bool = False,
) -> tuple[list[dict], list[str]]:
    if matrix is not None:
        with matrix.open() as handle:
            expected = [row["experiment_id"] for row in csv.DictReader(handle)]
    else:
        expected = sorted(
            {
                path.parent.name
                for path in root.glob("*/trial-*.json")
            }
        )
    rows = []
    errors = []
    if expected_count is not None and len(expected) != expected_count:
        errors.append(
            f"expected {expected_count} experiment directories, found "
            f"{len(expected)}"
        )
    for experiment_id in expected:
        paths = list((root / experiment_id).glob("trial-*.json"))
        if not paths:
            errors.append(
                f"{experiment_id}: expected a trial artifact, found 0"
            )
            continue
        try:
            trial, excluded = select_scientific_rl_attempt(paths)
        except ValueError as error:
            errors.append(f"{experiment_id}: {error}")
            continue
        if trial is None:
            reasons = ", ".join(reason for _, reason in excluded)
            errors.append(
                f"{experiment_id}: no scientific attempt among {len(paths)} "
                f"attempts: {reasons}"
            )
            continue
        trial["experiment_id"] = experiment_id
        rows.append(trial)
        errors.extend(
            f"{experiment_id}: {error}"
            for error in validate_environment_metadata(trial)
        )
        if require_telemetry:
            for field in ("phase_memory_csv", "gpu_telemetry_csv"):
                path = Path(str(trial.get(field, "")))
                if not path.is_file() or path.stat().st_size == 0:
                    errors.append(
                        f"{experiment_id}: missing nonempty {field}: {path}"
                    )
        if (
            required_phases
            or required_phase_steps
            or required_first_positive_step is not None
            or maximum_observed_step is not None
        ):
            path = Path(str(trial.get("phase_memory_csv", "")))
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(
                    f"{experiment_id}: cannot verify required phase trace in "
                    f"missing or empty phase_memory_csv: {path}"
                )
            else:
                try:
                    observed = observed_phase_labels(path)
                    observed_pairs = observed_phase_steps(path)
                except OSError as error:
                    errors.append(
                        f"{experiment_id}: cannot read phase_memory_csv "
                        f"{path}: {error}"
                    )
                else:
                    missing = sorted(set(required_phases) - observed)
                    if missing:
                        errors.append(
                            f"{experiment_id}: missing required phase labels "
                            f"{missing} in {path}"
                        )
                    missing_pairs = sorted(
                        set(required_phase_steps) - observed_pairs
                    )
                    if missing_pairs:
                        errors.append(
                            f"{experiment_id}: missing required phase-step "
                            f"pairs {missing_pairs} in {path}"
                        )
                    steps = sorted({step for _, step in observed_pairs})
                    positive_steps = [step for step in steps if step > 0]
                    if required_first_positive_step is not None and (
                        not positive_steps
                        or positive_steps[0] != required_first_positive_step
                    ):
                        observed_first = (
                            positive_steps[0] if positive_steps else None
                        )
                        errors.append(
                            f"{experiment_id}: first positive observed step "
                            f"is {observed_first}, expected "
                            f"{required_first_positive_step} in {path}"
                        )
                    if (
                        maximum_observed_step is not None
                        and steps
                        and steps[-1] > maximum_observed_step
                    ):
                        errors.append(
                            f"{experiment_id}: maximum observed step "
                            f"{steps[-1]} exceeds {maximum_observed_step} "
                            f"in {path}"
                        )
        if require_allocator:
            path = Path(str(trial.get("allocator_trace_dir", "")))
            if not path.is_dir() or not any(
                item.is_file() for item in path.iterdir()
            ):
                errors.append(
                    f"{experiment_id}: missing allocator traces: {path}"
                )
    extra = {
        path.parent.name
        for path in root.glob("*/trial-*.json")
        if path.parent.name not in set(expected)
    }
    if extra and not allow_extra:
        errors.append(f"unexpected experiment directories: {sorted(extra)}")
    return rows, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected", type=int)
    parser.add_argument("--require-telemetry", action="store_true")
    parser.add_argument("--require-allocator", action="store_true")
    parser.add_argument(
        "--require-phase",
        action="append",
        default=[],
        help=(
            "require an exact phase label in phase_memory_csv; repeat for "
            "multiple labels"
        ),
    )
    parser.add_argument(
        "--require-phase-step",
        action="append",
        default=[],
        metavar="PHASE:STEP",
        help=(
            "require an exact phase and global-step pair; repeat for "
            "multiple pairs"
        ),
    )
    parser.add_argument("--require-first-positive-step", type=int)
    parser.add_argument("--maximum-observed-step", type=int)
    parser.add_argument("--require-all-success", action="store_true")
    parser.add_argument(
        "--allow-extra",
        action="store_true",
        help=(
            "validate every requested matrix row while allowing unrelated "
            "experiment directories in a shared output root"
        ),
    )
    args = parser.parse_args()
    if args.matrix is None and args.expected is None:
        parser.error("one of --matrix or --expected is required")
    required_phase_steps = []
    for value in args.require_phase_step:
        phase, separator, raw_step = value.rpartition(":")
        if not separator or not phase:
            parser.error(
                f"--require-phase-step must be PHASE:STEP, got {value!r}"
            )
        try:
            step = int(raw_step)
        except ValueError:
            parser.error(
                f"--require-phase-step has invalid step in {value!r}"
            )
        required_phase_steps.append((phase, step))
    rows, errors = validate(
        args.matrix,
        args.root,
        expected_count=args.expected,
        require_telemetry=args.require_telemetry,
        require_allocator=args.require_allocator,
        required_phases=args.require_phase,
        required_phase_steps=required_phase_steps,
        required_first_positive_step=args.require_first_positive_step,
        maximum_observed_step=args.maximum_observed_step,
        allow_extra=args.allow_extra,
    )
    if args.require_all_success:
        errors.extend(
            f"{row['experiment_id']}: exit_code={row.get('exit_code')}"
            for row in rows
            if int(row.get("exit_code", 1)) != 0
        )
    print(
        f"validated {len(rows)} matrix trials; "
        f"successes={sum(int(row.get('exit_code', 1)) == 0 for row in rows)}; "
        f"issues={len(errors)}"
    )
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
