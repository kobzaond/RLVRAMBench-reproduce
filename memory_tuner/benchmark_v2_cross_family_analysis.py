#!/usr/bin/env python3
"""Analyze and advance the preregistered cross-family boundary study."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from memory_tuner.benchmark_characterization import dominant_phase
from memory_tuner.failure_taxonomy import last_observed_phase
from memory_tuner.log_parser import classify_failure_detailed
from memory_tuner.plan_benchmark_v2_cross_family import (
    FIELDS,
    build_rows,
)
from memory_tuner.rl_trial_attempts import select_scientific_rl_attempt


MEMORY_LIMIT_MIB = 38_912.0
EXPECTED_CASES = 8
MEMORY_FAILURE_KINDS = {
    "cuda_oom",
    "rollout_init_memory",
    "weight_sync_oom",
}
PHASE_MEMORY_FAILURES = {
    "actor_update": "actor_update_oom",
    "reference_logprob": "reference_logprob_oom",
    "rollout": "rollout_oom",
    "weight_sync": "weight_sync_oom",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]) if rows else ["experiment_id"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def case_key(row: Mapping) -> tuple[str, str, str]:
    return (
        str(row["model_family"]),
        str(row["algorithm"]),
        str(row["dataset"]),
    )


def load_results(matrix: Path, root: Path) -> list[dict]:
    output = []
    for specification in read_csv(matrix):
        experiment_id = specification["experiment_id"]
        paths = sorted((root / experiment_id).glob("trial-*.json"))
        if not paths:
            raise ValueError(f"{experiment_id}: missing trial artifact")
        trial, excluded = select_scientific_rl_attempt(paths)
        if trial is None:
            reasons = ", ".join(reason for _, reason in excluded)
            raise ValueError(
                f"{experiment_id}: no scientific trial: {reasons}"
            )
        success = int(int(trial.get("exit_code", 1)) == 0)
        peak = float(trial.get("peak_gpu_memory_mib", math.nan))
        safe = int(success and math.isfinite(peak) and peak <= MEMORY_LIMIT_MIB)
        terminal_phase = last_observed_phase(
            trial.get("phase_memory_csv")
        )
        failure_kind = str(trial.get("failure_kind", ""))
        if not success and not failure_kind:
            run_log = Path(str(trial.get("run_log", "")))
            diagnostic_text = (
                run_log.read_text(errors="replace")
                if run_log.is_file()
                else ""
            )
            failure_kind = (
                classify_failure_detailed(
                    diagnostic_text,
                    int(trial.get("exit_code", 1)),
                )
                or ""
            )
        if (
            not success
            and failure_kind in MEMORY_FAILURE_KINDS
            and terminal_phase in PHASE_MEMORY_FAILURES
        ):
            failure_kind = PHASE_MEMORY_FAILURES[terminal_phase]
        output.append(
            {
                "experiment_id": experiment_id,
                "study_stage": specification["study_stage"],
                "configuration_level": specification["configuration_level"],
                "model_family": specification["model_family"],
                "algorithm": specification["algorithm"],
                "dataset": specification["dataset"],
                "training_seed": specification["training_seed"],
                "success": success,
                "safe": safe,
                "peak_gpu_memory_mib": peak,
                "dominant_phase": dominant_phase(trial),
                "terminal_phase": terminal_phase,
                "failure_kind": failure_kind,
                "elapsed_seconds": trial.get("elapsed_seconds", ""),
                "job_id": trial.get("job_id", ""),
                "artifact_path": trial.get("artifact_path", ""),
            }
        )
    return output


def bracket_status(rows: Sequence[Mapping]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[Mapping]] = defaultdict(list)
    for row in rows:
        grouped[case_key(row)].append(row)
    status = []
    for key, values in sorted(grouped.items()):
        safe_count = sum(int(row["safe"]) for row in values)
        unsafe_count = len(values) - safe_count
        status.append(
            {
                "model_family": key[0],
                "algorithm": key[1],
                "dataset": key[2],
                "configurations": len(values),
                "safe_configurations": safe_count,
                "unsafe_configurations": unsafe_count,
                "bracketed": int(safe_count > 0 and unsafe_count > 0),
            }
        )
    return status


def build_confirmatory_rows(seeds: Sequence[int] = (41, 42, 43)) -> list[dict]:
    base_rows = build_rows("smoke") + build_rows("development")
    output = []
    for base in base_rows:
        for seed in seeds:
            row = {field: base[field] for field in FIELDS}
            row["experiment_id"] = (
                str(base["experiment_id"])
                .replace("-smoke-", "-confirm-")
                .replace("-development-", "-confirm-")
                + f"-s{seed}"
            )
            row["run_group"] = "benchmark_v2_cross_family_confirmatory"
            row["study_stage"] = "confirmatory"
            row["training_seed"] = seed
            output.append(row)
    return output


def render_development(
    rows: Sequence[Mapping],
    status: Sequence[Mapping],
) -> str:
    bracketed = sum(int(row["bracketed"]) for row in status)
    lines = [
        "# RLVRAMBench v2 cross-family boundary development",
        "",
        f"- Scientific development trials: **{len(rows)}/48**",
        f"- Family/algorithm/workload cases bracketed: "
        f"**{bracketed}/{EXPECTED_CASES}**",
        "- Development outcomes are used only to verify that the fixed "
        "six-level lattice spans both sides of the boundary.",
        "- If every case is bracketed, the confirmatory arm repeats all six "
        "predeclared levels with seeds 41, 42, and 43.",
        "",
        "| Family | Algorithm | Workload | Configurations | Safe | Unsafe | "
        "Bracketed |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in status:
        lines.append(
            f"| {row['model_family']} | {row['algorithm']} | "
            f"{row['dataset']} | {row['configurations']} | "
            f"{row['safe_configurations']} | "
            f"{row['unsafe_configurations']} | {row['bracketed']} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_confirmatory(rows: Sequence[Mapping]) -> str:
    grouped: dict[tuple[str, str, str, str], list[Mapping]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["model_family"]),
                str(row["algorithm"]),
                str(row["dataset"]),
                str(row["configuration_level"]),
            )
        ].append(row)
    complete = sum(len(values) == 3 for values in grouped.values())
    flips = sum(
        len({int(row["safe"]) for row in values}) > 1
        for values in grouped.values()
    )
    lines = [
        "# RLVRAMBench v2 cross-family confirmatory repetitions",
        "",
        f"- Scientific trial artifacts: **{len(rows)}/144**",
        f"- Complete three-repetition configurations: "
        f"**{complete}/48**",
        f"- Configurations with a safety-label flip: **{flips}/48**",
        "",
        "Confirmatory inference must use all trials, including scientific "
        "memory failures. Development outcomes are not pooled into "
        "confirmatory estimates.",
        "",
    ]
    return "\n".join(lines)


def development_mode(args: argparse.Namespace) -> int:
    smoke = load_results(args.smoke_matrix, args.smoke_root)
    development = load_results(
        args.development_matrix,
        args.development_root,
    )
    rows = smoke + development
    status = bracket_status(rows)
    write_csv(args.development_output, rows)
    write_csv(args.bracket_output, status)
    args.development_report.parent.mkdir(parents=True, exist_ok=True)
    args.development_report.write_text(
        render_development(rows, status),
        encoding="utf-8",
    )
    bracketed = sum(int(row["bracketed"]) for row in status)
    if len(status) != EXPECTED_CASES or bracketed != EXPECTED_CASES:
        print(
            f"boundary lattice insufficient: {bracketed}/"
            f"{EXPECTED_CASES} cases bracketed"
        )
        return 3
    confirmatory = build_confirmatory_rows()
    write_csv(args.confirmatory_matrix, confirmatory)
    print(
        f"all {EXPECTED_CASES} cases bracketed; wrote "
        f"{len(confirmatory)} confirmatory rows"
    )
    return 0


def confirmatory_mode(args: argparse.Namespace) -> int:
    rows = load_results(args.confirmatory_matrix, args.confirmatory_root)
    write_csv(args.confirmatory_output, rows)
    args.confirmatory_report.parent.mkdir(parents=True, exist_ok=True)
    args.confirmatory_report.write_text(
        render_confirmatory(rows),
        encoding="utf-8",
    )
    print(f"analyzed {len(rows)} cross-family confirmatory trials")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("development", "confirmatory"),
        default="development",
    )
    parser.add_argument(
        "--smoke-matrix",
        type=Path,
        default=Path("memory_tuner/rlvram_v2_cross_family_smoke.csv"),
    )
    parser.add_argument(
        "--smoke-root",
        type=Path,
        default=Path("output/benchmark_v2_cross_family_smoke"),
    )
    parser.add_argument(
        "--development-matrix",
        type=Path,
        default=Path(
            "memory_tuner/rlvram_v2_cross_family_development.csv"
        ),
    )
    parser.add_argument(
        "--development-root",
        type=Path,
        default=Path("output/benchmark_v2_cross_family_development"),
    )
    parser.add_argument(
        "--development-output",
        type=Path,
        default=Path(
            "profiles/benchmark_v2/cross-family-development-trials.csv"
        ),
    )
    parser.add_argument(
        "--bracket-output",
        type=Path,
        default=Path(
            "profiles/benchmark_v2/cross-family-development-brackets.csv"
        ),
    )
    parser.add_argument(
        "--development-report",
        type=Path,
        default=Path("paper/benchmark_v2_cross_family_development.md"),
    )
    parser.add_argument(
        "--confirmatory-matrix",
        type=Path,
        default=Path(
            "memory_tuner/rlvram_v2_cross_family_confirmatory.csv"
        ),
    )
    parser.add_argument(
        "--confirmatory-root",
        type=Path,
        default=Path("output/benchmark_v2_cross_family_confirmatory"),
    )
    parser.add_argument(
        "--confirmatory-output",
        type=Path,
        default=Path(
            "profiles/benchmark_v2/cross-family-confirmatory-trials.csv"
        ),
    )
    parser.add_argument(
        "--confirmatory-report",
        type=Path,
        default=Path("paper/benchmark_v2_cross_family_confirmatory.md"),
    )
    args = parser.parse_args()
    try:
        if args.mode == "development":
            return development_mode(args)
        return confirmatory_mode(args)
    except ValueError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    raise SystemExit(main())
