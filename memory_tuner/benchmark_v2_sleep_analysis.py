#!/usr/bin/env python3
"""Analyze failure-inclusive matched sleep/release experiments."""

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
from memory_tuner.rl_trial_attempts import select_scientific_rl_attempt


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


def load_results(matrix: Path, root: Path) -> list[dict]:
    output = []
    for specification in read_csv(matrix):
        experiment_id = specification["experiment_id"]
        paths = sorted((root / experiment_id).glob("trial-*.json"))
        trial, excluded = select_scientific_rl_attempt(paths)
        if trial is None:
            reasons = ", ".join(reason for _, reason in excluded)
            raise ValueError(
                f"{experiment_id}: no scientific trial: {reasons}"
            )
        success = int(int(trial.get("exit_code", 1)) == 0)
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
                "pair_id": specification["pair_id"],
                "model_family": specification["model_family"],
                "algorithm": specification["algorithm"],
                "dataset": specification["dataset"],
                "risk_tier": specification["risk_tier"],
                "condition": specification["condition"],
                "training_seed": specification["training_seed"],
                "success": success,
                "peak_gpu_memory_mib": trial.get(
                    "peak_gpu_memory_mib", math.nan
                ),
                "elapsed_seconds": trial.get("elapsed_seconds", ""),
                "dominant_phase": dominant_phase(trial),
                "terminal_phase": terminal_phase,
                "failure_kind": failure_kind,
                "job_id": trial.get("job_id", ""),
                "artifact_path": trial.get("artifact_path", ""),
            }
        )
    return output


def paired_results(rows: Sequence[Mapping]) -> list[dict]:
    grouped: dict[str, dict[str, Mapping]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["pair_id"])][str(row["condition"])] = row
    output = []
    for pair_id, conditions in sorted(grouped.items()):
        if set(conditions) != {"release", "resident"}:
            raise ValueError(f"{pair_id}: incomplete matched conditions")
        release = conditions["release"]
        resident = conditions["resident"]
        release_success = int(release["success"])
        resident_success = int(resident["success"])
        both = release_success and resident_success
        output.append(
            {
                "pair_id": pair_id,
                "model_family": release["model_family"],
                "algorithm": release["algorithm"],
                "dataset": release["dataset"],
                "risk_tier": release["risk_tier"],
                "training_seed": release["training_seed"],
                "release_success": release_success,
                "resident_success": resident_success,
                "release_rescues_failure": int(
                    release_success and not resident_success
                ),
                "resident_rescues_failure": int(
                    resident_success and not release_success
                ),
                "both_success": int(both),
                "resident_minus_release_peak_mib": (
                    float(resident["peak_gpu_memory_mib"])
                    - float(release["peak_gpu_memory_mib"])
                    if both
                    else math.nan
                ),
                "resident_minus_release_elapsed_seconds": (
                    float(resident["elapsed_seconds"])
                    - float(release["elapsed_seconds"])
                    if both
                    else math.nan
                ),
                "release_dominant_phase": release["dominant_phase"],
                "resident_dominant_phase": resident["dominant_phase"],
                "release_failure_kind": release["failure_kind"],
                "resident_failure_kind": resident["failure_kind"],
            }
        )
    return output


def write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def render(pairs: Sequence[Mapping]) -> str:
    joint = [row for row in pairs if int(row["both_success"])]
    deltas = [
        float(row["resident_minus_release_peak_mib"])
        for row in joint
    ]
    mean_delta = sum(deltas) / len(deltas) if deltas else math.nan
    return "\n".join(
        [
            "# RLVRAMBench v2 expanded sleep/release ablation",
            "",
            f"- Complete matched fresh-process pairs: **{len(pairs)}/36**",
            f"- Release succeeds while resident state fails: "
            f"**{sum(int(row['release_rescues_failure']) for row in pairs)}"
            f"/{len(pairs)}**",
            f"- Resident state succeeds while release fails: "
            f"**{sum(int(row['resident_rescues_failure']) for row in pairs)}"
            f"/{len(pairs)}**",
            f"- Both conditions succeed: **{len(joint)}/{len(pairs)}**",
            f"- Mean resident-minus-release peak among joint successes: "
            f"**{mean_delta:.1f} MiB**"
            if deltas
            else "- Mean paired peak difference: **not estimable**",
            "",
            "All scientific failures are retained as paired outcomes. Peak "
            "and runtime differences are computed only for joint successes.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("memory_tuner/rlvram_v2_sleep_expansion.csv"),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("output/benchmark_v2_sleep_expansion"),
    )
    parser.add_argument(
        "--trials-output",
        type=Path,
        default=Path("profiles/benchmark_v2/sleep-expansion-trials.csv"),
    )
    parser.add_argument(
        "--pairs-output",
        type=Path,
        default=Path("profiles/benchmark_v2/sleep-expansion-pairs.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("paper/benchmark_v2_sleep_expansion.md"),
    )
    args = parser.parse_args()
    try:
        rows = load_results(args.matrix, args.root)
        pairs = paired_results(rows)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    write_csv(args.trials_output, rows)
    write_csv(args.pairs_output, pairs)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render(pairs), encoding="utf-8")
    print(f"analyzed {len(pairs)} expanded sleep/release pairs")


if __name__ == "__main__":
    main()
