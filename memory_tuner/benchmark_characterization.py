#!/usr/bin/env python3
"""Configuration-level characterization of the failure-inclusive benchmark."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from memory_tuner.phaseguard_evaluation import (
    aggregate_configurations,
    configuration_id,
)


MEMORY_LIMIT_MIB = 38_912.0
PHASE_FIELDS = {
    "initializing": "phase_peak_initializing_mib",
    "rollout": "phase_peak_rollout_mib",
    "reference_logprob": "phase_peak_reference_logprob_mib",
    "actor_update": "phase_peak_actor_update_mib",
    "weight_sync": "phase_peak_weight_sync_mib",
    "idle": "phase_peak_idle_mib",
    "unknown": "phase_peak_unknown_mib",
}


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: object, default: float = math.nan) -> float:
    if value in (None, ""):
        return default
    return float(value)


def dominant_phase(row: Mapping) -> str:
    observed = [
        (number(row.get(field)), phase)
        for phase, field in PHASE_FIELDS.items()
        if math.isfinite(number(row.get(field)))
    ]
    return max(observed)[1] if observed else ""


def characterize(
    rows: Sequence[Mapping],
    *,
    memory_limit_mib: float = MEMORY_LIMIT_MIB,
) -> tuple[list[dict], list[dict], list[dict]]:
    configurations = aggregate_configurations(rows, memory_limit_mib)
    phase_counts = Counter(
        phase
        for row in configurations
        if int(row["success"])
        for phase in [dominant_phase(row)]
        if phase
    )
    phase_total = sum(phase_counts.values())
    phase_rows = [
        {
            "phase": phase,
            "configurations": count,
            "share": count / phase_total if phase_total else math.nan,
        }
        for phase, count in sorted(
            phase_counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]

    raw_by_case: dict[str, list[Mapping]] = defaultdict(list)
    configurations_by_case: dict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        raw_by_case[str(row["case_id"])].append(row)
    for row in configurations:
        configurations_by_case[str(row["case_id"])].append(row)

    case_rows = []
    for case_id, candidates in sorted(configurations_by_case.items()):
        raw = raw_by_case[case_id]
        failures = Counter(
            str(row.get("failure_kind") or "unclassified")
            for row in raw
            if not int(row.get("success", 0))
        )
        phases = Counter(
            phase
            for row in candidates
            if int(row["success"])
            for phase in [dominant_phase(row)]
            if phase
        )
        safe = [row for row in candidates if int(row["observed_safe"])]
        case_rows.append(
            {
                "case_id": case_id,
                "model_tag": candidates[0].get("model_tag", ""),
                "dataset": candidates[0].get("dataset", ""),
                "algorithm": candidates[0].get("algorithm", ""),
                "gpu_count": candidates[0].get("gpu_count", ""),
                "configurations": len(candidates),
                "successful_configurations": sum(
                    int(row["success"]) for row in candidates
                ),
                "safe_configurations": len(safe),
                "scientific_failure_artifacts": sum(failures.values()),
                "failure_kinds": ";".join(
                    f"{kind}:{count}"
                    for kind, count in sorted(failures.items())
                ),
                "modal_dominant_phase": (
                    phases.most_common(1)[0][0] if phases else ""
                ),
                "minimum_safe_peak_mib": (
                    min(number(row["peak_gpu_memory_mib"]) for row in safe)
                    if safe
                    else math.nan
                ),
                "maximum_safe_peak_mib": (
                    max(number(row["peak_gpu_memory_mib"]) for row in safe)
                    if safe
                    else math.nan
                ),
            }
        )

    outcome_counts = Counter(
        "success"
        if int(row.get("success", 0))
        else str(row.get("failure_kind") or "unclassified")
        for row in rows
    )
    outcome_rows = [
        {
            "outcome": outcome,
            "artifacts": count,
            "share": count / len(rows) if rows else math.nan,
        }
        for outcome, count in sorted(
            outcome_counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    return case_rows, phase_rows, outcome_rows


def configuration_outcomes(
    rows: Sequence[Mapping],
    *,
    memory_limit_mib: float = MEMORY_LIMIT_MIB,
) -> list[dict]:
    configurations = aggregate_configurations(rows, memory_limit_mib)
    raw_by_configuration: dict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        raw_by_configuration[configuration_id(row)].append(row)
    counts = Counter()
    for configuration in configurations:
        if int(configuration["success"]):
            counts["success"] += 1
            continue
        failure_kinds = sorted(
            {
                str(row.get("failure_kind") or "unclassified")
                for row in raw_by_configuration[
                    str(configuration["configuration_id"])
                ]
                if not int(row.get("success", 0))
            }
        )
        outcome = (
            failure_kinds[0]
            if len(failure_kinds) == 1
            else "mixed:" + "+".join(failure_kinds)
        )
        counts[outcome] += 1
    total = sum(counts.values())
    return [
        {
            "outcome": outcome,
            "configurations": count,
            "share": count / total if total else math.nan,
        }
        for outcome, count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]


def write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def render_markdown(
    case_rows: Sequence[Mapping],
    phase_rows: Sequence[Mapping],
    outcome_rows: Sequence[Mapping],
    configuration_outcome_rows: Sequence[Mapping],
) -> str:
    lines = [
        "# RLVRAMBench configuration-level characterization",
        "",
        "Repeated processes are collapsed to one configuration before phase "
        "dominance and case feasibility are counted. A configuration is safe "
        "only if every observed repetition succeeds and its worst observed "
        f"peak is at most {MEMORY_LIMIT_MIB:.0f} MiB.",
        "",
        "## Repetition-collapsed configuration outcomes",
        "",
        "| Outcome | Configurations | Share |",
        "|---|---:|---:|",
    ]
    lines.extend(
        f"| {row['outcome']} | {row['configurations']} | "
        f"{100 * number(row['share']):.1f}% |"
        for row in configuration_outcome_rows
    )
    lines.extend(
        [
        "",
        "## Raw scientific trial artifacts",
        "",
        "| Outcome | Trial artifacts | Share |",
        "|---|---:|---:|",
        ]
    )
    lines.extend(
        f"| {row['outcome']} | {row['artifacts']} | "
        f"{100 * number(row['share']):.1f}% |"
        for row in outcome_rows
    )
    lines.extend(
        [
            "",
            "## Dominant phase among successful configurations",
            "",
            "| Phase | Configurations | Share |",
            "|---|---:|---:|",
        ]
    )
    lines.extend(
        f"| {row['phase']} | {row['configurations']} | "
        f"{100 * number(row['share']):.1f}% |"
        for row in phase_rows
    )
    lines.extend(
        [
            "",
            "## Case feasibility",
            "",
            "| Case | Configs | Safe | Scientific failure artifacts | "
            "Modal dominant phase |",
            "|---|---:|---:|---:|---|",
        ]
    )
    lines.extend(
        f"| `{row['case_id']}` | {row['configurations']} | "
        f"{row['safe_configurations']} | "
        f"{row['scientific_failure_artifacts']} | "
        f"{row['modal_dominant_phase'] or '--'} |"
        for row in case_rows
    )
    lines.extend(
        [
            "",
            "Dominant phase is descriptive: it is the largest recorded phase "
            "peak in a successful configuration, not proof that the phase "
            "caused a failure. Scientific failure labels use logs and "
            "allocator evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("profiles/benchmark/benchmark-corpus.csv"),
    )
    parser.add_argument(
        "--case-output",
        type=Path,
        default=Path("profiles/benchmark/case-characterization.csv"),
    )
    parser.add_argument(
        "--phase-output",
        type=Path,
        default=Path("profiles/benchmark/phase-dominance.csv"),
    )
    parser.add_argument(
        "--outcome-output",
        type=Path,
        default=Path("profiles/benchmark/scientific-outcomes.csv"),
    )
    parser.add_argument(
        "--configuration-outcome-output",
        type=Path,
        default=Path("profiles/benchmark/configuration-outcomes.csv"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("paper/benchmark_characterization.md"),
    )
    args = parser.parse_args()
    rows = read_csv(args.corpus)
    if not rows:
        raise SystemExit(f"empty benchmark corpus: {args.corpus}")
    case_rows, phase_rows, outcome_rows = characterize(rows)
    configuration_outcome_rows = configuration_outcomes(rows)
    write_csv(args.case_output, case_rows)
    write_csv(args.phase_output, phase_rows)
    write_csv(args.outcome_output, outcome_rows)
    write_csv(
        args.configuration_outcome_output,
        configuration_outcome_rows,
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(
        render_markdown(
            case_rows,
            phase_rows,
            outcome_rows,
            configuration_outcome_rows,
        ),
        encoding="utf-8",
    )
    print(
        f"characterized {len(case_rows)} cases and "
        f"{sum(int(row['configurations']) for row in configuration_outcome_rows)} "
        "repetition-collapsed configurations"
    )


if __name__ == "__main__":
    main()
