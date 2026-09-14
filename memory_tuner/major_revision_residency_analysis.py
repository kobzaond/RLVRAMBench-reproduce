#!/usr/bin/env python3
"""Analyze the counterbalanced release/residency replication."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from memory_tuner.benchmark_v2_inference import (
    cluster_bootstrap,
    inference_row,
    mean_value,
)
from memory_tuner.benchmark_v2_sleep_analysis import (
    load_results,
    paired_results,
)

SEQUENCE_LABELS = {
    "release_first": "Release first",
    "resident_first": "Resident first",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]) if rows else ["empty"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def augment(
    matrices: Sequence[Path],
    root: Path,
) -> tuple[list[dict], list[dict]]:
    specifications = {
        row["experiment_id"]: row
        for matrix in matrices
        for row in read_csv(matrix)
    }
    trials = []
    for matrix in matrices:
        trials.extend(load_results(matrix, root))
    for row in trials:
        spec = specifications[row["experiment_id"]]
        row["sequence"] = spec["sequence"]
        row["period"] = int(spec["period"])
        row["source_config_id"] = spec["source_config_id"]
    pairs = paired_results(trials)
    pair_specs = {
        row["pair_id"]: row for row in specifications.values()
    }
    for row in pairs:
        spec = pair_specs[row["pair_id"]]
        row["sequence"] = spec["sequence"]
        row["source_config_id"] = spec["source_config_id"]
    return trials, pairs


def validate_same_node_provenance(
    trials: Sequence[Mapping],
    manifest_root: Path,
) -> list[dict]:
    """Verify that both selected periods came from one controlled allocation."""

    grouped: dict[str, list[Mapping]] = defaultdict(list)
    for row in trials:
        grouped[str(row["pair_id"])].append(row)
    output = []
    for pair_id, rows in sorted(grouped.items()):
        if len(rows) != 2:
            raise ValueError(
                f"{pair_id}: expected two selected period trials, "
                f"found {len(rows)}"
            )
        job_ids = {str(row["job_id"]) for row in rows}
        if len(job_ids) != 1:
            raise ValueError(
                f"{pair_id}: periods were selected from different jobs "
                f"{sorted(job_ids)}"
            )
        job_id = next(iter(job_ids))
        manifests = sorted((manifest_root / job_id).glob("pair-*.json"))
        if len(manifests) != 1:
            raise ValueError(
                f"{pair_id}: expected one pair manifest for job {job_id}, "
                f"found {len(manifests)}"
            )
        manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
        if str(manifest.get("job_id")) != job_id:
            raise ValueError(f"{pair_id}: manifest job ID mismatch")
        if manifest.get("pair_status") != "complete":
            raise ValueError(
                f"{pair_id}: incomplete pair status "
                f"{manifest.get('pair_status')!r}"
            )
        if manifest.get("same_node") is not True:
            raise ValueError(f"{pair_id}: same-node control not verified")
        if manifest.get("same_gpu_allocation") is not True:
            raise ValueError(
                f"{pair_id}: same-GPU allocation control not verified"
            )
        by_period = {int(row["period"]): row for row in rows}
        if set(by_period) != {1, 2}:
            raise ValueError(f"{pair_id}: selected periods are incomplete")
        for period in (1, 2):
            observed_success = int(by_period[period]["success"])
            manifest_success = int(
                int(manifest[f"period{period}_exit_code"]) == 0
            )
            if observed_success != manifest_success:
                raise ValueError(
                    f"{pair_id}: period {period} outcome disagrees with "
                    "the pair manifest"
                )
        output.append(
            {
                "pair_id": pair_id,
                "job_id": job_id,
                "pair_manifest": str(manifests[0]),
                "same_node": 1,
                "same_gpu_allocation": 1,
                "pair_status": "complete",
            }
        )
    return output


def completion_difference(rows: Sequence[Mapping]) -> float:
    return mean_value(
        [
            {
                "difference": float(row["release_success"])
                - float(row["resident_success"])
            }
            for row in rows
        ],
        "difference",
    )


def _cluster(row: Mapping) -> str:
    return str(row["source_config_id"])


def _sequence_difference(
    rows: Sequence[Mapping],
    sequence: str,
) -> float:
    return completion_difference(
        [row for row in rows if str(row["sequence"]) == sequence]
    )


def residency_inference(pairs: Sequence[Mapping]) -> list[dict]:
    output = [
        inference_row(
            study="counterbalanced_residency",
            estimand="release_minus_resident_process_success",
            rows=pairs,
            cluster=_cluster,
            statistic=completion_difference,
            unit="probability",
        ),
        inference_row(
            study="counterbalanced_residency",
            estimand="resident_minus_release_peak_joint_success",
            rows=pairs,
            cluster=_cluster,
            statistic=lambda values: mean_value(
                values,
                "resident_minus_release_peak_mib",
            ),
            unit="MiB",
        ),
        inference_row(
            study="counterbalanced_residency",
            estimand="resident_minus_release_elapsed_joint_success",
            rows=pairs,
            cluster=_cluster,
            statistic=lambda values: mean_value(
                values,
                "resident_minus_release_elapsed_seconds",
            ),
            unit="seconds",
        ),
    ]
    for sequence in ("release_first", "resident_first"):
        subset = [
            row for row in pairs if str(row["sequence"]) == sequence
        ]
        output.extend(
            [
                inference_row(
                    study="counterbalanced_residency",
                    subgroup=sequence,
                    estimand="release_minus_resident_process_success",
                    rows=subset,
                    cluster=_cluster,
                    statistic=completion_difference,
                    unit="probability",
                ),
                inference_row(
                    study="counterbalanced_residency",
                    subgroup=sequence,
                    estimand="resident_minus_release_peak_joint_success",
                    rows=subset,
                    cluster=_cluster,
                    statistic=lambda values: mean_value(
                        values,
                        "resident_minus_release_peak_mib",
                    ),
                    unit="MiB",
                ),
                inference_row(
                    study="counterbalanced_residency",
                    subgroup=sequence,
                    estimand="resident_minus_release_elapsed_joint_success",
                    rows=subset,
                    cluster=_cluster,
                    statistic=lambda values: mean_value(
                        values,
                        "resident_minus_release_elapsed_seconds",
                    ),
                    unit="seconds",
                ),
            ]
        )
    output.append(
        inference_row(
            study="counterbalanced_residency",
            estimand=(
                "release_first_minus_resident_first_completion_difference"
            ),
            rows=pairs,
            cluster=_cluster,
            statistic=lambda values: (
                _sequence_difference(values, "release_first")
                - _sequence_difference(values, "resident_first")
            ),
            unit="probability",
        )
    )
    return output


def sequence_summary(pairs: Sequence[Mapping]) -> list[dict]:
    grouped: dict[str, list[Mapping]] = defaultdict(list)
    for row in pairs:
        grouped[str(row["sequence"])].append(row)
    output = []
    for sequence, rows in sorted(grouped.items()):
        joint = [row for row in rows if int(row["both_success"])]
        completion, completion_low, completion_high, clusters = (
            cluster_bootstrap(
                rows,
                cluster=_cluster,
                statistic=completion_difference,
            )
        )
        peak, peak_low, peak_high, peak_clusters = cluster_bootstrap(
            rows,
            cluster=_cluster,
            statistic=lambda values: mean_value(
                values,
                "resident_minus_release_peak_mib",
            ),
        )
        elapsed, elapsed_low, elapsed_high, elapsed_clusters = (
            cluster_bootstrap(
                rows,
                cluster=_cluster,
                statistic=lambda values: mean_value(
                    values,
                    "resident_minus_release_elapsed_seconds",
                ),
            )
        )
        if {clusters, peak_clusters, elapsed_clusters} != {clusters}:
            raise RuntimeError(
                "sequence estimands do not use identical cluster counts"
            )
        output.append(
            {
                "sequence": sequence,
                "pairs": len(rows),
                "independent_clusters": clusters,
                "release_only_success": sum(
                    int(row["release_rescues_failure"]) for row in rows
                ),
                "resident_only_success": sum(
                    int(row["resident_rescues_failure"]) for row in rows
                ),
                "joint_successes": len(joint),
                "resident_minus_release_peak_mib_mean": peak,
                "peak_ci95_low": peak_low,
                "peak_ci95_high": peak_high,
                "resident_minus_release_elapsed_seconds_mean": elapsed,
                "elapsed_ci95_low": elapsed_low,
                "elapsed_ci95_high": elapsed_high,
                "release_minus_resident_completion": completion,
                "completion_ci95_low": completion_low,
                "completion_ci95_high": completion_high,
            }
        )
    return output


def pair_failure_counts(
    pairs: Sequence[Mapping],
) -> dict[str, Counter[str]]:
    output = {"release": Counter(), "resident": Counter()}
    for row in pairs:
        for condition in ("release", "resident"):
            if int(row[f"{condition}_success"]):
                continue
            kind = str(row.get(f"{condition}_failure_kind") or "unknown")
            output[condition][kind] += 1
    return output


def render(
    pairs: Sequence[Mapping],
    sequences: Sequence[Mapping],
    original_pairs: Sequence[Mapping],
    inference: Sequence[Mapping],
    *,
    same_node: bool = False,
) -> str:
    joint = [row for row in pairs if int(row["both_success"])]
    peaks = [
        float(row["resident_minus_release_peak_mib"])
        for row in joint
    ]
    release_only = sum(
        int(row["release_rescues_failure"]) for row in pairs
    )
    resident_only = sum(
        int(row["resident_rescues_failure"]) for row in pairs
    )
    old_release_only = sum(
        int(row["release_rescues_failure"]) for row in original_pairs
    )
    old_resident_only = sum(
        int(row["resident_rescues_failure"]) for row in original_pairs
    )
    inference_by_key = {
        (str(row["subgroup"]), str(row["estimand"])): row
        for row in inference
    }
    overall = inference_by_key[
        ("all", "release_minus_resident_process_success")
    ]
    order = inference_by_key[
        (
            "all",
            "release_first_minus_resident_first_completion_difference",
        )
    ]
    peak_interval = inference_by_key[
        ("all", "resident_minus_release_peak_joint_success")
    ]
    elapsed_interval = inference_by_key[
        ("all", "resident_minus_release_elapsed_joint_success")
    ]
    failures = pair_failure_counts(pairs)
    peak_text = (
        f"**{float(peak_interval['estimate']):.1f} MiB** "
        f"(base-configuration-cluster bootstrap 95% CI "
        f"{float(peak_interval['ci95_low']):.1f} to "
        f"{float(peak_interval['ci95_high']):.1f})"
        if math.isfinite(float(peak_interval["estimate"]))
        else "**not estimable**"
    )
    elapsed_text = (
        f"**{float(elapsed_interval['estimate']):.1f} seconds** "
        f"(95% CI {float(elapsed_interval['ci95_low']):.1f} to "
        f"{float(elapsed_interval['ci95_high']):.1f})"
        if math.isfinite(float(elapsed_interval["estimate"]))
        else "**not estimable**"
    )
    lines = [
        "# Counterbalanced release/residency replication",
        "",
        f"- Fresh matched pairs: **{len(pairs)}/36**.",
        f"- Release-only process success: **{release_only}/{len(pairs)}**.",
        f"- Resident-only process success: **{resident_only}/{len(pairs)}**.",
        f"- Joint successes: **{len(joint)}/{len(pairs)}**.",
        "- Release-minus-resident completion difference: "
        f"**{100 * float(overall['estimate']):.1f} percentage points** "
        f"(base-configuration-cluster bootstrap 95% CI "
        f"{100 * float(overall['ci95_low']):.1f} to "
        f"{100 * float(overall['ci95_high']):.1f}).",
        "- Release-first minus resident-first completion-effect contrast: "
        f"**{100 * float(order['estimate']):.1f} percentage points** "
        f"(95% CI {100 * float(order['ci95_low']):.1f} to "
        f"{100 * float(order['ci95_high']):.1f}).",
        "- Mean resident-minus-release peak among joint successes: "
        f"{peak_text}.",
        "- Mean resident-minus-release runtime among joint successes: "
        f"{elapsed_text}.",
        f"- Original replication discordance: release-only "
        f"**{old_release_only}/{len(original_pairs)}**, resident-only "
        f"**{old_resident_only}/{len(original_pairs)}**.",
        "- Scientific failure distribution: release "
        f"**{sum(failures['release'].values())}** "
        f"({', '.join(f'{kind}={count}' for kind, count in sorted(failures['release'].items())) or 'none'}); "
        "resident "
        f"**{sum(failures['resident'].values())}** "
        f"({', '.join(f'{kind}={count}' for kind, count in sorted(failures['resident'].items())) or 'none'}).",
        "",
        "| Sequence | Pairs | Release only | Resident only | Joint success | "
        "Completion difference [95% CI] | Resident-release peak [95% CI] | "
        "Resident-release runtime [95% CI] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sequences:
        peak = float(row["resident_minus_release_peak_mib_mean"])
        elapsed = float(
            row["resident_minus_release_elapsed_seconds_mean"]
        )
        peak_cell = (
            f"{peak:.1f} [{float(row['peak_ci95_low']):.1f}, "
            f"{float(row['peak_ci95_high']):.1f}] MiB"
            if math.isfinite(peak)
            else "not estimable"
        )
        elapsed_cell = (
            f"{elapsed:.1f} [{float(row['elapsed_ci95_low']):.1f}, "
            f"{float(row['elapsed_ci95_high']):.1f}] s"
            if math.isfinite(elapsed)
            else "not estimable"
        )
        lines.append(
            f"| {SEQUENCE_LABELS.get(str(row['sequence']), row['sequence'])} "
            f"| {row['pairs']} | "
            f"{row['release_only_success']} | "
            f"{row['resident_only_success']} | "
            f"{row['joint_successes']} | "
            f"{100 * float(row['release_minus_resident_completion']):.1f} "
            f"[{100 * float(row['completion_ci95_low']):.1f}, "
            f"{100 * float(row['completion_ci95_high']):.1f}] pp | "
            f"{peak_cell} | {elapsed_cell} |"
        )
    placement = (
        "The two conditions run in separate fresh processes within one Slurm "
        "allocation on the same node and same two GPUs. Period order is "
        "balanced across pairs, and an idle-memory criterion is enforced "
        "between periods. This controls node and GPU placement while retaining "
        "a fresh-process intervention."
        if same_node
        else
        "The two conditions run in separate fresh processes. Period order is "
        "balanced across pairs, so agreement between release-first and "
        "resident-first strata directly addresses the fixed-order concern "
        "without treating scheduler placement as experimentally controlled."
    )
    lines.extend(
        [
            "",
            placement,
            "Runtime and peak contrasts are restricted to joint successes. In "
            "particular, a negative resident-minus-release runtime contrast "
            "does not establish an overall resident-mode efficiency advantage "
            "because resident-mode failures are excluded from that conditional "
            "comparison.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--period1-matrix", type=Path, required=True)
    parser.add_argument("--period2-matrix", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--original-pairs",
        type=Path,
        default=Path("profiles/benchmark_v2/sleep-expansion-pairs.csv"),
    )
    parser.add_argument(
        "--trials-output",
        type=Path,
        default=Path(
            "profiles/major_revision/residency-counterbalanced-trials.csv"
        ),
    )
    parser.add_argument(
        "--pairs-output",
        type=Path,
        default=Path(
            "profiles/major_revision/residency-counterbalanced-pairs.csv"
        ),
    )
    parser.add_argument(
        "--sequence-output",
        type=Path,
        default=Path(
            "profiles/major_revision/residency-counterbalanced-sequences.csv"
        ),
    )
    parser.add_argument(
        "--inference-output",
        type=Path,
        default=Path(
            "profiles/major_revision/residency-counterbalanced-inference.csv"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("paper/major_revision_residency_counterbalanced.md"),
    )
    parser.add_argument(
        "--same-node",
        action="store_true",
        help="state that both periods used the same Slurm GPU allocation",
    )
    parser.add_argument(
        "--pair-manifest-root",
        type=Path,
        default=Path("profiles/strengthening/same_node"),
        help="same-node pair manifests used to verify selected retry provenance",
    )
    parser.add_argument(
        "--provenance-output",
        type=Path,
        default=Path("profiles/strengthening/same-node-provenance.csv"),
    )
    args = parser.parse_args()
    trials, pairs = augment(
        [args.period1_matrix, args.period2_matrix],
        args.root,
    )
    if args.same_node:
        provenance = validate_same_node_provenance(
            trials,
            args.pair_manifest_root,
        )
        if len(provenance) != len(pairs):
            raise SystemExit(
                "same-node provenance does not cover every selected pair"
            )
        write_csv(args.provenance_output, provenance)
    sequences = sequence_summary(pairs)
    inference = residency_inference(pairs)
    if {row["pairs"] for row in sequences} != {18}:
        raise SystemExit("counterbalanced sequences are not 18/18")
    write_csv(args.trials_output, trials)
    write_csv(args.pairs_output, pairs)
    write_csv(args.sequence_output, sequences)
    write_csv(args.inference_output, inference)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        render(
            pairs,
            sequences,
            read_csv(args.original_pairs),
            inference,
            same_node=args.same_node,
        ),
        encoding="utf-8",
    )
    print(f"analyzed {len(pairs)} counterbalanced pairs")


if __name__ == "__main__":
    main()
