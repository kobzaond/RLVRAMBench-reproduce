#!/usr/bin/env python3
"""Generate a failure-inclusive RLVRAMBench taxonomy and report."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def phase_labels(path_value: object) -> list[str]:
    if not path_value:
        return []
    path = Path(str(path_value))
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return []
    if "phase" in rows[0]:
        phase_index = rows[0].index("phase")
        rows = rows[1:]
    else:
        phase_index = 2
    labels = []
    for row in rows:
        if len(row) <= phase_index:
            continue
        phase = str(row[phase_index]).strip()
        if phase:
            labels.append(phase)
    return labels


def last_observed_phase(path_value: object) -> str:
    labels = phase_labels(path_value)
    return labels[-1] if labels else ""


def failure_label(
    row: dict,
    annotations: dict[str, str] | None = None,
    last_phase: str = "",
) -> str:
    if int(row.get("success", 0)):
        return "success"
    value = str(row.get("failure_kind", "")).strip()
    if value:
        return value
    annotation = (annotations or {}).get(row.get("experiment_id", ""), "")
    if annotation:
        return annotation
    if last_phase == "initializing":
        return "initialization_phase_failure_unclassified"
    return "scientific_failure_unspecified"


def load_failure_annotations(paths: list[Path]) -> dict[str, str]:
    annotations = {}
    for path in paths:
        for row in read_csv(path):
            experiment_id = row.get("experiment_id", "")
            label = row.get("failure_kind", "")
            if experiment_id and label:
                annotations[experiment_id] = label
    return annotations


def summarize(
    corpus: list[dict],
    annotations: dict[str, str] | None = None,
) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in corpus:
        last_phase = last_observed_phase(row.get("phase_memory_csv"))
        label = failure_label(row, annotations, last_phase)
        if label == "success":
            last_phase = ""
        key = (
            row.get("model_tag", ""),
            row.get("dataset", ""),
            row.get("algorithm", ""),
            row.get("gpu_count", ""),
            label,
            last_phase,
        )
        grouped[key].append(row)

    total = len(corpus)
    rows = []
    for key, members in sorted(grouped.items()):
        model, dataset, algorithm, gpu_count, label, last_phase = key
        rows.append(
            {
                "model_tag": model,
                "dataset": dataset,
                "algorithm": algorithm,
                "gpu_count": gpu_count,
                "outcome": label,
                "last_observed_phase": last_phase,
                "trials": len(members),
                "share_of_corpus": len(members) / total if total else 0.0,
                "distinct_cases": len(
                    {member.get("case_id", "") for member in members}
                ),
                "experiment_ids_json": json.dumps(
                    sorted(member["experiment_id"] for member in members)
                ),
            }
        )
    return rows


def render(
    corpus: list[dict],
    manifest: list[dict],
    summary_rows: list[dict],
    annotations: dict[str, str] | None = None,
) -> str:
    outcomes = Counter()
    for row in corpus:
        last_phase = last_observed_phase(row.get("phase_memory_csv"))
        outcomes[failure_label(row, annotations, last_phase)] += 1
    excluded = Counter(
        row.get("validity_reason", "") or "unspecified_exclusion"
        for row in manifest
        if str(row.get("scientific_valid", "1")) == "0"
    )
    scientific_failures = sum(
        count for label, count in outcomes.items() if label != "success"
    )
    lines = [
        "# RLVRAMBench failure taxonomy",
        "",
        "Scientific failures remain benchmark outcomes. Infrastructure, data, "
        "or instrumentation failures are excluded only through the explicit "
        "artifact manifest and are reported separately below.",
        "",
        f"- Valid scientific trials: {len(corpus)}",
        f"- Successful trials: {outcomes['success']}",
        f"- Scientific failures: {scientific_failures}",
        "",
        "## Scientific outcomes",
        "",
        "| Outcome | Trials | Share |",
        "|---|---:|---:|",
    ]
    for label, count in sorted(
        outcomes.items(), key=lambda item: (-item[1], item[0])
    ):
        lines.append(
            f"| {label} | {count} | "
            f"{100 * count / len(corpus) if corpus else 0:.1f}% |"
        )
    lines.extend(["", "## Failure phase/context breakdown", ""])
    failure_rows = [
        row for row in summary_rows if row["outcome"] != "success"
    ]
    if failure_rows:
        lines.extend(
            [
                "| Model | Workload | Algorithm | GPUs | Failure | Last phase | Trials |",
                "|---|---|---|---:|---|---|---:|",
            ]
        )
        for row in failure_rows:
            lines.append(
                f"| {row['model_tag']} | {row['dataset']} | "
                f"{row['algorithm']} | {row['gpu_count']} | "
                f"{row['outcome']} | "
                f"{row['last_observed_phase'] or 'unavailable'} | "
                f"{row['trials']} |"
            )
    else:
        lines.append("- No scientific failures in the current corpus.")
    lines.extend(["", "## Excluded non-scientific artifacts", ""])
    if excluded:
        for reason, count in sorted(excluded.items()):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "The last observed phase is diagnostic context, not necessarily "
            "the causal allocation site; log classification and allocator "
            "traces remain the primary evidence for the failure label.",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
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
        "--output",
        type=Path,
        default=Path("profiles/benchmark/failure-taxonomy.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("paper/failure_taxonomy.md"),
    )
    parser.add_argument(
        "--legacy-annotations",
        type=Path,
        action="append",
        default=[Path("profiles/screening/phase-screening.csv")],
    )
    args = parser.parse_args()
    corpus = read_csv(args.corpus)
    if not corpus:
        raise SystemExit("empty benchmark corpus")
    manifest = read_csv(args.manifest)
    annotations = load_failure_annotations(args.legacy_annotations)
    rows = summarize(corpus, annotations)
    write_csv(args.output, rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render(corpus, manifest, rows, annotations))
    print(f"wrote {len(rows)} failure-taxonomy groups")


if __name__ == "__main__":
    main()
