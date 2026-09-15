#!/usr/bin/env python3
"""Dependency-free interface to the public RLVRAMBench transfer tasks."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

STATES = ("within_margin", "above_margin", "memory_failure")
TRACKS = ("model_transfer", "workload_transfer", "gpu_count_transfer", "horizon_transfer")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("Refusing to write an unspecified empty table")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")


def unique(rows, key):
    result = {}
    for row in rows:
        value = row[key]
        if not value or value in result:
            raise ValueError(f"Missing or duplicate {key}: {value}")
        result[value] = row
    return result


def load(directory, track="all"):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, record in manifest["files"].items():
        path = directory / name
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unsafe manifest path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Benchmark checksum mismatch: {name}")
    queries = read_csv(directory / "queries.csv")
    unique(queries, "query_id")
    if track != "all":
        queries = [q for q in queries if q["track"] == track]
    if not queries:
        raise ValueError("No queries selected")
    configs = unique(read_csv(directory / "configurations.csv"), "configuration_id")
    outcomes = unique(read_csv(directory / "outcomes.csv"), "configuration_id")
    if configs.keys() != outcomes.keys():
        raise ValueError("Configuration/outcome keys disagree")
    for query in queries:
        source, target = query["source_configuration_id"], query["target_configuration_id"]
        if source not in configs or target not in configs:
            raise ValueError("Unknown query configuration")
        if (query["track"] not in TRACKS or configs[source]["case_id"] == configs[target]["case_id"]
                or query["source_case_id"] != configs[source]["case_id"]
                or query["target_case_id"] != configs[target]["case_id"]):
            raise ValueError("Invalid transfer case")
        if outcomes[source]["observed_state"] not in STATES or outcomes[target]["observed_state"] not in STATES:
            raise ValueError("Transfer query requires repeated labels")
    return queries, configs, outcomes


def task_inputs(directory, track="all"):
    """Return only allowed measurements and target settings, not target outcomes.

    Different tasks can reverse source and target roles. Evaluate tasks
    independently; pooling their inputs would leak target measurements.
    """
    queries, configurations, outcomes = load(directory, track)
    tasks = defaultdict(list)
    for query in queries:
        tasks[query["task_id"]].append(query)
    return [
        {
            "task_id": task_id,
            "track": part[0]["track"],
            "source": [
                {"settings": configurations[key], "measurements": outcomes[key]}
                for key in sorted({q["source_configuration_id"] for q in part})
            ],
            "targets": [
                {"query_id": q["query_id"],
                 "matched_source_configuration_id": q["source_configuration_id"],
                 "settings": configurations[q["target_configuration_id"]]}
                for q in part
            ],
        }
        for task_id, part in sorted(tasks.items())
    ]


def baseline(directory, track="all"):
    predictions = []
    for task in task_inputs(directory, track):
        allowed = {r["settings"]["configuration_id"]: r["measurements"]["observed_state"]
                   for r in task["source"]}
        for target in task["targets"]:
            prediction = allowed[target["matched_source_configuration_id"]]
            if prediction not in STATES:
                raise ValueError("Source does not have a repeated three-state label")
            predictions.append({"query_id": target["query_id"], "predicted_state": prediction})
    return sorted(predictions, key=lambda r: r["query_id"])


def rate(numerator, denominator):
    return numerator / denominator if denominator else None


def metrics(rows):
    counts = Counter((r["predicted_state"], r["observed_state"]) for r in rows)
    approved = sum(r["predicted_state"] == STATES[0] for r in rows)
    target_counts = Counter(r["observed_state"] for r in rows)
    oom = counts[STATES[0], STATES[2]]
    above = counts[STATES[0], STATES[1]]
    rejected = sum(r["predicted_state"] != STATES[0] and r["observed_state"] == STATES[0]
                   for r in rows)
    correct = sum(r["predicted_state"] == r["observed_state"] for r in rows)
    return {
        "queries": len(rows),
        "distinct_target_configurations": len({r["target_configuration_id"] for r in rows}),
        "correct_three_state": correct,
        "three_state_accuracy": rate(correct, len(rows)),
        "approved": approved,
        "approved_memory_failure": oom,
        "approved_above_margin": above,
        "rejected_within_margin": rejected,
        **{f"target_{state}": target_counts[state] for state in STATES},
        "memory_failure_among_approved": rate(oom, approved),
        "above_margin_among_approved": rate(above, approved),
        "approval_rate_on_memory_failure_targets": rate(oom, target_counts[STATES[2]]),
        "approval_rate_on_above_margin_targets": rate(above, target_counts[STATES[1]]),
        "rejection_rate_on_within_margin_targets": rate(rejected, target_counts[STATES[0]]),
        "confusion_matrix_predicted_rows_observed_columns": {
            a: {b: counts[a, b] for b in STATES} for a in STATES},
    }


def source_cost(rows, outcomes):
    source_ids = {r["source_configuration_id"] for r in rows}
    return {
        "distinct_source_configurations": len(source_ids),
        "eligible_source_processes": sum(int(outcomes[k]["eligible_processes"]) for k in source_ids),
        "recorded_source_attempts": sum(int(outcomes[k]["recorded_attempts"]) for k in source_ids),
    }


def evaluate(directory, predictions, track="all"):
    queries, _, outcomes = load(directory, track)
    provided = unique(predictions, "query_id")
    required = {q["query_id"] for q in queries}
    if set(provided) != required:
        raise ValueError(f"Prediction coverage mismatch: {len(required - set(provided))} missing, "
                         f"{len(set(provided) - required)} unexpected queries")
    if any(p["predicted_state"] not in STATES for p in predictions):
        raise ValueError(f"predicted_state must be one of {STATES}")
    scored = [{**q, "predicted_state": provided[q["query_id"]]["predicted_state"],
               "observed_state": outcomes[q["target_configuration_id"]]["observed_state"]}
              for q in queries]
    tasks, tracks = defaultdict(list), defaultdict(list)
    for row in scored:
        tasks[row["task_id"]].append(row)
        tracks[row["track"]].append(row)
    per_task = {key: {**metrics(rows), **source_cost(rows, outcomes)}
                for key, rows in sorted(tasks.items())}
    per_track = {}
    for key, rows in sorted(tracks.items()):
        task_ids = sorted({row["task_id"] for row in rows})
        per_track[key] = {
            **metrics(rows), **source_cost(rows, outcomes), "tasks": len(task_ids),
            "macro_task_three_state_accuracy": mean(
                per_task[t]["three_state_accuracy"] for t in task_ids),
        }
    return {
        "protocol_version": "1.0",
        "selected_track": track,
        "rate_with_zero_denominator": None,
        "cost_scope": "provided donor measurements, deduplicated within each reported scope",
        "cost_warning": "Track and task costs overlap; do not add them. Extra measurements must be disclosed.",
        "evaluation_warning": "Open descriptive tasks, not a blind leaderboard or independent query samples.",
        "overall": {**metrics(scored), **source_cost(scored, outcomes)},
        "per_track": per_track,
        "per_task": per_task,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inputs", "baseline", "evaluate", "verify"))
    parser.add_argument("--data", type=Path, default=Path("benchmark"))
    parser.add_argument("--track", choices=("all", *TRACKS), default="all")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        queries, configs, _ = load(args.data, args.track)
        print(json.dumps({"status": "passed", "queries": len(queries),
                          "configurations": len(configs)}))
        return
    if args.output is None:
        parser.error("--output is required")
    if args.command == "inputs":
        save_json(args.output, task_inputs(args.data, args.track))
    elif args.command == "baseline":
        write_csv(args.output, baseline(args.data, args.track))
    else:
        if args.predictions is None:
            parser.error("--predictions is required for evaluate")
        result = evaluate(args.data, read_csv(args.predictions), args.track)
        save_json(args.output, result)
        print(json.dumps(result["per_track"], indent=2))


if __name__ == "__main__":
    main()
