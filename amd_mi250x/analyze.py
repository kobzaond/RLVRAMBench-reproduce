#!/usr/bin/env python3
"""Compare MI250X outcomes with the A100 evidence and summarize each panel.

Inputs are the collector outputs (results/trials.csv, configurations.csv)
and the curated A100 tables in ../benchmark. Every comparison keeps its
denominator: unresolved configurations stay visible and never count as
agreement. Peak contrasts use completed runs only.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STATES = ("within_margin", "above_margin", "memory_failure", "unresolved")
A100_CAPACITY_MIB = 40960
A100_FAILURE_STAGE = {"rollout_init_memory": "startup admission", "rollout_oom": "generation",
                      "weight_sync_oom": "weight synchronization", "actor_update_oom": "actor update"}
AMD_FAILURE_STAGE = {"rollout_startup_admission": "startup admission", "rollout_startup_oom": "startup admission",
                     "gen_oom": "generation", "generation_wakeup_oom": "generation-state restore (wake-up)",
                     "update_weights_oom": "weight synchronization", "update_actor_oom": "actor update",
                     "update_critic_oom": "critic update", "old_log_prob_oom": "log-probability scoring"}


def read(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]) if rows else ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def short(model: str) -> str:
    return model.split("/")[-1].replace("-Instruct", "").replace("-instruct", "")


def mean(values):
    values = [float(value) for value in values if value not in ("", None)]
    return round(statistics.fmean(values), 1) if values else ""


def a100_reference() -> tuple[dict, dict, dict]:
    labels = {row["configuration_id"]: row["observed_state"] for row in read(ROOT / "benchmark/outcomes.csv")}
    peaks, stages = defaultdict(list), defaultdict(Counter)
    for row in read(ROOT / "benchmark/runs.csv"):
        if row["completed_run_peak_mib"]:
            peaks[row["configuration_id"]].append(float(row["completed_run_peak_mib"]))
        if row["observed_state"] == "memory_failure":
            stages[row["configuration_id"]][A100_FAILURE_STAGE.get(row["failure_kind"], row["failure_kind"])] += 1
    for row in read(ROOT / "benchmark/completion_replication/results/configurations.csv"):
        labels[row["configuration_id"]] = "unresolved" if row["unresolved"] == "True" else row["observed_state"]
    for row in read(ROOT / "benchmark/completion_replication/results/slots.csv"):
        if row["completed_peak_mib"]:
            peaks[row["configuration_id"]].append(float(row["completed_peak_mib"]))
        if row["known_memory_failure"] == "True":
            stages[row["configuration_id"]][A100_FAILURE_STAGE.get(row["failure_kind"], row["failure_kind"] or "unknown")] += 1
    return labels, peaks, stages


def transfer(configurations: list[dict], trials: list[dict], panel: str, output: Path) -> dict:
    labels, a100_peaks, a100_stages = a100_reference()
    by_key = defaultdict(list)
    for trial in trials:
        by_key[trial["configuration_key"]].append(trial)
    rows, matrix = [], Counter()
    for config in configurations:
        if config["panel"] != panel:
            continue
        source = config["source_configuration_id"]
        a100 = labels.get(source, "unresolved")
        mi250x = config["observed_state"]
        members = by_key[config["configuration_key"]]
        amd_stages = Counter(AMD_FAILURE_STAGE.get(item["failure_mode"], item["failure_mode"])
                             for item in members if item["state"] == "memory_failure")
        reservation = float(config["vllm_gpu_memory_utilization"])
        amd_capacity = next((int(item["device_capacity_mib"]) for item in members if item.get("device_capacity_mib")), 65520)
        amd_peak = config["max_completed_peak_mib"]
        a100_peak = max(a100_peaks.get(source, []), default="")
        rows.append({
            "source_configuration_id": source, "model": short(config["model"]), "workload": config["workload"],
            "level": config["configuration_level"], "gpu_count": config["gpu_count"],
            "actor_micro_batch": config["actor_micro_batch"], "reservation": reservation,
            "parameter_offload": config["parameter_offload"], "state_release": config["free_cache_engine"],
            "a100_label": a100, "mi250x_label": mi250x,
            "a100_failure_stages": ";".join(f"{k}:{v}" for k, v in sorted(a100_stages[source].items())),
            "mi250x_failure_stages": ";".join(f"{k}:{v}" for k, v in sorted(amd_stages.items())),
            "a100_max_completed_peak_mib": a100_peak, "mi250x_max_completed_peak_mib": amd_peak,
            # Load beyond the vLLM reservation: what the rest of the colocated
            # system adds on top of the requested generation budget.
            "a100_non_reservation_mib": round(a100_peak - reservation * A100_CAPACITY_MIB) if a100_peak != "" else "",
            "mi250x_non_reservation_mib": round(float(amd_peak) - reservation * amd_capacity) if amd_peak != "" else "",
            "mi250x_eligible": config["eligible_processes"], "mi250x_planned": config["planned_processes"],
        })
        matrix[(a100, mi250x)] += 1
    write(output / f"{panel}_labels.csv", rows)
    both = [row for row in rows if row["a100_label"] != "unresolved" and row["mi250x_label"] != "unresolved"]
    approvals = [row for row in both if row["a100_label"] == "within_margin"]
    usable = [row for row in both if row["mi250x_label"] == "within_margin"]
    load = [(row["a100_non_reservation_mib"], row["mi250x_non_reservation_mib"]) for row in rows
            if row["a100_non_reservation_mib"] != "" and row["mi250x_non_reservation_mib"] != ""]
    return {
        "configurations": len(rows), "resolved_on_both": len(both),
        "label_agreement": sum(row["a100_label"] == row["mi250x_label"] for row in both),
        "confusion_a100_to_mi250x": {f"{a}->{b}": count for (a, b), count in sorted(matrix.items())},
        "copied_a100_approvals": len(approvals),
        "copied_approvals_failing_on_mi250x": sum(row["mi250x_label"] == "memory_failure" for row in approvals),
        "copied_approvals_above_margin_on_mi250x": sum(row["mi250x_label"] == "above_margin" for row in approvals),
        "mi250x_within_margin_rejected_by_copy": sum(row["a100_label"] != "within_margin" for row in usable),
        "mi250x_within_margin": len(usable),
        "median_non_reservation_load_mib": {
            "a100": statistics.median(a for a, _ in load) if load else None,
            "mi250x": statistics.median(b for _, b in load) if load else None, "n": len(load)},
    }


def stage_masking(trials: list[dict], output: Path) -> dict:
    """Batch 8 -> 16 and reservation 0.6 -> 0.7 contrasts (batch_reservation grid)."""
    cells = defaultdict(lambda: defaultdict(list))
    for trial in trials:
        if (trial["panel"] != "hardware_transfer" or trial["state"] not in ("within_margin", "above_margin")
                or trial["configuration_level"] not in ("base", "actor_only", "reservation_only", "joint")):
            continue
        case = (short(trial["model"]), trial["workload"])
        cell = (int(trial["actor_micro_batch"]), float(trial["vllm_gpu_memory_utilization"]))
        cells[case][cell].append((float(trial["peak_mib"]), float(trial["peak_update_actor_mib"] or "nan")))
    rows = []
    for (model, workload), cell in sorted(cells.items()):
        if len(cell) < 4:
            continue
        m = {key: (statistics.fmean(v[0] for v in values), statistics.fmean(v[1] for v in values))
             for key, values in cell.items()}
        batch_whole = statistics.fmean(m[(16, r)][0] - m[(8, r)][0] for r in (0.6, 0.7))
        batch_actor = statistics.fmean(m[(16, r)][1] - m[(8, r)][1] for r in (0.6, 0.7))
        reservation_whole = statistics.fmean(m[(b, 0.7)][0] - m[(b, 0.6)][0] for b in (8, 16))
        rows.append({"model": model, "workload": workload,
                     "completed_processes": sum(len(values) for values in cell.values()),
                     "actor_stage_batch_contrast_mib": round(batch_actor),
                     "whole_run_batch_contrast_mib": round(batch_whole),
                     "whole_run_reservation_contrast_mib": round(reservation_whole)})
    write(output / "stage_masking_cases.csv", rows)
    if not rows:
        return {"cases": 0}
    return {"cases": len(rows),
            "equal_case_actor_stage_batch_contrast_mib": round(statistics.fmean(r["actor_stage_batch_contrast_mib"] for r in rows), 1),
            "equal_case_whole_run_batch_contrast_mib": round(statistics.fmean(r["whole_run_batch_contrast_mib"] for r in rows), 1),
            "equal_case_whole_run_reservation_contrast_mib": round(statistics.fmean(r["whole_run_reservation_contrast_mib"] for r in rows), 1),
            "actor_stage_batch_contrast_positive_cases": sum(r["actor_stage_batch_contrast_mib"] > 0 for r in rows)}


def panel_table(configurations: list[dict], trials: list[dict], panel: str, output: Path) -> list[dict]:
    by_key = defaultdict(list)
    for trial in trials:
        if trial["panel"] == panel:
            by_key[(trial["configuration_key"], trial["algorithm"], trial["adaptation"])].append(trial)
    rows = []
    for config in configurations:
        if config["panel"] != panel:
            continue
        members = by_key[(config["configuration_key"], config["algorithm"], config["adaptation"])]
        completed = [m for m in members if m["state"] in ("within_margin", "above_margin")]
        peak_stage = Counter()
        for member in completed:
            stage_peaks = {key.removeprefix("peak_").removesuffix("_mib"): float(value)
                           for key, value in member.items()
                           if key.startswith("peak_") and key.endswith("_mib") and key != "peak_mib" and value}
            if stage_peaks:
                peak_stage[max(stage_peaks, key=stage_peaks.get)] += 1
        rows.append({
            "model": short(config["model"]), "workload": config["workload"], "level": config["configuration_level"],
            "gpu_count": config["gpu_count"], "algorithm": config["algorithm"], "adaptation": config["adaptation"],
            "reservation": config["vllm_gpu_memory_utilization"], "parameter_offload": config["parameter_offload"],
            "steps": config["total_training_steps"], "label": config["observed_state"],
            "eligible": config["eligible_processes"], "planned": config["planned_processes"],
            "failure_modes": config["failure_modes"], "max_completed_peak_mib": config["max_completed_peak_mib"],
            "mean_actor_stage_peak_mib": mean(m["peak_update_actor_mib"] for m in completed),
            "mean_generation_peak_mib": mean(m["peak_gen_mib"] for m in completed),
            "whole_run_peak_stage": ";".join(f"{k}:{v}" for k, v in peak_stage.most_common()),
            "mean_step_seconds": mean(m["time_step_s"] for m in completed),
            "mean_update_actor_seconds": mean(m["time_update_actor_s"] for m in completed),
            "mean_update_weights_seconds": mean(m["time_update_weights_s"] for m in completed),
            "mean_elapsed_seconds": mean(m["elapsed_seconds"] for m in completed),
        })
    write(output / f"{panel}_configurations.csv", rows)
    return rows


def horizon(trials: list[dict], output: Path) -> dict:
    groups = defaultdict(dict)
    for trial in trials:
        if trial["panel"] == "horizon":
            groups[(trial["source_configuration_id"], trial["training_seed"])][int(trial["total_training_steps"])] = trial
    rows = []
    for (source, seed), pair in sorted(groups.items()):
        short_run, long_run = pair.get(1), pair.get(40)
        if not short_run or not long_run:
            continue
        rows.append({"source_configuration_id": source, "model": short(short_run["model"]),
                     "workload": short_run["workload"], "seed": seed,
                     "screen_state": short_run["state"], "long_state": long_run["state"],
                     "screen_peak_mib": short_run.get("peak_mib", ""), "long_peak_mib": long_run.get("peak_mib", ""),
                     "margin_limit_mib": short_run.get("margin_limit_mib", "")})
    write(output / "horizon_pairs.csv", rows)
    resolved = [r for r in rows if r["screen_state"] not in ("missing", "other_failure")
                and r["long_state"] not in ("missing", "other_failure")]
    growth = [float(r["long_peak_mib"]) - float(r["screen_peak_mib"]) for r in resolved
              if r["screen_state"] != "memory_failure" and r["long_state"] != "memory_failure"]
    return {"pairs": len(rows), "resolved_pairs": len(resolved),
            "agreeing_pairs": sum(r["screen_state"] == r["long_state"] for r in resolved),
            "disagreements": [f"{r['model']}/{r['workload']}/s{r['seed']}: {r['screen_state']}->{r['long_state']}"
                              for r in resolved if r["screen_state"] != r["long_state"]],
            "mean_long_minus_screen_peak_mib": round(statistics.fmean(growth), 1) if growth else None,
            "max_long_minus_screen_peak_mib": max(growth) if growth else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=HERE / "results")
    args = parser.parse_args()
    trials = read(args.results / "trials.csv")
    configurations = read(args.results / "configurations.csv")
    analysis = args.results / "analysis"
    summary = {
        "trial_states": dict(Counter((t["panel"], t["state"]) for t in trials).most_common()),
        "failure_modes": dict(Counter(t["failure_mode"] for t in trials if t["state"] == "memory_failure")),
        "hardware_transfer": transfer(configurations, trials, "hardware_transfer", analysis),
        "larger_models": transfer(configurations, trials, "larger_models", analysis),
        "stage_masking": stage_masking(trials, analysis),
        "horizon": horizon(trials, analysis),
    }
    summary["trial_states"] = {f"{p}/{s}": n for (p, s), n in summary["trial_states"].items()}
    for panel in ("reservation_sweep", "algorithm_adaptation", "larger_models", "horizon"):
        panel_table(configurations, trials, panel, analysis)
    (analysis / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
