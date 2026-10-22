"""Audit and summarize the frozen Electronics follow-ups without replacing slots.

Completed-run costs use durable optimizer-step records, not phase-marker
durations. Missing secondary measurements never become zero-valued observations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

import numpy as np

from memory_tuner.collect_electronics import collect
from memory_tuner.electronics_experiment import ROOT, require
from memory_tuner.electronics_failure_evidence import diagnose, startup_budget_diagnostics

PANELS = ("adaptation", "reservation", "release", "timing")
COMPLETED = {"within_margin", "above_margin"}
TERMINAL = COMPLETED | {"memory_failure"}
TIMINGS = ("gen", "old_log_prob", "ref", "adv", "update_actor", "update_weights", "step")


def write_rows(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()} for row in rows)


def metrics(record):
    text = record["console_line"]
    require(hashlib.sha256(text.encode()).hexdigest() == record["console_line_sha256"],
            "Durable metric text hash differs")
    values = {}
    for part in text.split(" - "):
        key, value = part.rsplit(":", 1)
        require(key not in values, "Duplicate metric")
        values[key] = float(value)
    require(values["step"] == record["step"] == values["training/global_step"],
            "Metric step identity differs")
    return values


def performance(attempt, row):
    records = [json.loads(p.read_text()) for p in (attempt/"trainer-metrics").glob("step-*.json")]
    records.sort(key=lambda r: r["step"])
    requested = int(row["total_training_steps"])
    require([r["step"] for r in records] == list(range(1, requested+1)),
            "Performance comparison requires all requested steps")
    parsed = [metrics(r) for r in records]
    for values in parsed:
        require(all(key in values and math.isfinite(values[key])
                    for key in ("actor/loss", "actor/pg_loss", "actor/grad_norm")),
                "Completed-work cost requires finite actor loss and gradient norm")
    steady = parsed[2:] if requested > 2 else []
    result = dict(completed_steps=requested, warmup_steps=min(2, requested),
                  steady_steps=len(steady), finite_actor_optimization_metrics=True,
                  actor_gradient_norm_min=min(v["actor/grad_norm"] for v in parsed),
                  actor_gradient_norm_max=max(v["actor/grad_norm"] for v in parsed),
                  performance_scope="Completed run; step 1–2 separate from step 3 onward")
    for name, selected in (("all", parsed), ("warmup", parsed[:2]), ("steady", steady)):
        if not selected:
            continue
        for key in TIMINGS:
            present = [v["timing_s/"+key] for v in selected if "timing_s/"+key in v]
            if len(present) == len(selected):
                require(all(math.isfinite(v) and v >= 0 for v in present), "Invalid stage timing")
                result[f"{name}_{key}_seconds"] = mean(present)
        seconds = [v["timing_s/step"] for v in selected]
        require(all(math.isfinite(v) and v > 0 for v in seconds), "Invalid step duration")
        tokens = [v["perf/total_num_tokens"] for v in selected]
        require(all(math.isfinite(v) and v > 0 for v in tokens), "Invalid token count")
        result[f"{name}_step_median_seconds"] = median(seconds)
        result[f"{name}_step_seconds_sum"] = sum(seconds)
        result[f"{name}_total_tokens"] = sum(tokens)
        result[f"{name}_aggregate_tokens_per_second"] = sum(tokens)/sum(seconds)
        result[f"{name}_task_gpu_seconds"] = sum(seconds)*int(row["gpu_count"])
    paths = list(attempt.glob("response-lengths-*.jsonl"))
    require(len(paths) == 1, "Missing or ambiguous response-length trace")
    lengths = [json.loads(line) for line in paths[0].read_text().splitlines()]
    require(len(lengths) == requested, "Response-length records differ from completed steps")
    for index, (length, record, values) in enumerate(zip(lengths, records, parsed)):
        require(length["pid"] == record["pid"], "Response-length producer differs")
        require(length["timestamp_ns"] < record["timestamp_ns"], "Lengths recorded after step completion")
        if index:
            require(length["timestamp_ns"] > records[index-1]["timestamp_ns"],
                    "Length trace cannot be matched to durable step order")
        prompt, response = length["prompt_lengths"], length["response_lengths"]
        require(len(prompt) == len(response) and len(prompt) > 0, "Invalid lengths")
        require(all(isinstance(v, int) and v >= 0 for v in prompt+response), "Invalid length value")
        require(sum(prompt)+sum(response) == values["perf/total_num_tokens"],
                "Realized token trace differs from logged token count")
        require(abs(mean(response)-values["response_length/mean"]) < 1e-3,
                "Realized response mean differs")
    for name, selected in (("all", lengths), ("steady", lengths[2:])):
        if not selected:
            continue
        result[f"{name}_sequences_processed"] = sum(len(r["response_lengths"]) for r in selected)
        for kind in ("prompt", "response"):
            values = [v for r in selected for v in r[kind+"_lengths"]]
            result[f"{name}_{kind}_tokens_mean"] = mean(values)
            result[f"{name}_{kind}_tokens_p95"] = float(np.quantile(values, .95))
            result[f"{name}_{kind}_tokens_max"] = max(values)
    return result


def host_memory(attempt, row):
    path = attempt/"host-memory-metadata.json"
    if not path.exists():
        return dict(host_memory_status="not_instrumented")
    metadata = json.loads(path.read_text())
    require(str(metadata["job_id"]) == str(row["job_id"]), "Host memory belongs to another job")
    if metadata["error"] or not metadata["samples"]:
        return dict(host_memory_status="unavailable", host_memory_error=metadata["error"])
    samples = [json.loads(line) for line in (attempt/"host-memory.jsonl").read_text().splitlines()]
    require(len(samples) == metadata["samples"], "Host sample count differs")
    require(all(a["timestamp_ns"] < b["timestamp_ns"] for a, b in zip(samples, samples[1:])),
            "Host memory clock is not monotonic")
    result = dict(host_memory_status="observed", host_memory_scope=metadata["scope"],
                  host_memory_cgroup=metadata["cgroup"])
    for field in ("current", "anonymous", "file"):
        values = [r[field+"_bytes"] for r in samples]
        require(all(isinstance(v, int) and v >= 0 for v in values), "Invalid host memory")
        result[f"host_{field}_baseline_gib"] = values[0]/2**30
        result[f"host_{field}_peak_gib"] = max(values)/2**30
        result[f"host_{field}_increase_gib"] = (max(values)-values[0])/2**30
    return result


def enrich(row):
    if row["status"] not in TERMINAL:
        return row
    row = dict(row)
    attempt = Path(row["raw_path"])
    job = row["job_id"]
    row["secondary_errors"] = []
    if row["status"] == "memory_failure":
        samples = list(csv.reader((attempt/f"phase-memory-{job}.csv").open()))
        last_phase = samples[-1][2]
        text = (attempt/f"training-{job}.log").read_text(errors="replace")
        row.update(diagnose(text, last_phase), terminal_phase=last_phase)
        budgets = startup_budget_diagnostics(text)
        if budgets:
            row["startup_budget_messages"] = budgets
    else:
        row["diagnosis"] = "completed"
        try:
            row.update(performance(attempt, row))
        except (KeyError, ValueError, TypeError, OSError) as error:
            row["secondary_errors"].append("performance: "+str(error))
    try:
        row.update(host_memory(attempt, row))
    except (KeyError, ValueError, TypeError, OSError) as error:
        row.update(host_memory_status="invalid")
        row["secondary_errors"].append("host_memory: "+str(error))
    # Phase markers identify the host-side operation, not allocation ownership.
    row["actor_stage_peak_mib"] = row.get("stage_peaks_mib", {}).get("actor_update")
    if row["status"] in COMPLETED:
        row["global_peak_stages"] = sorted(
            stage for stage, value in row.get("stage_peaks_mib", {}).items()
            if value == row["completed_peak_mib"])
        row["actor_reaches_global_peak"] = "actor_update" in row["global_peak_stages"]
    return row


def summarize_cases(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["configuration_id"]].append(row)
    cases = []
    for config, group in sorted(groups.items()):
        example = group[0]
        item = {k: example[k] for k in ("model", "dataset", "lora_rank", "use_kl_loss",
                                        "parameter_offload", "vllm_gpu_memory_utilization",
                                        "free_cache_engine", "gpu_count", "condition")}
        states = Counter(r["status"] for r in group)
        item.update(configuration_id=config, planned=len(group), states=dict(states),
                    complete_configuration=sum(states[s] for s in TERMINAL) == len(group))
        if item["complete_configuration"]:
            item["configuration_label"] = ("memory_failure" if states["memory_failure"] else
                                           "above_margin" if states["above_margin"] else "within_margin")
        else:
            item["configuration_label"] = "unresolved"
        for key in ("completed_peak_mib", "actor_stage_peak_mib", "invocation_seconds",
                    "task_gpu_seconds", "steady_step_seconds", "steady_aggregate_tokens_per_second",
                    "host_current_peak_gib", "host_anonymous_peak_gib", "host_file_peak_gib"):
            eligible = [r[key] for r in group if r["status"] in COMPLETED and r.get(key) is not None]
            item[key+"_n"] = len(eligible)
            if eligible:
                item[key+"_mean"] = mean(eligible)
                item[key+"_min"], item[key+"_max"] = min(eligible), max(eligible)
        cases.append(item)
    return cases


def matched_contrasts(rows, panel):
    comparisons = []
    blocks = defaultdict(list)
    for row in rows:
        blocks[row["block_id"]].append(row)
    for block, group in sorted(blocks.items()):
        if panel == "adaptation":
            pairings = [(a, b, "full_minus_lora") for a in group for b in group
                        if a["lora_rank"] == "64" and b["lora_rank"] == "0"
                        and a["parameter_offload"] == b["parameter_offload"]
                        and a["vllm_gpu_memory_utilization"] == b["vllm_gpu_memory_utilization"]]
        elif panel == "release":
            pairings = [(a, b, "resident_minus_release") for a in group for b in group
                        if a["free_cache_engine"] == "True" and b["free_cache_engine"] == "False"
                        and a["vllm_gpu_memory_utilization"] == b["vllm_gpu_memory_utilization"]]
        elif panel == "timing":
            anchor = next(r for r in group if r["condition"] == "off1-r45-release1")
            pairings = [(anchor, b, b["condition"]+"_minus_anchor") for b in group if b != anchor]
        else:
            pairings = [(a, b, "offload_minus_resident") for a in group for b in group
                        if a["parameter_offload"] == "False" and b["parameter_offload"] == "True"
                        and a["vllm_gpu_memory_utilization"] == b["vllm_gpu_memory_utilization"]]
        for a, b, contrast in pairings:
            item = dict(block_id=block, model=a["model"], dataset=a["dataset"],
                        training_seed=a["training_seed"], contrast=contrast,
                        reservation=a["vllm_gpu_memory_utilization"], use_kl_loss=a["use_kl_loss"],
                        parameter_offload=a["parameter_offload"], reference_id=a["experiment_id"],
                        alternative_id=b["experiment_id"], reference_state=a["status"], alternative_state=b["status"],
                        resolved=a["status"] in TERMINAL and b["status"] in TERMINAL,
                        both_completed=a["status"] in COMPLETED and b["status"] in COMPLETED)
            if item["both_completed"]:
                for key in ("completed_peak_mib", "actor_stage_peak_mib", "invocation_seconds",
                            "steady_step_seconds", "steady_aggregate_tokens_per_second",
                            "steady_update_actor_seconds", "steady_update_weights_seconds",
                            "steady_gen_seconds", "steady_response_tokens_mean",
                            "host_current_peak_gib", "host_anonymous_peak_gib", "host_file_peak_gib"):
                    if a.get(key) is not None and b.get(key) is not None:
                        item[key+"_difference"] = b[key]-a[key]
                        if a[key] > 0:
                            item[key+"_ratio"] = b[key]/a[key]
            comparisons.append(item)
    return comparisons


def adaptation_offload_contrasts(rows):
    """Compare offload within an adaptation mode, never across rank or KL."""
    strata = defaultdict(list)
    for row in rows:
        strata[(row["lora_rank"], row["use_kl_loss"])].append(row)
    comparisons = []
    for (rank, kl), group in sorted(strata.items()):
        for contrast in matched_contrasts(group, "reservation"):
            require(contrast["use_kl_loss"] == kl, "Mixed objectives in offload contrast")
            comparisons.append(dict(contrast, lora_rank=rank))
    return comparisons


def score_predictions(directory, cases):
    predictions = json.loads((directory/"predictions.json").read_text())
    targets = {r["configuration_id"]: r for r in cases}
    records = []
    for prediction in predictions:
        case = targets[prediction["configuration_id"]]
        record = dict(prediction, actual=case["configuration_label"],
                      model=case["model"], dataset=case["dataset"],
                      actor_offload=case["parameter_offload"],
                      reservation=case["vllm_gpu_memory_utilization"],
                      reservation_group="endpoint" if float(case["vllm_gpu_memory_utilization"]) in (.45,.70)
                      else "intermediate")
        supported = prediction["prediction_type"] != "unsupported"
        resolved = case["configuration_label"] in TERMINAL
        record.update(supported=supported, resolved_target=resolved, scored=supported and resolved)
        if record["scored"]:
            approved = prediction["approved"]
            state = case["configuration_label"]
            record.update(within_recovered=approved and state=="within_margin",
                          within_rejected=not approved and state=="within_margin",
                          failure_approved=approved and state=="memory_failure",
                          above_approved=approved and state=="above_margin")
            if prediction["predicted_state"] is not None:
                record["state_correct"] = prediction["predicted_state"] == state
            if state in COMPLETED and prediction["predicted_peak_mib"] is not None:
                target = case["completed_peak_mib_max"]
                record.update(numerical_target_mib=target,
                              signed_error_mib=prediction["predicted_peak_mib"]-target,
                              absolute_error_mib=abs(prediction["predicted_peak_mib"]-target))
        records.append(record)
    summaries = []
    strata = [("all", "all", "all")] + sorted({
        (r["model"], r["dataset"], r["actor_offload"]) for r in records})
    for scope, stratum in ((scope, stratum) for scope in ("all", "endpoint", "intermediate")
                          for stratum in strata):
        for method in dict.fromkeys(p["method"] for p in predictions):
            group = [r for r in records if r["method"]==method
                     and (scope=="all" or r["reservation_group"]==scope)
                     and (stratum == ("all", "all", "all") or
                          (r["model"], r["dataset"], r["actor_offload"]) == stratum)]
            eligible = [r for r in group if r["scored"]]
            summary = dict(scope=scope, method=method, planned=len(group),
                           model=stratum[0], dataset=stratum[1], actor_offload=stratum[2],
                           supported=sum(r["supported"] for r in group), scored=len(eligible),
                           unresolved_targets=sum(not r["resolved_target"] for r in group),
                           actual_within=sum(r["actual"]=="within_margin" for r in eligible),
                           actual_failure=sum(r["actual"]=="memory_failure" for r in eligible),
                           actual_above=sum(r["actual"]=="above_margin" for r in eligible))
            for key in ("within_recovered", "within_rejected", "failure_approved", "above_approved"):
                summary[key] = sum(r[key] for r in eligible)
            summary["within_recall"] = (summary["within_recovered"]/summary["actual_within"]
                                       if summary["actual_within"] else None)
            errors = [r["absolute_error_mib"] for r in eligible if "absolute_error_mib" in r]
            summary.update(numerical_targets=len(errors), mae_mib=mean(errors) if errors else None)
            summaries.append(summary)
    return records, summaries


def analyze_panel(root, panel, output_root, results):
    directory = root/"benchmark/electronics_revision_20261016"/panel
    receipt = json.loads((directory/"submission.json").read_text())
    summary, rows = collect(directory, output_root, receipt["protocol_sha256"], receipt["source_commit"])
    rows = [enrich(row) for row in rows]
    cases = summarize_cases(rows)
    contrasts = matched_contrasts(rows, panel)
    protocol = json.loads((directory/"protocol.json").read_text())
    blocks = output_root/protocol["group"]/"blocks"
    unfinished = [block for block in sorted({r["block_id"] for r in rows})
                  if not (blocks/block/"summary.json").exists()
                  and not (blocks/block/"setup-error.json").exists()]
    summary.update(panel=panel, planned_configurations=len(cases),
                   configuration_states=dict(Counter(r["configuration_label"] for r in cases)),
                   planned_contrasts=len(contrasts), resolved_contrasts=sum(r["resolved"] for r in contrasts),
                   completed_contrasts=sum(r["both_completed"] for r in contrasts),
                   secondary_error_runs=sum(bool(r.get("secondary_errors")) for r in rows),
                   diagnoses=dict(Counter(r.get("diagnosis", "unavailable") for r in rows)),
                   all_slots_terminal=all(r["status"] in TERMINAL for r in rows),
                   all_blocks_finished=not unfinished, unfinished_blocks=unfinished)
    target = results/panel
    target.mkdir(parents=True, exist_ok=True)
    (target/"outcomes.json").write_text(json.dumps(rows, indent=2)+"\n")
    (target/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    write_rows(target/"outcomes.csv", rows)
    write_rows(target/"cases.csv", cases)
    write_rows(target/"contrasts.csv", contrasts)
    if panel == "adaptation":
        write_rows(target/"offload_contrasts.csv", adaptation_offload_contrasts(rows))
    if (directory/"predictions.json").exists():
        predictions, scores = score_predictions(directory, cases)
        write_rows(target/"predictions.csv", predictions)
        write_rows(target/"prediction_scores.csv", scores)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--panel", choices=PANELS, action="append")
    parser.add_argument("--output-root", type=Path, default=ROOT/"output")
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    results = args.results or args.root/"profiles/electronics_revision"
    for panel in args.panel or PANELS:
        print(json.dumps(analyze_panel(args.root, panel, args.output_root, results), indent=2), flush=True)


if __name__ == "__main__":
    main()
