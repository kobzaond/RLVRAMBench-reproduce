"""Unified configuration-level view across the benchmark and the later panels.

Each row: explicit settings, repeated outcome label, completed peaks and,
where available, stage peaks / timings. Panels stay distinguishable through
``source``; nothing is pooled into a single experiment count.
"""
from __future__ import annotations

import json
from collections import defaultdict
from statistics import mean

from estimator_comparison.common import (MARGIN_LIMIT_MIB, Benchmark, family_from_model, label_from_peak,
                                       normalise_workload, to_bool, to_float)

SETTING_KEYS = ("model_family", "model_id", "workload", "gpu_count", "rollout_tp_size", "actor_micro_batch",
                "rollout_logprob_micro_batch", "ref_logprob_micro_batch", "vllm_gpu_memory_utilization",
                "parameter_offload", "optimizer_offload", "free_cache_engine", "max_prompt_length",
                "max_response_length", "max_model_len", "max_num_seqs", "rollout_n", "train_batch_size",
                "total_training_steps", "device_capacity_mib", "margin_limit_mib")


def _norm_settings(raw: dict) -> dict:
    s = {k: raw.get(k, "") for k in SETTING_KEYS}
    s["model_id"] = raw.get("model_id") or raw.get("model", "")
    s["model_family"] = family_from_model(s["model_id"] or raw.get("model_family", ""))
    s["workload"] = normalise_workload(raw.get("workload_id") or raw.get("dataset", ""))
    s["device_capacity_mib"] = s["device_capacity_mib"] or "40960"
    s["margin_limit_mib"] = s["margin_limit_mib"] or "38912"
    for k in ("vllm_gpu_memory_utilization",):
        if s[k] != "":
            s[k] = str(float(s[k]))
    return s


def _stage_means(rows, key):
    vals = [to_float(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    return mean(vals) if vals else None


def collect(bench: Benchmark) -> list[dict]:
    out = []
    stage = bench.stage_peaks()
    runs_by_cfg = defaultdict(list)
    for r in bench.runs:
        runs_by_cfg[r["configuration_id"]].append(r)
    for cid, cfg in sorted(bench.configurations.items()):
        o = bench.outcomes[cid]
        runs = [r for r in runs_by_cfg[cid] if r["completed_requested_work"] == "1"]
        st = {}
        for name in ("rollout", "actor_update", "weight_sync", "reference_logprob", "initializing"):
            vals = [stage[(r["run_id"], name)][0] for r in runs
                    if (r["run_id"], name) in stage and stage[(r["run_id"], name)][1] == "completed_run"]
            st[f"{name}_peak_mean_mib"] = mean(vals) if vals else None
        el = [to_float(r["elapsed_seconds"]) for r in runs if to_float(r["elapsed_seconds"]) is not None]
        out.append({"source": "benchmark", "study": cfg["study"], "configuration_id": cid,
                    "configuration_level": cfg["configuration_level"], "condition": cfg["condition"],
                    "settings": _norm_settings(cfg), "state": o["observed_state"],
                    "max_completed_peak_mib": to_float(o["max_completed_run_peak_mib"]),
                    "mean_completed_peak_mib": to_float(o["mean_completed_run_peak_mib"]),
                    "eligible_processes": int(o["eligible_processes"]),
                    "completed_processes": int(o["completed_eligible_processes"]),
                    "mean_elapsed_seconds": mean(el) if el else None,
                    "used_in_transfer_tasks": None, **st})

    # replication control: one configuration per (model, workload, condition)
    matrix = {r["experiment_id"]: r for r in bench.optional_csv("replication_control/replication_control_matrix.csv")}
    groups = defaultdict(list)
    for t in bench.optional_csv("replication_control/results/replication_control_trials.csv"):
        groups[(t["model_family"], t["dataset"], t["condition"])].append(t)
    for key in sorted(groups):
        trials = groups[key]
        settings = _norm_settings(matrix[trials[0]["experiment_id"]])
        elig = [t for t in trials if to_bool(t["scientific_eligible"])]
        peaks = [to_float(t["whole_run_peak_mib"]) for t in elig]
        if len(elig) == 3 and all(p is not None for p in peaks):
            state = "within_margin" if max(peaks) <= MARGIN_LIMIT_MIB else "above_margin"
        else:
            state = "unresolved"  # fewer than three eligible seeds; known peaks are still reported
        out.append({"source": "replication_control", "study": "replication_control",
                    "configuration_id": f"rc-{key[0]}-{key[1]}-{key[2]}", "configuration_level": key[2],
                    "condition": key[2], "settings": settings, "state": state,
                    "max_completed_peak_mib": max(peaks) if peaks else None,
                    "mean_completed_peak_mib": mean(peaks) if peaks else None,
                    "eligible_processes": len(elig), "completed_processes": len(elig),
                    "mean_elapsed_seconds": _stage_means(elig, "invocation_gpu_seconds") / 2.0 if elig else None,
                    "rollout_peak_mean_mib": _stage_means(elig, "rollout_peak_mib"),
                    "actor_update_peak_mean_mib": _stage_means(elig, "actor_update_peak_mib"),
                    "weight_sync_peak_mean_mib": _stage_means(elig, "weight_sync_peak_mib")})

    # completion replication
    cr_matrix = {}
    for r in bench.optional_csv("completion_replication/matrix.csv"):
        cr_matrix.setdefault(r["configuration_id"], r)
    for c in bench.optional_csv("completion_replication/results/configurations.csv"):
        settings = _norm_settings(cr_matrix[c["configuration_id"]])
        out.append({"source": "completion_replication", "study": "completion_replication",
                    "configuration_id": c["configuration_id"], "configuration_level": c["configuration_level"],
                    "condition": "", "settings": settings, "state": c["observed_state"],
                    "max_completed_peak_mib": to_float(c["completed_peak_mib"]) or to_float(
                        c["maximum_known_completed_peak_mib"]),
                    "mean_completed_peak_mib": None, "eligible_processes": int(c["eligible_seed_count"] or 0),
                    "completed_processes": int(c["completed_processes"] or 0), "mean_elapsed_seconds": None})

    for panel, rel in (("estimation", "estimation/results/configurations.csv"),
                       ("host_capacity", "host_capacity/results/configurations.csv")):
        for c in bench.optional_csv(rel):
            settings = _norm_settings(json.loads(c["settings"]))
            out.append({"source": panel, "study": panel, "configuration_id": c["configuration_id"],
                        "configuration_level": json.loads(c["settings"]).get("configuration_level", ""),
                        "condition": "", "settings": settings, "state": c["state"],
                        "max_completed_peak_mib": to_float(c["completed_peak_mib"]) or to_float(
                            c["maximum_known_completed_peak_mib"]),
                        "mean_completed_peak_mib": None, "eligible_processes": int(c["eligible_processes"] or 0),
                        "completed_processes": int(c["completed_processes"] or 0), "mean_elapsed_seconds": None})

    dmatrix = {}
    for r in bench.optional_csv("decision/matrix.csv"):
        dmatrix.setdefault(r["candidate_id"], r)
    for c in bench.optional_csv("decision/results/candidate_results.csv"):
        settings = _norm_settings(dmatrix[c["candidate_id"]])
        out.append({"source": "decision", "study": "decision", "configuration_id": c["candidate_id"],
                    "configuration_level": c["configuration_level"], "condition": "", "settings": settings,
                    "state": c["evaluation_state"],
                    "max_completed_peak_mib": to_float(c["evaluation_max_completed_peak_mib"]),
                    "mean_completed_peak_mib": None, "eligible_processes": 3,
                    "completed_processes": None, "mean_elapsed_seconds": None})
    return out


__all__ = ["collect", "label_from_peak"]
