"""Score the static analytical estimator on the benchmark tasks.

Information boundary (BENCHMARK.md): predictions are produced from the task
objects exported by ``benchmark.py inputs`` only. The analytic variants read
nothing but the *target settings* (plus architecture metadata). The
calibrated variant additionally reads the task's own *source* measurements
(``max_completed_run_peak_mib``) to fit one additive constant; it never sees a
target outcome and tasks are predicted independently. Target outcomes are
read only afterwards, by ``benchmark.py evaluate`` and for peak-error
reporting.
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from statistics import mean, median

from estimator_comparison.common import (MODEL_LABELS, STATES, Benchmark, to_float, write_csv, write_json)
from estimator_comparison.estimator import (BF16, DEFAULT_OVERHEAD_MIB, FP32, fit_overhead, load_architectures,
                                          predict, stage_estimates)
from estimator_comparison import plotting

TRACKS = ("model_transfer", "workload_transfer", "gpu_count_transfer", "horizon_transfer")
VARIANTS = {
    "analytic_bf16": {"param_bytes": BF16, "calibrated": False,
                      "description": "a priori: bf16 FSDP weight shard, 1,024 MiB overhead; target settings only"},
    "analytic_fp32_master": {"param_bytes": FP32, "calibrated": False,
                             "description": "a priori alternative: fp32 FSDP master-weight shard; target settings only"},
    "calibrated_bf16": {"param_bytes": BF16, "calibrated": True,
                        "description": "analytic_bf16 with the overhead constant replaced by the mean residual of the "
                                       "task's completed source configurations (source measurements only)"},
}
REFERENCE_RULES = ("source_label", "always_approve", "always_reject")
KEY_METRICS = ("queries", "three_state_accuracy", "approved", "approved_within_margin", "within_margin_recall",
               "within_margin_approval_precision", "approved_memory_failure", "approved_above_margin",
               "rejected_within_margin", "target_within_margin", "target_above_margin", "target_memory_failure")


def predict_task(task: dict, archs: dict, variant: str) -> list[dict]:
    spec = VARIANTS[variant]
    overhead, n_src = DEFAULT_OVERHEAD_MIB, 0
    if spec["calibrated"]:
        overhead, n_src = fit_overhead(task["source"], archs, param_bytes=spec["param_bytes"])
    rows = []
    for target in task["targets"]:
        settings = target["settings"]
        est = predict(settings, archs[settings["model_id"]], overhead_mib=overhead, param_bytes=spec["param_bytes"])
        rows.append({"query_id": target["query_id"], "predicted_state": est["predicted_state"],
                     "task_id": task["task_id"], "track": task["track"],
                     "target_configuration_id": settings["configuration_id"],
                     "predicted_peak_mib": est["peak_mib"], "predicted_peak_stage": est["peak_stage"],
                     **{f"pred_{k}_mib": v for k, v in est["stages"].items()},
                     "overhead_mib": overhead, "calibration_sources": n_src})
    return rows


def _bench_cli(repo_root, *args):
    cmd = [sys.executable, str(repo_root / "benchmark.py"), *map(str, args)]
    result = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"benchmark.py failed: {' '.join(cmd)}\n{result.stderr}")
    return result.stdout


def run(bench: Benchmark, out, fig_dir, repo_root, configs) -> dict:
    archs = load_architectures(bench.root)
    data = bench.root.resolve()
    inputs_dir = out / "task_inputs"
    pred_dir = out / "analytic_predictions"
    score_dir = out / "analytic_scores"
    for d in (inputs_dir, pred_dir, score_dir):
        d.mkdir(parents=True, exist_ok=True)
    all_preds = defaultdict(list)
    for track in TRACKS:
        path = inputs_dir / f"{track}.json"
        _bench_cli(repo_root, "inputs", "--data", data, "--track", track, "--output", path)
        tasks = json.loads(path.read_text())
        for variant in VARIANTS:
            rows = []
            for task in tasks:  # each task predicted independently
                rows.extend(predict_task(task, archs, variant))
            rows.sort(key=lambda r: r["query_id"])
            all_preds[variant].extend(rows)
            write_csv(pred_dir / f"{variant}_{track}.csv", rows, fieldnames=["query_id", "predicted_state"])
            write_csv(pred_dir / f"{variant}_{track}_detail.csv", rows)
            _bench_cli(repo_root, "evaluate", "--data", data, "--track", track,
                       "--predictions", pred_dir / f"{variant}_{track}.csv",
                       "--output", score_dir / f"{variant}_{track}.json")
    for variant, rows in all_preds.items():
        rows.sort(key=lambda r: r["query_id"])
        write_csv(pred_dir / f"{variant}_all.csv", rows, fieldnames=["query_id", "predicted_state"])
        _bench_cli(repo_root, "evaluate", "--data", data, "--track", "all",
                   "--predictions", pred_dir / f"{variant}_all.csv", "--output", score_dir / f"{variant}_all.json")

    reference = json.loads((bench.root / "reference_scores.json").read_text())
    comparison = []
    for track in TRACKS + ("all",):
        for method in list(VARIANTS) + list(REFERENCE_RULES):
            if method in VARIANTS:
                score = json.loads((score_dir / f"{method}_{track}.json").read_text())
                block = score["overall"]
            else:
                block = reference[method]["overall"] if track == "all" else reference[method]["per_track"][track]
            comparison.append({"track": track, "method": method, **{k: block.get(k) for k in KEY_METRICS}})
    write_csv(out / "analytic_score_comparison.csv", comparison)

    # ---- peak error on queries whose target completed (scoring only) -----------------
    q_err = []
    for variant, rows in all_preds.items():
        for r in rows:
            measured = to_float(bench.outcomes[r["target_configuration_id"]]["max_completed_run_peak_mib"])
            observed = bench.outcomes[r["target_configuration_id"]]["observed_state"]
            if measured is not None and observed != "memory_failure":
                q_err.append({"variant": variant, "track": r["track"], "query_id": r["query_id"],
                              "error_mib": r["predicted_peak_mib"] - measured})
    q_summary = {}
    for variant in VARIANTS:
        for track in TRACKS + ("all",):
            errs = [e["error_mib"] for e in q_err if e["variant"] == variant and (track == "all" or e["track"] == track)]
            if errs:
                q_summary[f"{variant}/{track}"] = _err_stats(errs)

    # ---- configuration-level predicted vs measured peak ------------------------------
    cfg_rows = []
    for c in configs:
        s = c["settings"]
        if s["model_id"] not in archs or c["max_completed_peak_mib"] is None or c["state"] == "memory_failure" \
                or c["completed_processes"] == 0:
            continue
        if c["state"] == "unresolved" and c["source"] != "replication_control":
            continue
        rec = {"source": c["source"], "study": c["study"], "configuration_id": c["configuration_id"],
               "model_family": s["model_family"], "workload": s["workload"], "gpu_count": s["gpu_count"],
               "vllm_gpu_memory_utilization": s["vllm_gpu_memory_utilization"],
               "actor_micro_batch": s["actor_micro_batch"], "parameter_offload": s["parameter_offload"],
               "free_cache_engine": s["free_cache_engine"], "observed_state": c["state"],
               "measured_peak_mib": c["max_completed_peak_mib"],
               "measured_rollout_mib": c.get("rollout_peak_mean_mib"),
               "measured_actor_update_mib": c.get("actor_update_peak_mean_mib"),
               "measured_weight_sync_mib": c.get("weight_sync_peak_mean_mib"),
               "metadata_source": archs[s["model_id"]]["metadata_source"]}
        for variant in ("analytic_bf16", "analytic_fp32_master"):
            est = predict(s, archs[s["model_id"]], param_bytes=VARIANTS[variant]["param_bytes"])
            rec[f"{variant}_peak_mib"] = est["peak_mib"]
            rec[f"{variant}_peak_stage"] = est["peak_stage"]
            rec[f"{variant}_state"] = est["predicted_state"]
            rec[f"{variant}_peak_error_mib"] = est["peak_mib"] - c["max_completed_peak_mib"]
            for stage, meas in (("generation", "measured_rollout_mib"), ("actor_update", "measured_actor_update_mib"),
                                ("weight_sync", "measured_weight_sync_mib")):
                rec[f"{variant}_{stage}_mib"] = est["stages"][stage]
                rec[f"{variant}_{stage}_error_mib"] = (est["stages"][stage] - rec[meas]) if rec[meas] is not None else None
        cfg_rows.append(rec)
    write_csv(out / "analytic_peak_error_configurations.csv", cfg_rows)
    cfg_summary = {}
    for variant in ("analytic_bf16", "analytic_fp32_master"):
        block = {"peak": _err_stats([r[f"{variant}_peak_error_mib"] for r in cfg_rows])}
        for stage in ("generation", "actor_update", "weight_sync"):
            errs = [r[f"{variant}_{stage}_error_mib"] for r in cfg_rows if r[f"{variant}_{stage}_error_mib"] is not None]
            if errs:
                block[stage] = _err_stats(errs)
        by_model = {}
        for m in sorted({r["model_family"] for r in cfg_rows}):
            by_model[m] = _err_stats([r[f"{variant}_peak_error_mib"] for r in cfg_rows if r["model_family"] == m])
        block["peak_by_model"] = by_model
        state_agree = sum(r[f"{variant}_state"] == r["observed_state"] for r in cfg_rows
                          if r["observed_state"] in STATES)
        block["completed_config_label_agreement"] = f"{state_agree}/{sum(r['observed_state'] in STATES for r in cfg_rows)}"
        cfg_summary[variant] = block

    # ---- failing configurations: does the estimator flag them? ------------------------
    fail_rows = []
    for c in configs:
        s = c["settings"]
        if c["state"] != "memory_failure" or s["model_id"] not in archs:
            continue
        rec = {"source": c["source"], "configuration_id": c["configuration_id"],
               "model_family": s["model_family"], "workload": s["workload"], "gpu_count": s["gpu_count"],
               "vllm_gpu_memory_utilization": s["vllm_gpu_memory_utilization"],
               "parameter_offload": s["parameter_offload"], "actor_micro_batch": s["actor_micro_batch"],
               "free_cache_engine": s["free_cache_engine"]}
        for variant in ("analytic_bf16", "analytic_fp32_master"):
            est = predict(s, archs[s["model_id"]], param_bytes=VARIANTS[variant]["param_bytes"])
            rec.update({f"{variant}_peak_mib": est["peak_mib"], f"{variant}_stage": est["peak_stage"],
                        f"{variant}_state": est["predicted_state"]})
        fail_rows.append(rec)
    write_csv(out / "analytic_failing_configurations_predictions.csv", fail_rows)

    summary = {
        "variants": {k: v["description"] for k, v in VARIANTS.items()},
        "score_comparison": comparison,
        "query_peak_error_mib": q_summary,
        "configuration_peak_error_mib": cfg_summary,
        "completed_configurations_scored": len(cfg_rows),
        "failing_configurations": len(fail_rows),
        "failing_predicted": {v: {st: sum(r[f"{v}_state"] == st for r in fail_rows) for st in STATES}
                              for v in ("analytic_bf16", "analytic_fp32_master")},
        "failing_predicted_by_source": {v: {src: {st: sum(r[f"{v}_state"] == st for r in fail_rows if r["source"] == src)
                                                  for st in STATES} for src in sorted({r["source"] for r in fail_rows})}
                                        for v in ("analytic_bf16", "analytic_fp32_master")},
        "calibrated_overhead_constants_mib": _overheads(all_preds["calibrated_bf16"]),
    }
    write_json(out / "analytic_summary.json", summary)
    summary["figures"] = _plot(cfg_rows, fig_dir)
    return summary


def _overheads(rows):
    vals = sorted({(r["task_id"], round(r["overhead_mib"], 3), r["calibration_sources"]) for r in rows})
    fitted = [v[1] for v in vals if v[2] > 0]
    return {"tasks": len(vals), "fitted_tasks": len(fitted), "fallback_tasks": len(vals) - len(fitted),
            "min": min(fitted) if fitted else None, "median": median(fitted) if fitted else None,
            "max": max(fitted) if fitted else None}


def _err_stats(errs):
    errs = [e for e in errs if e is not None]
    if not errs:
        return None
    return {"n": len(errs), "mean_signed_error_mib": mean(errs), "mean_absolute_error_mib": mean(abs(e) for e in errs),
            "median_absolute_error_mib": median(abs(e) for e in errs), "min_error_mib": min(errs),
            "max_error_mib": max(errs), "share_underestimated": sum(e < 0 for e in errs) / len(errs)}


def _plot(cfg_rows, fig_dir):
    plt = plotting.setup()
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    models = sorted({r["model_family"] for r in cfg_rows},
                    key=lambda m: ("qwen25_1p5b", "granite33_2b", "qwen25_3b", "phi4_mini", "qwen25_7b", "mistral").index(m))
    panels = (("peak", "measured_peak_mib", "analytic_bf16_peak_mib", "whole-run peak"),
              ("generation", "measured_rollout_mib", "analytic_bf16_generation_mib", "generation stage"),
              ("actor", "measured_actor_update_mib", "analytic_bf16_actor_update_mib", "actor-update stage"))
    for ax, (_, mk, pk, title) in zip(axes, panels):
        for i, m in enumerate(models):
            rows = [r for r in cfg_rows if r["model_family"] == m and r[mk] is not None]
            if not rows:
                continue
            ax.scatter([r[mk] / 1024 for r in rows], [r[pk] / 1024 for r in rows], s=14, color=plotting.SERIES[i],
                       edgecolor=plotting.SURFACE, linewidth=0.5, label=MODEL_LABELS.get(m, m))
        ax.plot([0, 70], [0, 70], color=plotting.MUTED, lw=0.8, ls=":")
        ax.axhline(38912 / 1024, color=plotting.STATUS["memory_failure"], lw=0.7, ls=":")
        ax.axvline(38912 / 1024, color=plotting.STATUS["memory_failure"], lw=0.7, ls=":")
        ax.set_xlim(5, 45)
        ax.set_ylim(5, 70)
        ax.set_title(f"analytic_bf16: {title}")
        ax.set_xlabel("measured (GiB)")
    axes[0].set_ylabel("predicted (GiB)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    return plotting.save(fig, fig_dir, "analytic_predicted_vs_measured")
