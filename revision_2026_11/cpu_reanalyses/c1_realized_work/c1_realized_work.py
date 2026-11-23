#!/usr/bin/env python3
"""C1 (reviewer R1.3): realized work behind the micro-batch contrast.

CPU-only re-analysis of existing measurements. For the same-device
replication (144 invocations) and the original factorial experiment (144
processes) it tabulates prompt/response lengths, cap reaching and token
volume per process, sets the within-block token difference between actor
micro-batches 16 and 8 next to the actor-stage contrast, and fits a small
descriptive within-case adjustment.

Inputs are read as data only:
  * repository CSV/JSON tables (paths below, relative to --repo);
  * archive members extracted from the released b0018d0 archive
    (--extract): replication ``trainer-metrics/step-1.json`` records and the
    retained original-factorial ``training-<job>.log`` files.

Run:  python -I c1_realized_work.py --repo R --extract DIR --archive FILE --out OUT
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

METRIC = re.compile(r"([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+):([-+0-9.eE]+)")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
LEVELS = {"base": (8, 0.6), "actor_only": (16, 0.6),
          "reservation_only": (8, 0.7), "joint": (16, 0.7)}
COND_BY_LEVEL = {v: k for k, v in LEVELS.items()}
MODEL_LABEL = {"granite33_2b": "Granite", "phi4_mini": "Phi", "qwen25_3b": "Qwen2.5-3B"}
WORK_LABEL = {"gsm8k": "GSM8K", "math": "MATH", "code_standard": "Standard code",
              "code_heavy_tail": "Longer-prompt code"}
CASES = [(m, w) for m in MODEL_LABEL for w in WORK_LABEL]
THRESHOLDS = (0.05, 0.02, 0.01)
# Manuscript Table tab:factorial (CCPE version): actor-stage and whole-run
# batch contrasts per case, rounded MiB, and the equal-case means.
PAPER_RC = {
    ("granite33_2b", "gsm8k"): (1374, -1), ("granite33_2b", "math"): (1932, -37),
    ("granite33_2b", "code_standard"): (2067, -18), ("granite33_2b", "code_heavy_tail"): (3460, 7),
    ("phi4_mini", "gsm8k"): (2424, 15), ("phi4_mini", "math"): (4635, -9),
    ("phi4_mini", "code_standard"): (4915, 5), ("phi4_mini", "code_heavy_tail"): (8093, 18),
    ("qwen25_3b", "gsm8k"): (2682, 14), ("qwen25_3b", "math"): (3723, 28),
    ("qwen25_3b", "code_standard"): (4942, 67), ("qwen25_3b", "code_heavy_tail"): (6608, 1),
}
PAPER_RC_MEAN = (3905, 7)
PAPER_OF_MEAN = (3980, -31)

RC_TRIALS = "benchmark/replication_control/results/replication_control_trials.csv"
RC_BLOCKS = "benchmark/replication_control/results/replication_control_blocks.csv"
RC_SUMMARY = "benchmark/replication_control/results/replication_control_summary.json"
RC_MATRIX = "benchmark/replication_control/replication_control_matrix.csv"
RC_PAIRS = "profiles/electronics_revision/stage_paired_contrasts.csv"
RC_CASES = "profiles/electronics_revision/stage_case_contrasts.csv"
OF_TRIALS = "profiles/standard_grpo/factorial-trials.csv"
OF_PHASE = "profiles/standard_grpo/factorial-phase-trials.csv"
OF_BLOCKS = "profiles/standard_grpo/factorial-phase-blocks.csv"
OF_REALIZED = "profiles/review_revision/realized-workload-trials.csv"
RC_MEMBER = "output/replication_control/attempts/{eid}/trainer-metrics/step-1.json"
OF_MEMBERS = ("output/strengthening_factorial/{eid}/training-{job}.log",
              "output/strengthening_factorial/{eid}/attempt-{job}/training-{job}.log")


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt(row.get(k)) for k in fields})


def fmt(value):
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.6g}" if abs(value) < 1e6 else f"{value:.1f}"
    return "" if value is None else value


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_metrics(line):
    return {k: float(v) for k, v in METRIC.findall(ANSI.sub("", line))}


def half_up(value):
    """Round half away from zero, as in the manuscript tables (66.5 -> 67)."""
    return int(math.copysign(math.floor(abs(value) + 0.5), value))


def require(condition, message):
    if not condition:
        raise SystemExit(f"CHECK FAILED: {message}")


def realized(values, row):
    """Realized-work fields from one logged training step (step 1)."""
    n = int(row["train_batch_size"]) * int(row["rollout_n"])
    gpus = int(row["gpu_count"])
    cap = int(row["max_response_length"])
    total = values["perf/total_num_tokens"]
    prompt_tokens = values["prompt_length/mean"] * n
    response_tokens = values["response_length/mean"] * n
    require(abs(prompt_tokens + response_tokens - total) < 0.5,
            f"prompt+response tokens differ from perf/total_num_tokens ({row['experiment_id']})")
    require(abs(values["global_seqlen/mean"] * gpus - total) < 0.5 * gpus,
            f"global_seqlen/mean x GPUs differs from total tokens ({row['experiment_id']})")
    reached = values["response_length/max"] >= cap
    # verl pads responses to the batch maximum, so response_length/clip_ratio
    # is the share at the batch maximum; it equals the share at the configured
    # cap only when the batch maximum reaches the cap.
    at_cap = round(values["response_length/clip_ratio"] * n) if reached else 0
    return dict(
        sequences=n, max_prompt_length=int(row["max_prompt_length"]), max_response_length=cap,
        prompt_mean=values["prompt_length/mean"], prompt_max=values["prompt_length/max"],
        prompt_min=values["prompt_length/min"],
        response_mean=values["response_length/mean"], response_max=values["response_length/max"],
        response_min=values["response_length/min"],
        response_clip_ratio_logged=values["response_length/clip_ratio"],
        response_cap_reached=int(reached), responses_at_cap=at_cap,
        responses_at_cap_share=at_cap / n,
        aborted_ratio=values.get("response/aborted_ratio", math.nan),
        prompt_tokens=round(prompt_tokens), response_tokens=round(response_tokens),
        total_tokens=int(total),
        per_gpu_tokens_mean=values["global_seqlen/mean"],
        per_gpu_tokens_balanced_max=values["global_seqlen/balanced_max"],
        per_gpu_tokens_balanced_min=values["global_seqlen/balanced_min"],
        per_gpu_tokens_unbalanced_max=values["global_seqlen/max"],
        per_gpu_tokens_unbalanced_min=values["global_seqlen/min"],
        actor_update_seconds=values.get("timing_s/update_actor", math.nan),
    )


# ----------------------------------------------------------------- loading

def load_replication(repo, extract):
    matrix = {r["experiment_id"]: r for r in read_csv(repo / RC_MATRIX)}
    trials = read_csv(repo / RC_TRIALS)
    require(len(trials) == 144 and set(matrix) == {t["experiment_id"] for t in trials},
            "replication trials and matrix do not cover the same 144 slots")
    rows = []
    for trial in trials:
        eid = trial["experiment_id"]
        spec = matrix[eid]
        path = extract / RC_MEMBER.format(eid=eid)
        record = json.loads(path.read_text())
        require(record["invocation_id"] == eid and record["step"] == 1
                and record["training_global_step"] == 1, f"step record identity differs: {eid}")
        require(hashlib.sha256(record["console_line"].encode()).hexdigest()
                == record["console_line_sha256"], f"console line hash differs: {eid}")
        mb, res = int(spec["actor_micro_batch"]), float(spec["vllm_gpu_memory_utilization"])
        require(LEVELS[trial["condition"]] == (mb, res), f"condition/level mismatch: {eid}")
        completed = trial["status"] == "completed" and trial["scientific_eligible"] == "True"
        row = dict(
            study="same_device_replication", experiment_id=eid, block_id=trial["block_id"],
            model_family=trial["model_family"], dataset=trial["dataset"],
            case=f"{MODEL_LABEL[trial['model_family']]} / {WORK_LABEL[trial['dataset']]}",
            training_seed=int(trial["training_seed"]), period=int(trial["period"]),
            condition=trial["condition"], actor_micro_batch=mb, reservation=res,
            status=trial["status"], scientific_eligible=int(completed),
            failure_or_reason=trial["reason"] if not completed else "",
            gpu_count=int(spec["gpu_count"]),
            source_member=RC_MEMBER.format(eid=eid),
        )
        row.update(realized(parse_metrics(record["console_line"]), {**spec, "experiment_id": eid}))
        for phase in ("actor_update", "rollout", "weight_sync", "whole_run"):
            value = trial[f"{phase}_peak_mib"]
            row[f"{phase}_peak_mib"] = float(value) if completed and value else math.nan
        if completed:
            require(math.isfinite(row["actor_update_peak_mib"]) and math.isfinite(row["whole_run_peak_mib"]),
                    f"completed slot lacks peaks: {eid}")
        rows.append(row)
    return rows


def load_original(repo, extract):
    trials = [r for r in read_csv(repo / OF_TRIALS) if r["algorithm"] == "grpo"]
    phase = {r["experiment_id"]: r for r in read_csv(repo / OF_PHASE)}
    realized_rows = {r["experiment_id"]: r for r in read_csv(repo / OF_REALIZED)
                     if r["study"] == "factorial"}
    require(len(trials) == 144 and set(phase) == {t["experiment_id"] for t in trials}
            == set(realized_rows), "original factorial tables do not cover the same 144 processes")
    rows = []
    for trial in trials:
        eid, job = trial["experiment_id"], trial["job_id"]
        members = [m.format(eid=eid, job=job) for m in OF_MEMBERS]
        found = [m for m in members if (extract / m).is_file()]
        require(len(found) == 1, f"retained log not uniquely present: {eid}")
        lines = [l for l in (extract / found[0]).read_text(errors="replace").splitlines()
                 if "training/global_step:" in l]
        values = [parse_metrics(l) for l in lines]
        values = [v for v in values if v.get("training/global_step") == 1]
        require(len(values) == 1, f"expected one logged step-1 metric line: {eid}")
        values = values[0]
        mb, res = int(trial["actor_micro_batch"]), float(trial["vllm_gpu_memory_utilization"])
        require(LEVELS[trial["configuration_level"]] == (mb, res), f"level mismatch: {eid}")
        p = phase[eid]
        require(trial["success"] == "1" and p["success"] == "1", f"process not completed: {eid}")
        require(abs(float(p["actor_nvml_mib"]) - float(trial["phase_peak_actor_update_mib"])) < 0.01
                and abs(float(p["whole_run_mib"]) - float(trial["peak_gpu_memory_mib"])) < 0.01,
                f"phase table differs from trial table: {eid}")
        row = dict(
            study="original_factorial", experiment_id=eid,
            block_id=f"{trial['model_family']}-{trial['dataset']}-s{trial['training_seed']}",
            model_family=trial["model_family"], dataset=trial["dataset"],
            case=f"{MODEL_LABEL[trial['model_family']]} / {WORK_LABEL[trial['dataset']]}",
            training_seed=int(trial["training_seed"]), period="",
            condition=trial["configuration_level"], actor_micro_batch=mb, reservation=res,
            status="completed", scientific_eligible=1, failure_or_reason="",
            gpu_count=int(trial["gpu_count"]), job_id=job, source_member=found[0],
        )
        row.update(realized(values, {**trial, "experiment_id": eid}))
        rw = realized_rows[eid]
        require(abs(float(rw["response_mean_over_logged_batches"]) - row["response_mean"]) < 1e-6
                and abs(float(rw["prompt_mean_over_logged_batches"]) - row["prompt_mean"]) < 1e-6
                and int(rw["cap_reaching_batches"]) == row["response_cap_reached"],
                f"log metrics differ from realized-workload-trials.csv: {eid}")
        row["actor_update_peak_mib"] = float(p["actor_nvml_mib"])
        row["actor_update_allocated_mib"] = float(p["actor_allocated_mib"])
        row["whole_run_peak_mib"] = float(p["whole_run_mib"])
        rows.append(row)
    return rows


# ------------------------------------------------------------- contrasts

def complete_blocks(rows):
    blocks = defaultdict(dict)
    for r in rows:
        blocks[r["block_id"]][r["condition"]] = r
    return {b: c for b, c in sorted(blocks.items())
            if set(c) == set(LEVELS) and all(r["scientific_eligible"] for r in c.values())}


def batch_contrast(cells, field):
    a, b, c, d = (cells[k][field] for k in ("base", "actor_only", "reservation_only", "joint"))
    return ((b - a) + (d - c)) / 2


def equal_case_mean(per_block, field):
    by_case = defaultdict(list)
    for r in per_block:
        by_case[(r["model_family"], r["dataset"])].append(r[field])
    means = {k: statistics.fmean(v) for k, v in by_case.items()}
    return statistics.fmean(means.values()), means


def pair_rows(blocks):
    out = []
    for block, cells in blocks.items():
        for res in (0.6, 0.7):
            lo, hi = cells[COND_BY_LEVEL[(8, res)]], cells[COND_BY_LEVEL[(16, res)]]
            row = dict(block_id=block, model_family=lo["model_family"], dataset=lo["dataset"],
                       case=lo["case"], training_seed=lo["training_seed"], reservation=res,
                       mb8_experiment_id=lo["experiment_id"], mb16_experiment_id=hi["experiment_id"])
            for f in ("total_tokens", "response_tokens", "per_gpu_tokens_balanced_max",
                      "response_mean", "response_max", "responses_at_cap",
                      "actor_update_peak_mib", "whole_run_peak_mib"):
                row[f"mb8_{f}"], row[f"mb16_{f}"] = lo[f], hi[f]
                row[f"diff_{f}"] = hi[f] - lo[f]
            for f in ("total_tokens", "response_tokens", "per_gpu_tokens_balanced_max"):
                row[f"rel_diff_{f}"] = row[f"diff_{f}"] / lo[f]
            row["abs_rel_diff_total_tokens"] = abs(row["rel_diff_total_tokens"])
            row["abs_rel_diff_response_tokens"] = abs(row["rel_diff_response_tokens"])
            out.append(row)
    return out


def block_rows(blocks):
    out = []
    for block, cells in blocks.items():
        first = cells["base"]
        row = dict(block_id=block, model_family=first["model_family"], dataset=first["dataset"],
                   case=first["case"], training_seed=first["training_seed"])
        for f in ("actor_update_peak_mib", "whole_run_peak_mib", "total_tokens",
                  "response_tokens", "per_gpu_tokens_balanced_max"):
            row[f"batch_contrast_{f}"] = batch_contrast(cells, f)
        mean8 = (cells["base"]["total_tokens"] + cells["reservation_only"]["total_tokens"]) / 2
        row["rel_batch_contrast_total_tokens"] = row["batch_contrast_total_tokens"] / mean8
        out.append(row)
    return out


def describe(values):
    v = np.asarray(values, float)
    q = np.quantile(v, [0, .25, .5, .75, 1])
    return dict(n=len(v), mean=float(v.mean()), mean_abs=float(np.abs(v).mean()),
                min=q[0], q25=q[1], median=q[2], q75=q[3], max=q[4],
                n_positive=int((v > 0).sum()), n_negative=int((v < 0).sum()), n_zero=int((v == 0).sum()),
                **{f"n_abs_below_{int(t * 100)}pct": int((np.abs(v) < t).sum()) for t in (0.01, 0.02, 0.05, 0.10)})


def centered(pairs, field):
    by_case = defaultdict(list)
    for r in pairs:
        by_case[(r["model_family"], r["dataset"])].append(r[field])
    means = {k: statistics.fmean(v) for k, v in by_case.items()}
    return [r[field] - means[(r["model_family"], r["dataset"])] for r in pairs]


def correlations(pairs, blocks_out, label):
    out = []

    def add(name, x, y, unit):
        rho, p = stats.spearmanr(x, y)
        out.append(dict(study=label, unit=unit, comparison=name, n=len(x), spearman_rho=float(rho),
                        spearman_p_descriptive=float(p)))

    actor = [r["diff_actor_update_peak_mib"] for r in pairs]
    add("abs token diff vs actor-stage contrast (pooled)",
        [r["diff_total_tokens"] for r in pairs], actor, "block-reservation pair")
    add("relative token diff vs actor-stage contrast (pooled)",
        [r["rel_diff_total_tokens"] for r in pairs], actor, "block-reservation pair")
    add("token diff vs actor-stage contrast, both centered within case",
        centered(pairs, "diff_total_tokens"), centered(pairs, "diff_actor_update_peak_mib"),
        "block-reservation pair")
    add("relative token diff vs actor-stage contrast, both centered within case",
        centered(pairs, "rel_diff_total_tokens"), centered(pairs, "diff_actor_update_peak_mib"),
        "block-reservation pair")
    add("per-GPU balanced-max token diff vs actor-stage contrast, centered within case",
        centered(pairs, "diff_per_gpu_tokens_balanced_max"),
        centered(pairs, "diff_actor_update_peak_mib"), "block-reservation pair")
    add("response-max diff vs actor-stage contrast, centered within case",
        centered(pairs, "diff_response_max"), centered(pairs, "diff_actor_update_peak_mib"),
        "block-reservation pair")
    add("token diff vs whole-run contrast, centered within case",
        centered(pairs, "diff_total_tokens"), centered(pairs, "diff_whole_run_peak_mib"),
        "block-reservation pair")
    add("block token contrast vs block actor-stage contrast (pooled)",
        [r["batch_contrast_total_tokens"] for r in blocks_out],
        [r["batch_contrast_actor_update_peak_mib"] for r in blocks_out], "complete block")
    add("block token contrast vs block actor-stage contrast, centered within case",
        centered(blocks_out, "batch_contrast_total_tokens"),
        centered(blocks_out, "batch_contrast_actor_update_peak_mib"), "complete block")
    return out


def restricted(pairs, label):
    """Equal-case mean of pair contrasts restricted by |relative token diff|."""
    rows, overall = [], []
    specs = [("all pairs", None, None)] + [
        (f"|rel total-token diff| < {int(t * 100)}%", "abs_rel_diff_total_tokens", t) for t in THRESHOLDS
    ] + [(f"|rel response-token diff| < {int(t * 100)}%", "abs_rel_diff_response_tokens", t)
         for t in THRESHOLDS[:2]]
    all_by_case = defaultdict(list)
    for r in pairs:
        all_by_case[(r["model_family"], r["dataset"])].append(r)
    for name, field, t in specs:
        kept = defaultdict(list)
        for key, values in all_by_case.items():
            kept[key] = [r for r in values if field is None or r[field] < t]
        for key in CASES:
            values = kept.get(key, [])
            rows.append(dict(study=label, restriction=name, model_family=key[0], dataset=key[1],
                             case=f"{MODEL_LABEL[key[0]]} / {WORK_LABEL[key[1]]}",
                             pairs_available=len(all_by_case[key]), pairs_kept=len(values),
                             mean_actor_contrast_kept_mib=(statistics.fmean(r["diff_actor_update_peak_mib"] for r in values)
                                                           if values else math.nan),
                             mean_actor_contrast_all_mib=statistics.fmean(
                                 r["diff_actor_update_peak_mib"] for r in all_by_case[key]),
                             mean_whole_run_contrast_kept_mib=(statistics.fmean(r["diff_whole_run_peak_mib"] for r in values)
                                                               if values else math.nan),
                             min_actor_contrast_kept_mib=(min(r["diff_actor_update_peak_mib"] for r in values)
                                                          if values else math.nan)))
        contributing = [k for k in CASES if kept.get(k)]
        kept_means = [statistics.fmean(r["diff_actor_update_peak_mib"] for r in kept[k]) for k in contributing]
        same_case_all = [statistics.fmean(r["diff_actor_update_peak_mib"] for r in all_by_case[k])
                         for k in contributing]
        whole = [statistics.fmean(r["diff_whole_run_peak_mib"] for r in kept[k]) for k in contributing]
        flat = [r for k in contributing for r in kept[k]]
        overall.append(dict(study=label, restriction=name, pairs_kept=len(flat),
                            pairs_available=len(pairs), contributing_cases=len(contributing),
                            equal_case_mean_actor_contrast_kept_mib=statistics.fmean(kept_means) if kept_means else math.nan,
                            equal_case_mean_actor_contrast_all_pairs_same_cases_mib=(
                                statistics.fmean(same_case_all) if same_case_all else math.nan),
                            equal_case_mean_whole_run_contrast_kept_mib=statistics.fmean(whole) if whole else math.nan,
                            kept_pairs_with_positive_actor_contrast=sum(r["diff_actor_update_peak_mib"] > 0 for r in flat),
                            min_actor_contrast_kept_mib=min((r["diff_actor_update_peak_mib"] for r in flat), default=math.nan),
                            kept_per_case=";".join(f"{MODEL_LABEL[k[0]]}/{WORK_LABEL[k[1]]}:{len(kept.get(k, []))}"
                                                   for k in CASES)))
    return rows, overall


def ols(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    df = len(y) - np.linalg.matrix_rank(X)
    sigma2 = float(resid @ resid) / df if df > 0 else math.nan
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    return beta, np.sqrt(np.diag(cov)), df


def adjustment(rows, blocks, label):
    """Per case: actor peak ~ mb16 + res70 + covariate + block fixed effects."""
    out = []
    by_case = defaultdict(list)
    for block, cells in blocks.items():
        for r in cells.values():
            by_case[(r["model_family"], r["dataset"])].append(r)
    for key in CASES:
        obs = by_case[key]
        ids = sorted({r["block_id"] for r in obs})
        y = np.array([r["actor_update_peak_mib"] for r in obs])
        base = [[1.0, float(r["actor_micro_batch"] == 16), float(r["reservation"] == 0.7)]
                + [float(r["block_id"] == b) for b in ids[1:]] for r in obs]
        b0, se0, df0 = ols(y, np.array(base))
        row = dict(study=label, model_family=key[0], dataset=key[1],
                   case=f"{MODEL_LABEL[key[0]]} / {WORK_LABEL[key[1]]}", processes=len(obs),
                   blocks=len(ids), unadjusted_mb16_coef_mib=b0[1], unadjusted_se_mib=se0[1],
                   unadjusted_resid_df=df0)
        for cov_name, field, scale in (("total_tokens", "total_tokens", 1000.0),
                                       ("per_gpu_balanced_max_tokens", "per_gpu_tokens_balanced_max", 1000.0),
                                       ("response_max", "response_max", 100.0)):
            X = np.array([b + [r[field] / scale] for b, r in zip(base, obs)])
            beta, se, df = ols(y, X)
            row[f"adj_{cov_name}_mb16_coef_mib"] = beta[1]
            row[f"adj_{cov_name}_mb16_se_mib"] = se[1]
            row[f"adj_{cov_name}_slope_mib_per_{int(scale)}"] = beta[-1]
            row[f"adj_{cov_name}_slope_se"] = se[-1]
            row[f"adj_{cov_name}_resid_df"] = df
            row[f"adj_{cov_name}_change_mib"] = beta[1] - b0[1]
        tokens = np.array([r["total_tokens"] for r in obs], float)
        row["token_range_within_case"] = float(tokens.max() - tokens.min())
        row["token_mean_within_case"] = float(tokens.mean())
        out.append(row)
    return out


def case_condition_summary(rows, label):
    out = []
    groups = defaultdict(list)
    for r in rows:
        if r["scientific_eligible"]:
            groups[(r["model_family"], r["dataset"], r["actor_micro_batch"], r["reservation"])].append(r)
    for (m, w) in CASES:
        for mb, res in ((8, 0.6), (16, 0.6), (8, 0.7), (16, 0.7)):
            g = groups[(m, w, mb, res)]

            def stat(field, fn):
                return fn([r[field] for r in g]) if g else math.nan

            out.append(dict(
                study=label, model_family=m, dataset=w, case=f"{MODEL_LABEL[m]} / {WORK_LABEL[w]}",
                actor_micro_batch=mb, reservation=res, condition=COND_BY_LEVEL[(mb, res)],
                completed=len(g), sequences_per_process=stat("sequences", min),
                max_prompt_length=stat("max_prompt_length", min), max_response_length=stat("max_response_length", min),
                prompt_mean=stat("prompt_mean", statistics.fmean), prompt_max=stat("prompt_max", max),
                response_mean_mean=stat("response_mean", statistics.fmean),
                response_mean_min=stat("response_mean", min), response_mean_max=stat("response_mean", max),
                response_max_min=stat("response_max", min), response_max_max=stat("response_max", max),
                processes_reaching_cap=stat("response_cap_reached", sum),
                responses_at_cap_total=stat("responses_at_cap", sum),
                responses_total=stat("sequences", sum),
                total_tokens_mean=stat("total_tokens", statistics.fmean),
                total_tokens_min=stat("total_tokens", min), total_tokens_max=stat("total_tokens", max),
                per_gpu_tokens_balanced_max_mean=stat("per_gpu_tokens_balanced_max", statistics.fmean),
                actor_update_peak_mean_mib=stat("actor_update_peak_mib", statistics.fmean),
                actor_update_peak_min_mib=stat("actor_update_peak_mib", min),
                actor_update_peak_max_mib=stat("actor_update_peak_mib", max),
                whole_run_peak_mean_mib=stat("whole_run_peak_mib", statistics.fmean),
                whole_run_peak_min_mib=stat("whole_run_peak_mib", min),
                whole_run_peak_max_mib=stat("whole_run_peak_mib", max)))
    return out


# --------------------------------------------------------- reproduction

def reproduce_replication(repo, blocks, rc_blocks_out, pairs):
    checks = []
    actor_mean, actor_cases = equal_case_mean(rc_blocks_out, "batch_contrast_actor_update_peak_mib")
    whole_mean, whole_cases = equal_case_mean(rc_blocks_out, "batch_contrast_whole_run_peak_mib")
    summary = json.loads((repo / RC_SUMMARY).read_text())["overall"]
    checks.append(dict(check="replication complete blocks", computed=len(blocks), reference=35,
                       source="manuscript", ok=len(blocks) == 35))
    checks.append(dict(check="replication completed processes",
                       computed=sum(len(c) for c in blocks.values()), reference=140,
                       source="manuscript", ok=sum(len(c) for c in blocks.values()) == 140))
    checks.append(dict(check="replication equal-case actor-stage batch contrast (MiB)",
                       computed=actor_mean, reference=summary["actor_update_batch_mib"]["equal_case_mean"],
                       source=RC_SUMMARY,
                       ok=abs(actor_mean - summary["actor_update_batch_mib"]["equal_case_mean"]) < 1e-6))
    checks.append(dict(check="replication equal-case whole-run batch contrast (MiB)",
                       computed=whole_mean, reference=summary["whole_run_batch_mib"]["equal_case_mean"],
                       source=RC_SUMMARY,
                       ok=abs(whole_mean - summary["whole_run_batch_mib"]["equal_case_mean"]) < 1e-6))
    checks.append(dict(check="replication actor-stage equal-case mean, rounded", computed=half_up(actor_mean),
                       reference=PAPER_RC_MEAN[0], source="manuscript tab:factorial",
                       ok=half_up(actor_mean) == PAPER_RC_MEAN[0]))
    checks.append(dict(check="replication whole-run equal-case mean, rounded", computed=half_up(whole_mean),
                       reference=PAPER_RC_MEAN[1], source="manuscript tab:factorial",
                       ok=half_up(whole_mean) == PAPER_RC_MEAN[1]))
    case_ok = all(half_up(actor_cases[k]) == PAPER_RC[k][0] and half_up(whole_cases[k]) == PAPER_RC[k][1]
                  for k in CASES)
    checks.append(dict(check="replication 12 case rows of tab:factorial (actor, whole-run)",
                       computed="all match" if case_ok else "mismatch", reference="12 rows",
                       source="manuscript tab:factorial", ok=case_ok))
    released = {r["block_id"]: r for r in read_csv(repo / RC_BLOCKS)}
    block_ok = all(abs(float(released[r["block_id"]]["actor_update_batch_mib"])
                       - r["batch_contrast_actor_update_peak_mib"]) < 1e-6
                   and abs(float(released[r["block_id"]]["whole_run_batch_mib"])
                           - r["batch_contrast_whole_run_peak_mib"]) < 1e-6 for r in rc_blocks_out)
    checks.append(dict(check="replication per-block batch contrasts vs released blocks CSV",
                       computed="all 35 match" if block_ok else "mismatch", reference=RC_BLOCKS,
                       source=RC_BLOCKS, ok=block_ok))
    paired = {(r["block_id"], float(r["reservation"]), r["metric"]): float(r["difference_mib"])
              for r in read_csv(repo / RC_PAIRS)}
    pair_ok = len(paired) == 140 and all(
        abs(paired[(r["block_id"], r["reservation"], "actor_update")] - r["diff_actor_update_peak_mib"]) < 1e-6
        and abs(paired[(r["block_id"], r["reservation"], "whole_run")] - r["diff_whole_run_peak_mib"]) < 1e-6
        for r in pairs)
    checks.append(dict(check="replication 70 reservation-specific pair contrasts vs stage_paired_contrasts.csv",
                       computed="all 70 match" if pair_ok else "mismatch", reference=RC_PAIRS,
                       source=RC_PAIRS, ok=pair_ok))
    return checks, actor_mean, whole_mean


def reproduce_original(repo, blocks, of_blocks_out):
    checks = []
    actor_mean, _ = equal_case_mean(of_blocks_out, "batch_contrast_actor_update_peak_mib")
    whole_mean, _ = equal_case_mean(of_blocks_out, "batch_contrast_whole_run_peak_mib")
    released = {(r["model_family"], r["dataset"], int(r["training_seed"])): r for r in read_csv(repo / OF_BLOCKS)}
    block_ok = len(of_blocks_out) == 36 and all(
        abs(float(released[(r["model_family"], r["dataset"], r["training_seed"])]["actor_main_effect_actor_nvml_mib"])
            - r["batch_contrast_actor_update_peak_mib"]) < 1e-6 for r in of_blocks_out)
    checks.append(dict(check="original factorial complete blocks", computed=len(blocks), reference=36,
                       source="manuscript", ok=len(blocks) == 36))
    checks.append(dict(check="original factorial per-block actor-stage contrasts vs factorial-phase-blocks.csv",
                       computed="all 36 match" if block_ok else "mismatch", reference=OF_BLOCKS,
                       source=OF_BLOCKS, ok=block_ok))
    checks.append(dict(check="original factorial equal-case actor-stage batch contrast, rounded",
                       computed=half_up(actor_mean), reference=PAPER_OF_MEAN[0], source="manuscript",
                       ok=half_up(actor_mean) == PAPER_OF_MEAN[0]))
    checks.append(dict(check="original factorial equal-case whole-run batch contrast, rounded",
                       computed=half_up(whole_mean), reference=PAPER_OF_MEAN[1], source="manuscript",
                       ok=half_up(whole_mean) == PAPER_OF_MEAN[1]))
    return checks, actor_mean, whole_mean


# ------------------------------------------------------------------ main

def analyse(rows, label, out, prefix):
    blocks = complete_blocks(rows)
    pairs = pair_rows(blocks)
    blocks_out = block_rows(blocks)
    write_csv(out / f"{prefix}_block_pairs.csv", pairs)
    write_csv(out / f"{prefix}_blocks.csv", blocks_out)
    write_csv(out / f"{prefix}_case_condition_summary.csv", case_condition_summary(rows, label))
    dist = []
    for field in ("rel_diff_total_tokens", "rel_diff_response_tokens", "rel_diff_per_gpu_tokens_balanced_max"):
        dist.append(dict(study=label, quantity=field, unit="block-reservation pair",
                         **describe([r[field] for r in pairs])))
    dist.append(dict(study=label, quantity="rel_batch_contrast_total_tokens", unit="complete block",
                     **describe([r["rel_batch_contrast_total_tokens"] for r in blocks_out])))
    # Reference: token differences between the two reservations at a fixed
    # micro-batch, i.e. between two conditions whose actor setting is equal.
    reference = []
    for cells in blocks.values():
        for mb in (8, 16):
            lo, hi = cells[COND_BY_LEVEL[(mb, 0.6)]], cells[COND_BY_LEVEL[(mb, 0.7)]]
            reference.append((hi["total_tokens"] - lo["total_tokens"]) / lo["total_tokens"])
    dist.append(dict(study=label, quantity="reference_rel_diff_total_tokens_res070_vs_res060_same_mb",
                     unit="block-micro-batch pair", **describe(reference)))
    write_csv(out / f"{prefix}_token_difference_distribution.csv", dist)
    corr = correlations(pairs, blocks_out, label)
    write_csv(out / f"{prefix}_correlations.csv", corr)
    per_case, overall = restricted(pairs, label)
    write_csv(out / f"{prefix}_restricted_by_case.csv", per_case)
    write_csv(out / f"{prefix}_restricted_overall.csv", overall)
    adj = adjustment(rows, blocks, label)
    write_csv(out / f"{prefix}_case_adjustment.csv", adj)
    # Per-case token-difference summary (pairs).
    case_tok = []
    for key in CASES:
        values = [r for r in pairs if (r["model_family"], r["dataset"]) == key]
        rel = [r["rel_diff_total_tokens"] for r in values]
        case_tok.append(dict(study=label, model_family=key[0], dataset=key[1],
                             case=f"{MODEL_LABEL[key[0]]} / {WORK_LABEL[key[1]]}", pairs=len(values),
                             rel_token_diff_min=min(rel), rel_token_diff_max=max(rel),
                             rel_token_diff_median=statistics.median(rel),
                             mean_abs_rel_token_diff=statistics.fmean(abs(v) for v in rel),
                             mean_token_diff=statistics.fmean(r["diff_total_tokens"] for r in values),
                             mean_actor_contrast_mib=statistics.fmean(r["diff_actor_update_peak_mib"] for r in values),
                             min_actor_contrast_mib=min(r["diff_actor_update_peak_mib"] for r in values),
                             max_actor_contrast_mib=max(r["diff_actor_update_peak_mib"] for r in values),
                             pairs_with_fewer_mb16_tokens=sum(r["diff_total_tokens"] < 0 for r in values)))
    write_csv(out / f"{prefix}_case_token_differences.csv", case_tok)
    return dict(blocks=blocks, pairs=pairs, blocks_out=blocks_out, dist=dist, corr=corr,
                restricted_case=per_case, restricted_overall=overall, adjustment=adj, case_tok=case_tok)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--extract", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-sha256", default="3dd90eaca2f2d9b5eceb955fe67b6a3528210a6bb9bd7684ac3eb3c36d84a5ab")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo, out = args.repo.resolve(), args.out.resolve()
    digest = sha256(args.archive)
    require(digest == args.archive_sha256, f"archive SHA-256 {digest} differs from expected")

    rc_rows = load_replication(repo, args.extract)
    of_rows = load_original(repo, args.extract)
    attempt_fields = list(rc_rows[0]) + [k for k in of_rows[0] if k not in rc_rows[0]]
    write_csv(out / "rc_attempts.csv", rc_rows, [k for k in attempt_fields if k in rc_rows[0]])
    write_csv(out / "of_attempts.csv", of_rows, [k for k in attempt_fields if k in of_rows[0]])

    rc = analyse(rc_rows, "same_device_replication", out, "rc")
    of = analyse(of_rows, "original_factorial", out, "of")

    rc_checks, rc_actor, rc_whole = reproduce_replication(repo, rc["blocks"], rc["blocks_out"], rc["pairs"])
    of_checks, of_actor, of_whole = reproduce_original(repo, of["blocks"], of["blocks_out"])
    checks = rc_checks + of_checks
    write_csv(out / "reproduction_check.csv", checks)
    for c in checks:
        print(("OK   " if c["ok"] else "FAIL ") + f"{c['check']}: computed={c['computed']} reference={c['reference']}")
    if not all(c["ok"] for c in rc_checks):
        raise SystemExit("Replication reproduction check failed; downstream results are not trusted.")
    if not all(c["ok"] for c in of_checks):
        raise SystemExit("Original-factorial reproduction check failed; downstream results are not trusted.")

    # Unresolved invocations: realized work only (no eligible peaks).
    unresolved = [r for r in rc_rows if not r["scientific_eligible"]]

    def equal_case_adj(adj, field):
        return statistics.fmean(r[field] for r in adj)

    key = dict(
        archive=dict(path=str(args.archive), sha256=digest),
        reproduction=dict(rc_actor_equal_case_mean_mib=rc_actor, rc_whole_run_equal_case_mean_mib=rc_whole,
                          of_actor_equal_case_mean_mib=of_actor, of_whole_run_equal_case_mean_mib=of_whole),
        unresolved=[dict(experiment_id=r["experiment_id"], status=r["status"], reason=r["failure_or_reason"],
                         total_tokens=r["total_tokens"], response_mean=r["response_mean"],
                         response_max=r["response_max"]) for r in unresolved],
    )
    for name, res in (("rc", rc), ("of", of)):
        rows = rc_rows if name == "rc" else [r for r in of_rows]
        eligible = [r for r in rows if r["scientific_eligible"]]
        key[name] = dict(
            processes=len(eligible),
            prompt_identical_within_case=all(
                len({(r["prompt_mean"], r["prompt_max"]) for r in rows
                     if (r["model_family"], r["dataset"]) == k}) == 1 for k in CASES),
            processes_reaching_cap=sum(r["response_cap_reached"] for r in eligible),
            processes_reaching_cap_by_condition={c: sum(r["response_cap_reached"] for r in eligible if r["condition"] == c)
                                                 for c in LEVELS},
            responses_at_cap_by_condition={c: sum(r["responses_at_cap"] for r in eligible if r["condition"] == c)
                                           for c in LEVELS},
            responses_by_condition={c: sum(r["sequences"] for r in eligible if r["condition"] == c) for c in LEVELS},
            cap_reaching_by_case={f"{MODEL_LABEL[k[0]]}/{WORK_LABEL[k[1]]}": sum(
                r["response_cap_reached"] for r in eligible if (r["model_family"], r["dataset"]) == k) for k in CASES},
            distribution=res["dist"], correlations=res["corr"], restricted=res["restricted_overall"],
            adjustment_equal_case=dict(
                unadjusted=equal_case_adj(res["adjustment"], "unadjusted_mb16_coef_mib"),
                total_tokens=equal_case_adj(res["adjustment"], "adj_total_tokens_mb16_coef_mib"),
                per_gpu_balanced_max_tokens=equal_case_adj(res["adjustment"], "adj_per_gpu_balanced_max_tokens_mb16_coef_mib"),
                response_max=equal_case_adj(res["adjustment"], "adj_response_max_mb16_coef_mib")),
            pairs_with_fewer_mb16_tokens=sum(r["diff_total_tokens"] < 0 for r in res["pairs"]),
            actor_contrast_in_pairs_with_fewer_mb16_tokens=dict(
                n=sum(r["diff_total_tokens"] < 0 for r in res["pairs"]),
                min_mib=min((r["diff_actor_update_peak_mib"] for r in res["pairs"] if r["diff_total_tokens"] < 0),
                            default=math.nan),
                positive=sum(r["diff_actor_update_peak_mib"] > 0 for r in res["pairs"] if r["diff_total_tokens"] < 0)),
            pairs_positive_actor_contrast=sum(r["diff_actor_update_peak_mib"] > 0 for r in res["pairs"]),
            pairs=len(res["pairs"]),
            pair_actor_contrast_range=(min(r["diff_actor_update_peak_mib"] for r in res["pairs"]),
                                       max(r["diff_actor_update_peak_mib"] for r in res["pairs"])),
        )
    (out / "key_results.json").write_text(json.dumps(key, indent=2, default=float) + "\n")
    compact, full = latex_tables(rc, of, rc_rows, of_rows)
    (out / "si_table_compact.tex").write_text(compact)
    (out / "si_table_per_condition.tex").write_text(full)
    write_report(out, repo, args, digest, key, checks, rc, of, rc_rows, of_rows, compact)
    print(json.dumps({k: key[k] for k in ("reproduction",)}, indent=1))


# ------------------------------------------------------- tables, report

def n0(value):
    return f"{half_up(value):,}"


def tex(value, decimals=0):
    """Manuscript style: text-mode thousands commas; negatives in math mode."""
    if decimals:
        text = f"{abs(value):,.{decimals}f}"
        rounded_zero = float(text.replace(",", "")) == 0
    else:
        text = f"{abs(half_up(value)):,}"
        rounded_zero = half_up(value) == 0
    if value < 0 and not rounded_zero:
        return "\\(-" + text.replace(",", "{,}") + "\\)"
    return text


def cap_share(blocks, key, mb):
    cells = [r for c in blocks.values() for r in c.values()
             if (r["model_family"], r["dataset"]) == key and r["actor_micro_batch"] == mb]
    return sum(r["responses_at_cap"] for r in cells) / sum(r["sequences"] for r in cells)


def overall_cap_share(blocks, mb):
    cells = [r for c in blocks.values() for r in c.values() if r["actor_micro_batch"] == mb]
    return sum(r["responses_at_cap"] for r in cells) / sum(r["sequences"] for r in cells)


def restricted_lookup(res, name):
    by_case = {(r["model_family"], r["dataset"]): r for r in res["restricted_case"] if r["restriction"] == name}
    overall = next(r for r in res["restricted_overall"] if r["restriction"] == name)
    return by_case, overall


def latex_tables(rc, of, rc_rows, of_rows):
    two, two_all = restricted_lookup(rc, "|rel total-token diff| < 2%")
    adj = {(r["model_family"], r["dataset"]): r for r in rc["adjustment"]}
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{Realized work behind the micro-batch contrast in the same-device replication.",
        "Each micro-batch pair compares batch 16 with batch 8 at one reservation within one complete",
        "block; \\(n\\) is the number of pairs. Prompts are identical across conditions within a case,",
        "so token differences arise from generated responses. Tokens are prompt plus response tokens",
        "per process (mean over pairs). Responses at cap: share of generated responses whose length",
        "equals the configured response cap. Token difference: range over pairs of",
        "\\((T_{16}-T_{8})/T_{8}\\). Actor-stage contrast: mean pair difference in MiB (as in",
        "Table~\\ref{tab:factorial}). Token-adjusted: micro-batch coefficient of a within-case",
        "least-squares fit with block fixed effects, a reservation indicator, and total tokens.",
        "Pairs within 2\\%: pairs with \\(|T_{16}-T_{8}|/T_{8}<0.02\\) and their mean contrast.",
        "Summary rows: pooled cap shares, the overall token-difference range, and equal-case means",
        "of the contrasts (twelve cases weighted equally); the original factorial row uses the 36",
        "complete blocks of the separate-allocation experiment.}",
        "\\label{tab:si-realized-work}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}llrrrrrcrrrr@{}}",
        "\\toprule",
        " & & & \\multicolumn{2}{c}{Tokens (\\(\\times 10^3\\))} & \\multicolumn{2}{c}{At cap (\\%)} & "
        "Token diff. & Actor-stage & Token- & \\multicolumn{2}{c}{Pairs within 2\\%} \\\\",
        "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){11-12}",
        "Model & Workload & \\(n\\) & 8 & 16 & 8 & 16 & 16 vs 8 (\\%) & contrast & adjusted & \\(n\\) & Mean \\\\",
        "\\midrule",
    ]
    previous = None
    for key in CASES:
        pairs = [p for p in rc["pairs"] if (p["model_family"], p["dataset"]) == key]
        rel = [p["rel_diff_total_tokens"] * 100 for p in pairs]
        if previous is not None and previous != key[0]:
            lines.append("\\addlinespace[3pt]")
        model = MODEL_LABEL[key[0]] if previous != key[0] else ""
        previous = key[0]
        kept = two[key]
        lines.append(
            f"{model} & {WORK_LABEL[key[1]]} & {len(pairs)} & "
            f"{statistics.fmean(p['mb8_total_tokens'] for p in pairs) / 1000:.1f} & "
            f"{statistics.fmean(p['mb16_total_tokens'] for p in pairs) / 1000:.1f} & "
            f"{cap_share(rc['blocks'], key, 8) * 100:.1f} & {cap_share(rc['blocks'], key, 16) * 100:.1f} & "
            f"{tex(min(rel), 1)} to {tex(max(rel), 1)} & "
            f"{tex(statistics.fmean(p['diff_actor_update_peak_mib'] for p in pairs))} & "
            f"{tex(adj[key]['adj_total_tokens_mb16_coef_mib'])} & {kept['pairs_kept']} & "
            f"{tex(kept['mean_actor_contrast_kept_mib']) if kept['pairs_kept'] else '--'} \\\\")
    lines.append("\\midrule")
    for label, res in (("All cases", rc), ("Original factorial", of)):
        rel = [p["rel_diff_total_tokens"] * 100 for p in res["pairs"]]
        _, kept = restricted_lookup(res, "|rel total-token diff| < 2%")
        adjusted = statistics.fmean(r["adj_total_tokens_mb16_coef_mib"] for r in res["adjustment"])
        unadjusted = statistics.fmean(r["unadjusted_mb16_coef_mib"] for r in res["adjustment"])
        lines.append(
            f"{label} & & {len(res['pairs'])} & & & {overall_cap_share(res['blocks'], 8) * 100:.1f} & "
            f"{overall_cap_share(res['blocks'], 16) * 100:.1f} & {tex(min(rel), 1)} to {tex(max(rel), 1)} & "
            f"{tex(unadjusted)} & {tex(adjusted)} & {kept['pairs_kept']} & "
            f"{tex(kept['equal_case_mean_actor_contrast_kept_mib'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular*}", "\\end{table}", ""]
    compact = "\n".join(lines)

    summary = case_condition_summary(rc_rows, "same_device_replication")
    full = [
        "% Optional long table: one row per replication case and condition (140 completed processes).",
        "% Requires longtable and booktabs (both loaded by the Supporting Information preamble).",
        "{\\footnotesize\\setlength{\\tabcolsep}{4pt}",
        "\\begin{longtable}{@{}rrrrrrrrrr@{}}",
        "\\caption{Realized work per same-device replication condition (completed processes only).",
        "MB: actor micro-batch; Res.: generation reservation; \\(n\\): completed processes.",
        "Response lengths in tokens: mean of the per-process means with the range over seeds, and the",
        "longest response. Cap: processes with at least one response at the configured cap and the",
        "share of all responses at the cap. Tokens: prompt plus response tokens per process",
        "(\\(\\times 10^3\\)), mean with the range over seeds. Peaks: means in MiB.",
        "Prompt lengths are identical across conditions and seeds within a case.}",
        "\\label{tab:si-realized-work-conditions}\\\\",
        "\\toprule",
        "MB & Res. & \\(n\\) & \\multicolumn{2}{c}{Response length} & \\multicolumn{2}{c}{Cap} & "
        "Tokens (\\(\\times 10^3\\)) & \\multicolumn{2}{c}{Peak (MiB)} \\\\",
        "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){9-10}",
        " & & & Mean (range) & Max & Proc. & \\% & Mean (range) & Actor stage & Whole run \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        "MB & Res. & \\(n\\) & \\multicolumn{2}{c}{Response length} & \\multicolumn{2}{c}{Cap} & "
        "Tokens (\\(\\times 10^3\\)) & \\multicolumn{2}{c}{Peak (MiB)} \\\\",
        "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){9-10}",
        " & & & Mean (range) & Max & Proc. & \\% & Mean (range) & Actor stage & Whole run \\\\",
        "\\midrule",
        "\\endhead",
        "\\bottomrule",
        "\\endlastfoot",
    ]
    previous = None
    for r in summary:
        key = (r["model_family"], r["dataset"])
        if previous != key:
            if previous is not None:
                full.append("\\addlinespace[2pt]")
            first = next(x for x in summary if (x["model_family"], x["dataset"]) == key)
            full.append(f"\\multicolumn{{10}}{{@{{}}l}}{{\\rlap{{\\textit{{{MODEL_LABEL[key[0]]} / {WORK_LABEL[key[1]]}}} "
                        f"(prompt mean {first['prompt_mean']:.0f}, prompt max {first['prompt_max']:.0f}; "
                        f"response cap {first['max_response_length']}; {first['sequences_per_process']} "
                        "responses per process)}} \\\\")
        previous = key
        full.append(
            f"{r['actor_micro_batch']} & {r['reservation']:.2f} & {r['completed']} & "
            f"{r['response_mean_mean']:.0f} ({r['response_mean_min']:.0f}--{r['response_mean_max']:.0f}) & "
            f"{r['response_max_max']:.0f} & {r['processes_reaching_cap']} & "
            f"{r['responses_at_cap_total'] / r['responses_total'] * 100:.1f} & "
            f"{r['total_tokens_mean'] / 1000:.1f} ({r['total_tokens_min'] / 1000:.1f}--{r['total_tokens_max'] / 1000:.1f}) & "
            f"{n0(r['actor_update_peak_mean_mib'])} & {n0(r['whole_run_peak_mean_mib'])} \\\\")
    full += ["\\end{longtable}}", ""]
    return compact, "\n".join(full)


def dist_row(res, quantity):
    return next(r for r in res["dist"] if r["quantity"] == quantity)


def corr_row(res, prefix):
    return next(r for r in res["corr"] if r["comparison"].startswith(prefix))


def write_report(out, repo, args, digest, key, checks, rc, of, rc_rows, of_rows, compact):
    L = []
    add = L.append
    tok = {s: dist_row(r, "rel_diff_total_tokens") for s, r in (("rc", rc), ("of", of))}
    resp = {s: dist_row(r, "rel_diff_response_tokens") for s, r in (("rc", rc), ("of", of))}
    ref = {s: dist_row(r, "reference_rel_diff_total_tokens_res070_vs_res060_same_mb") for s, r in (("rc", rc), ("of", of))}
    blk = {s: dist_row(r, "rel_batch_contrast_total_tokens") for s, r in (("rc", rc), ("of", of))}
    within = {s: corr_row(r, "token diff vs actor-stage contrast, both centered") for s, r in (("rc", rc), ("of", of))}
    within_rel = {s: corr_row(r, "relative token diff vs actor-stage contrast, both centered") for s, r in (("rc", rc), ("of", of))}
    pooled = {s: corr_row(r, "relative token diff vs actor-stage contrast (pooled)") for s, r in (("rc", rc), ("of", of))}
    gpu = {s: corr_row(r, "per-GPU balanced-max") for s, r in (("rc", rc), ("of", of))}
    rmax = {s: corr_row(r, "response-max diff") for s, r in (("rc", rc), ("of", of))}
    whole = {s: corr_row(r, "token diff vs whole-run") for s, r in (("rc", rc), ("of", of))}
    blockc = {s: corr_row(r, "block token contrast vs block actor-stage contrast, centered") for s, r in (("rc", rc), ("of", of))}
    restr = {s: {r["restriction"]: r for r in res["restricted_overall"]} for s, res in (("rc", rc), ("of", of))}
    adj = {s: key[s]["adjustment_equal_case"] for s in ("rc", "of")}
    k = key

    def caps(s):
        c = k[s]["responses_at_cap_by_condition"]
        n = k[s]["responses_by_condition"]
        return {x: c[x] / n[x] * 100 for x in LEVELS}

    cap = {s: caps(s) for s in ("rc", "of")}
    adj_rows = {s: res["adjustment"] for s, res in (("rc", rc), ("of", of))}
    changes = {s: [r["adj_total_tokens_mb16_coef_mib"] - r["unadjusted_mb16_coef_mib"] for r in adj_rows[s]]
               for s in ("rc", "of")}
    rel_changes = {s: [(r["adj_total_tokens_mb16_coef_mib"] - r["unadjusted_mb16_coef_mib"]) / r["unadjusted_mb16_coef_mib"]
                       for r in adj_rows[s]] for s in ("rc", "of")}
    slopes = {s: [r["adj_total_tokens_slope_mib_per_1000"] for r in adj_rows[s]] for s in ("rc", "of")}

    def by_cond_tokens(rows):
        out = {}
        for c in LEVELS:
            means = []
            for case in CASES:
                v = [r["total_tokens"] for r in rows if r["scientific_eligible"] and r["condition"] == c
                     and (r["model_family"], r["dataset"]) == case]
                means.append(statistics.fmean(v))
            out[c] = statistics.fmean(means)
        return out

    tokc = {"rc": by_cond_tokens(rc_rows), "of": by_cond_tokens(of_rows)}
    fewer = {s: k[s]["actor_contrast_in_pairs_with_fewer_mb16_tokens"] for s in ("rc", "of")}

    add("# C1. Realized work behind the micro-batch contrast (reviewer R1.3)")
    add("")
    add("CPU-only re-analysis of existing measurements; nothing was run on GPUs or clusters, nothing was "
        "pushed or uploaded, and no manuscript file was edited. All numbers below are produced by "
        "`c1_realized_work.py` in this directory (run with `python -I`).")
    add("")
    add("## 1. Data sources")
    add("")
    add(f"Repository root R = `{repo}`.")
    add("")
    add("| Content | Path |")
    add("|---|---|")
    for label, path in (("Replication slots, status, stage peaks (144 rows)", RC_TRIALS),
                        ("Replication configuration matrix (micro-batch, reservation, caps)", RC_MATRIX),
                        ("Replication per-block contrasts (cross-check)", RC_BLOCKS),
                        ("Replication equal-case summary (cross-check)", RC_SUMMARY),
                        ("Replication reservation-specific pair contrasts (cross-check)", RC_PAIRS),
                        ("Original factorial trial table (job ids, settings)", OF_TRIALS),
                        ("Original factorial stage peaks (actor NVML, whole run)", OF_PHASE),
                        ("Original factorial per-block contrasts (cross-check)", OF_BLOCKS),
                        ("Original factorial realized workload (cross-check of log parsing)", OF_REALIZED)):
        add(f"| {label} | `R/{path}` |")
    add("")
    add("Per-process trainer metrics are not in the working tree; they exist only in the released archive. "
        "The archive was downloaded once into a fresh `mktemp -d` directory and verified before use:")
    add("")
    add("- URL: `https://huggingface.co/datasets/kobzaond/RLVRAMBench/resolve/00e61d8aa329cee64c1cf899f11678971b5115e0/"
        "releases/RLVRAMBench-review-artifact-submission-20260917-b0018d0.tar.zst`")
    add(f"- Local copy: `{args.archive}` (210,126,067 bytes)")
    add(f"- SHA-256: `{digest}` (matches the expected value and `R/paper/electronics_delivery_manifest.json`)")
    add(f"- Extracted members only (into `{args.extract}`): the 144 files "
        "`output/replication_control/attempts/<experiment_id>/trainer-metrics/step-1.json` and the 144 retained "
        "original-factorial logs `output/strengthening_factorial/<experiment_id>/training-<job_id>.log` "
        "(job ids from `R/" + OF_TRIALS + "`). Nothing from the archive was executed.")
    add("")
    add("Each `step-1.json` holds the trainer's step-1 console line; the script checks that its "
        "`invocation_id` equals the attempt directory and that `console_line_sha256` matches the line.")
    add("")
    add("**Definitions.**")
    add("")
    add("- *Total tokens* = `perf/total_num_tokens` (prompt plus response tokens over all sequences of the "
        "step; 64 sequences = 32 prompts x 2 responses, MATH 128). The script verifies "
        "total = sequences x (prompt mean + response mean) and total = 2 x `global_seqlen/mean` for every process.")
    add("- *Per-GPU tokens* = `global_seqlen/mean` (total/2) and `global_seqlen/balanced_max` (larger of the two "
        "per-GPU token loads after verl's sequence balancing).")
    add("- *Cap reaching* follows `extract_workload_log` in `R/memory_tuner/review_attempt_workload_audit.py`: a "
        "process reaches the cap when `response_length/max` >= configured `max_response_length`. "
        "`response_length/clip_ratio` is the share at the padded batch maximum, so it is the share at the cap "
        "only when the maximum reaches the cap; the number of responses at the cap is computed as "
        "clip_ratio x sequences for cap-reaching processes and 0 otherwise. The logged clip ratios are kept "
        "unchanged in `rc_attempts.csv`.")
    add("- *Actor-stage peak* = `actor_update_peak_mib` (replication) or `actor_nvml_mib` (original factorial): "
        "the largest NVML sample on either GPU during the actor-update interval. *Whole-run peak* = whole-process peak.")
    add("- *Pair* = the two conditions with micro-batch 16 and 8 at one reservation in one complete block "
        "(70 pairs in the replication, 72 in the original factorial). The block batch contrast is the mean of its "
        "two pairs, so equal-case means over pairs equal the published equal-case means.")
    add("")
    add("## 2. Reproduction check (passed)")
    add("")
    add("| Check | Computed | Reference | OK |")
    add("|---|---|---|---|")
    for c in checks:
        computed = c["computed"]
        computed = f"{computed:.4f}" if isinstance(computed, float) else computed
        reference = c["reference"]
        reference = f"{reference:.4f}" if isinstance(reference, float) else reference
        add(f"| {c['check']} | {computed} | {reference} | {'yes' if c['ok'] else 'NO'} |")
    add("")
    add(f"The unadjusted equal-case means reproduce the manuscript exactly: actor stage "
        f"{k['reproduction']['rc_actor_equal_case_mean_mib']:.2f} MiB -> 3,905 MiB and whole run "
        f"{k['reproduction']['rc_whole_run_equal_case_mean_mib']:.2f} MiB -> 7 MiB (35 complete blocks, 140 "
        f"processes); all twelve case rows of Table tab:factorial match (half-up rounding; Qwen/standard code "
        f"whole run is 66.5 -> 67). The original factorial gives "
        f"{k['reproduction']['of_actor_equal_case_mean_mib']:.2f} -> 3,980 MiB and "
        f"{k['reproduction']['of_whole_run_equal_case_mean_mib']:.2f} -> -31 MiB. Individual replication block "
        f"contrasts range from {n0(min(r['batch_contrast_actor_update_peak_mib'] for r in rc['blocks_out']))} to "
        f"{n0(max(r['batch_contrast_actor_update_peak_mib'] for r in rc['blocks_out']))} MiB, as in the text.")
    add("")
    add("## 3. Key results: same-device replication")
    add("")
    add("**Prompts.** Prompt-length statistics (mean, max, min) are identical across all conditions and all seeds "
        "of each case (the launch script uses `data.shuffle=False`, and each process trains one step on the first "
        "batch). Token differences between conditions are therefore entirely response-token differences.")
    add("")
    add(f"**Cap reaching.** {k['rc']['processes_reaching_cap']} of 140 completed processes have at least one "
        f"response at the cap; by condition (MB8/0.60, MB16/0.60, MB8/0.70, MB16/0.70): "
        + ", ".join(f"{k['rc']['processes_reaching_cap_by_condition'][c]}" for c in LEVELS)
        + " of 35. Share of responses at the cap: "
        + ", ".join(f"{cap['rc'][c]:.1f}%" for c in LEVELS)
        + f" (pooled over {k['rc']['responses_by_condition']['base']:,} responses per condition). Cap reaching is "
        "concentrated in MATH (cap 512; every process of every model), Phi standard and longer-prompt code, and "
        "Qwen longer-prompt code; by case: "
        + "; ".join(f"{c} {v}" for c, v in k["rc"]["cap_reaching_by_case"].items()) + " (processes of 12; Qwen standard code of 8).")
    add("")
    add("**Token volume by condition.** Equal-case mean total tokens per process: "
        + ", ".join(f"MB{LEVELS[c][0]}/{LEVELS[c][1]:.2f} {n0(tokc['rc'][c])}" for c in LEVELS) + ". Per case x condition values (means and ranges "
        "over seeds) are in `rc_case_condition_summary.csv`; per-process values in `rc_attempts.csv`.")
    add("")
    t = tok["rc"]
    add(f"**Within-pair token differences (MB16 - MB8), n = {t['n']} pairs.** Relative total-token difference "
        f"(T16 - T8)/T8: median {t['median'] * 100:.2f}%, IQR {t['q25'] * 100:.2f}% to {t['q75'] * 100:.2f}%, "
        f"range {t['min'] * 100:.1f}% to {t['max'] * 100:.1f}%, mean {t['mean'] * 100:.2f}%, mean absolute "
        f"{t['mean_abs'] * 100:.2f}%. Pairs within 1% / 2% / 5% / 10%: {t['n_abs_below_1pct']} / "
        f"{t['n_abs_below_2pct']} / {t['n_abs_below_5pct']} / {t['n_abs_below_10pct']}. MB16 has more tokens in "
        f"{t['n_positive']} pairs and fewer in {t['n_negative']}. Response tokens alone: median "
        f"{resp['rc']['median'] * 100:.2f}%, range {resp['rc']['min'] * 100:.1f}% to {resp['rc']['max'] * 100:.1f}%, "
        f"mean absolute {resp['rc']['mean_abs'] * 100:.2f}%. Block-level (averaged over reservations, n = 35): "
        f"median {blk['rc']['median'] * 100:.2f}%, range {blk['rc']['min'] * 100:.1f}% to {blk['rc']['max'] * 100:.1f}%. "
        f"Reference: between the two reservations at a fixed micro-batch (70 pairs), the mean absolute difference "
        f"is {ref['rc']['mean_abs'] * 100:.2f}% (range {ref['rc']['min'] * 100:.1f}% to {ref['rc']['max'] * 100:.1f}%), "
        "i.e. the micro-batch pairs differ in realized work by about as much as any two conditions of a block. "
        "Per-case ranges: `rc_case_token_differences.csv`.")
    add("")
    add(f"**Sign of the actor-stage contrast.** The actor-stage peak is higher at MB16 in all "
        f"{k['rc']['pairs_positive_actor_contrast']} of {k['rc']['pairs']} pairs (range "
        f"{n0(k['rc']['pair_actor_contrast_range'][0])} to {n0(k['rc']['pair_actor_contrast_range'][1])} MiB), including "
        f"all {fewer['rc']['n']} pairs in which MB16 processed fewer tokens (smallest contrast there: "
        f"{n0(fewer['rc']['min_mib'])} MiB).")
    add("")
    add(f"**Correlation (Spearman).** Pooled across cases, the relative token difference and the pair actor-stage "
        f"contrast are weakly related (rho = {pooled['rc']['spearman_rho']:.2f}, n = 70), because case differences "
        f"dominate the contrast. After centering both quantities within case, rho = {within['rc']['spearman_rho']:.2f} "
        f"(absolute tokens) and {within_rel['rc']['spearman_rho']:.2f} (relative tokens); per-GPU balanced-max tokens "
        f"{gpu['rc']['spearman_rho']:.2f}; longest response {rmax['rc']['spearman_rho']:.2f}; at block level (n = 35, "
        f"centered) {blockc['rc']['spearman_rho']:.2f}. For the whole-run contrast, rho = "
        f"{whole['rc']['spearman_rho']:.2f}. Thus realized work explains a substantial part of the block-to-block "
        "spread of the actor-stage contrast within a case, not its sign or mean. (p-values in "
        "`rc_correlations.csv` are descriptive only; pairs share blocks.)")
    add("")
    add("**Contrast restricted to token-matched pairs** (equal-case mean over cases with at least one kept pair):")
    add("")
    add("| Restriction | Pairs kept | Cases | Equal-case actor-stage contrast (MiB) | Same cases, all pairs (MiB) | Whole-run contrast, kept (MiB) | Kept pairs per case |")
    add("|---|---|---|---|---|---|---|")
    for name in ("all pairs", "|rel total-token diff| < 5%", "|rel total-token diff| < 2%",
                 "|rel total-token diff| < 1%", "|rel response-token diff| < 5%", "|rel response-token diff| < 2%"):
        r = restr["rc"][name]
        add(f"| {name.replace('|', '&#124;')} | {r['pairs_kept']} / {r['pairs_available']} | {r['contributing_cases']} | "
            f"{n0(r['equal_case_mean_actor_contrast_kept_mib'])} | "
            f"{n0(r['equal_case_mean_actor_contrast_all_pairs_same_cases_mib'])} | "
            f"{r['equal_case_mean_whole_run_contrast_kept_mib']:.1f} | {r['kept_per_case'].replace(';', '; ')} |")
    add("")
    add(f"**Within-case adjustment.** Per case, OLS of the actor-stage peak on an MB16 indicator, a reservation-0.70 "
        "indicator, block fixed effects, and total tokens (12 processes and 6 residual df per case; Qwen standard "
        "code 8 and 3). Without the token term the MB16 coefficient equals the published case contrast. "
        f"Equal-case mean MB16 coefficient: unadjusted {n0(adj['rc']['unadjusted'])} MiB; adjusted for total tokens "
        f"{n0(adj['rc']['total_tokens'])} MiB; for per-GPU balanced-max tokens "
        f"{n0(adj['rc']['per_gpu_balanced_max_tokens'])} MiB; for the longest response {n0(adj['rc']['response_max'])} MiB. "
        f"Per-case change from token adjustment: {n0(min(changes['rc']))} to {n0(max(changes['rc']))} MiB "
        f"({min(rel_changes['rc']) * 100:.1f}% to {max(rel_changes['rc']) * 100:.1f}%); token slopes range from "
        f"{min(slopes['rc']):.0f} to {max(slopes['rc']):.0f} MiB per 1,000 tokens. Per-case coefficients and OLS "
        "standard errors: `rc_case_adjustment.csv`. The adjustment assumes one token slope for both micro-batches and "
        "is descriptive (small n).")
    add("")
    add("**Unresolved block.** The four unresolved slots (Qwen standard code, seed 211; collector reason "
        "`Trial scheduler/exit identity differs`) also logged step 1; their total tokens are "
        + ", ".join(f"{u['experiment_id'].rsplit('-', 1)[1]} {u['total_tokens']:,}" for u in k["unresolved"])
        + ". They remain outside all contrasts and are listed with `scientific_eligible = 0` in `rc_attempts.csv`.")
    add("")
    add("## 4. Key results: original factorial (separate allocations, 144 processes, 36 blocks)")
    add("")
    t = tok["of"]
    add(f"Token data come from the retained training logs (`perf/total_num_tokens` etc.); the parsed prompt and "
        f"response means and cap flags match `realized-workload-trials.csv` for all 144 processes. Prompts are again "
        f"identical within each case. Processes reaching the cap: {k['of']['processes_reaching_cap']} of 144 (by "
        f"condition " + ", ".join(f"{k['of']['processes_reaching_cap_by_condition'][c]}" for c in LEVELS)
        + " of 36); responses at the cap: " + ", ".join(f"{cap['of'][c]:.1f}%" for c in LEVELS) + ". "
        f"Relative token difference over 72 pairs: median {t['median'] * 100:.2f}%, range {t['min'] * 100:.1f}% to "
        f"{t['max'] * 100:.1f}%, mean absolute {t['mean_abs'] * 100:.2f}% (reservation reference "
        f"{ref['of']['mean_abs'] * 100:.2f}%); within 2% / 5%: {t['n_abs_below_2pct']} / {t['n_abs_below_5pct']}. "
        f"All 72 pair contrasts are positive ({n0(k['of']['pair_actor_contrast_range'][0])} to "
        f"{n0(k['of']['pair_actor_contrast_range'][1])} MiB), including the {fewer['of']['n']} pairs with fewer MB16 "
        f"tokens. Within-case Spearman rho = {within['of']['spearman_rho']:.2f} (pooled "
        f"{pooled['of']['spearman_rho']:.2f}). Restricted equal-case means: <5% "
        f"{n0(restr['of']['|rel total-token diff| < 5%']['equal_case_mean_actor_contrast_kept_mib'])} MiB "
        f"({restr['of']['|rel total-token diff| < 5%']['pairs_kept']} pairs, "
        f"{restr['of']['|rel total-token diff| < 5%']['contributing_cases']} cases); <2% "
        f"{n0(restr['of']['|rel total-token diff| < 2%']['equal_case_mean_actor_contrast_kept_mib'])} MiB "
        f"({restr['of']['|rel total-token diff| < 2%']['pairs_kept']} pairs, "
        f"{restr['of']['|rel total-token diff| < 2%']['contributing_cases']} cases). Token-adjusted equal-case "
        f"coefficient {n0(adj['of']['total_tokens'])} MiB versus {n0(adj['of']['unadjusted'])} MiB unadjusted "
        f"(per-case change {n0(min(changes['of']))} to {n0(max(changes['of']))} MiB). Files: `of_*.csv`.")
    add("")
    add("## 5. Interpretation and limits")
    add("")

    def alternatives(s):
        rows = restr[s]
        rel = []
        for name in ("|rel total-token diff| < 5%", "|rel total-token diff| < 2%", "|rel total-token diff| < 1%"):
            r = rows[name]
            rel.append(r["equal_case_mean_actor_contrast_kept_mib"] / r["equal_case_mean_actor_contrast_all_pairs_same_cases_mib"] - 1)
        for field in ("total_tokens", "per_gpu_balanced_max_tokens", "response_max"):
            rel.append(adj[s][field] / adj[s]["unadjusted"] - 1)
        return rel

    alt = alternatives("rc") + alternatives("of")
    max_tok = max(max(abs(tok[s]["min"]), abs(tok[s]["max"])) for s in ("rc", "of")) * 100
    max_resp = max(max(abs(resp[s]["min"]), abs(resp[s]["max"])) for s in ("rc", "of")) * 100
    sparse = [r["case"] for r in rc["restricted_case"]
              if r["restriction"] == "|rel total-token diff| < 2%" and r["pairs_kept"] <= 1]
    dfs = [r["adj_total_tokens_resid_df"] for r in rc["adjustment"]]
    add(f"- Realized work does differ between conditions with matched seeds: the mean absolute difference is "
        f"{tok['rc']['mean_abs'] * 100:.1f}% of total tokens, and single pairs differ by up to {max_tok:.1f}% of total "
        f"tokens ({max_resp:.1f}% of response tokens) across both experiments. The differences have no systematic "
        f"direction (median {tok['rc']['median'] * 100:.2f}%; {tok['rc']['n_positive']} positive and "
        f"{tok['rc']['n_negative']} negative) and are as large between the two reservations at a fixed micro-batch "
        f"({ref['rc']['mean_abs'] * 100:.1f}% mean absolute), which is consistent with run-to-run variation in "
        "generation. In the launch configuration the varied setting is `actor.ppo_micro_batch_size_per_gpu`, which "
        "applies to the actor update after generation in the single measured step; the old- and reference-log-prob "
        "micro-batches are fixed at 4 in all conditions.")
    add(f"- Within a case, token differences track the size of the pair contrast (Spearman rho "
        f"{within['rc']['spearman_rho']:.2f} in the replication, {within['of']['spearman_rho']:.2f} in the original "
        "factorial), so part of the block-to-block range of the contrast reflects realized work. The mean contrast is "
        f"insensitive: the token-matched subsets (5%, 2%, 1%) and the three adjusted fits change the equal-case mean by "
        f"{min(alt) * 100:.1f}% to {max(alt) * 100:.1f}% across both experiments (restricted means compared with all "
        "pairs of the same contributing cases).")
    add("- The trainer logs only step-level aggregates. Tokens per actor micro-batch, which most directly drive "
        "actor-update activations, are not recorded; the per-GPU balanced-max load and the longest response are the "
        "closest available proxies and lead to the same conclusion.")
    add(f"- The 2% and 5% thresholds are post hoc. At 2%, some cases keep at most one pair ({'; '.join(sparse)}), "
        f"so per-case restricted means are noisy; the regression has {min(dfs)}-{max(dfs)} residual degrees of "
        "freedom per case and assumes one token slope for both micro-batches.")
    add("- The wording \"observed configuration contrast\" therefore remains appropriate: tokens differ materially in "
        "some pairs, even though the mean contrast does not depend on them.")
    add("")
    add("## 6. Proposed manuscript text")
    add("")
    r2, r5 = restr["rc"]["|rel total-token diff| < 2%"], restr["rc"]["|rel total-token diff| < 5%"]
    o2 = restr["of"]["|rel total-token diff| < 2%"]
    t = tok["rc"]
    rho = within["rc"]["spearman_rho"]
    shares = [cap["rc"][c] for c in ("base", "actor_only", "reservation_only", "joint")]

    def paragraph(latex):
        def pct(v):
            text = f"{v:.1f}"
            if latex:
                return (f"\\({text}\\)" if v < 0 and text != "-0.0" else text.replace("-0.0", "0.0")) + "\\%"
            return text + "%"

        def mib(v):
            return n0(v) + ("~MiB" if latex else " MiB")

        rho_text = f"\\(\\rho = {rho:.2f}\\)" if latex else f"ρ = {rho:.2f}"
        pct_sign = "\\%" if latex else "%"
        table = "Table~S[X]" if latex else "Table S[X]"
        return (
            f"Because generated responses can differ between conditions despite matched seeds, {table} reports the "
            "realized work of every replication condition, and the released per-process records give prompt and "
            "response lengths, cap reaching, and token volumes. Prompt lengths are identical across all conditions and "
            "seeds of a case, because each process trains one step on the same unshuffled first batch; token volumes "
            "therefore differ only through the generated responses. In each of the four conditions, between "
            f"{pct(min(shares))} and {pct(max(shares))} of the responses reach the length cap. Across the "
            f"{t['n']} reservation-specific micro-batch pairs of the 35 complete blocks, the total token volume at "
            f"micro-batch 16 differs from that at micro-batch 8 by a median of {pct(t['median'] * 100)} (range "
            f"{pct(t['min'] * 100)} to {pct(t['max'] * 100)}), about as much as the two reservations differ at a fixed "
            f"micro-batch (mean absolute difference {pct(t['mean_abs'] * 100)} versus {pct(ref['rc']['mean_abs'] * 100)}). "
            f"Within a case, larger token differences accompany larger actor-stage contrasts (Spearman {rho_text}), "
            f"yet the actor-stage peak is higher at micro-batch 16 in all {k['rc']['pairs']} pairs, including the "
            f"{fewer['rc']['n']} pairs in which the micro-batch-16 process handled fewer tokens. Restricting the "
            f"comparison to the {r2['pairs_kept']} pairs whose token volumes differ by less than 2{pct_sign}, which cover "
            f"all twelve cases, gives an equal-case mean of {mib(r2['equal_case_mean_actor_contrast_kept_mib'])} "
            f"({mib(r5['equal_case_mean_actor_contrast_kept_mib'])} for the {r5['pairs_kept']} pairs below 5{pct_sign}), "
            "and a within-case fit with block fixed effects and total tokens as a covariate gives "
            f"{mib(adj['rc']['total_tokens'])}, compared with {mib(adj['rc']['unadjusted'])} without adjustment. The "
            f"original factorial experiment shows the same pattern ({mib(adj['of']['unadjusted'])} unadjusted, "
            f"{mib(o2['equal_case_mean_actor_contrast_kept_mib'])} in the {o2['pairs_kept']} pairs within 2{pct_sign}, "
            f"and {mib(adj['of']['total_tokens'])} after adjustment). The {mib(adj['rc']['unadjusted'])} value "
            "therefore remains an observed configuration contrast under full instrumentation: realized work varies by "
            "a few percent between conditions and contributes to the spread between blocks, but it does not account "
            "for the mean increase.")

    add(paragraph(False))
    add("")
    add("LaTeX-ready version (same content):")
    add("")
    add("```latex")
    add(paragraph(True))
    add("```")
    add("")
    add("## 7. Proposed Supporting Information table (booktabs)")
    add("")
    add("Compact table (also written to `si_table_compact.tex`):")
    add("")
    add("```latex")
    add(compact.rstrip())
    add("```")
    add("")
    add("A per-condition long table (48 rows: every case x condition with response mean/range, longest response, "
        "cap reaching, token volume, and mean peaks; requires `longtable`) is written to `si_table_per_condition.tex`; "
        "it answers the request for every replication condition literally if the editor prefers it.")
    add("")
    add("## 8. Output files")
    add("")
    for name, what in (
            ("c1_realized_work.py", "analysis script (stdlib + numpy + scipy; run with python -I)"),
            ("reproduction_check.csv", "all reproduction checks"),
            ("key_results.json", "machine-readable key numbers"),
            ("rc_attempts.csv / of_attempts.csv", "per process: case, seed, block, condition, lengths, cap reaching, tokens (total, per GPU), stage peaks"),
            ("rc_case_condition_summary.csv / of_case_condition_summary.csv", "per case x condition means and ranges"),
            ("rc_block_pairs.csv / of_block_pairs.csv", "per block x reservation: MB16 - MB8 token, length, cap and peak differences"),
            ("rc_blocks.csv / of_blocks.csv", "per block: reservation-averaged batch contrasts (peaks and tokens)"),
            ("rc_token_difference_distribution.csv / of_...", "distribution of relative token differences, incl. reservation reference"),
            ("rc_case_token_differences.csv / of_...", "per-case token-difference ranges and contrast ranges"),
            ("rc_correlations.csv / of_...", "Spearman correlations (pooled and within-case)"),
            ("rc_restricted_overall.csv / rc_restricted_by_case.csv / of_...", "token-matched restricted contrasts"),
            ("rc_case_adjustment.csv / of_...", "per-case OLS: unadjusted vs token-adjusted micro-batch coefficient"),
            ("si_table_compact.tex, si_table_per_condition.tex", "proposed SI tables")):
        add(f"- `{name}`: {what}")
    add("")
    (out / "REPORT.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
