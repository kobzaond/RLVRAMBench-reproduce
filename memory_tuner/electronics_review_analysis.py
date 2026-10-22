"""Reconstruct three review analyses from the historical, eligible trial tables.

No fitting or new GPU outcomes enter this descriptive analysis. Input hashes,
individual observations, excluded slots, and denominators accompany the figures.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from memory_tuner.electronics_failure_evidence import diagnose
from memory_tuner.artifact_paths import recorded_path

MODELS = {"qwen25_1p5b": "Qwen 1.5B", "qwen25_3b": "Qwen 3B",
          "phi4_mini": "Phi 4 mini", "granite33_2b": "Granite 2B"}
WORKLOADS = {"gsm8k": "GSM8K", "math": "MATH", "code_standard": "Code",
             "code_heavy_tail": "Long code"}
FAILURES = {"startup_budget_rejection": "Startup budget rejection",
            "startup_allocation_oom": "Startup allocation OOM", "rollout_oom": "Generation OOM",
            "weight_sync_oom": "Weight-sync OOM", "actor_update_oom": "Actor-update OOM"}


def read(root, relative, hashes):
    path = root / relative
    hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    with path.open("w", newline="") as handle:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save(fig, directory, name):
    fig.savefig(directory / (name + ".pdf"), bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(directory / (name + ".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def stage_analysis(root, output, figures, hashes):
    rows = read(root, "benchmark/replication_control/results/replication_control_trials.csv", hashes)
    groups = defaultdict(dict)
    for row in rows:
        if row["condition"] in groups[row["block_id"]]:
            raise ValueError("Duplicate condition in a block")
        groups[row["block_id"]][row["condition"]] = row
    contrasts, excluded = [], []
    for block, conditions in sorted(groups.items()):
        if (set(conditions) != {"base", "actor_only", "reservation_only", "joint"}
                or any(r["scientific_eligible"] != "True" for r in conditions.values())):
            excluded.extend(conditions.values())
            continue
        for reservation, low, high in [(.60, "base", "actor_only"), (.70, "reservation_only", "joint")]:
            a, b = conditions[low], conditions[high]
            for metric in ("actor_update", "whole_run"):
                contrasts.append(dict(block_id=block, model_family=a["model_family"],
                                      dataset=a["dataset"], training_seed=a["training_seed"],
                                      reservation=reservation, metric=metric,
                                      batch8_experiment_id=a["experiment_id"],
                                      batch16_experiment_id=b["experiment_id"],
                                      batch8_peak_mib=float(a[metric+"_peak_mib"]),
                                      batch16_peak_mib=float(b[metric+"_peak_mib"]),
                                      difference_mib=float(b[metric+"_peak_mib"])-float(a[metric+"_peak_mib"])))
    write(output / "stage_paired_contrasts.csv", contrasts)
    write(output / "stage_excluded_slots.csv", excluded)
    cases = sorted({(r["model_family"], r["dataset"]) for r in contrasts},
                   key=lambda c: (c[0], list(WORKLOADS).index(c[1])))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.4), sharey=True, sharex=True)
    colors = {"actor_update": "#0072B2", "whole_run": "#D55E00"}
    summary = []
    for ax, reservation in zip(axes, (.60, .70)):
        for i, case in enumerate(cases):
            for metric, offset in [("actor_update", -.13), ("whole_run", .13)]:
                selected = [r for r in contrasts if (r["model_family"], r["dataset"]) == case
                            and r["metric"] == metric and r["reservation"] == reservation]
                values = [r["difference_mib"] / 1024 for r in selected]
                ax.scatter(values, i + offset + np.linspace(-.05, .05, len(values)),
                           s=18, alpha=.55, c=colors[metric], edgecolors="none")
                ax.scatter(mean(values), i+offset, c=colors[metric], marker="D", s=28)
                summary.append(dict(model_family=case[0], dataset=case[1], reservation=reservation,
                                    metric=metric, complete_blocks=len(values),
                                    mean_difference_mib=mean(values)*1024))
        ax.axvline(0, color=".55", lw=.8)
        ax.set_title(f"Generation reservation {reservation:.2f}")
        ax.set_xlabel("Batch 16 − batch 8 peak (GiB)")
        ax.grid(axis="x", alpha=.18)
    axes[0].set_yticks(range(len(cases)),
                      [f"{MODELS[m]} / {WORKLOADS[d]}" for m, d in cases])
    axes[0].invert_yaxis()
    for metric, label in [("actor_update", "Actor stage"), ("whole_run", "Whole run")]:
        axes[1].scatter([], [], c=colors[metric], label=label)
    axes[1].legend(loc="upper right", frameon=False)
    fig.suptitle("35 complete same-device blocks; points: seeds; diamonds: case means", fontsize=11)
    fig.tight_layout()
    save(fig, figures, "electronics_stage_contrasts")
    write(output / "stage_case_contrasts.csv", summary)
    return dict(planned_slots=len(rows), complete_blocks=len(contrasts)//4,
                excluded_slots=len(excluded),
                equal_case_actor_batch_mib=mean(r["mean_difference_mib"] for r in summary
                                               if r["metric"] == "actor_update"),
                equal_case_whole_batch_mib=mean(r["mean_difference_mib"] for r in summary
                                               if r["metric"] == "whole_run"))


def horizon_analysis(root, output, figures, hashes):
    observations = []
    for relative, protocol, horizon, paired in [
        ("profiles/standard_grpo/temporal-40-audit.csv", "Historical screen / 40 steps", 40, False),
        ("profiles/standard_grpo/temporal-100-trials.csv", "Historical screen / 100 steps", 100, False),
        ("profiles/standard_grpo/revision-temporal-pairs.csv", "Paired screen / 100 steps", 100, True),
    ]:
        rows = read(root, relative, hashes)
        for row in rows:
            source = row["source_safe"] if paired else row["source_one_step_safe"]
            target = row["long_safe"] if paired else row["long_run_safe"]
            observations.append(dict(protocol=protocol, horizon=horizon, model_family=row["model_family"],
                                     dataset=row["dataset"], training_seed=row["training_seed"],
                                     observation_id=row["pair_id"] if paired else row["experiment_id"],
                                     source_within_margin=int(source), long_within_margin=int(target),
                                     agreement=int(source == target),
                                     missed_margin_or_failure=int(source == "1" and target == "0"),
                                     outcome_scope="eligible memory outcomes; extra excluded attempts not replaced here"))
    write(output / "horizon_observations.csv", observations)
    groups = defaultdict(list)
    for row in observations:
        groups[(row["protocol"], row["model_family"], row["dataset"])].append(row)
    summary = []
    for (protocol, model, dataset), values in groups.items():
        summary.append(dict(protocol=protocol, model_family=model, dataset=dataset,
                            comparisons=len(values), agreements=sum(r["agreement"] for r in values),
                            within_to_unsuitable=sum(r["missed_margin_or_failure"] for r in values),
                            source_within=sum(r["source_within_margin"] for r in values),
                            long_within=sum(r["long_within_margin"] for r in values)))
    write(output / "horizon_agreement.csv", summary)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7), sharex=True)
    for ax, protocol in zip(axes, dict.fromkeys(r["protocol"] for r in summary)):
        selected = [r for r in summary if r["protocol"] == protocol]
        for i, r in enumerate(selected):
            n, yes = r["comparisons"], r["agreements"]
            ax.barh(i, yes / n, color="#0072B2", height=.6)
            ax.barh(i, (n-yes) / n, left=yes / n, color="#D55E00", height=.6)
            ax.text(.02, i, f"{yes}/{n} agree", va="center", color="white", fontsize=11)
        ax.set_yticks(range(len(selected)),
                      [f"{MODELS[r['model_family']]} / {WORKLOADS[r['dataset']]}" for r in selected])
        ax.invert_yaxis()
        ax.set_title(protocol.replace(" / ", "\n"), fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Fraction of eligible comparisons")
    fig.tight_layout()
    save(fig, figures, "electronics_horizon_agreement")
    repeated = read(root, "profiles/standard_grpo/revision-temporal-cases.csv", hashes)
    return dict(comparisons=len(observations),
                agreements=sum(r["agreement"] for r in observations),
                paired_case_labels_agree=sum(r["source_conservative_safe"] == r["long_conservative_safe"]
                                             for r in repeated),
                paired_case_denominator=len(repeated),
                caveat="Historical and paired protocols are separate; no pooled accuracy estimate. "
                       "Original 100-step panel: two extra excluded attempts; paired panel: three. "
                       "Binary approval agreement does not resolve the type of an unsuitable outcome.")


def failure_analysis(root, output, figures, hashes, raw_root):
    observations, seen = [], set()
    for relative, study in [
        ("profiles/standard_grpo/boundary-trials.csv", "Boundary"),
        ("profiles/standard_grpo/same-node-trials.csv", "State release"),
        ("benchmark/offload_factorial/results/slots.csv", "Independent offload"),
    ]:
        rows = read(root, relative, hashes)
        for row in rows:
            if row["experiment_id"] in seen:
                raise ValueError("Repeated run across failure panels")
            seen.add(row["experiment_id"])
            historical = row["failure_kind"]
            evidence = {}
            kind = "completed"
            if historical:
                if row.get("run_log"):
                    log = recorded_path(row["run_log"], raw_root)
                else:
                    # Historical tables carry absolute raw paths except for the
                    # independent panel, whose protocol has one permanent attempt.
                    matches = list((raw_root/"output"/row["run_group"]/row["experiment_id"]).rglob("training-*.log"))
                    if len(matches) != 1:
                        raise ValueError(f"Expected one raw failure log for {row['experiment_id']}")
                    log = matches[0]
                raw = log.read_bytes()
                terminal = row.get("terminal_phase", "initializing")
                evidence = diagnose(raw.decode(errors="replace"), terminal)
                kind = evidence["diagnosis"]
                if kind == "unclassified":
                    raise ValueError(f"Insufficient diagnostic evidence: {row['experiment_id']}")
                evidence.update(raw_log=str(log.relative_to(raw_root)),
                                raw_log_sha256=hashlib.sha256(raw).hexdigest())
                evidence["evidence_lines"] = json.dumps(evidence["evidence_lines"])
            observations.append(dict(study=study, experiment_id=row["experiment_id"],
                                     model=row.get("model", row["model_family"]), dataset=row["dataset"],
                                     actor_micro_batch=row["actor_micro_batch"],
                                     reservation=row["vllm_gpu_memory_utilization"],
                                     actor_offload=row["parameter_offload"],
                                     release=row["free_cache_engine"], diagnosis=kind,
                                     historical_diagnosis=historical,
                                     terminal_marker=row.get("terminal_phase", ""),
                                     evidence_path=row.get("artifact_path", relative),
                                     source_table=relative, **{k: v for k, v in evidence.items() if k != "diagnosis"}))
    write(output / "failure_observations.csv", observations)
    summary = []
    for study in dict.fromkeys(r["study"] for r in observations):
        selected = [r for r in observations if r["study"] == study]
        counts = Counter(r["diagnosis"] for r in selected)
        summary.append(dict(study=study, eligible_invocations=len(selected),
                            **{kind: counts[kind] for kind in ("completed", *FAILURES)}))
    write(output / "failure_counts.csv", summary)
    grouped = defaultdict(Counter)
    for row in observations:
        key = tuple(row[k] for k in ("study", "model", "dataset", "actor_micro_batch",
                                     "reservation", "actor_offload", "release"))
        grouped[key][row["diagnosis"]] += 1
    write(output / "failure_by_condition.csv",
          [dict(zip(("study", "model", "dataset", "actor_micro_batch", "reservation",
                     "actor_offload", "release"), key),
                eligible_invocations=sum(counts.values()),
                **{kind: counts[kind] for kind in ("completed", *FAILURES)})
           for key, counts in sorted(grouped.items())])
    fig, ax = plt.subplots(figsize=(9, 3.4))
    x, bottom = np.arange(len(summary)), np.zeros(len(summary))
    for kind, color in zip(FAILURES, ["#0072B2", "#009E73", "#56B4E9", "#E69F00", "#CC79A7"]):
        values = np.array([r[kind] for r in summary])
        ax.bar(x, values, bottom=bottom, color=color, label=FAILURES[kind], width=.55)
        for i, value in enumerate(values):
            if value:
                ax.text(i, bottom[i]+value/2, str(value), ha="center", va="center", fontsize=10)
        bottom += values
    ax.set_xticks(x, [f"{r['study']}\n{int(n)}/{r['eligible_invocations']} memory failures"
                     for r, n in zip(summary, bottom)])
    ax.set_ylabel("Diagnosed memory failures")
    ax.set_ylim(0, max(bottom)*1.13)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    save(fig, figures, "electronics_failure_modes")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--figures", type=Path)
    parser.add_argument("--raw-root", type=Path,
                        help="Artifact containing output/; defaults to --root, without external fallback")
    args = parser.parse_args()
    output = args.output or args.root / "profiles/electronics_revision"
    figures = args.figures or args.root / "paper/submission/mdpi_submission/figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    hashes = {}
    summary = dict(
        stage=stage_analysis(args.root, output, figures, hashes),
        horizon=horizon_analysis(args.root, output, figures, hashes),
        failures=failure_analysis(args.root, output, figures, hashes, args.raw_root or args.root),
        input_sha256=hashes,
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "input_sha256"}, indent=2))


if __name__ == "__main__":
    main()
