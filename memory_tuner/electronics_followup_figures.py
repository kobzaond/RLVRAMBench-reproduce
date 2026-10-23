"""Render audited follow-up evidence, preserving failures and missing slots."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np

from memory_tuner.electronics_followup_analysis import (
    COMPLETED, matched_contrasts, summarize_cases, write_rows,
)
from memory_tuner.electronics_review_analysis import save

SHORT = {"Qwen/Qwen2.5-1.5B-Instruct": "Qwen 1.5B",
         "Qwen/Qwen2.5-3B-Instruct": "Qwen 3B",
         "Qwen/Qwen2.5-7B-Instruct": "Qwen 7B",
         "mistralai/Mistral-7B-Instruct-v0.3": "Mistral 7B"}
WORKLOAD = {"gsm8k": "GSM8K", "code_heavy_tail": "Long code"}
COLORS = {"within_margin": "#b8d9e9", "above_margin": "#f6d996",
          "memory_failure": "#e7aba7", "unresolved": "#dddddd"}


def outcome_label(rows):
    counts = Counter(r["status"] for r in rows)
    eligible = sum(counts[s] for s in ("within_margin", "above_margin", "memory_failure"))
    label = "/".join(str(counts[s]) for s in ("within_margin", "above_margin", "memory_failure"))
    if eligible != len(rows):
        label += "\n-"
    state = ("unresolved" if eligible != len(rows) else "memory_failure" if counts["memory_failure"]
             else "above_margin" if counts["above_margin"] else "within_margin")
    return label, state


def reservation_figure(rows, figures, suffix):
    keys = sorted({(r["model"], r["dataset"], r["parameter_offload"]) for r in rows})
    fractions = sorted({float(r["vllm_gpu_memory_utilization"]) for r in rows})
    fig, ax = plt.subplots(figsize=(5.45, 3.7))
    for i, (model, dataset, offload) in enumerate(keys):
        for j, reservation in enumerate(fractions):
            group = [r for r in rows if (r["model"], r["dataset"], r["parameter_offload"]) ==
                     (model, dataset, offload) and float(r["vllm_gpu_memory_utilization"])==reservation]
            label, state = outcome_label(group)
            ax.add_patch(Rectangle((j-.5, i-.5), 1, 1, facecolor=COLORS[state],
                                   edgecolor="white", linewidth=2))
            ax.text(j, i, label, ha="center", va="center", fontsize=8)
    ax.set_xlim(-.5,len(fractions)-.5)
    ax.set_ylim(len(keys)-.5,-.5)
    ax.set_xticks(range(len(fractions)), [f"{r:.2f}" for r in fractions])
    ax.set_yticks(range(len(keys)),
                  [f"{SHORT[m]} / {WORKLOAD[d]} / offload {'on' if o=='True' else 'off'}"
                   for m,d,o in keys])
    ax.set_xlabel("Generation reservation")
    ax.set_title("Counts: within / above / memory failure", fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    save(fig, figures, "electronics_reservation_sensitivity"+suffix)


def adaptation_figure(rows, figures, results, suffix):
    native = [r for r in rows if r["use_kl_loss"]=="True"]
    contrasts = matched_contrasts(native, "adaptation")
    keys = sorted({(r["model"], r["dataset"]) for r in native})
    fig, axes = plt.subplots(1, 2, figsize=(5.45, 3.65), sharex=True, sharey=True)
    summaries = []
    colors = {"completed_peak_mib": "#0072B2", "actor_stage_peak_mib": "#D55E00"}
    for ax, reservation in zip(axes, (.45, .65)):
        labels = []
        for i, (model, dataset, offload) in enumerate((m,d,o) for m,d in keys for o in ("False","True")):
            group = [r for r in contrasts if (r["model"],r["dataset"],r["parameter_offload"]) ==
                     (model,dataset,offload) and float(r["reservation"])==reservation]
            completed = [r for r in group if r["both_completed"]]
            labels.append(f"{SHORT[model]} / {WORKLOAD[dataset]} / {'on' if offload=='True' else 'off'}")
            for metric, offset in (("completed_peak_mib",-.12),("actor_stage_peak_mib",.12)):
                values = [r[metric+"_difference"]/1024 for r in completed if metric+"_difference" in r]
                if values:
                    ax.scatter(values, i+offset+np.linspace(-.035,.035,len(values)),
                               c=colors[metric], s=10, alpha=.6)
                    ax.scatter(mean(values),i+offset,c=colors[metric],s=18,marker="D")
                summaries.append(dict(model=model,dataset=dataset,actor_offload=offload,
                                      use_kl_loss=True,reservation=reservation,metric=metric,planned_pairs=len(group),
                                      completed_pairs=len(values),
                                      mean_difference_mib=mean(values)*1024 if values else None,
                                      min_difference_mib=min(values)*1024 if values else None,
                                      max_difference_mib=max(values)*1024 if values else None))
            ax.text(.98,i,f"{len(completed)}/{len(group)}",va="center",ha="right",
                    transform=ax.get_yaxis_transform(),fontsize=7.5,color=".25")
        ax.axvline(0,color=".55",lw=.8)
        ax.set_yticks(range(len(labels)),labels)
        ax.set_title(f"Reservation {reservation:.2f}",fontsize=9)
        ax.set_xlim(-1, 25)
        ax.set_xticks((0, 10, 20))
        ax.grid(axis="x",alpha=.16)
    axes[0].set_ylim(len(keys)*2-.5,-.5)
    for metric,label in (("completed_peak_mib","Whole run"),("actor_stage_peak_mib","Actor stage")):
        axes[1].scatter([],[],c=colors[metric],label=label)
    handles,legend_labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc="lower center",ncol=2,frameon=False,fontsize=8)
    fig.supxlabel("Native full − LoRA peak (GiB)", y=.10, fontsize=9)
    fig.tight_layout(rect=(0,.15,1,1), w_pad=.8)
    save(fig, figures, "electronics_full_parameter_contrasts"+suffix)
    controls = matched_contrasts([r for r in rows if r["use_kl_loss"]=="False"], "adaptation")
    for offload in ("False", "True"):
        group = [r for r in controls if r["parameter_offload"]==offload]
        for metric in ("completed_peak_mib", "actor_stage_peak_mib"):
            values = [r[metric+"_difference"] for r in group
                      if r["both_completed"] and metric+"_difference" in r]
            if group:
                summaries.append(dict(
                    model=group[0]["model"],dataset=group[0]["dataset"],actor_offload=offload,
                    use_kl_loss=False,reservation=float(group[0]["reservation"]),metric=metric,
                    planned_pairs=len(group),completed_pairs=len(values),
                    mean_difference_mib=mean(values) if values else None,
                    min_difference_mib=min(values) if values else None,
                    max_difference_mib=max(values) if values else None))
    write_rows(results/("adaptation_contrast_summary"+suffix+".csv"), summaries)


def timing_figure(rows, figures, results, suffix):
    cases = sorted({(r["model"], r["dataset"]) for r in rows})
    conditions = ("off1-r45-release1", "off0-r45-release1", "off1-r70-release1", "off1-r45-release0")
    labels = ("Anchor", "Offload off", "Reservation .70", "State resident")
    fig, axes = plt.subplots(1, len(cases), figsize=(6.5, 3.5), squeeze=False)
    summaries = []
    for ax, case in zip(axes[0], cases):
        for index, (condition,label) in enumerate(zip(conditions,labels)):
            group = [r for r in rows if (r["model"],r["dataset"])==case and r["condition"]==condition]
            completed = [r for r in group if r["status"] in COMPLETED and "steady_step_seconds" in r]
            values = [r["steady_step_seconds"] for r in completed]
            ax.scatter(np.full(len(values),index)+np.linspace(-.07,.07,len(values)),values,
                       c="#0072B2",s=25,alpha=.65)
            if values:
                ax.scatter(index,mean(values),c="#0072B2",s=42,marker="D")
            annotation,_=outcome_label(group)
            ax.text(index,.98,annotation,transform=ax.get_xaxis_transform(),ha="center",va="top",fontsize=9)
            item=dict(model=case[0],dataset=case[1],condition=condition,planned=len(group),
                      completed=len(completed),states=dict(Counter(r["status"] for r in group)))
            for key in ("steady_step_seconds","steady_gen_seconds","steady_update_actor_seconds",
                        "steady_update_weights_seconds","steady_ref_seconds","steady_aggregate_tokens_per_second",
                        "steady_response_tokens_mean","invocation_seconds",
                        "host_current_peak_gib","host_anonymous_peak_gib","host_file_peak_gib"):
                observed=[r[key] for r in completed if key in r]
                item[key+"_n"]=len(observed)
                if observed:
                    item.update({key+"_mean":mean(observed),key+"_min":min(observed),key+"_max":max(observed)})
            summaries.append(item)
        ax.set_title(f"{SHORT[case[0]]} / {WORKLOAD[case[1]]}",fontsize=10)
        ax.set_xticks(range(len(conditions)),labels,rotation=15,ha="right")
        ax.set_ylim(bottom=0)
        low,high=ax.get_ylim()
        ax.set_ylim(low,high*1.25)
        ax.set_ylabel("Mean iteration time, updates 3–20 (s)")
        ax.grid(axis="y",alpha=.16)
    fig.suptitle("Points: seeds; diamonds: means; counts: within / above / failed",fontsize=9)
    fig.tight_layout()
    save(fig,figures,"electronics_iteration_cost"+suffix)
    write_rows(results/("timing_condition_summary"+suffix+".csv"),summaries)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--results",type=Path,help="Audited inputs and generated summary tables")
    parser.add_argument("--figures",type=Path,help="Output directory for figures")
    parser.add_argument("--panel",choices=("adaptation","reservation","timing"),action="append")
    parser.add_argument("--allow-partial",action="store_true")
    args=parser.parse_args()
    results=args.results or args.root/"profiles/electronics_revision"
    figures=args.figures or args.root/"paper/submission/mdpi_submission/figures"
    figures.mkdir(parents=True, exist_ok=True)
    suffix="_preview" if args.allow_partial else ""
    plt.rcParams.update({"font.size":8,"axes.labelsize":9,"axes.spines.top":False,"axes.spines.right":False,
                         "pdf.fonttype":42,"ps.fonttype":42})
    for panel in args.panel or ("adaptation","reservation","timing"):
        summary=json.loads((results/panel/"summary.json").read_text())
        if not args.allow_partial and not summary["all_blocks_finished"]:
            raise ValueError(f"{panel}: experiments still running; use explicitly marked preview only")
        rows=json.loads((results/panel/"outcomes.json").read_text())
        if panel=="reservation":
            reservation_figure(rows,figures,suffix)
        elif panel=="adaptation":
            adaptation_figure(rows,figures,results,suffix)
        else:
            timing_figure(rows,figures,results,suffix)
        print(panel+suffix+" rendered")


if __name__=="__main__":
    main()
