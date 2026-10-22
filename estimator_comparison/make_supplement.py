#!/usr/bin/env python3
"""Write the DeepSpeed/analytic estimator supplement section from results/."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
METHODS = [
    ("ds3_full", "ZeRO-3, all parameters trainable"),
    ("ds3_lora+frozen_base", "ZeRO-3, LoRA states + frozen BF16 base"),
    ("ds3_lora+frozen_base+reservation", "\\quad + vLLM reservation"),
    ("ds3_lora+frozen_base+reservation+activations", "\\quad + reservation + activations"),
    ("ds2_full", "ZeRO-2, all parameters trainable"),
    ("analytic_bf16", "Analytic component budget (BF16)"),
    ("source_label", "Source-label copy (reference)"),
    ("always_approve", "Always approve (reference)"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with (args.results / "score_comparison_all_methods.csv").open() as handle:
        scores = {row["method"]: row for row in csv.DictReader(handle) if row["track"] == "all"}
    summary = json.loads((args.results / "deepspeed_summary.json").read_text())
    rows = []
    for key, name in METHODS:
        row = scores[key]
        rows.append(f"{name} & {float(row['three_state_accuracy']):.3f} & {float(row['within_margin_recall']):.3f} & "
                    f"{row['approved_memory_failure']}/{row['target_memory_failure']} & "
                    f"{row['approved_above_margin']}/{row['target_above_margin']} \\\\")
    any_row = next(iter(scores.values()))
    text = [
        r"\section{Query-level comparison with DeepSpeed memory estimators}\label{ssec:deepspeed}",
        "",
        "This comparison complements the native LLMem validation. It applies DeepSpeed's published "
        f"model-state estimators (DeepSpeed {summary['deepspeed_version']}, called directly; "
        r"\nolinkurl{estimate_zero3_model_states_mem_needs} and its ZeRO-2 counterpart) to the "
        f"{any_row['queries']} public benchmark queries, using only target settings and model architecture. "
        "These queries reuse 94 distinct target configurations, so they are not independent experiments. "
        "Predicted per-device memory is compared with the operating limit and capacity to give a three-state "
        "label, scored with the released evaluator. Composites add the requested vLLM reservation and an "
        "activation term. A transparent analytic component budget is shown for reference. These estimators "
        "target training state; applying them to colocated RL is outside their intended scope, so the "
        "comparison shows which terms they omit, not their accuracy for training.",
        "",
        r"\begin{table}[!ht]\centering\small",
        r"\caption{Estimator decisions on all benchmark targets. Recall is within-margin recall; approval means a within-margin prediction; failures and above-margin approvals are counted against all such targets.}",
        r"\label{stab:deepspeed}",
        r"\begin{tabular}{@{}lrrrr@{}}\toprule",
        r"Predictor & Acc. & Recall & Appr.\ failures & Appr.\ above \\\midrule",
        *rows,
        r"\bottomrule\end{tabular}\end{table}",
        "",
        "Model states alone approve every memory-failing target. Adding the reservation removes few of these "
        "approvals, and adding activations rejects many usable targets because stages that never coexist "
        "under state release are summed. The released artifact (\\texttt{estimator\\_comparison/}) records "
        "every estimator call with its arguments, returned bytes, and source hashes.",
        "",
    ]
    args.output.write_text("\n".join(text) + "\n")


if __name__ == "__main__":
    main()
