#!/usr/bin/env python3
"""Failure-inclusive multi-shape, multi-load serving-to-RL transfer analysis."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from memory_tuner.benchmark_v2_cross_family_analysis import load_results
from memory_tuner.benchmark_v2_inference import inference_row, mean_value


MEMORY_LIMIT_MIB = 38_912.0
MODEL_LABELS = {
    "qwen25_1p5b": "Qwen2.5-1.5B",
    "qwen25_3b": "Qwen2.5-3B",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]) if rows else ["empty"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def number(value: object, default: float = math.nan) -> float:
    if value in ("", None):
        return default
    return float(value)


def shape_key(row: Mapping) -> tuple[str, int, int, int, int, float]:
    return (
        str(row["model"]),
        int(float(row["input_len"])),
        int(float(row["output_len"])),
        int(float(row["max_model_len"])),
        int(float(row["max_num_seqs"])),
        number(row["gpu_memory_utilization"]),
    )


def rl_shape_key(specification: Mapping) -> tuple[str, int, int, int, int, float]:
    return (
        str(specification["model"]),
        int(float(specification["max_prompt_length"])),
        int(float(specification["max_response_length"])),
        int(float(specification["max_model_len"])),
        int(float(specification["max_num_seqs"])),
        number(specification["vllm_gpu_memory_utilization"]),
    )


def safe(rows: Sequence[Mapping]) -> int:
    return int(
        len(rows) == 3
        and all(
            int(float(row.get("success", 0))) == 1
            and number(row.get("peak_gpu_memory_mib")) <= MEMORY_LIMIT_MIB
            for row in rows
        )
    )


def mean(rows: Sequence[Mapping], field: str) -> float:
    values = [number(row.get(field)) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    return statistics.mean(values) if values else math.nan


def has_service_metrics(row: Mapping) -> bool:
    return all(
        math.isfinite(number(row.get(field)))
        for field in (
            "result_request_throughput",
            "result_output_throughput",
            "result_p95_ttft_ms",
            "result_p95_e2el_ms",
        )
    )


def sequence_signature(specification: Mapping) -> str:
    """Return the frozen sequence-shape label, not only the source dataset."""
    dataset = str(specification["dataset"]).strip().lower()
    prompt = int(float(specification["max_prompt_length"]))
    response = int(float(specification["max_response_length"]))
    max_sequences = int(float(specification["max_num_seqs"]))
    known = {
        ("gsm8k", 1024, 2048, 64): "GSM8K-long",
        ("math", 512, 512, 128): "MATH-short",
    }
    return known.get(
        (dataset, prompt, response, max_sequences),
        (
            f"{dataset.upper()}-{prompt}x{response}"
            f"-n{max_sequences}"
        ),
    )


def zero_event_two_sided_upper95(independent_cases: int) -> float:
    """Exact Clopper--Pearson upper bound for zero events."""
    if independent_cases <= 0:
        return math.nan
    return 1.0 - (0.025 ** (1.0 / independent_cases))


def metric(value: float, suffix: str, digits: int = 1) -> str:
    return f"{value:.{digits}f} {suffix}" if math.isfinite(value) else "--"


def _transfer_cluster(row: Mapping) -> str:
    return f"{row['model_family']}|{row['sequence_signature']}"


def transfer_inference(
    base_rows: Sequence[Mapping],
    load_rows: Sequence[Mapping],
) -> list[dict]:
    output = [
        inference_row(
            study="multishape_transfer",
            estimand="offline_safe_colocated_unsafe",
            rows=base_rows,
            cluster=_transfer_cluster,
            statistic=lambda values: mean_value(
                values,
                "offline_false_safe",
            ),
            unit="probability",
        ),
        inference_row(
            study="multishape_transfer",
            estimand="online_safe_colocated_unsafe",
            rows=load_rows,
            cluster=_transfer_cluster,
            statistic=lambda values: mean_value(
                values,
                "online_false_safe",
            ),
            unit="probability",
        ),
    ]
    for tier in ("low", "capacity", "overload"):
        subset = [
            row for row in load_rows if str(row["load_tier"]) == tier
        ]
        output.append(
            inference_row(
                study="multishape_transfer",
                subgroup=tier,
                estimand="online_safe_colocated_unsafe",
                rows=subset,
                cluster=_transfer_cluster,
                statistic=lambda values: mean_value(
                    values,
                    "online_false_safe",
                ),
                unit="probability",
            )
        )
    return output


def summarize(
    rl_matrix: Sequence[Mapping],
    rl_results: Sequence[Mapping],
    offline: Sequence[Mapping],
    online: Sequence[Mapping],
) -> tuple[list[dict], list[dict]]:
    rl_specs = {row["experiment_id"]: row for row in rl_matrix}
    rl_groups: dict[tuple, list[dict]] = defaultdict(list)
    for result in rl_results:
        spec = rl_specs[result["experiment_id"]]
        rl_groups[rl_shape_key(spec)].append(result)
    offline_groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in offline:
        offline_groups[shape_key(row)].append(row)
    online_groups: dict[tuple[tuple, float], list[dict]] = defaultdict(list)
    for row in online:
        online_groups[(shape_key(row), number(row["request_rate"]))].append(row)

    base_rows = []
    load_rows = []
    if set(rl_groups) != set(offline_groups):
        raise ValueError("colocated and offline shape cells do not match")
    for key in sorted(rl_groups):
        rl_rows = rl_groups[key]
        offline_rows = offline_groups[key]
        if len(rl_rows) != 3 or len(offline_rows) != 3:
            raise ValueError(f"{key}: expected three RL and offline repetitions")
        spec = rl_specs[rl_rows[0]["experiment_id"]]
        signature = sequence_signature(spec)
        rates = sorted(
            rate for (online_key, rate) in online_groups if online_key == key
        )
        if len(rates) != 3:
            raise ValueError(f"{key}: expected three online load rates")
        rl_safe = safe(rl_rows)
        offline_safe = safe(offline_rows)
        base_rows.append(
            {
                "model_family": spec["model_family"],
                "model": spec["model"],
                "dataset": spec["dataset"],
                "sequence_signature": signature,
                "input_len": key[1],
                "output_len": key[2],
                "max_num_seqs": key[4],
                "gpu_memory_utilization": key[5],
                "rl_safe": rl_safe,
                "rl_success_rate": mean(rl_rows, "success"),
                "rl_peak_mib_mean": mean(rl_rows, "peak_gpu_memory_mib"),
                "offline_safe": offline_safe,
                "offline_success_rate": mean(offline_rows, "success"),
                "offline_peak_mib_mean": mean(
                    offline_rows, "peak_gpu_memory_mib"
                ),
                "offline_output_tokens_per_second_mean": mean(
                    offline_rows, "output_tokens_per_second"
                ),
                "offline_false_safe": int(offline_safe and not rl_safe),
            }
        )
        for rate_index, rate in enumerate(rates):
            online_rows = online_groups[(key, rate)]
            if len(online_rows) != 3:
                raise ValueError(
                    f"{key} rate={rate}: expected three online repetitions"
                )
            online_safe = safe(online_rows)
            load_rows.append(
                {
                    "model_family": spec["model_family"],
                    "model": spec["model"],
                    "dataset": spec["dataset"],
                    "sequence_signature": signature,
                    "input_len": key[1],
                    "output_len": key[2],
                    "max_num_seqs": key[4],
                    "gpu_memory_utilization": key[5],
                    "load_tier": ("low", "capacity", "overload")[rate_index],
                    "request_rate": rate,
                    "online_safe": online_safe,
                    "online_success_rate": mean(online_rows, "success"),
                    "online_peak_mib_mean": mean(
                        online_rows, "peak_gpu_memory_mib"
                    ),
                    "online_request_throughput_mean": mean(
                        online_rows, "result_request_throughput"
                    ),
                    "online_output_throughput_mean": mean(
                        online_rows, "result_output_throughput"
                    ),
                    "online_p95_ttft_ms_mean": mean(
                        online_rows, "result_p95_ttft_ms"
                    ),
                    "online_p95_e2el_ms_mean": mean(
                        online_rows, "result_p95_e2el_ms"
                    ),
                    "online_service_metric_repetitions": sum(
                        has_service_metrics(row) for row in online_rows
                    ),
                    "rl_safe": rl_safe,
                    "online_false_safe": int(online_safe and not rl_safe),
                }
            )
    return base_rows, load_rows


def render(
    base_rows: Sequence[Mapping],
    load_rows: Sequence[Mapping],
    inference: Sequence[Mapping],
) -> str:
    offline_false = sum(int(row["offline_false_safe"]) for row in base_rows)
    online_false = sum(int(row["online_false_safe"]) for row in load_rows)
    inference_by_key = {
        (str(row["subgroup"]), str(row["estimand"])): row
        for row in inference
    }
    offline_interval = inference_by_key[
        ("all", "offline_safe_colocated_unsafe")
    ]
    online_interval = inference_by_key[
        ("all", "online_safe_colocated_unsafe")
    ]
    by_tier = defaultdict(list)
    for row in load_rows:
        by_tier[row["load_tier"]].append(row)
    by_case_base: dict[tuple[str, str], list[Mapping]] = defaultdict(list)
    by_case_load: dict[tuple[str, str], list[Mapping]] = defaultdict(list)
    for row in base_rows:
        by_case_base[
            (
                str(row["model_family"]),
                str(row["sequence_signature"]),
            )
        ].append(row)
    for row in load_rows:
        by_case_load[
            (
                str(row["model_family"]),
                str(row["sequence_signature"]),
            )
        ].append(row)
    lines = [
        "# Major-revision multi-shape, multi-load transfer study",
        "",
        f"- Matched model/shape/utilization cells: **{len(base_rows)}**.",
        f"- Standalone-safe but colocated-unsafe cells: "
        f"**{offline_false}/{len(base_rows)}** "
        f"(descriptive four-case cluster-bootstrap 95% interval "
        f"{100 * float(offline_interval['ci95_low']):.1f}% to "
        f"{100 * float(offline_interval['ci95_high']):.1f}%).",
        f"- Loaded-online-safe but colocated-unsafe cells across load tiers: "
        f"**{online_false}/{len(load_rows)}** "
        f"(descriptive four-case cluster-bootstrap 95% interval "
        f"{100 * float(online_interval['ci95_low']):.1f}% to "
        f"{100 * float(online_interval['ci95_high']):.1f}%).",
        "",
        "| Load tier | Online-safe cells | False-safe transfers | "
        "Service-metric repetitions (complete cells) | "
        "Mean achieved request rate | "
        "Mean p95 TTFT |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for tier in ("low", "capacity", "overload"):
        rows = by_tier[tier]
        interval = inference_by_key[
            (tier, "online_safe_colocated_unsafe")
        ]
        throughput = mean(rows, "online_request_throughput_mean")
        ttft = mean(rows, "online_p95_ttft_ms_mean")
        service_metric_repetitions = sum(
            int(row["online_service_metric_repetitions"]) for row in rows
        )
        complete_service_metric_cells = sum(
            int(row["online_service_metric_repetitions"]) == 3
            for row in rows
        )
        lines.append(
            f"| {tier} | {sum(int(row['online_safe']) for row in rows)}/"
            f"{len(rows)} | "
            f"{sum(int(row['online_false_safe']) for row in rows)}/"
            f"{len(rows)} "
            f"[{100 * float(interval['ci95_low']):.1f}%, "
            f"{100 * float(interval['ci95_high']):.1f}%] | "
            f"{service_metric_repetitions}/{3 * len(rows)} "
            f"({complete_service_metric_cells}/{len(rows)}) | "
            f"{metric(throughput, 'req/s', digits=2)} | "
            f"{metric(ttft, 'ms')} |"
        )
    lines.extend(
        [
            "",
            "| Model | Sequence signature | Offline false-safe | "
            "Online false-safe |",
            "|---|---|---:|---:|",
        ]
    )
    for case in sorted(by_case_base):
        case_base = by_case_base[case]
        case_load = by_case_load[case]
        lines.append(
            f"| {MODEL_LABELS.get(case[0], case[0])} | {case[1]} | "
            f"{sum(int(row['offline_false_safe']) for row in case_base)}/"
            f"{len(case_base)} | "
            f"{sum(int(row['online_false_safe']) for row in case_load)}/"
            f"{len(case_load)} |"
        )
    lines.extend(
        [
            "",
            "The study matches model, sequence caps, concurrency cap, GPU "
            "count, and vLLM reservation. Isolated serving remains TP=2, "
            "whereas colocated RL uses TP=1 rollout workers and retains "
            "training state. The topology difference is the transfer target, "
            "not an uncontrolled claim of identical execution.",
            "Service-rate and latency means are computed over finite "
            "repetition-level service metrics and then averaged within "
            "load-specific cells. The table reports both contributing "
            "repetitions and cells with all three service repetitions; "
            "failed scientific attempts remain unsafe in the primary "
            "analysis.",
            (
                "No mismatch occurs in any of the four independent "
                "model--sequence cases. The resulting 0--0% percentile "
                "cluster-bootstrap intervals are descriptive rather than "
                "evidence of zero population risk: a two-sided exact "
                "binomial calculation for zero events in four cases has an "
                f"upper 95% bound of "
                f"{100 * zero_event_two_sided_upper95(4):.1f}%."
                if not offline_false and not online_false
                else ""
            ),
            (
                "The mismatch remains observable under the prescribed "
                "loaded-online schedules."
                if online_false
                else "No loaded-online false-safe transfer is observed under "
                "the prescribed schedules; this arm therefore does not "
                "extend the original online mismatch."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl-matrix", type=Path, required=True)
    parser.add_argument("--rl-root", type=Path, required=True)
    parser.add_argument("--offline-matrix", type=Path, required=True)
    parser.add_argument("--offline", type=Path, required=True)
    parser.add_argument("--online", type=Path, required=True)
    parser.add_argument(
        "--base-output",
        type=Path,
        default=Path("profiles/major_revision/transfer-base-cells.csv"),
    )
    parser.add_argument(
        "--load-output",
        type=Path,
        default=Path("profiles/major_revision/transfer-load-cells.csv"),
    )
    parser.add_argument(
        "--inference-output",
        type=Path,
        default=Path("profiles/major_revision/transfer-inference.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("paper/major_revision_transfer.md"),
    )
    args = parser.parse_args()
    rl_matrix = read_csv(args.rl_matrix)
    rl_results = load_results(args.rl_matrix, args.rl_root)
    offline_ids = {
        row["experiment_id"] for row in read_csv(args.offline_matrix)
    }
    offline = [
        row
        for row in read_csv(args.offline)
        if row["experiment_id"] in offline_ids
    ]
    if len(offline) != len(offline_ids):
        raise SystemExit(
            f"expected {len(offline_ids)} offline profiles, "
            f"found {len(offline)}"
        )
    base_rows, load_rows = summarize(
        rl_matrix,
        rl_results,
        offline,
        read_csv(args.online),
    )
    inference = transfer_inference(base_rows, load_rows)
    write_csv(args.base_output, base_rows)
    write_csv(args.load_output, load_rows)
    write_csv(args.inference_output, inference)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        render(base_rows, load_rows, inference),
        encoding="utf-8",
    )
    print(
        f"analyzed {len(base_rows)} base cells and "
        f"{len(load_rows)} loaded-online cells"
    )


if __name__ == "__main__":
    main()
