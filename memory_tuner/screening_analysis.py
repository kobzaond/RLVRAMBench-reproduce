#!/usr/bin/env python3
"""Validate and summarize the RLVRAMBench cross-model screening study."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def read_matrix(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    ids = [row["experiment_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate experiment IDs in {path}")
    return rows


def load_vllm(
    matrix_path: Path, root: Path
) -> tuple[list[dict], list[str]]:
    expected = {row["experiment_id"]: row for row in read_matrix(matrix_path)}
    artifacts: dict[str, tuple[Path, dict]] = {}
    for path in root.glob("*/trial.json"):
        trial = json.loads(path.read_text())
        experiment_id = trial.get("experiment_id")
        if experiment_id not in expected:
            continue
        if experiment_id in artifacts:
            raise ValueError(f"duplicate vLLM artifact: {experiment_id}")
        artifacts[experiment_id] = (path, trial)

    rows = []
    errors = []
    for experiment_id, config in expected.items():
        if experiment_id not in artifacts:
            continue
        trial_path, trial = artifacts[experiment_id]
        result_path = trial_path.with_name("result.json")
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        expected_tokens = int(config["num_prompts"]) * (
            int(config["input_len"]) + int(config["output_len"])
        )
        success = (
            trial.get("exit_code") == 0
            and bool(result)
            and int(result.get("total_num_tokens", -1)) == expected_tokens
        )
        if not success:
            errors.append(f"invalid vLLM trial: {experiment_id}")
        rows.append(
            {
                **config,
                "job_id": trial.get("job_id"),
                "success": int(success),
                "elapsed_seconds": trial.get("elapsed_seconds"),
                "peak_gpu_memory_mib": trial.get("peak_gpu_memory_mib"),
                "tokens_per_second": result.get("tokens_per_second"),
                "requests_per_second": result.get("requests_per_second"),
                "engine_startup_seconds": result.get("engine_startup_seconds"),
            }
        )
    missing = sorted(set(expected) - set(artifacts))
    return rows, [*errors, *(f"missing vLLM trial: {item}" for item in missing)]


def phase_peaks(path: Path) -> dict[str, float]:
    peaks: dict[str, float] = {}
    if not path.exists():
        return peaks
    with path.open() as handle:
        for sample in csv.reader(handle):
            if len(sample) < 4:
                continue
            _, _, phase, *gpu_memory = sample
            value = max(float(item) for item in gpu_memory)
            peaks[phase] = max(peaks.get(phase, 0.0), value)
    return peaks


def load_phase(
    matrix_path: Path, root: Path, memory_limit_mib: float = 38_912
) -> tuple[list[dict], list[str]]:
    matrix_rows = read_matrix(matrix_path)
    expected = {row["experiment_id"]: row for row in matrix_rows}
    rows = []
    errors = []
    for experiment_id, config in expected.items():
        paths = list((root / experiment_id).glob("trial-*.json"))
        if not paths:
            continue
        if len(paths) != 1:
            raise ValueError(f"duplicate phase artifacts: {experiment_id}")
        trial = json.loads(paths[0].read_text())
        peaks = phase_peaks(Path(trial["phase_memory_csv"]))
        completed_step = False
        failure_kind = ""
        job_id = str(trial["job_id"])
        index = next(
            index
            for index, row in enumerate(matrix_rows)
            if row["experiment_id"] == experiment_id
        )
        out_paths = sorted(
            Path("logs").glob(f"verl-screen-*_{index}.out"),
            key=lambda path: path.stat().st_mtime,
        )
        err_paths = sorted(
            Path("logs").glob(f"verl-screen-*_{index}.err"),
            key=lambda path: path.stat().st_mtime,
        )
        for log_path in out_paths[-1:]:
            if log_path.exists():
                completed_step = "training/global_step:1" in log_path.read_text(errors="replace")
        diagnostic_text = "\n".join(
            path.read_text(errors="replace")
            for path in [*out_paths[-1:], *err_paths[-1:]]
            if path.exists()
        )
        lower_text = diagnostic_text.lower()
        if trial.get("exit_code") != 0:
            if (
                "actor_rollout_update_weights" in diagnostic_text
                or "update_weights" in diagnostic_text
            ) and "out of memory" in lower_text:
                failure_kind = "weight_sync_oom"
            elif (
                "free memory" in lower_text
                or "gpu memory utilization" in lower_text
                or "cache blocks" in lower_text
            ) and ("out of memory" in lower_text or "valueerror" in lower_text):
                failure_kind = "rollout_init_memory"
            elif "out of memory" in lower_text:
                failure_kind = "cuda_oom"
            else:
                failure_kind = "other_failure"
        success = trial.get("exit_code") == 0 and completed_step
        if trial.get("exit_code") == 0 and not completed_step:
            errors.append(f"phase trial lacks completed step evidence: {experiment_id}")
        row = {
            **config,
            **trial,
            "success": int(success),
            "completed_step": int(completed_step),
            "within_safety_margin": int(
                success
                and float(trial.get("peak_gpu_memory_mib", 1e9))
                <= memory_limit_mib
            ),
            "failure_kind": failure_kind,
        }
        row.update({f"phase_peak_{key}_mib": value for key, value in peaks.items()})
        rows.append(row)
    missing = sorted(set(expected) - {row["experiment_id"] for row in rows})
    return rows, [*errors, *(f"missing phase trial: {item}" for item in missing)]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def cross_model_effects(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], dict[float, dict]] = defaultdict(dict)
    for row in rows:
        if not row["success"]:
            continue
        grouped[(row["model"], int(row["max_num_seqs"]))][
            float(row["gpu_memory_utilization"])
        ] = row
    effects = []
    for (model, max_num_seqs), values in sorted(grouped.items()):
        if 0.50 not in values or 0.80 not in values:
            continue
        low, high = values[0.50], values[0.80]
        low_t = float(low["tokens_per_second"])
        high_t = float(high["tokens_per_second"])
        effects.append(
            {
                "model": model,
                "max_num_seqs": max_num_seqs,
                "throughput_u50": low_t,
                "throughput_u80": high_t,
                "throughput_change_pct": 100 * (high_t / low_t - 1),
                "memory_u50_mib": float(low["peak_gpu_memory_mib"]),
                "memory_u80_mib": float(high["peak_gpu_memory_mib"]),
                "memory_increase_mib": (
                    float(high["peak_gpu_memory_mib"])
                    - float(low["peak_gpu_memory_mib"])
                ),
            }
        )
    return effects


def render_report(
    vllm_rows: list[dict],
    phase_rows: list[dict],
    effects: list[dict],
    errors: list[str],
) -> str:
    lines = [
        "# RLVRAMBench screening evidence",
        "",
        f"- Standalone artifacts: {len(vllm_rows)}/12",
        f"- Colocated artifacts: {len(phase_rows)}/12",
        f"- Validation issues (including unfinished trials): {len(errors)}",
        "",
        "## Standalone transfer-cost result",
        "",
        "| Model | max_num_seqs | Throughput change u0.50→u0.80 | Memory increase |",
        "|---|---:|---:|---:|",
    ]
    for row in effects:
        lines.append(
            f"| {row['model']} | {row['max_num_seqs']} | "
            f"{row['throughput_change_pct']:+.3f}% | "
            f"{row['memory_increase_mib'] / 1024:.2f} GiB |"
        )
    lines += [
        "",
        "## Colocated trials",
        "",
        "| Experiment | Completed | Safe margin | Failure phase | Peak VRAM | Runtime |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for row in sorted(phase_rows, key=lambda item: item["experiment_id"]):
        lines.append(
            f"| {row['experiment_id']} | {row['success']} | "
            f"{row['within_safety_margin']} | {row['failure_kind'] or '—'} | "
            f"{float(row['peak_gpu_memory_mib']) / 1024:.2f} GiB | "
            f"{row['elapsed_seconds']} s |"
        )
    if errors:
        lines += ["", "## Pending or invalid artifacts", ""]
        lines.extend(f"- {item}" for item in errors)
    if effects:
        changes = [abs(float(row["throughput_change_pct"])) for row in effects]
        memory = [float(row["memory_increase_mib"]) for row in effects]
        lines += [
            "",
            "## Current interpretation",
            "",
            f"Across the completed standalone model/concurrency pairs, the median "
            f"absolute throughput change is {statistics.median(changes):.3f}% while "
            f"the median memory increase is {statistics.median(memory) / 1024:.2f} GiB.",
            "This supports a configuration-transfer cost claim, but not yet a novel "
            "autotuner claim; held-out safety prediction and search-budget baselines "
            "remain required.",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vllm-matrix",
        type=Path,
        default=Path("memory_tuner/rlvram_screening_vllm.csv"),
    )
    parser.add_argument(
        "--phase-matrix",
        type=Path,
        default=Path("memory_tuner/rlvram_screening_phase.csv"),
    )
    parser.add_argument("--vllm-root", type=Path, default=Path("profiles/vllm"))
    parser.add_argument(
        "--phase-root", type=Path, default=Path("output/phase_screening")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("profiles/screening")
    )
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--memory-limit-mib", type=float, default=38_912)
    args = parser.parse_args()

    vllm_rows, vllm_errors = load_vllm(args.vllm_matrix, args.vllm_root)
    phase_rows, phase_errors = load_phase(
        args.phase_matrix, args.phase_root, args.memory_limit_mib
    )
    errors = [*vllm_errors, *phase_errors]
    effects = cross_model_effects(vllm_rows)
    write_csv(args.output_dir / "vllm-screening.csv", vllm_rows)
    write_csv(args.output_dir / "phase-screening.csv", phase_rows)
    write_csv(args.output_dir / "standalone-transfer-effects.csv", effects)
    (args.output_dir / "screening-report.md").write_text(
        render_report(vllm_rows, phase_rows, effects, errors)
    )
    print(
        f"validated {len(vllm_rows)}/12 standalone and "
        f"{len(phase_rows)}/12 colocated artifacts; issues={len(errors)}"
    )
    if args.require_complete and errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
