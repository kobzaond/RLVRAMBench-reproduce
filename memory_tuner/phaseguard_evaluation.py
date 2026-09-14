#!/usr/bin/env python3
"""Grouped held-out evaluation for PhaseGuard and selection baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from memory_tuner.phaseguard import (
    DEFAULT_FEATURES,
    PhaseGuard,
    conformal_upper_quantile,
    feature_matrix,
)

TARGET_ALPHA = 0.10
MIN_CALIBRATION_CASES = 9
MIN_FIT_CASES = 2
MIN_PRIMARY_CONFIGURATIONS = 4


CONFIG_FIELDS = (
    "case_id",
    "rollout_tp_size",
    "actor_micro_batch",
    "rollout_logprob_micro_batch",
    "ref_logprob_micro_batch",
    "vllm_gpu_memory_utilization",
    "parameter_offload",
    "optimizer_offload",
    "free_cache_engine",
    "max_prompt_length",
    "max_response_length",
    "max_model_len",
    "rollout_n",
    "train_batch_size",
    "max_num_seqs",
)


def _number(value, default=math.nan) -> float:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes"}:
            return 1.0
        if lowered in {"false", "no"}:
            return 0.0
    return float(value)


def configuration_id(row: Mapping) -> str:
    return "|".join(str(row.get(field, "")) for field in CONFIG_FIELDS)


def observed_success(row: Mapping) -> bool:
    if row.get("success") not in (None, ""):
        return bool(int(row["success"]))
    return int(row.get("exit_code", 1)) == 0


def aggregate_configurations(
    rows: Sequence[Mapping], memory_limit_mib: float
) -> list[dict]:
    grouped: dict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        grouped[configuration_id(row)].append(row)
    aggregated = []
    for config_id, repetitions in sorted(grouped.items()):
        representative = dict(repetitions[0])
        successes = [observed_success(row) for row in repetitions]
        peaks = [
            _number(row.get("peak_gpu_memory_mib"))
            for row in repetitions
            if row.get("peak_gpu_memory_mib") not in (None, "")
        ]
        runtimes = [
            _number(row.get("elapsed_seconds"))
            for row in repetitions
            if observed_success(row)
            and row.get("elapsed_seconds") not in (None, "")
        ]
        representative.update(
            {
                "configuration_id": config_id,
                "replicate_count": len(repetitions),
                "success": int(all(successes)),
                "peak_gpu_memory_mib": max(peaks) if peaks else math.nan,
                "elapsed_seconds": (
                    statistics.mean(runtimes) if runtimes else math.nan
                ),
                "observed_safe": int(
                    all(successes)
                    and bool(peaks)
                    and max(peaks) <= memory_limit_mib
                ),
            }
        )
        phase_fields = sorted(
            {
                field
                for row in repetitions
                for field in row
                if field.startswith("phase_peak_") and field.endswith("_mib")
            }
        )
        for field in phase_fields:
            values = [
                _number(row.get(field))
                for row in repetitions
                if row.get(field) not in (None, "")
            ]
            if values:
                representative[field] = max(values)
        aggregated.append(representative)
    return aggregated


def primary_case_groups(
    rows: Sequence[Mapping],
) -> dict[str, list[Mapping]]:
    """Return cases eligible for every frozen primary method analysis."""
    by_case: dict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        by_case[str(row["case_id"])].append(row)
    return {
        case_id: candidates
        for case_id, candidates in by_case.items()
        if len(candidates) >= MIN_PRIMARY_CONFIGURATIONS
        and any(int(row["observed_safe"]) for row in candidates)
    }


def split_fit_calibration(
    rows: Sequence[Mapping],
    *,
    minimum_calibration_cases: int = MIN_CALIBRATION_CASES,
    minimum_fit_cases: int = MIN_FIT_CASES,
) -> tuple[list, list]:
    by_case: dict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        case_id = str(row.get("case_id", ""))
        if case_id:
            by_case[case_id].append(row)
    if len(by_case) >= minimum_fit_cases + 1:
        ordered_cases = sorted(
            by_case,
            key=lambda case_id: hashlib.sha256(
                case_id.encode()
            ).hexdigest(),
        )
        calibration_count = min(
            minimum_calibration_cases,
            len(ordered_cases) - minimum_fit_cases,
        )
        calibration_cases = set(ordered_cases[:calibration_count])
        calibration = [
            row
            for case_id in ordered_cases
            if case_id in calibration_cases
            for row in by_case[case_id]
        ]
        train = [
            row
            for case_id in ordered_cases
            if case_id not in calibration_cases
            for row in by_case[case_id]
        ]
        if (
            len({str(row["case_id"]) for row in train})
            >= minimum_fit_cases
            and calibration
        ):
            return train, calibration

    # Unit tests and tiny legacy corpora may not carry case identifiers.
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            str(row["configuration_id"]).encode()
        ).hexdigest(),
    )
    if len(ordered) < 4:
        raise ValueError("at least four non-held-out configurations are required")
    calibration_count = max(1, int(round(0.25 * len(ordered))))
    calibration_count = min(calibration_count, len(ordered) - 2)
    return ordered[calibration_count:], ordered[:calibration_count]


class TotalPeakGuard:
    """Strong ablation: conformal total-peak model without phase structure."""

    def __init__(
        self,
        feature_names=DEFAULT_FEATURES,
        alpha=TARGET_ALPHA,
    ):
        self.feature_names = tuple(feature_names)
        self.alpha = alpha
        self.peak_model = None
        self.runtime_model = None
        self.correction = 0.0
        self.calibration_case_count = 0

    @staticmethod
    def _regressor():
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=1.0),
        )

    def fit(self, train: Sequence[Mapping], calibration: Sequence[Mapping]):
        successful = [
            row
            for row in train
            if observed_success(row)
            and row.get("peak_gpu_memory_mib") not in (None, "")
        ]
        if not successful:
            raise ValueError("total-peak baseline needs a successful training row")
        self.peak_model = self._regressor()
        self.peak_model.fit(
            feature_matrix(successful, self.feature_names),
            np.asarray([_number(row["peak_gpu_memory_mib"]) for row in successful]),
        )
        residuals_by_case: dict[str, list[float]] = defaultdict(list)
        for row_index, row in enumerate(calibration):
            if not observed_success(row):
                continue
            predicted = float(
                self.peak_model.predict(
                    feature_matrix([row], self.feature_names)
                )[0]
            )
            case_id = str(
                row.get("case_id")
                or row.get("configuration_id")
                or f"calibration-row-{row_index}"
            )
            residuals_by_case[case_id].append(
                _number(row["peak_gpu_memory_mib"]) - predicted
            )
        case_maxima = [
            max(residuals) for residuals in residuals_by_case.values()
        ]
        self.calibration_case_count = len(case_maxima)
        self.correction = max(
            0.0,
            conformal_upper_quantile(case_maxima, self.alpha),
        )
        runtime_rows = [
            row
            for row in successful
            if row.get("elapsed_seconds") not in (None, "")
        ]
        self.runtime_model = self._regressor()
        self.runtime_model.fit(
            feature_matrix(runtime_rows, self.feature_names),
            np.asarray([_number(row["elapsed_seconds"]) for row in runtime_rows]),
        )
        return self

    def select(self, rows: Sequence[Mapping], memory_limit_mib: float):
        choices = []
        for row in rows:
            x = feature_matrix([row], self.feature_names)
            upper = float(self.peak_model.predict(x)[0]) + self.correction
            runtime = max(0.0, float(self.runtime_model.predict(x)[0]))
            if upper <= memory_limit_mib:
                choices.append((row, runtime, upper))
        if not choices:
            return None
        return min(choices, key=lambda item: (item[1], item[2]))


def static_choice(candidates: Sequence[Mapping], mode: str) -> Mapping:
    if mode == "conservative":
        return min(
            candidates,
            key=lambda row: (
                _number(row.get("vllm_gpu_memory_utilization"), 1.0),
                _number(row.get("actor_micro_batch"), math.inf),
                -_number(row.get("parameter_offload"), 0.0),
            ),
        )
    if mode == "aggressive":
        return max(
            candidates,
            key=lambda row: (
                _number(row.get("actor_micro_batch"), 0.0),
                _number(row.get("vllm_gpu_memory_utilization"), 0.0),
                -_number(row.get("parameter_offload"), 0.0),
            ),
        )
    raise ValueError(mode)


def score_choice(
    case_id: str,
    method: str,
    selected: Mapping | None,
    candidates: Sequence[Mapping],
) -> dict:
    safe_candidates = [
        row
        for row in candidates
        if int(row["observed_safe"])
        and math.isfinite(_number(row.get("elapsed_seconds")))
    ]
    oracle_runtime = min(
        (_number(row["elapsed_seconds"]) for row in safe_candidates),
        default=math.nan,
    )
    if selected is None:
        return {
            "case_id": case_id,
            "method": method,
            "selected": 0,
            "selected_configuration": "",
            "unsafe_selected": math.nan,
            "observed_runtime_s": math.nan,
            "oracle_runtime_s": oracle_runtime,
            "regret_s": math.nan,
        }
    selected_safe = bool(int(selected["observed_safe"]))
    runtime = _number(selected.get("elapsed_seconds"))
    regret = (
        runtime - oracle_runtime
        if selected_safe and math.isfinite(runtime) and math.isfinite(oracle_runtime)
        else math.nan
    )
    return {
        "case_id": case_id,
        "method": method,
        "selected": 1,
        "selected_configuration": selected["configuration_id"],
        "unsafe_selected": int(not selected_safe),
        "observed_runtime_s": runtime,
        "oracle_runtime_s": oracle_runtime,
        "regret_s": regret,
    }


def evaluate(rows: Sequence[Mapping], memory_limit_mib: float) -> list[dict]:
    configs = aggregate_configurations(rows, memory_limit_mib)
    by_case = primary_case_groups(configs)
    primary_configs = [
        row for candidates in by_case.values() for row in candidates
    ]
    results = []
    for case_id, candidates in sorted(by_case.items()):
        nonheld = [
            row for row in primary_configs if row["case_id"] != case_id
        ]
        try:
            train, calibration = split_fit_calibration(nonheld)
        except ValueError:
            continue
        methods: list[tuple[str, Mapping | None]] = []
        try:
            phaseguard = PhaseGuard().fit(train, calibration)
            selected = phaseguard.select(candidates, memory_limit_mib)
            methods.append(("phaseguard", selected[0] if selected else None))
        except (ValueError, RuntimeError):
            methods.append(("phaseguard", None))
        try:
            total = TotalPeakGuard().fit(train, calibration)
            selected = total.select(candidates, memory_limit_mib)
            methods.append(("total_peak_conformal", selected[0] if selected else None))
        except (ValueError, RuntimeError):
            methods.append(("total_peak_conformal", None))
        methods.extend(
            [
                ("static_conservative", static_choice(candidates, "conservative")),
                ("static_aggressive", static_choice(candidates, "aggressive")),
                (
                    "oracle",
                    min(
                        (
                            row
                            for row in candidates
                            if int(row["observed_safe"])
                        ),
                        key=lambda row: _number(row["elapsed_seconds"]),
                    ),
                ),
            ]
        )
        results.extend(
            score_choice(case_id, method, selected, candidates)
            for method, selected in methods
        )
    return results


def summarize(results: Sequence[Mapping]) -> list[dict]:
    grouped: dict[str, list[Mapping]] = defaultdict(list)
    for row in results:
        grouped[str(row["method"])].append(row)
    summary = []
    for method, rows in sorted(grouped.items()):
        unsafe = [
            _number(row["unsafe_selected"])
            for row in rows
            if row["unsafe_selected"] not in (None, "")
            and math.isfinite(_number(row["unsafe_selected"]))
        ]
        regrets = [
            _number(row["regret_s"])
            for row in rows
            if row["regret_s"] not in (None, "")
            and math.isfinite(_number(row["regret_s"]))
        ]
        summary.append(
            {
                "method": method,
                "cases": len(rows),
                "selection_rate": statistics.mean(
                    int(row["selected"]) for row in rows
                ),
                "unsafe_selection_rate": (
                    statistics.mean(unsafe) if unsafe else math.nan
                ),
                "mean_safe_regret_s": (
                    statistics.mean(regrets) if regrets else math.nan
                ),
                "median_safe_regret_s": (
                    statistics.median(regrets) if regrets else math.nan
                ),
            }
        )
    return summary


def write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path(
            "profiles/benchmark/prospective-method-corpus.csv"
        ),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("profiles/phaseguard/heldout-predictions.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("profiles/phaseguard/heldout-summary.csv"),
    )
    parser.add_argument("--memory-limit-mib", type=float, default=38_912)
    args = parser.parse_args()
    with args.corpus.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    results = evaluate(rows, args.memory_limit_mib)
    if not results:
        raise SystemExit(
            "no evaluable held-out cases; collect more cross-case benchmark cells"
        )
    write_csv(args.predictions, results)
    write_csv(args.summary, summarize(results))
    print(f"evaluated {len(results)} method-case selections")


if __name__ == "__main__":
    main()
