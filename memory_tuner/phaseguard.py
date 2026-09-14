"""Phase-calibrated safety and runtime selector for colocated LLM-RL."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURES = (
    "model_parameters_b",
    "gpu_count",
    "rollout_tp_size",
    "prompt_p50_tokens",
    "prompt_p95_tokens",
    "prompt_p99_tokens",
    "prompt_max_tokens",
    "max_prompt_length",
    "max_response_length",
    "max_model_len",
    "rollout_n",
    "train_batch_size",
    "actor_micro_batch",
    "rollout_logprob_micro_batch",
    "ref_logprob_micro_batch",
    "vllm_gpu_memory_utilization",
    "max_num_seqs",
    "parameter_offload",
    "optimizer_offload",
    "free_cache_engine",
)

PHASE_FEATURES = {
    "initializing": (
        "model_parameters_b",
        "inverse_gpu_count",
        "inverse_rollout_tp_size",
        "max_model_len",
        "vllm_gpu_memory_utilization",
        "max_num_seqs",
        "parameter_resident",
    ),
    "rollout": (
        "model_parameters_b",
        "inverse_rollout_tp_size",
        "prompt_p95_tokens",
        "prompt_p99_tokens",
        "prompt_max_tokens",
        "max_prompt_length",
        "max_response_length",
        "max_model_len",
        "rollout_n",
        "vllm_gpu_memory_utilization",
        "max_num_seqs",
    ),
    "reference_logprob": (
        "model_parameters_b",
        "inverse_gpu_count",
        "prompt_p95_tokens",
        "max_prompt_length",
        "max_response_length",
        "ref_logprob_micro_batch",
        "parameter_resident",
    ),
    "actor_update": (
        "model_parameters_b",
        "inverse_gpu_count",
        "prompt_p95_tokens",
        "max_prompt_length",
        "max_response_length",
        "train_batch_size",
        "actor_micro_batch",
        "parameter_resident",
        "optimizer_resident",
    ),
    "weight_sync": (
        "model_parameters_b",
        "inverse_gpu_count",
        "inverse_rollout_tp_size",
        "max_model_len",
        "vllm_gpu_memory_utilization",
        "parameter_resident",
    ),
}


def _number(value) -> float:
    if isinstance(value, bool):
        return float(value)
    if value is None or value == "":
        return math.nan
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes"}:
            return 1.0
        if lowered in {"false", "no"}:
            return 0.0
    return float(value)


def _feature_value(row: Mapping, name: str) -> float:
    if name == "inverse_gpu_count":
        value = _number(row.get("gpu_count"))
        return 1.0 / value if math.isfinite(value) and value > 0 else math.nan
    if name == "inverse_rollout_tp_size":
        value = _number(row.get("rollout_tp_size"))
        return 1.0 / value if math.isfinite(value) and value > 0 else math.nan
    if name == "parameter_resident":
        value = _number(row.get("parameter_offload"))
        return 1.0 - value if math.isfinite(value) else math.nan
    if name == "optimizer_resident":
        value = _number(row.get("optimizer_offload"))
        return 1.0 - value if math.isfinite(value) else math.nan
    return _number(row.get(name))


def feature_matrix(
    rows: Sequence[Mapping], feature_names: Sequence[str]
) -> np.ndarray:
    return np.asarray(
        [[_feature_value(row, name) for name in feature_names] for row in rows],
        dtype=float,
    )


def conformal_upper_quantile(residuals: Sequence[float], alpha: float) -> float:
    if not residuals:
        return 0.0
    ordered = np.sort(np.asarray(residuals, dtype=float))
    rank = math.ceil((len(ordered) + 1) * (1 - alpha))
    rank = min(max(rank, 1), len(ordered))
    return float(ordered[rank - 1])


def maximum_finite_sample_coverage(calibration_units: int) -> float:
    """Best distribution-free coverage resolution with finite calibration."""
    if calibration_units <= 0:
        return 0.0
    return calibration_units / (calibration_units + 1)


@dataclass(frozen=True)
class Prediction:
    safe: bool
    unsafe_probability: float
    predicted_runtime_s: float
    predicted_peak_mib: float
    memory_headroom_mib: float
    phase_upper_mib: dict[str, float]


class PhaseGuard:
    def __init__(
        self,
        *,
        feature_names: Sequence[str] = DEFAULT_FEATURES,
        phases: Sequence[str] = (
            "initializing",
            "rollout",
            "reference_logprob",
            "actor_update",
            "weight_sync",
        ),
        alpha: float = 0.10,
        unsafe_probability_limit: float = 0.50,
        ridge_alpha: float = 1.0,
    ) -> None:
        self.feature_names = tuple(feature_names)
        self.phases = tuple(phases)
        self.alpha = alpha
        self.unsafe_probability_limit = unsafe_probability_limit
        self.ridge_alpha = ridge_alpha
        self.phase_models = {}
        self.phase_feature_names = {}
        self.phase_corrections = {}
        self.joint_phase_correction = 0.0
        self.calibration_case_count = 0
        self.runtime_model = None
        self.failure_model = None
        self.constant_unsafe_probability = None

    def _regressor(self, *, positive: bool = False):
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=self.ridge_alpha, positive=positive),
        )

    @staticmethod
    def _success(row: Mapping) -> bool:
        if "success" in row and row["success"] != "":
            return bool(int(row["success"]))
        return int(row.get("exit_code", 1)) == 0

    def fit(
        self,
        train_rows: Sequence[Mapping],
        calibration_rows: Sequence[Mapping],
    ) -> "PhaseGuard":
        if not train_rows:
            raise ValueError("train_rows must not be empty")
        successful_train = [row for row in train_rows if self._success(row)]
        if not successful_train:
            raise ValueError("at least one successful training row is required")
        for phase in self.phases:
            target_name = f"phase_peak_{phase}_mib"
            phase_rows = [
                row
                for row in successful_train
                if row.get(target_name) not in (None, "")
            ]
            if not phase_rows:
                continue
            phase_features = (
                PHASE_FEATURES.get(phase, self.feature_names)
                if self.feature_names == DEFAULT_FEATURES
                else self.feature_names
            )
            model = self._regressor(positive=True)
            x = feature_matrix(phase_rows, phase_features)
            y = np.asarray([_number(row[target_name]) for row in phase_rows])
            model.fit(x, y)
            self.phase_models[phase] = model
            self.phase_feature_names[phase] = phase_features
            self.phase_corrections[phase] = 0.0

        # Calibrate the maximum residual jointly across phases and all
        # configurations from the same benchmark case. This treats the case,
        # rather than an individual configuration/phase, as the exchangeable
        # unit and protects the complete phase envelope with one correction.
        residuals_by_case: dict[str, list[float]] = {}
        for row_index, row in enumerate(calibration_rows):
            if not self._success(row):
                continue
            residuals = []
            for phase, model in self.phase_models.items():
                target_name = f"phase_peak_{phase}_mib"
                if row.get(target_name) in (None, ""):
                    continue
                prediction = float(
                    model.predict(
                        feature_matrix(
                            [row], self.phase_feature_names[phase]
                        )
                    )[0]
                )
                residuals.append(_number(row[target_name]) - prediction)
            if not residuals:
                continue
            case_id = str(
                row.get("case_id")
                or row.get("configuration_id")
                or f"calibration-row-{row_index}"
            )
            residuals_by_case.setdefault(case_id, []).append(max(residuals))
        case_maxima = [
            max(residuals) for residuals in residuals_by_case.values()
        ]
        self.calibration_case_count = len(case_maxima)
        self.joint_phase_correction = max(
            0.0,
            conformal_upper_quantile(case_maxima, self.alpha),
        )

        runtime_rows = [
            row
            for row in successful_train
            if row.get("elapsed_seconds") not in (None, "")
        ]
        self.runtime_model = self._regressor(positive=False)
        self.runtime_model.fit(
            feature_matrix(runtime_rows, self.feature_names),
            np.asarray([_number(row["elapsed_seconds"]) for row in runtime_rows]),
        )

        classification_rows = list(train_rows) + list(calibration_rows)
        labels = np.asarray([int(not self._success(row)) for row in classification_rows])
        if len(set(labels.tolist())) == 1:
            self.constant_unsafe_probability = float(labels[0])
            self.failure_model = None
        else:
            self.failure_model = make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=True),
                StandardScaler(),
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=20260826,
                ),
            )
            self.failure_model.fit(
                feature_matrix(classification_rows, self.feature_names), labels
            )
            self.constant_unsafe_probability = None
        return self

    def predict(self, row: Mapping, memory_limit_mib: float) -> Prediction:
        if self.runtime_model is None or not self.phase_models:
            raise RuntimeError("PhaseGuard must be fitted before prediction")
        x = feature_matrix([row], self.feature_names)
        phase_upper = {}
        for phase, model in self.phase_models.items():
            phase_x = feature_matrix(
                [row], self.phase_feature_names[phase]
            )
            phase_upper[phase] = max(
                0.0,
                float(model.predict(phase_x)[0])
                + self.phase_corrections[phase]
                + self.joint_phase_correction,
            )
        predicted_peak = max(phase_upper.values())
        if self.failure_model is None:
            unsafe_probability = float(self.constant_unsafe_probability or 0.0)
        else:
            unsafe_probability = float(self.failure_model.predict_proba(x)[0, 1])
        predicted_runtime = max(0.0, float(self.runtime_model.predict(x)[0]))
        safe = (
            predicted_peak <= memory_limit_mib
            and unsafe_probability <= self.unsafe_probability_limit
        )
        return Prediction(
            safe=safe,
            unsafe_probability=unsafe_probability,
            predicted_runtime_s=predicted_runtime,
            predicted_peak_mib=predicted_peak,
            memory_headroom_mib=memory_limit_mib - predicted_peak,
            phase_upper_mib=phase_upper,
        )

    def select(
        self, rows: Iterable[Mapping], memory_limit_mib: float
    ) -> tuple[Mapping, Prediction] | None:
        candidates = []
        for row in rows:
            prediction = self.predict(row, memory_limit_mib)
            if prediction.safe:
                candidates.append((row, prediction))
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda item: (
                item[1].predicted_runtime_s,
                -item[1].memory_headroom_mib,
            ),
        )
