"""Shared loading, normalisation and output helpers (standard library only)."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable, Mapping, Sequence

DEVICE_CAPACITY_MIB = 40960.0
MARGIN_LIMIT_MIB = 38912.0
STATES = ("within_margin", "above_margin", "memory_failure")
STATE_ORDER = {"within_margin": 0, "above_margin": 1, "memory_failure": 2}

# Workload identifiers differ between the curated benchmark tables and the
# later panels; ``code_heavy_tail`` is the legacy spelling of the
# "Longer-prompt code" workload (see DATA_DICTIONARY.md).
WORKLOAD_ALIASES = {
    "code_heavy_tail": "longer_prompt_code",
    "longer_prompt_code": "longer_prompt_code",
    "code_standard": "code_standard",
    "gsm8k": "gsm8k",
    "math": "math",
}
WORKLOAD_LABELS = {
    "gsm8k": "GSM8K",
    "math": "MATH",
    "code_standard": "Std. code",
    "longer_prompt_code": "Longer-prompt code",
}
WORKLOAD_ORDER = ("gsm8k", "math", "code_standard", "longer_prompt_code")

MODEL_FAMILY_IDS = {
    "qwen25_1p5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "granite33_2b": "ibm-granite/granite-3.3-2b-instruct",
    "qwen25_3b": "Qwen/Qwen2.5-3B-Instruct",
    "phi4_mini": "microsoft/Phi-4-mini-instruct",
    "qwen25_7b": "Qwen/Qwen2.5-7B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}
MODEL_ID_FAMILIES = {value: key for key, value in MODEL_FAMILY_IDS.items()}
MODEL_LABELS = {
    "qwen25_1p5b": "Qwen2.5-1.5B",
    "granite33_2b": "Granite-3.3-2B",
    "qwen25_3b": "Qwen2.5-3B",
    "phi4_mini": "Phi-4-mini (3.8B)",
    "qwen25_7b": "Qwen2.5-7B",
    "mistral": "Mistral-7B-v0.3",
}
# Parameter counts (billions) from the published architecture metadata;
# used only for ordering/labelling by model size.
MODEL_SIZE_B = {
    "qwen25_1p5b": 1.54,
    "granite33_2b": 2.53,
    "qwen25_3b": 3.09,
    "phi4_mini": 3.84,
    "qwen25_7b": 7.62,
    "mistral": 7.25,
}
MODEL_ORDER = ("qwen25_1p5b", "granite33_2b", "qwen25_3b", "phi4_mini",
               "qwen25_7b", "mistral")


def normalise_workload(value: str) -> str:
    return WORKLOAD_ALIASES.get(value, value)


def family_from_model(value: str) -> str:
    if value in MODEL_FAMILY_IDS:
        return value
    if value in MODEL_ID_FAMILIES:
        return MODEL_ID_FAMILIES[value]
    lowered = value.lower()
    if "mistral" in lowered:
        return "mistral"
    if "qwen25_7b" in lowered or "qwen2.5-7b" in lowered:
        return "qwen25_7b"
    return value


def read_csv(path: Path | str) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path | str, rows: Sequence[Mapping], fieldnames: Sequence[str] | None = None) -> None:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n",
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key)) for key in fieldnames})


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if value == int(value) and abs(value) < 1e15:
            return f"{value:.1f}"
        return f"{value:.4f}"
    return value


def _clean(value):
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 6)
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def write_json(path: Path | str, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(value), indent=2, sort_keys=True, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def to_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def to_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no"):
        return False
    return None


def safe_mean(values: Iterable[float | None]) -> float | None:
    values = [v for v in values if v is not None]
    return mean(values) if values else None


def label_from_peak(peak_mib: float | None, completed: bool, memory_failure: bool) -> str:
    if memory_failure:
        return "memory_failure"
    if not completed or peak_mib is None:
        return "unresolved"
    return "within_margin" if peak_mib <= MARGIN_LIMIT_MIB else "above_margin"


def group_by(rows: Iterable[Mapping], *keys: str) -> dict:
    groups: dict = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    return dict(groups)


class Benchmark:
    """Lazy accessor for the curated benchmark tables."""

    def __init__(self, data_dir: Path | str):
        self.root = Path(data_dir)
        if not (self.root / "configurations.csv").is_file():
            raise FileNotFoundError(f"{self.root} does not contain configurations.csv")
        self.repo_root = self.root.resolve().parent
        self.configurations = {r["configuration_id"]: r for r in read_csv(self.root / "configurations.csv")}
        self.outcomes = {r["configuration_id"]: r for r in read_csv(self.root / "outcomes.csv")}
        self.runs = read_csv(self.root / "runs.csv")
        self.attempts = read_csv(self.root / "attempts.csv")
        self.stages = read_csv(self.root / "stage_measurements.csv")
        self.queries = read_csv(self.root / "queries.csv")
        self._stage_index = None

    def stage_peaks(self) -> dict:
        """(run_id, stage) -> (peak, coverage)."""
        if self._stage_index is None:
            index = {}
            for row in self.stages:
                index[(row["run_id"], row["stage"])] = (to_float(row["observed_peak_mib"]), row["coverage"])
            self._stage_index = index
        return self._stage_index

    def optional_csv(self, relative: str) -> list[dict]:
        path = self.root / relative
        return read_csv(path) if path.is_file() else []

    def optional_json(self, relative: str):
        path = self.root / relative
        return json.loads(path.read_text()) if path.is_file() else None
