"""Resolve recorded evidence paths inside the artifact being inspected.

Raw JSON is immutable. Its historical absolute paths are provenance, not
permission to read a different copy of the artifact during regeneration.
"""

from __future__ import annotations

import json
from pathlib import Path


ORIGINAL_ROOT = Path("/mnt/proj3/open-35-44/verl/rl")
PATH_FIELDS = (
    "run_log", "phase_memory_csv", "gpu_telemetry_csv", "allocator_trace_dir",
    "environment_json", "response_length_trace", "data_dir",
)


def artifact_root(path: Path) -> Path | None:
    for parent in path.absolute().parents:
        if (parent / "memory_tuner").is_dir() and (parent / "output").is_dir():
            return parent
    return None


def recorded_path(value: str, root: Path) -> Path:
    path = Path(value)
    if ".." in path.parts:
        raise ValueError(f"evidence path escapes artifact: {value}")
    if path.is_absolute():
        if path.is_relative_to(root.absolute()):
            return path
        if path.is_relative_to(ORIGINAL_ROOT):
            path = path.relative_to(ORIGINAL_ROOT)
        else:
            # Regenerated tables may record the absolute root of another
            # extracted copy. Evidence lives in these fixed artifact folders.
            parts = path.parts
            indices = [i for i, part in enumerate(parts)
                       if part in ("output", "profiles", "data")]
            if not indices:
                raise ValueError(f"unmapped evidence path outside artifact: {value}")
            path = Path(*parts[indices[0]:])
    return root / path


def load_trial_record(path: Path, root: Path | None = None) -> dict:
    trial = json.loads(path.read_text(encoding="utf-8"))
    root = root or artifact_root(path)
    if root is not None:
        for field in PATH_FIELDS:
            if trial.get(field):
                trial[field] = str(recorded_path(str(trial[field]), root))
    return trial
