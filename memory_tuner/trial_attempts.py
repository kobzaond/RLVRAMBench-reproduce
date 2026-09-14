"""Select one analyzable attempt while retaining infrastructure retry history."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def _attempt_order(attempt: dict) -> tuple[int, int]:
    trial = attempt["trial"]
    job_id = str(trial.get("job_id", "") or "")
    numeric_job_id = int(job_id) if job_id.isdigit() else -1
    return numeric_job_id, attempt["trial_path"].stat().st_mtime_ns


def select_trial_attempts(
    trial_paths: list[Path],
) -> dict[str, dict]:
    """Choose the sole successful attempt, or the newest failed attempt.

    Failed infrastructure attempts remain counted in the selected record.
    More than one successful attempt is rejected because it would make the
    frozen replicate definition ambiguous.
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    for trial_path in trial_paths:
        trial = json.loads(trial_path.read_text())
        experiment_id = str(
            trial.get("experiment_id", "") or trial_path.parent.name
        )
        result_path = trial_path.with_name("result.json")
        result = json.loads(result_path.read_text()) if result_path.is_file() else {}
        grouped[experiment_id].append(
            {
                "trial_path": trial_path,
                "trial": trial,
                "result": result,
                "successful": (
                    int(trial.get("exit_code", 1)) == 0 and bool(result)
                ),
            }
        )

    selected = {}
    for experiment_id, attempts in grouped.items():
        successful = [attempt for attempt in attempts if attempt["successful"]]
        if len(successful) > 1:
            paths = ", ".join(
                str(attempt["trial_path"]) for attempt in successful
            )
            raise ValueError(
                f"multiple successful attempts for {experiment_id}: {paths}"
            )
        chosen = (
            successful[0]
            if successful
            else max(attempts, key=_attempt_order)
        )
        chosen = dict(chosen)
        chosen["attempt_count"] = len(attempts)
        chosen["failed_attempt_count"] = sum(
            not attempt["successful"] for attempt in attempts
        )
        selected[experiment_id] = chosen
    return selected
