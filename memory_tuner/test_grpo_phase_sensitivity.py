import pytest

from memory_tuner.grpo_phase_sensitivity import threshold_sensitivity


def lattice():
    return [
        {"algorithm": "grpo", "model_family": model, "dataset": "workload",
         "configuration_level": f"c{level}", "training_seed": seed,
         "success": int(level < 4),
         "peak_gpu_memory_mib": 39000 if level == 3 and model == "target" else 30000}
        for model in ("source", "target")
        for level in range(6) for seed in (1, 2, 3)
    ]


def test_margin_changes_completed_labels_but_never_rescues_failed_processes():
    cases, transfers = threshold_sensitivity(lattice())
    indexed = {(r["threshold_fraction"], r["model_family"]): r for r in cases}
    assert indexed[0.95, "target"]["highest_safe_level"] == 2
    assert indexed[1.0, "target"]["highest_safe_level"] == 3
    assert indexed[1.0, "source"]["highest_safe_level"] == 3
    by_fraction = {r["threshold_fraction"]: r for r in transfers
                   if r["source_family"] == "source"}
    assert by_fraction[0.95]["false_safe_count"] == 1
    assert by_fraction[1.0]["false_safe_count"] == 0
    assert by_fraction[1.0]["unsafe_targets"] == 2


def test_missing_repeat_or_wrong_scope_fails_closed():
    with pytest.raises(ValueError, match="three distinct"):
        threshold_sensitivity(lattice()[:-1])
    rows = lattice()
    rows[0]["algorithm"] = "other"
    with pytest.raises(ValueError, match="GRPO only"):
        threshold_sensitivity(rows)
