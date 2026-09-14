import math
import json

import pytest

from memory_tuner.scientific_revision_analysis import (
    paired_rows, repeated_cells, temporal_analysis, topology_analysis,
    instrumentation_analysis,
    excluded_allocations,
    validate_fixed_stack, CONTAINER_SHA256, RUNTIME_PACKAGES,
)


def trial(seed, condition, safe=1, success=1, **extra):
    return {
        "pair_id": f"phi-gsm-c3-s{seed}", "model_family": "phi",
        "dataset": "gsm", "configuration_level": "c3", "training_seed": seed,
        "condition": condition, "period": 1 if condition in ("source", "external") else 2,
        "safe": safe, "success": success, "peak_gpu_memory_mib": 38000,
        "elapsed_seconds": 100, "failure_kind": "" if success else "cuda_oom",
        "observed_positive_steps": 100 if success else 0,
        "maximum_later_step_increase_mib": 500,
        "phase_steps_checkpoint": "25,50,75,100",
        "phase_steps_validation": "0,20,40,60,80,100",
        "experiment_id": f"phi-{seed}-{condition}",
        **extra,
    }


def test_fixed_stack_rejects_runtime_drift():
    environment = {
        "container": {"sha256": CONTAINER_SHA256},
        "software": {"python": "3.12.3", "torch_cuda": "12.9",
                     "packages": dict(RUNTIME_PACKAGES)},
    }
    validate_fixed_stack(environment)
    environment["software"]["packages"]["vllm"] = "other"
    with pytest.raises(ValueError, match="vllm differs"):
        validate_fixed_stack(environment)


def test_oom_is_retained_without_counterfactual_peak():
    rows = [trial(1, "external"), trial(1, "full", safe=0, success=0)]
    pair = paired_rows(rows, "external", "full")[0]
    assert pair["completion_difference"] == -1
    assert pair["safety_discordance"] == 1
    assert math.isnan(pair["peak_difference_mib"])


def test_incomplete_pair_and_duplicate_seed_fail():
    with pytest.raises(ValueError, match="paired conditions"):
        paired_rows([trial(1, "external")], "external", "full")
    with pytest.raises(ValueError, match="distinct repetitions"):
        repeated_cells([trial(1, "source")] * 3)


def test_repeated_label_and_directional_topology_errors():
    historical = [trial(i, "source", safe=0) for i in range(3)]
    new = [trial(i, "four_gpu") for i in range(3)]
    cells, summary = topology_analysis(new, historical)
    assert cells[0]["safe"] == 1
    assert summary["two_to_four_false_safe"] == 0
    assert summary["four_to_two_false_safe"] == 1
    assert summary["two_to_four_false_unsafe"] == 1
    assert summary["two_gpu_unsafe_configurations"] == 1
    assert summary["four_gpu_unsafe_configurations"] == 0


def test_long_run_requires_all_checkpoint_cycles():
    rows = [trial(i, condition) for i in range(3) for condition in ("source", "long")]
    _, _, summary = temporal_analysis(rows)
    assert summary["source_safe_cells"] == 1
    rows[1]["phase_steps_checkpoint"] = "25,50,100"
    with pytest.raises(ValueError, match="missing checkpoint cycles"):
        temporal_analysis(rows)


def test_failed_long_run_is_a_valid_temporal_outcome():
    rows = [trial(i, condition, safe=int(condition == "source"),
                  success=int(condition == "source"))
            for i in range(3) for condition in ("source", "long")]
    _, _, summary = temporal_analysis(rows)
    assert summary["source_safe_to_long_unsafe_runs"] == 3
    assert summary["source_safe_to_long_unsafe_cells"] == 1


def test_conditional_peak_keeps_all_pretreatment_resampling_cells(monkeypatch):
    def fake_inference(**kwargs):
        rows = kwargs["rows"]
        return {
            "estimand": kwargs["estimand"], "observations": len(rows),
            "independent_clusters": len({kwargs["cluster"](row) for row in rows}),
        }

    monkeypatch.setattr(
        "memory_tuner.scientific_revision_analysis.inference_row", fake_inference)
    rows = [
        trial(seed, condition, safe=int(cell < 8), success=int(cell < 8),
              pair_id=f"m{cell}-s{seed}", model_family=f"m{cell}")
        for cell in range(12) for seed in range(3)
        for condition in ("external", "full")
    ]
    _, _, intervals, summary = instrumentation_analysis(rows)
    peak = next(r for r in intervals if r["estimand"] == "peak_difference_mib")
    assert peak["observations"] == 36
    assert peak["independent_clusters"] == 12
    assert peak["contributing_pairs"] == 24
    assert peak["contributing_base_cells"] == 8
    assert 0.26 < summary["zero_discordance_cell_upper95"] < 0.27


def test_all_excluded_allocations_must_be_documented(tmp_path):
    path = tmp_path / "output/revision_instrumentation/pairs/p/pair-1.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "pair_status": "infrastructure_invalid", "job_id": "1", "pair_id": "p",
        "source_head": "abc", "hostname": "node", "periods": [
            {"condition": "external"}], "error": "timeout",
    }))
    with pytest.raises(ValueError, match="undocumented"):
        excluded_allocations(tmp_path, "instrumentation")
    ledger = tmp_path / "memory_tuner/attempt_exclusions.csv"
    ledger.parent.mkdir()
    ledger.write_text("job_id,reason,evidence_note\n1,revision_startup_stall_timeout,raw log\n")
    observed = excluded_allocations(tmp_path, "instrumentation")
    assert observed[0]["last_attempted_condition"] == "external"
    assert observed[0]["reason"] == "revision_startup_stall_timeout"
