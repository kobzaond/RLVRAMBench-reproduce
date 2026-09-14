from pathlib import Path

from memory_tuner.plan_scientific_revision import build_matrices


ROOT = Path(__file__).resolve().parents[1]


def test_control_pairs_change_only_measurement_and_order_metadata():
    rows = build_matrices(ROOT)["instrumentation"]
    assert len(rows) == 72
    pairs = {}
    for row in rows:
        pairs.setdefault(row["pair_id"], []).append(row)
    ignored = {"experiment_id", "condition", "period", "allocator_trace_enabled"}
    first_conditions = []
    for values in pairs.values():
        assert len(values) == 2
        assert {r["condition"] for r in values} == {"external", "full"}
        assert {r["period"] for r in values} == {1, 2}
        first_conditions.append(values[0]["condition"])
        assert {k: v for k, v in values[0].items() if k not in ignored} == {
            k: v for k, v in values[1].items() if k not in ignored}
    assert first_conditions.count("external") == first_conditions.count("full")


def test_temporal_pairs_cannot_resume_or_select_on_source_outcomes():
    rows = build_matrices(ROOT)["temporal"]
    assert len(rows) == 24
    pairs = {}
    for row in rows:
        pairs.setdefault(row["pair_id"], []).append(row)
        assert row["resume_mode"] == "disable"
        assert row["save_freq"] == 25
        assert row["train_max_samples"] == -1
    ignored = {"experiment_id", "condition", "period", "total_training_steps"}
    for source, long in pairs.values():
        assert (source["total_training_steps"], long["total_training_steps"]) == (1, 100)
        assert {k: v for k, v in source.items() if k not in ignored} == {
            k: v for k, v in long.items() if k not in ignored}


def test_topology_keeps_global_workload_and_covers_every_boundary_cell():
    rows = build_matrices(ROOT)["topology"]
    assert len(rows) == 54
    assert {r["gpu_count"] for r in rows} == {4}
    assert {r["rollout_tp_size"] for r in rows} == {"1"}
    assert {r["configuration_level"] for r in rows} == {"c2", "c3", "c4"}
    assert len({r["experiment_id"] for r in rows}) == len(rows)
