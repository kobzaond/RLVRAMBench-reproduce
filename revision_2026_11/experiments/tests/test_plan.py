"""Matrix structure, randomization balance, seeds, settings provenance and caps."""
from collections import Counter

from ieee_resub import common as C
from ieee_resub import plan

E1, E1_DESIGN = plan.e1_rows()
E2, E2_DESIGN = plan.e2_rows()
E3, E3_DESIGN = plan.e3_rows()


def blocks(rows):
    out = {}
    for row in rows:
        out.setdefault(row["block_id"], []).append(row)
    return {k: sorted(v, key=lambda r: int(r["period"])) for k, v in out.items()}


def test_counts():
    assert (len(E1), len(E2), len(E3)) == (48, 40, 28)
    assert len(blocks(E1)) == 12 and len(blocks(E2)) == 10 and len(blocks(E3)) == 12
    assert sum(r["role"] == "restore" for r in E3) == 4
    assert len(plan.smoke1_rows()) == 3 and len(plan.smoke2_rows()) == 2


def test_e1_williams_balance_and_same_case_settings():
    position = Counter((r["condition"], r["period"]) for r in E1)
    assert set(position.values()) == {3} and len(position) == 16
    adjacency = Counter()
    for members in blocks(E1).values():
        conditions = [r["condition"] for r in members]
        assert sorted(conditions) == ["b08-ext", "b08-full", "b16-ext", "b16-full"]
        adjacency.update(zip(conditions, conditions[1:]))
    assert len(adjacency) == 12 and set(adjacency.values()) == {3}
    by_case = {}
    for members in blocks(E1).values():
        by_case.setdefault(members[0]["case_id"], []).append(tuple(r["condition"] for r in members))
    for sequences in by_case.values():
        assert len(sequences) == 3 and len(set(sequences)) == 3
    fixed = ("model", "dataset", "max_prompt_length", "max_response_length", "max_model_len", "max_num_seqs",
             "rollout_n", "train_batch_size", "train_max_samples", "vllm_gpu_memory_utilization",
             "rollout_logprob_micro_batch", "ref_logprob_micro_batch", "free_cache_engine", "parameter_offload")
    for members in blocks(E1).values():
        for field in fixed:
            assert len({r[field] for r in members}) == 1, field
        assert {r["training_seed"] for r in members} == {members[0]["training_seed"]}
        assert {(r["actor_micro_batch"], r["allocator_trace_enabled"]) for r in members} == {
            ("8", "False"), ("8", "True"), ("16", "False"), ("16", "True")}
        for r in members:
            assert r["fixed_response_length"] == r["canonical_batch_order"] == r["shape_trace"] == "True"
            assert r["vllm_gpu_memory_utilization"] == "0.60" and r["total_training_steps"] == "1"
            assert r["gpu_count"] == "2" and r["gpu_subset_ordinals"] == "0;1"


def test_e1_workloads_match_replication_templates():
    replication = C.read_rows(C.REPO_ROOT / plan.REPLICATION_MATRIX)
    for row in E1:
        template = next(t for t in replication if t["model"] == row["model"] and t["dataset"] == row["dataset"]
                        and t["condition"] == "base")
        for field in ("max_prompt_length", "max_response_length", "max_model_len", "max_num_seqs", "rollout_n",
                      "train_batch_size", "train_max_samples", "rollout_logprob_micro_batch"):
            assert row[field] == template[field], field


def test_e1_selected_cases_predicted_below_capacity():
    predictions = {(p["model_family"], p["dataset"]): p for p in plan.e1_predictions()["cases"]}
    for model, dataset in plan.E1_CASES:
        p = predictions[(C.FAMILY[model], dataset)]
        assert p["selected"] and not p["exceeds_margin_b16"]
    assert predictions[("phi4_mini", "code_heavy_tail")]["exceeds_capacity_b16"]
    assert predictions[("qwen25_3b", "gsm8k")]["exceeds_margin_b16"]


def test_e2_templates_and_balance():
    for name, (source, template_id, gpus, _) in plan.E2_CONFIGS.items():
        template = plan.find(source, template_id)
        members = [r for r in E2 if r["condition"] == name]
        assert len(members) == 10 and {r["gpu_count"] for r in members} == {str(gpus)}
        for field in plan.WORKLOAD_FIELDS:
            assert {r[field] for r in members} == {template[field]}, (name, field)
        assert {r["instrumentation"] for r in members} == {"full"}
    assert sorted(int(r["training_seed"]) for r in E2 if r["condition"] == "g2-math-c3") == list(range(611, 621))
    counts = E2_DESIGN["position_counts"]
    assert all(set(v) <= {2, 3} and sum(v) == 10 for v in counts.values())
    for members in blocks(E2).values():
        assert sorted(r["condition"] for r in members) == sorted(plan.E2_CONFIGS)


def test_e3_schedule_and_restore_links():
    for members in blocks(E3).values():
        roles = [r["role"] for r in members]
        assert roles[:2] == ["screen", "long"]
        assert (roles[2:] == ["restore"]) == (members[0]["training_seed"] == "631")
        for r in members:
            assert (r["actor_micro_batch"], r["gpu_count"], r["instrumentation"]) == ("8", "4", "external")
            assert (r["save_freq"], r["test_freq"], r["max_actor_ckpt_to_keep"]) == ("25", "20", "3")
            assert r["total_epochs"] == "100" and r["train_max_samples"] == "-1"
        restore = [r for r in members if r["role"] == "restore"]
        for r in restore:
            assert r["resume_source_experiment_id"] == members[1]["experiment_id"]
            assert r["resume_step"] == "50" and r["val_before_train"] == "False"
    settings = {r["condition"]: (r["parameter_offload"], r["vllm_gpu_memory_utilization"]) for r in E3}
    assert settings == {"q7-gsm8k-off1-r70": ("True", "0.70"), "q7-gsm8k-off0-r60": ("False", "0.60"),
                        "m7-code_heavy_tail-off1-r70": ("True", "0.70"),
                        "m7-code_heavy_tail-off0-r55": ("False", "0.55")}


def test_seeds_fresh_and_documented():
    used = {int(r["training_seed"]) for r in E1 + E2 + E3}
    assert used == set(C.SEEDS["e1"]) | set(C.SEEDS["e2"]) | set(C.SEEDS["e3"])
    audit = plan.seed_audit(sorted(used | {641, 642}))
    assert audit["fresh"] and not audit["collisions"]
    assert 211 in audit["previously_used"] and 501 in audit["previously_used"]


def test_models_pinned_and_hashes_available():
    inputs = plan.expected_inputs()
    for row in E1 + E2 + E3:
        assert row["model_revision"] == C.REVISIONS[row["model"]]
        assert row["data_dir"].startswith(str(C.CLUSTER_DATA_ROOT))
    assert set(inputs["models"]) >= set(C.REVISIONS)
    for subdir in C.DATA_SUBDIRS.values():
        for split in ("train.parquet", "test.parquet"):
            assert f"data/{subdir}/{split}" in inputs["data"]


def test_timeouts_fit_wall_caps():
    from ieee_resub.render import wall_budget
    for study, rows in (("e1", E1), ("e2", E2), ("e3", E3), ("smoke1", plan.smoke1_rows()),
                        ("smoke2", plan.smoke2_rows())):
        for members in blocks(rows).values():
            assert wall_budget(members, C.CAPS[study])["fits"], study
