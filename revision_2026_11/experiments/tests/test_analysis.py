"""Planned estimands: reproduction of published numbers and synthetic E1-E3 outcomes."""
from ieee_resub import analysis as N


def test_reproduces_published_numbers_from_existing_outputs():
    result = N.check_existing()
    failed = {k: v for k, v in result["checks"].items() if not v["ok"]}
    assert result["all_ok"], failed
    assert round(result["checks"]["replication_actor_stage_batch_contrast_equal_case_mean_mib"]["observed"]) == 3905
    assert round(result["checks"]["replication_whole_run_batch_contrast_equal_case_mean_mib"]["observed"]) == 7


def e1_outcome(block, case, batch, logging, actor, whole, shape=("p", "l"), tokens=None, status="within_margin"):
    return dict(experiment="e1", block_id=block, case_id=case, actor_micro_batch=str(batch), instrumentation=logging,
                status=status, stage_peaks_mib=dict(actor_update=actor), completed_peak_mib=whole,
                shape_signature=list(shape), max_actor_micro_batch_tokens=tokens or batch * 1000)


def test_e1_identical_shape_pool_and_decisions():
    outcomes = []
    for seed, bump in ((1, 0), (2, 100)):
        block = f"e1-x-s{seed}"
        outcomes += [e1_outcome(block, "e1-x", 8, "ext", 20000 + bump, 33000),
                     e1_outcome(block, "e1-x", 16, "ext", 26000 + bump, 33000),
                     e1_outcome(block, "e1-x", 8, "full", 20100 + bump, 33010),
                     e1_outcome(block, "e1-x", 16, "full", 26300 + bump, 33020)]
    outcomes += [e1_outcome("e1-y-s1", "e1-y", 8, "ext", 20000, 33000),
                 e1_outcome("e1-y-s1", "e1-y", 16, "ext", 34500, 34500),
                 e1_outcome("e1-y-s1", "e1-y", 8, "full", 20000, 33000),
                 e1_outcome("e1-y-s1", "e1-y", 16, "full", 34400, 34400, shape=("p2", "l"))]
    result = N.analyze_e1(outcomes)
    x = result["identical_shape_cases"]["e1-x"]
    assert x["pooled_blocks"] == 2 and x["actor_batch_ext"]["mean"] == 6000 and x["actor_interaction"]["mean"] == 200
    assert x["actor_mib_per_token_ext"]["mean"] == 6000 / 8000
    assert result["identical_shape_cases"]["e1-y"]["pooled_blocks"] == 0  # shape mismatch excluded
    assert result["all_complete_cases"]["e1-y"]["pooled_blocks"] == 1
    assert result["equal_case_means"]["actor_batch_ext"]["missing"] == ["e1-y"]
    assert result["decisions"]["D1_actor_contrast_positive_every_identical_block"]
    assert result["decisions"]["D2_logging_interaction_within_512_every_identical_block"]
    assert result["decisions"]["shape_mismatch_blocks"] == ["e1-y-s1"]


def test_e1_incomplete_block_is_reported_not_pooled():
    outcomes = [e1_outcome("b", "c", 8, "ext", 1, 2), e1_outcome("b", "c", 16, "ext", 1, 2, status="memory_failure"),
                e1_outcome("b", "c", 8, "full", 1, 2), e1_outcome("b", "c", 16, "full", 1, 2)]
    result = N.analyze_e1(outcomes)
    assert result["blocks"][0]["complete"] is False
    assert result["outcome_counts"]["c"] == {"within_margin": 3, "memory_failure": 1}


def test_e2_frequencies_and_exact_intervals():
    statuses = ["within_margin"] * 7 + ["above_margin"] * 2 + ["unresolved"]
    outcomes = [dict(experiment="e2", configuration_id="e2-q3-gsm8k-rel-r65", status=s,
                     completed_peak_mib=38000.0 + i if s.endswith("margin") else None) for i, s in enumerate(statuses)]
    table = N.analyze_e2(outcomes)["e2-q3-gsm8k-rel-r65"]
    assert table["eligible"] == 9 and table["unresolved"] == 1 and table["label_10"] == "unresolved"
    assert table["within_margin"]["k"] == 7
    lo, hi = table["within_margin"]["ci95_exact"]
    assert abs(lo - 0.3999) < 1e-3 and abs(hi - 0.9719) < 1e-3
    assert table["history"]["within"] == 5


def test_e3_pairs_growth_and_restore_delta():
    def row(role, seed, status, peak, growth=None, steps=None, extra=None):
        out = dict(experiment="e3", configuration_id="e3-c", role=role, training_seed=str(seed),
                   experiment_id=f"e3-c-s{seed}-{role}", status=status, completed_peak_mib=peak,
                   maximum_later_step_increase_mib=growth, step_peaks_mib=steps or {})
        out.update(extra or {})
        return out
    long_steps = {str(s): 36000.0 + (300 if s == 70 else 0) for s in range(1, 101)}
    restore_steps = {str(s): 36100.0 for s in range(51, 101)}
    outcomes = [row("screen", 631, "within_margin", 35900), row("long", 631, "within_margin", 36300, 300, long_steps),
                row("screen", 632, "within_margin", 35950), row("long", 632, "above_margin", 39000, 2900),
                row("restore", 631, "within_margin", 36500, None, restore_steps,
                    dict(resume_source_experiment_id="e3-c-s631-long", resume_step="50", checkpoint_load_peak_mib=21000))]
    table = N.analyze_e3(outcomes)["e3-c"]
    assert table["resolved_pairs"] == 2 and table["individual_agreements"] == 1
    assert table["long_label"] == "above_margin" and table["screen_label"] == "within_margin"
    restore = table["restore"][0]
    assert restore["uninterrupted_late_peak_mib"] == 36300 and restore["restore_minus_uninterrupted_late_mib"] == 200
    assert restore["restore_steps_minus_uninterrupted_late_mib"] == -200


def test_e1_collector_shaped_outcomes_use_external_instrumentation():
    """e3-amendment-01, defect 3: collect.py copies the matrix row into each outcome,
    so instrumentation is "external"/"full" while the frozen estimands use "ext"."""
    from ieee_resub import plan
    rows, _ = plan.e1_rows()
    members = [r for r in rows if r["block_id"] == rows[0]["block_id"]]
    assert {r["instrumentation"] for r in members} == {"external", "full"}
    actor = {("8", "external"): 20000.0, ("16", "external"): 26000.0,
             ("8", "full"): 20100.0, ("16", "full"): 26300.0}
    outcomes = [dict(r, status="within_margin", validation_error=None, completed_peak_mib=33000.0,
                     stage_peaks_mib=dict(actor_update=actor[(r["actor_micro_batch"], r["instrumentation"])]),
                     shape_signature=["p", "l"], max_actor_micro_batch_tokens=int(r["actor_micro_batch"]) * 1000)
                for r in members]
    result = N.analyze_e1(outcomes)
    block = result["blocks"][0]
    assert block["complete"] and block["shape_identical"]
    assert (block["actor_batch_ext"], block["actor_batch_full"], block["actor_interaction"]) == (6000, 6200, 200)
    assert result["identical_shape_cases"][members[0]["case_id"]]["pooled_blocks"] == 1
    assert result["decisions"]["D1_actor_contrast_positive_every_identical_block"]
