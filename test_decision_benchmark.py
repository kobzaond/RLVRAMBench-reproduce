import copy
import pytest
import decision_benchmark as db


def fixture():
    candidates = [{
        "candidate_id": f"c{i}", "actor_micro_batch": 8, "reservation": .6,
        "slot_ids": {"screen": [f"s{i}"], "evaluation": [f"e{i}-{j}" for j in range(3)]}}
        for i in range(3)]
    protocol = {
        "protocol_version": "decision-1.0", "screen_seed": 141,
        "evaluation_seeds": [142, 143, 144], "memory_limit_mib": 38912,
        "additional_donor_guard_mib": 2048, "budgets_per_case": [0, 1, 2, 3],
        "cases": [{"case_id": "case", "model_family": "model", "dataset": "task",
                   "candidates": candidates, "query_order": ["c1", "c0", "c2"],
                   "donors": {f"c{i}": {
                       "configuration_id": f"d{i}", "observed_state": "within_margin",
                       "max_completed_run_peak_mib": [33000, 38000, 35000][i],
                       "recorded_attempt_ids": [f"d{i}-{j}" for j in range(3)]}
                       for i in range(3)}}]}
    records = []
    for i, c in enumerate(candidates):
        for role, slots in c["slot_ids"].items():
            for slot in slots:
                state = ["within_margin", "above_margin", "memory_failure"][i]
                records.append({"attempt_id": slot, "slot_id": slot,
                                "attempt_index": 1, "role": role, "state": state,
                                "peak_mib": 33000 if i == 0 else 40000 if i == 1 else None,
                                "completed_final_operations": i != 2})
    return protocol, records


def test_screen_is_not_its_own_validation():
    protocol, records = fixture()
    screen = next(r for r in records if r["slot_id"] == "s2")
    screen.update(state="within_margin", peak_mib=34000, completed_final_operations=True)
    result = db.replay(protocol, records)
    rows = [r for r in result["per_case"] if r["budget_target_attempts"] == 3]
    assert all(r["metrics"]["approved_memory_failure"] == 1 for r in rows)
    assert all(r["decisions"] == rows[0]["decisions"] for r in rows)


def test_hidden_outcomes_cannot_change_acquisition_or_decisions():
    protocol, records = fixture()
    original = db.replay(protocol, records)
    for row in records:
        if row["role"] == "evaluation":
            row.update(state="within_margin", peak_mib=33000, completed_final_operations=True)
    changed = db.replay(protocol, records)
    for a, b in zip(original["per_case"], changed["per_case"]):
        for key in ("decisions", "transcript", "spent_target_attempts", "donor_attempts"):
            assert a[key] == b[key]
    assert original["candidate_evaluation"] != changed["candidate_evaluation"]


def test_zero_budget_and_cost_scenarios():
    protocol, records = fixture()
    rows = db.replay(protocol, records)["per_case"]
    for row in rows:
        assert row["spent_target_attempts"] == row["budget_target_attempts"]
        assert row["donor_attempts"] == (0 if row["rule"] == "direct_screen" else 9)
        assert len({r["attempt_id"] for r in row["transcript"]}) == len(row["transcript"])
    copy0 = next(r for r in rows if r["rule"] == "source_copy" and r["budget_target_attempts"] == 0)
    assert copy0["metrics"]["approved_memory_failure"] == 1
    assert copy0["metrics"]["approved_above_margin"] == 1
    assert copy0["metrics"]["within_margin_recall"] == 1


def test_repair_charged_before_result_and_never_peeked_to_skip():
    protocol, records = fixture()
    first = next(r for r in records if r["slot_id"] == "s1")
    repaired = {**first, "attempt_id": "s1-repair", "attempt_index": 2}
    first.update(state="unresolved", peak_mib=None, completed_final_operations=False,
                 repair_reason="port_binding_collision")
    records.append(repaired)
    rows = db.replay(protocol, records)["per_case"]
    one = next(r for r in rows if r["rule"] == "source_copy" and r["budget_target_attempts"] == 1)
    two = next(r for r in rows if r["rule"] == "source_copy" and r["budget_target_attempts"] == 2)
    assert one["decisions"]["c1"] == "abstain"
    assert two["decisions"]["c1"] == "reject"
    assert len(one["transcript"]) == 1 and len(two["transcript"]) == 2
    assert all(r["candidate_id"] == "c1" for r in two["transcript"])


def test_unresolved_evaluation_not_silently_dropped():
    protocol, records = fixture()
    row = next(r for r in records if r["slot_id"] == "e0-0")
    row.update(state="unresolved", peak_mib=None, completed_final_operations=False)
    result = db.replay(protocol, records)
    assert result["candidate_evaluation"]["c0"]["state"] == "unresolved"
    metrics = result["per_case"][0]["metrics"]
    assert metrics["unresolved_targets"] == 1
    assert metrics["approved_unresolved"] == 1
    assert metrics["within_margin_recall"] is None
    assert metrics["missing_outcome_sensitivity"]["within_margin_recall_max"] == 1
    assert metrics["missing_outcome_sensitivity"]["unsuitable_approvals_min"] == 2
    assert metrics["missing_outcome_sensitivity"]["unsuitable_approvals_max"] == 3


def test_unstarted_not_fabricated_as_charged_attempt():
    protocol, records = fixture()
    records = [r for r in records if r["slot_id"] != "s1"]
    rows = db.replay(protocol, records)["per_case"]
    row = next(r for r in rows if r["rule"] == "direct_screen" and r["budget_target_attempts"] == 1)
    assert row["spent_target_attempts"] == 0
    assert row["transcript"] == [{"candidate_id": "c1", "status": "not_started", "charged_attempts": 0}]


def test_input_objects_exclude_hidden_ids_and_outcomes():
    protocol, _ = fixture()
    task = db.task_inputs(protocol, "direct_screen")[0]
    assert task["donors"] == {}
    assert all("slot_ids" not in c and "observed_state" not in c for c in task["candidates"])


@pytest.mark.parametrize("corruption", ("duplicate", "wrong_role", "bad_peak",
                                      "missing_final_operations", "unauthorized_retry"))
def test_invalid_records_rejected(corruption):
    protocol, records = fixture()
    if corruption == "duplicate":
        records.append(copy.deepcopy(records[0]))
    elif corruption == "wrong_role":
        records[0]["role"] = "evaluation"
    elif corruption == "bad_peak":
        records[0]["peak_mib"] = float("nan")
    elif corruption == "missing_final_operations":
        records[0]["completed_final_operations"] = False
    else:
        records.append({**records[0], "attempt_id": "retry", "attempt_index": 2})
    with pytest.raises(ValueError):
        db.replay(protocol, records)


def test_overlapping_roles_rejected():
    protocol, _ = fixture()
    protocol["cases"][0]["candidates"][0]["slot_ids"]["evaluation"][0] = "s0"
    with pytest.raises(ValueError):
        db.validate_protocol(protocol)


def test_known_oom_overrides_artifact_repair_permission():
    protocol, records = fixture()
    first = next(r for r in records if r["slot_id"] == "s1")
    first.update(state="unresolved", peak_mib=None, completed_final_operations=False,
                 known_memory_failure=True, repair_reason="corrupted_artifact_writing")
    with pytest.raises(ValueError, match="setup"):
        db.replay(protocol, records)


def test_known_margin_crossing_precludes_safe_assignment():
    truth = {
        "known": {"state": "within_margin", "known_memory_failure": False},
        "partial": {"state": "unresolved", "known_memory_failure": False,
                    "known_above_margin": True}}
    metrics = db.score({"known": "reject", "partial": "approve"}, truth)
    bounds = metrics["missing_outcome_sensitivity"]
    assert bounds["within_margin_recall_min"] == bounds["within_margin_recall_max"] == 0
    assert bounds["unsuitable_approvals_min"] == bounds["unsuitable_approvals_max"] == 1


def test_partial_failure_evidence_survives_truth_and_summary():
    protocol, records = fixture()
    # The other two evaluation runs have known OOM; missing one must not
    # turn that evidence into zero failure risk in the summary.
    records = [r for r in records if r["slot_id"] != "e2-0"]
    result = db.replay(protocol, records)
    truth = result["candidate_evaluation"]["c2"]
    assert truth["state"] == "unresolved" and truth["known_memory_failure"]
    row = next(r for r in result["summary"] if r["rule"] == "source_copy"
               and r["budget_target_attempts_per_case"] == 0)
    assert row["approved_memory_failure"] == 0
    assert row["approved_known_memory_failure"] == 1
