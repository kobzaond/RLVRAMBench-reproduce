"""e3-amendment-01: split configuration record (defect 2), amendment chain, rerun guards."""
import json
import os
from pathlib import Path
import shutil

import pytest

from ieee_resub import amendment as AM
from ieee_resub import assess as A
from ieee_resub import common as C
from ieee_resub import collect, plan, runner

DATA = Path(__file__).resolve().parent / "fixtures"
RECORDED = C.EXPERIMENT_DIR / "amendments" / "e3-amendment-01.json"
E3_PROTOCOL_SHA256 = "f6ef9c52ddae8ef97e832e85215aa79341c29f262e00d59611e088751acd53de"


# --------------------------------------------------------------------------- defect 2
def test_split_configuration_start_from_the_real_log_parses_like_an_unsplit_log():
    """Heads (ANSI codes stripped) of training-5381356.log: the long run, whose first
    configuration record Ray split into "{" and "'actor_rollout_ref': ...", and the
    screen of the same block (seed 633, same configuration), printed normally."""
    from memory_tuner.electronics_experiment import applied_actor_configuration as frozen
    split = (DATA / "training-5381356-long-head.log").read_text()
    unsplit = (DATA / "training-5381356-screen-head.log").read_text()
    assert "(TaskRunnerV1 pid=1360627) {\n(TaskRunnerV1 pid=1360627) 'actor_rollout_ref': {'actor':" in split
    with pytest.raises(ValueError, match="Missing printed actor configuration"):
        frozen(split)  # the defect
    pid, actual = A.applied_actor_configuration(split)
    reference_pid, reference = frozen(unsplit)
    assert A.applied_actor_configuration(unsplit) == (reference_pid, reference)  # unchanged where it worked
    assert pid == 1360627 and reference_pid == 1318471
    # Same actor section; only the trace name embeds the experiment id (screen vs long).
    assert actual["rollout"]["trace"].pop("experiment_name").endswith("-s633-long/attempt-5381356")
    assert reference["rollout"]["trace"].pop("experiment_name").endswith("-s633-screen/attempt-5381356")
    assert actual == reference
    assert actual["actor"]["ppo_micro_batch_size_per_gpu"] == 8 and actual["model"]["lora_rank"] == 64
    # The rejoined text differs from the original only by the joined brace record.
    rejoined = A.rejoin_split_configuration_start(split).splitlines()
    assert len(rejoined) == len(split.splitlines()) - 1
    assert rejoined[19].startswith("(TaskRunnerV1 pid=1360627) {'actor_rollout_ref': {'actor':")


def test_a_brace_record_is_a_start_only_before_the_actor_section_of_the_same_process():
    lines = ["(TaskRunnerV1 pid=7) {", "(TaskRunnerV1 pid=7) 'reward': {'x': 1},",
             "(TaskRunnerV1 pid=7)  'algorithm': {'a': 1}}"]
    with pytest.raises(ValueError, match="Missing printed actor configuration"):
        A.applied_actor_configuration("\n".join(lines))
    other = ["(TaskRunnerV1 pid=7) {", "(TaskRunnerV1 pid=8) 'actor_rollout_ref': {'actor': {'x': 1}},",
             "(TaskRunnerV1 pid=8)  'algorithm': {'a': 1}}"]
    with pytest.raises(ValueError, match="Missing printed actor configuration"):
        A.applied_actor_configuration("\n".join(other))
    good = ["(TaskRunnerV1 pid=7) {", "noise from another actor",
            "(TaskRunnerV1 pid=7) 'actor_rollout_ref': {'actor': {'x': 1}},",
            "(TaskRunnerV1 pid=7)  'algorithm': {'a': 1}}"]
    assert A.applied_actor_configuration("\n".join(good)) == (7, {"actor": {"x": 1}})
    incomplete = ["(TaskRunnerV1 pid=7) {", "(TaskRunnerV1 pid=7) 'actor_rollout_ref': {'actor': {'x': ",
                  "(TaskRunnerV1 pid=7)  'algorithm': {'a': 1}}"]
    with pytest.raises(ValueError, match="incomplete"):
        A.applied_actor_configuration("\n".join(incomplete))


# --------------------------------------------------------------------------- amendment chain
def protocol_of(amendment):
    """The frozen protocol's source hashes, reconstructed from the recorded changes."""
    before = dict(amendment["source_files_sha256"])
    for name, change in amendment["changed_source_files"].items():
        if change["before"] is None:
            before.pop(name)
        else:
            before[name] = change["before"]
    return dict(study="e3", group=C.GROUPS["e3"], source_files_sha256=before)


def test_recorded_amendment_is_chained_and_matches_this_tree_and_the_matrix():
    amendment = json.loads(RECORDED.read_text())
    assert amendment["protocol_sha256"] == E3_PROTOCOL_SHA256
    AM.validate(amendment, protocol_of(amendment), E3_PROTOCOL_SHA256, "e3")
    AM.check_against_matrix(amendment, C.read_rows(C.EXPERIMENT_DIR / "matrices/e3_matrix.csv"))
    # Fails after any edit of a hashed source file until `amendment.py refresh` ran.
    assert amendment["source_files_sha256"] == plan.source_hashes()
    assert [r["experiment_id"] for r in amendment["reruns"]] == ["e3-m7-code_heavy_tail-off1-r70-s631-restore",
                                                                "e3-m7-code_heavy_tail-off1-r70-s632-long"]
    assert "restore_source" not in amendment["reruns"][1] and amendment["reruns"][1]["original_job_id"] == "5382462"
    assert amendment["amended_launch_blocks"]["array_indices"] == list(range(5, 12))
    frozen = C.EXPERIMENT_DIR / "FROZEN_PROTOCOLS.txt"
    if frozen.is_file():
        assert f"e3 {E3_PROTOCOL_SHA256}" in frozen.read_text().splitlines()


def test_amendment_load_refuses_wrong_hash_location_and_chain(tmp_path):
    amendment = json.loads(RECORDED.read_text())
    protocol = protocol_of(amendment)
    protocol_path = tmp_path / "e3_protocol.json"
    protocol_path.write_text("{}")
    target = tmp_path / "amendment-01.json"
    shutil.copy(RECORDED, target)
    sha = C.digest(target)
    assert AM.load(target, sha, protocol_path, protocol, E3_PROTOCOL_SHA256, "e3")["number"] == 1
    with pytest.raises(ValueError, match="changed"):
        AM.load(target, "0" * 64, protocol_path, protocol, E3_PROTOCOL_SHA256, "e3")
    with pytest.raises(ValueError, match="not chained"):
        AM.load(target, sha, protocol_path, protocol, "1" * 64, "e3")
    elsewhere = tmp_path / "x" / "amendment-01.json"
    elsewhere.parent.mkdir()
    shutil.copy(RECORDED, elsewhere)
    with pytest.raises(ValueError, match="next to their protocol"):
        AM.load(elsewhere, sha, protocol_path, protocol, E3_PROTOCOL_SHA256, "e3")
    drifted = dict(protocol, source_files_sha256=dict(protocol["source_files_sha256"], **{"ieee_resub/x.py": "f"}))
    with pytest.raises(ValueError, match="source changes"):
        AM.validate(amendment, drifted, E3_PROTOCOL_SHA256, "e3")
    memory = json.loads(RECORDED.read_text())
    memory["reruns"][0]["required_original_status"] = "memory_failure"
    with pytest.raises(ValueError, match="Only unresolved"):
        AM.validate(memory, protocol, E3_PROTOCOL_SHA256, "e3")


# --------------------------------------------------------------------------- rerun guards
def fake_checkpoint(folder, gpus=4):
    (folder / "actor").mkdir(parents=True)
    (folder / "data.pt").write_text("x")
    for rank in range(gpus):
        for kind in ("model", "optim", "extra_state"):
            (folder / "actor" / f"{kind}_world_size_{gpus}_rank_{rank}.pt").write_text(kind + str(rank))


@pytest.fixture
def cluster(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "CLUSTER_SCRATCH_ROOT", tmp_path / "scratch")
    group = tmp_path / "output" / C.GROUPS["e3"]
    original = tmp_path / "scratch/checkpoints/g/long/attempt-5381357/global_step_50"
    fake_checkpoint(original)
    preserved = tmp_path / "scratch/preserved-checkpoints/long-attempt-5381357/global_step_50"
    shutil.copytree(original, preserved, copy_function=shutil.copy2)
    restore = "e3-m7-code_heavy_tail-off1-r70-s631-restore"
    attempt = group / restore / "attempt-5381357"
    attempt.mkdir(parents=True)
    record = attempt / "restore-source.json"
    record.write_text(json.dumps(dict(folder=str(original), manifest=runner.checkpoint_manifest(original))))
    block = group / "blocks" / "e3-m7-code_heavy_tail-off1-r70-s631"
    block.mkdir(parents=True)
    (block / "block.json").write_text(json.dumps(dict(allocation=dict(job_id="5381357"))))
    (block / "period-3.json").write_text(json.dumps(dict(experiment_id=restore, launched=True, status="unresolved",
                                                         attempt_path=str(attempt), validation_error="Timed out")))
    (block / "summary.json").write_text("{}")
    entry = dict(json.loads(RECORDED.read_text())["reruns"][0], original_attempt=str(attempt))
    entry["restore_source"] = dict(entry["restore_source"], folder=str(preserved), original_folder=str(original),
                                   manifest_record=str(record), manifest_record_sha256=C.digest(record),
                                   manifest_files=13)
    row = AM.matrix_row(plan.e3_rows()[0], restore)
    return dict(group=group, entry=entry, row=row, preserved=preserved, block=block)


def test_preserved_checkpoint_must_equal_the_original_manifest(cluster):
    folder, manifest, expected = runner.prepare_preserved_source(cluster["entry"], cluster["row"], cluster["group"], 4)
    assert folder == cluster["preserved"] and manifest == expected and len(manifest) == 13
    assert not (folder / "data.pt").stat().st_mode & 0o222
    shard = folder / "actor/optim_world_size_4_rank_2.pt"
    shard.chmod(0o644)
    shard.write_text("changed")
    with pytest.raises(ValueError, match="differs from the original manifest"):
        runner.prepare_preserved_source(cluster["entry"], cluster["row"], cluster["group"], 4)
    outside = dict(cluster["entry"], restore_source=dict(cluster["entry"]["restore_source"],
                                                         folder=str(cluster["preserved"]).replace("preserved-", "")))
    with pytest.raises(ValueError, match="not a preserved"):
        runner.prepare_preserved_source(outside, cluster["row"], cluster["group"], 4)


def test_rerun_needs_a_finished_unresolved_original(cluster):
    original = runner.original_slot(cluster["entry"], cluster["group"])
    assert original["receipt_status"] == "unresolved" and "period-3.json" in original["files_sha256"]
    receipt = cluster["block"] / "period-3.json"
    for status in ("within_margin", "above_margin", "memory_failure"):
        receipt.write_text(json.dumps(dict(json.loads(receipt.read_text()), status=status)))
        with pytest.raises(ValueError, match="never rerun"):
            runner.original_slot(cluster["entry"], cluster["group"])
    receipt.write_text(json.dumps(dict(json.loads(receipt.read_text()), status="unresolved")))
    os.remove(cluster["block"] / "summary.json")
    with pytest.raises(ValueError, match="not finished"):
        runner.original_slot(cluster["entry"], cluster["group"])


# --------------------------------------------------------------------------- collector rule
def test_collector_accepts_a_raw_status_only_for_the_listed_reassessment(tmp_path, monkeypatch):
    rows, _ = plan.e3_rows()
    row = AM.matrix_row(rows, "e3-q7-gsm8k-off1-r70-s633-long")
    group = tmp_path / C.GROUPS["e3"]
    block = group / "blocks" / row["block_id"]
    block.mkdir(parents=True)
    protocol = dict(allocation_gpus=4)
    (block / "block.json").write_text(json.dumps(dict(source_commit="c" * 40, protocol_sha256="p", protocol=protocol,
                                                      rows=[row], allocation=dict(job_id="5381356"))))
    error = "ValueError: Missing printed actor configuration"
    (block / "period-2.json").write_text(json.dumps(dict(launched=True, status="unresolved", validation_error=error)))
    attempt = group / row["experiment_id"] / "attempt-5381356"
    attempt.mkdir(parents=True)
    (attempt / "dispatch.json").write_text(json.dumps(dict(row=row, source_commit="c" * 40, protocol_sha256="p")))
    (attempt / "execution.json").write_text(json.dumps(dict(invocation_elapsed_seconds=10.0)))
    monkeypatch.setattr(A, "assess", lambda *a: dict(status="within_margin", completed_peak_mib=31000.0))
    listed = dict(receipt_status="unresolved", receipt_validation_error=error, defect="defect-2")
    amendment = dict(record=dict(amendment_id="e3-amendment-01"), sha256="a" * 64, source_commit="d" * 40,
                     reassessed={(row["experiment_id"], "5381356"): listed})

    def assess(amendment):
        changes = []
        result = collect.assess_claim(row, block, group=group, protocol=protocol, protocol_hash="p",
                                      source_commit="c" * 40, base=None, amendment=amendment,
                                      verify_inventory=False, claim_kind="original", changes=changes)
        return result, changes
    result, changes = assess(amendment)
    assert result["status"] == "within_margin" and result["reassessed_under"] == "e3-amendment-01 defect-2"
    assert changes[0]["receipt_status"] == "unresolved" and changes[0]["raw_status"] == "within_margin"
    result, changes = assess(None)  # without the amendment the receipt wins, and the change is reported
    assert result["status"] == "unresolved" and result["validation_error"].startswith("Executor rejected")
    assert changes[0]["accepted_under"] is None
    other = dict(amendment, reassessed={(row["experiment_id"], "999"): listed})
    assert assess(other)[0]["status"] == "unresolved"


def test_collector_uses_the_rerun_as_slot_outcome_and_keeps_the_original_in_the_ledger(tmp_path, monkeypatch):
    rows, _ = plan.e3_rows()
    record = json.loads(RECORDED.read_text())
    restore = record["reruns"][0]["experiment_id"]
    group = tmp_path / C.GROUPS["e3"]
    (group / record["reruns"][0]["claim"]).mkdir(parents=True)
    (group / record["reruns"][0]["claim"] / "block.json").write_text("{}")
    protocol = dict(group=C.GROUPS["e3"], caps=C.CAPS["e3"], allocation_gpus=4)
    monkeypatch.setattr(runner, "load", lambda *a: (protocol, rows, record))
    monkeypatch.setattr(runner, "runtime", lambda *a: None)

    def fake(row, block_dir, *, claim_kind, **_):
        if row["experiment_id"] != restore:
            return dict(status="not_launched", launched=False, claim=block_dir.name, claim_kind=claim_kind)
        rerun = claim_kind == "amendment_rerun"
        return dict(status="within_margin" if rerun else "unresolved", launched=True,
                    job_id="8000001" if rerun else "5381357", claim=block_dir.name, claim_kind=claim_kind)
    monkeypatch.setattr(collect, "assess_claim", fake)
    summary, results, attempts = collect.collect("e3", tmp_path / "p.json", "p", "c" * 40, tmp_path,
                                                 amendment_path=tmp_path / "a.json", amendment_hash="a" * 64,
                                                 amended_source_commit="d" * 40)
    slot = next(r for r in results if r["experiment_id"] == restore)
    assert slot["status"] == "within_margin" and slot["job_id"] == "8000001"
    assert slot["outcome_source"] == "e3-amendment-01 rerun" and slot["original_attempt"]["status"] == "unresolved"
    ledger = [a for a in attempts if a["experiment_id"] == restore]
    assert [(a["job_id"], a["status"], a["slot_outcome"]) for a in ledger] == [
        ("5381357", "unresolved", False), ("8000001", "within_margin", True)]
    pending = record["reruns"][1]["experiment_id"]  # no rerun claim yet: the original stays the outcome
    assert summary["amendment"]["reruns"] == {restore: "within_margin", pending: "not_launched"}
    assert next(r for r in results if r["experiment_id"] == pending)["amendment_rerun_pending"] == record["reruns"][1]["claim"]
    assert len(results) == 28
