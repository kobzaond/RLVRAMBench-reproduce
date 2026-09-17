import json

import pytest

from memory_tuner import replication_control_accounting as accounting
from memory_tuner.capture_allocation_accounting import capture, timestamp


@pytest.fixture
def evidence(tmp_path):
    root = tmp_path / "raw"
    block = root / "blocks" / "control-a"
    block.mkdir(parents=True)
    start = timestamp("2026-09-16T10:00:00")
    (block / "replication_control_block.json").write_text(json.dumps({
        "job_id": "123", "block_id": "control-a",
        "allocation": {"started_at_epoch": start, "hostname": "node.example"},
    }))
    (block / "replication_control_ledger.jsonl").write_text(json.dumps({
        "job_id": "123", "block_id": "control-a", "event": "claimed",
        "timestamp_ns": int((start + 1) * 1e9),
    }) + "\n" + json.dumps({
        "event": "terminal", "timestamp_ns": int((start + 60.5) * 1e9),
    }) + "\n")
    # FAILED is deliberate: invalid scientific outcomes still incur full cost.
    text = "\n".join(
        f"123_0{suffix}|123{suffix}|{name}|FAILED|2:0|"
        "2026-09-16T10:00:00|2026-09-16T10:01:00|60|"
        "cpu=32,gres/gpu=2|node|NonZeroExitCode"
        for suffix, name in (("", "control"), (".batch", "batch"))
    )
    claims, _ = accounting.block_claims(root)
    return root, capture(text, claims)


def test_failed_allocation_cost_and_precision_without_double_counting(evidence):
    root, records = evidence
    result = accounting.reconstruct(root, records)
    assert result["allocation_count"] == 1
    assert result["whole_allocation_gpu_seconds"] == 120
    assert result["allocations"][0]["batch_step_gpu_seconds"] == 120
    assert result["allocations"][0]["scheduler_state"] == "FAILED"


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "block", "ledger", "node"])
def test_cost_reconstruction_rejects_changed_evidence(evidence, mutation):
    root, records = evidence
    block = root / "blocks/control-a"
    if mutation == "missing":
        records.clear()
    elif mutation == "duplicate":
        records.append(records[0])
    elif mutation == "block":
        (block / "replication_control_block.json").write_text("{}")
    elif mutation == "ledger":
        with (block / "replication_control_ledger.jsonl").open("a") as handle:
            handle.write("\n")
    else:
        records[0]["scheduler_node_list"] = "different"
    with pytest.raises((ValueError, KeyError)):
        accounting.reconstruct(root, records)


def test_linked_evidence_is_rejected(evidence, tmp_path):
    root, records = evidence
    path = root / "blocks/control-a/replication_control_ledger.jsonl"
    moved = tmp_path / "elsewhere.jsonl"
    path.rename(moved)
    path.symlink_to(moved)
    with pytest.raises(ValueError, match="linked"):
        accounting.reconstruct(root, records)


def test_longer_batch_does_not_inflate_cancelled_outer_cost(evidence):
    root, _ = evidence
    text = (
        "123_0|123|control|CANCELLED|0:0|2026-09-16T10:00:00|"
        "2026-09-16T10:01:00|60|cpu=32,gres/gpu=2|node|None\n"
        "123_0.batch|123.batch|batch|CANCELLED|0:15|2026-09-16T10:00:00|"
        "2026-09-16T10:01:05|65|cpu=32,gres/gpu=2|node|None\n"
    )
    claims, _ = accounting.block_claims(root)
    result = accounting.reconstruct(root, capture(text, claims))
    assert result["whole_allocation_gpu_seconds"] == 120
    assert result["allocations"][0]["batch_step_gpu_seconds"] == 130


@pytest.mark.parametrize("ledger_present", [False, True])
def test_incomplete_second_claim_cannot_disappear_from_total(evidence, ledger_present):
    root, records = evidence
    partial = root / "blocks/control-b"
    partial.mkdir()
    if ledger_present:
        (partial / "replication_control_ledger.jsonl").write_text(json.dumps({
            "event": "claimed", "block_id": "control-b", "job_id": "456",
            "timestamp_ns": 1789552800000000000,
        }) + "\n")
    with pytest.raises(ValueError, match="Incomplete control allocation claim"):
        accounting.reconstruct(root, records)


@pytest.mark.parametrize("boundary,accepted", [
    ("before_start", False), ("exact_start", True),
    ("before_upper_bound", True), ("exact_upper_bound", False),
])
def test_ledger_bounds_use_exact_integer_nanoseconds(evidence, boundary, accepted):
    root, records = evidence
    path = root / "blocks/control-a/replication_control_ledger.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    start_ns = int(records[0]["started_at_epoch"]) * 1_000_000_000
    upper_ns = (int(records[0]["ended_at_epoch"]) + 1) * 1_000_000_000
    if boundary in ("before_start", "exact_start"):
        events[0]["timestamp_ns"] = start_ns - (boundary == "before_start")
    else:
        events[-1]["timestamp_ns"] = upper_ns - (boundary == "before_upper_bound")
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n")
    claims, _ = accounting.block_claims(root)
    records[0].update(claims["123"])
    if accepted:
        assert accounting.reconstruct(root, records)["whole_allocation_gpu_seconds"] == 120
    else:
        with pytest.raises(ValueError, match="outside"):
            accounting.reconstruct(root, records)
