"""Portable scheduler accounting tests; no scheduler access."""
import copy
from datetime import datetime, timezone

import pytest

from memory_tuner import capture_allocation_accounting as accounting

START = 1789482736
CLAIMS = {"5054345": dict(pair_id="synthetic-pair",
                          pair_manifest_sha256="a" * 64, pair_ledger_sha256="b" * 64)}


def raw_line(job="5054345", *, start=START, end=START + 53,
             state="CANCELLED by 7547", step=""):
    def iso(seconds):
        return datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    suffix = "." + step if step else ""
    return "|".join((
        "5054326_1" + suffix, job + suffix, step or "rlvram-matched-gpu", state, "0:0",
        iso(start), iso(end), str(end - start), "cpu=64,gres/gpu=4,mem=384G,node=1",
        "synthetic-node", ""))


def accounting_row(*, job="5054345", start=START, end=START + 53, batch_end=START + 56,
                   state="CANCELLED by 7547", claim=None):
    text = raw_line(job, start=start, end=end, state=state) + "\n"
    if batch_end is not None:
        text += raw_line(job, start=start, end=batch_end, state="FAILED", step="batch") + "\n"
    return accounting.capture(text, {job: claim or CLAIMS["5054345"]})[0]


def test_outer_and_batch_times_and_raw_evidence_are_separate():
    row = accounting_row()
    assert row["ended_at_epoch"] == START + 53
    assert row["batch_ended_at_epoch"] == START + 56
    assert row["timestamp_precision_seconds"] == 1
    assert row["timestamp_interval"] == accounting.INTERVAL
    assert row["scheduler_scope"] == "outer_job_record"
    assert row["scheduler_state"] == "CANCELLED by 7547"
    assert accounting.validate_record(row)["allocated_gpu_count"] == 4
    assert row["raw_record_provenance"]["batch"]["line_number"] == 2


@pytest.mark.parametrize("field,value", [
    ("ended_at_epoch", START + 54), ("timestamp_precision_seconds", 2),
    ("scheduler_state", "COMPLETED"), ("job_id", "99999"),
    ("allocated_gpu_count", 8), ("batch_ended_at_epoch", START + 100),
    ("scheduler_scope", "outer_plus_batch"), ("pair_manifest_sha256", "invalid"),
    ("timestamp_precision_seconds", True), ("accounting_schema_version", True)])
def test_metadata_cannot_override_raw_records(field, value):
    row = accounting_row()
    row[field] = value
    with pytest.raises(ValueError):
        accounting.validate_record(row)


@pytest.mark.parametrize("mutation", ["raw_hash", "raw_line", "timezone", "format",
                                    "source_hash", "step_job", "node"])
def test_raw_record_provenance_and_step_identity_are_binding(mutation):
    row = accounting_row()
    raw = row["raw_record_provenance"]
    if mutation == "raw_hash":
        raw["outer"]["sha256"] = "0" * 64
    elif mutation == "raw_line":
        raw["outer"]["line"] += "|unexpected"
    elif mutation in {"step_job", "node"}:
        replacement = "99999" if mutation == "step_job" else "another-node"
        before = "5054345" if mutation == "step_job" else "synthetic-node"
        raw["batch"]["line"] = raw["batch"]["line"].replace(before, replacement)
        raw["batch"]["sha256"] = accounting.sha_text(raw["batch"]["line"])
    else:
        raw[{"timezone": "timezone", "format": "format_fields",
             "source_hash": "source_sha256"}[mutation]] = "wrong"
    with pytest.raises(ValueError):
        accounting.validate_record(row)


@pytest.mark.parametrize("text", [
    raw_line() + "\n" + raw_line(),
    raw_line("99999"), raw_line().replace("|53|", "|52|"),
    raw_line().replace("|CANCELLED by 7547|", "|RUNNING|"),
    raw_line().replace("14:33:09", "14:33:09.5"),
    raw_line().replace("gres/gpu=4", "gres/gpu=0"),
    raw_line().replace("2026-09-15T14:33:09", "2026-09-15T14:32:00"),
    raw_line(step="batch"),
])
def test_bad_intervals_unknown_jobs_and_missing_outer_are_rejected(text):
    with pytest.raises(ValueError):
        accounting.capture(text, copy.deepcopy(CLAIMS))


def test_batch_extension_requires_explicit_outer_cancellation():
    with pytest.raises(ValueError, match="non-cancelled"):
        accounting_row(state="FAILED")
    row = accounting_row(batch_end=None)
    assert row["batch_ended_at_epoch"] is None
