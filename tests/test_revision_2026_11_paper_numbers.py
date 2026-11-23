"""Numbers of the November 2026 revision experiments (E1-E3), recomputed from
the curated collector outputs and analyses in revision_2026_11/results/.

The full raw-evidence reconstruction of these files is a separate step of
reproduce_all.py; this test only checks the reported values.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "revision_2026_11" / "results"
LIMIT_MIB = 38912
CAPACITY_MIB = 40960


def load(study, name):
    return json.loads((RESULTS / study / name).read_text())


def test_operating_limit_is_a_convention_2048_below_capacity():
    release = json.loads((ROOT / "release.json").read_text())
    limit = release["operating_limit"]
    assert limit["a100_capacity_mib"] - limit["a100_limit_mib"] == 2048 == CAPACITY_MIB - LIMIT_MIB
    assert limit["mi250x_capacity_mib"] - limit["mi250x_limit_mib"] == 2048


def test_slot_inventories():
    assert load("e1", "summary.json")["states"] == {"within_margin": 48}
    assert load("e2", "summary.json")["states"] == {"within_margin": 30, "above_margin": 10}
    e3 = load("e3", "summary.json")
    assert e3["planned"] == 28 and e3["attempts"] == 30 and e3["states"] == {"within_margin": 28}
    assert e3["amendment"]["sha256"] == "966067e46423acd8dc256b313357a21a19010218243865d672127d9e88f62ee8"
    assert e3["amendment"]["amended_source_commit"] == "cc685c09a446c62f4507432e1962c2f49b277b1d"
    for study in ("e1", "e2"):
        assert load(study, "summary.json")["source_commit"] == "9f6faf94b472f62f910858be838710aedb3cbc55"


def test_controlled_length_equal_case_means():
    means = {k: v["value"] for k, v in load("e1", "analysis.json")["equal_case_means"].items()}
    assert round(means["actor_batch_ext"]) == 6369
    assert round(means["actor_batch_full"]) == 6358
    assert round(means["actor_batch_mean"]) == 6364
    assert round(means["actor_interaction"]) == -11
    assert abs(means["whole_batch_mean"]) < 512
    decisions = load("e1", "analysis.json")["decisions"]
    assert decisions["D1_actor_contrast_positive_every_identical_block"] is True
    assert decisions["D2_logging_interaction_within_512_every_identical_block"] is True
    assert decisions["shape_mismatch_blocks"] == []


def test_near_boundary_repetitions_keep_their_labels():
    table = load("e2", "analysis.json")
    labels = {k: v["label_10"] for k, v in table.items()}
    assert labels == {"e2-g2-math-c3": "within_margin", "e2-p4-gsm8k-c3": "above_margin",
                      "e2-q3-gsm8k-rel-r65": "within_margin", "e2-q7-gsm8k-off0-r60": "within_margin"}
    assert all(v["memory_failure"]["k"] == 0 and v["eligible"] == 10 for v in table.values())


def test_duration_and_restore():
    table = load("e3", "analysis.json")
    assert sum(v["individual_agreements"] for v in table.values()) == 12
    assert sum(v["resolved_pairs"] for v in table.values()) == 12
    restores = [r["restore_minus_uninterrupted_late_mib"] for v in table.values() for r in v["restore"]]
    assert min(restores) == -82 and max(restores) == 24
    outcomes = load("e3", "outcomes.json")
    assert Counter(o["status"] for o in outcomes) == {"within_margin": 28}
    no_offload = [o for o in outcomes if o["parameter_offload"] == "False" and o["role"] == "long"]
    headroom = {o["configuration_id"]: LIMIT_MIB - max(x["completed_peak_mib"] for x in no_offload
                                                         if x["configuration_id"] == o["configuration_id"])
                for o in no_offload}
    assert sorted(headroom.values()) == [36, 92]
