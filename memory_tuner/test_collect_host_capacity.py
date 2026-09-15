"""CPU-only follow-up collector controls with synthetic, parseable raw evidence."""
import json
import shutil

import pytest

from memory_tuner import collect_host_capacity as host
from memory_tuner import collect_estimation_study as core
from memory_tuner import host_capacity_protocol as design
from memory_tuner import estimation_baselines as eb
from memory_tuner.device_contract import digest
from memory_tuner.test_collect_estimation_study import (
    SOURCE, COMMIT, frozen_template, panel, make_pair, make_nonlaunch,
    put, read, journal, sync_period, explicit_accounting,
)


@pytest.fixture
def host_panel(panel):
    # Only a pytest temporary artifact is frozen here, never a real study.
    for name in design.IMPLEMENTATIONS:
        target = panel / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / name, target)
    design.freeze(panel)
    return panel


def frozen(root):
    return host.load_frozen(core.Evidence(root))


def run(root, **kwargs):
    return host.collect(root, source_commit=COMMIT, **kwargs)


def test_empty_followup_inventory_is_separate_and_all_unresolved(host_panel):
    old = core.load_frozen(core.Evidence(host_panel))
    f = frozen(host_panel)
    result = run(host_panel)
    assert len(old["rows"]) == 36
    assert (len(f["rows"]), len(f["pairs"]), len(f["targets"])) == (18, 9, 6)
    assert len(f["predictions"]) == 18
    assert (result["summary"]["planned_processes"], result["summary"]["planned_pairs"],
            result["summary"]["planned_configurations"]) == (18, 9, 6)
    assert result["summary"]["unresolved_processes"] == 18
    assert result["evaluation"]["resolved_subset_metrics"] is None
    assert all(c["planned_seeds"] == [161, 162, 163] for c in result["configurations"])
    assert all(p["pair_id"].startswith("host-") for p in result["processes"])
    provenance = result["provenance"]
    assert provenance["prediction_seal_sha256"] == design.ORIGINAL_SEAL_SHA256
    assert provenance["original_protocol_sha256"] == design.ORIGINAL_PROTOCOL_SHA256
    assert provenance["protocol_sha256"] == digest(host_panel / design.DIRECTORY / "protocol.json")
    assert provenance["study_freeze_sha256"] == digest(host_panel / design.DIRECTORY / "study_freeze.json")
    assert set(provenance["protocol_input_evidence"]) == {
        "protocol.json", "study_freeze.json", "matrix.csv", "targets.csv", "predictions.json"}
    assert set(design.IMPLEMENTATIONS) <= set(provenance["source_input_evidence"])
    assert "benchmark/estimation/prediction_freeze.json" in provenance["source_input_evidence"]
    assert provenance["study_frozen_at_utc"] != provenance["prediction_frozen_at_utc"]


def test_all_three_host_seeds_can_resolve_success_margin_and_failure_without_fitting(
        host_panel, monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("Fitting, predicting or GPU/scheduler access forbidden")
    for name in ("fit", "predict"):
        monkeypatch.setattr(eb, name, forbidden)
    for name in ("query_gpus", "allocation_context", "execute_pair", "run_payload"):
        monkeypatch.setattr(core.runner, name, forbidden)
    f = frozen(host_panel)
    before = {p: digest(p) for p in (host_panel / design.DIRECTORY).iterdir()}
    expected_states = {"gsm8k": "within_margin", "math": "above_margin",
                       "code_heavy_tail": "memory_failure"}
    for pair_id, rows in f["pairs"].items():
        state = expected_states[rows[0]["dataset"]]
        make_pair(host_panel, f, pair_id, outcomes={2: state, 4: state})
    result = run(host_panel)
    assert result["summary"]["pair_errors"] == {}
    assert result["summary"]["eligible_processes"] == 18, result["processes"]
    assert result["summary"]["completed_processes"] == 12
    assert result["summary"]["resolved_configurations"] == 6
    for config in result["configurations"]:
        assert config["eligible_seeds"] == [161, 162, 163]
        assert config["state"] == expected_states[config["settings"]["legacy_dataset_id"]]
    for config in result["configurations_flat"]:
        assert config["required_seeds"] == 3 and config["all_three_seeds_eligible"]
        for method in eb.METHODS:
            pred = next(p for p in f["predictions"] if p["method"] == method
                        and p["configuration_id"] == config["configuration_id"])
            assert config[method + "_predicted_state"] == pred["predicted_state"]
    assert all(digest(path) == value for path, value in before.items())
    output = tmp_path / "host-results"
    core.write_results(result, output)
    assert len(list(output.iterdir())) == 13
    assert len(read(output / "output_manifest.json")) == 12


@pytest.mark.parametrize("mutation", ["missing_seed", "missing_allocator", "wrong_host_memory",
                                    "wrong_cpu_request", "claim_before_study"])
def test_host_resolution_requires_every_seed_and_real_allocation_evidence(host_panel, mutation):
    f = frozen(host_panel)
    pairs = [pid for pid, rows in f["pairs"].items() if rows[0]["dataset"] == "gsm8k"]
    bundles = [make_pair(host_panel, f, pid) for pid in pairs]
    bundle = bundles[-1]
    if mutation == "missing_seed":
        shutil.rmtree(bundle["directory"])
    elif mutation == "missing_allocator":
        shutil.rmtree(bundle["attempts"][2] / "allocator-traces")
        sync_period(bundle, 2)
    elif mutation in {"wrong_host_memory", "wrong_cpu_request"}:
        path = bundle["directory"] / "pair.json"
        manifest = read(path)
        old, new = ("384G", "192G") if mutation == "wrong_host_memory" else ("CPUs/Task=64", "CPUs/Task=16")
        manifest["allocation"]["scheduler_record"] = manifest["allocation"]["scheduler_record"].replace(old, new)
        put(path, manifest)
    else:
        path = bundle["directory"] / "ledger.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()]
        events[0]["timestamp_ns"] = int((core.epoch(f["seal"]["frozen_at_utc"]) + 1) * 10**9)
        journal(path, events)
    result = run(host_panel)
    two = next(c for c in result["configurations"] if c["settings"]["legacy_dataset_id"] == "gsm8k"
               and c["settings"]["gpu_count"] == "2")
    assert two["state"] == "unresolved" and two["eligible_processes"] == 2
    assert len(result["processes"]) == 18
    if mutation == "claim_before_study":
        assert "before follow-up study freeze" in str(result["summary"]["pair_errors"])


@pytest.mark.parametrize("mutation", [
    "original_seal", "original_amendment", "original_input", "prediction_inheritance",
    "study_freeze", "execution_source", "original_execution_source", "external_symlink"])
def test_all_frozen_inputs_and_source_closure_are_verified_before_raw_read(
        host_panel, monkeypatch, tmp_path, mutation):
    old, new = host_panel / "benchmark/estimation", host_panel / design.DIRECTORY
    if mutation == "original_seal":
        path = old / "prediction_freeze.json"
    elif mutation == "original_amendment":
        path = next(old.glob("amendment-*.json"))
    elif mutation == "original_input":
        path = host_panel / next(iter(read(old / "protocol.json")["input_sha256"]))
    elif mutation == "execution_source":
        path = host_panel / "memory_tuner/run_host_capacity.py"
    elif mutation == "original_execution_source":
        path = host_panel / "memory_tuner/run_matched_gpu.py"
    elif mutation == "external_symlink":
        path = new / "targets.csv"
        external = tmp_path / "external-targets.csv"
        shutil.copyfile(path, external)
        path.unlink()
        path.symlink_to(external)
    elif mutation == "prediction_inheritance":
        path = new / "predictions.json"
        predictions = read(path)
        predictions[0]["predicted_state"] = "above_margin"
        put(path, predictions)
        seal = read(new / "study_freeze.json")
        seal["files_sha256"]["predictions.json"] = digest(path)
        put(new / "study_freeze.json", seal)
    else:
        path = new / "study_freeze.json"
    if mutation not in {"external_symlink", "prediction_inheritance"}:
        path.write_text(path.read_text() + "\nchanged")
    def forbidden(*args, **kwargs):
        raise AssertionError("Raw target read before complete frozen validation")
    monkeypatch.setattr(core, "validate_pair", forbidden)
    with pytest.raises((ValueError, json.JSONDecodeError)):
        run(host_panel)


def test_nonlaunch_and_accounting_remain_in_separate_host_denominator(host_panel):
    f = frozen(host_panel)
    bundle = make_nonlaunch(host_panel, f, next(iter(f["pairs"])))
    result = run(host_panel, accounting=[explicit_accounting(
        bundle, duration=9, batch_duration=9, state="FAILED")])
    assert result["costs"]["whole_allocation_time"]["known_gpu_seconds"] == 36
    assert result["costs"]["whole_allocation_time"]["unknown_records"] == 8
    assert result["summary"]["unresolved_processes"] == 18
    assert result["summary"]["known_launched_processes"] == 0


def test_portable_host_copy_has_identical_output_and_ignores_original_outcomes(host_panel, tmp_path):
    f = frozen(host_panel)
    make_pair(host_panel, f, next(iter(f["pairs"])))
    original = host_panel / "output/prospective_estimation/pairs/not-a-followup"
    original.mkdir(parents=True)
    (original / "pair.json").write_text("MUST NOT BE READ")
    result = run(host_panel)
    copied = tmp_path / "relocated"
    shutil.copytree(host_panel, copied)
    assert run(copied) == result


def test_followup_external_anchors_and_commit_map_are_not_original_defaults(host_panel):
    with pytest.raises(ValueError, match="anchor changed"):
        run(host_panel, protocol_sha256=design.ORIGINAL_PROTOCOL_SHA256)
    with pytest.raises(ValueError):
        run(host_panel, prediction_seal_sha256="0" * 64)
    old = core.load_frozen(core.Evidence(host_panel))
    with pytest.raises(ValueError, match="9 planned pairs"):
        host.collect(host_panel, execution_commits={pid: COMMIT for pid in old["pairs"]})


def test_followup_freeze_cannot_predate_original_predictor_even_with_rehashed_files(host_panel):
    directory = host_panel / design.DIRECTORY
    protocol = read(directory / "protocol.json")
    freeze = read(directory / "study_freeze.json")
    protocol["specified_at_utc"] = "2026-01-01T00:00:00+00:00"
    freeze["frozen_at_utc"] = "2026-01-01T00:00:01+00:00"
    put(directory / "protocol.json", protocol)
    freeze["protocol_sha256"] = digest(directory / "protocol.json")
    put(directory / "study_freeze.json", freeze)
    with pytest.raises(ValueError, match="predates original prediction seal"):
        run(host_panel)


def test_implicit_followup_directory_symlink_cannot_escape_artifact(host_panel, tmp_path):
    directory = host_panel / design.DIRECTORY
    external = tmp_path / "external-protocol"
    directory.rename(external)
    directory.symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="escape"):
        run(host_panel)
    with pytest.raises(ValueError, match="escape"):
        frozen(host_panel)
