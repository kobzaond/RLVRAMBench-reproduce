"""CPU-only adapter regressions; no scheduler, CUDA, model, or target-data runs."""
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from memory_tuner import device_contract as devices
from memory_tuner import run_host_capacity as host
from memory_tuner import run_matched_gpu as runner
from memory_tuner import test_matched_gpu as support
from memory_tuner.test_matched_gpu import frozen_run


SOURCE = Path(__file__).resolve().parents[1]


@pytest.fixture
def host_run(frozen_run, monkeypatch):
    """Synthetic panel; the parent-owned protocol's scientific checks are separate."""
    args = frozen_run
    with args.matrix.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["configuration_level"] == "c2"]
    for row in rows:
        row["training_seed"] = str(int(row["training_seed"]) + 10)
        row["pair_id"] = f"host-qwen25-7b-{row['dataset']}-c2-s{row['training_seed']}"
        row["experiment_id"] = row["pair_id"] + f"-{row['gpu_count']}gpu"
        row["run_group"] = host.GROUP
    with args.matrix.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.matrix_sha256 = devices.digest(args.matrix)
    calls = []
    protocol = ModuleType("memory_tuner.host_capacity_protocol")
    def validate_matrix(selected):
        calls.append("matrix")
        assert len(selected) == 18
        pairs = {}
        for row in selected:
            assert row["run_group"] == host.GROUP
            assert row["configuration_level"] == "c2"
            assert int(row["training_seed"]) in (161, 162, 163)
            pairs.setdefault(row["pair_id"], []).append(row)
        assert len(pairs) == 9
        return pairs  # Preserve the original row objects used by the core.
    def verify_protocol(source, received_args, selected):
        calls.append("protocol")
        assert source == SOURCE and received_args is args and len(selected) == 18
        # Reuse only the fixture's synthetic data hashes. No predictions are fit.
        value = json.loads(args.protocol.read_text())
        runner.verify_data(selected, value)
        return value
    original_allocation = runner.allocation_context
    def allocation():
        value = original_allocation()
        value["scheduler_record"] = (
            "JobId=123 NumCPUs=64 CPUs/Task=64 "
            "AllocTRES=cpu=64,mem=384G,node=1,gres/gpu=4")
        return value
    def validate_allocation(value):
        calls.append("allocation")
        assert "CPUs/Task=64" in value["scheduler_record"]
        assert "mem=384G" in value["scheduler_record"]
    protocol.validate_matrix = validate_matrix
    protocol.verify_protocol = verify_protocol
    protocol.validate_allocation = validate_allocation
    monkeypatch.setitem(sys.modules, protocol.__name__, protocol)
    monkeypatch.setattr(runner, "allocation_context", allocation)
    return SimpleNamespace(args=args, rows=rows, calls=calls, protocol=protocol)


def test_adapter_passes_parent_callbacks_without_mutating_original_globals(monkeypatch):
    protocol = ModuleType("memory_tuner.host_capacity_protocol")
    for name in ("validate_matrix", "verify_protocol", "validate_allocation"):
        setattr(protocol, name, object())
    monkeypatch.setitem(sys.modules, protocol.__name__, protocol)
    before = (runner.GROUP, runner.validate_matrix, runner.verify_protocol, runner.assess_period)
    args = SimpleNamespace()
    def execute(received, **kwargs):
        assert received is args
        assert kwargs == dict(group="estimation_host_capacity",
                              matrix_validator=protocol.validate_matrix,
                              protocol_verifier=protocol.verify_protocol,
                              allocation_validator=protocol.validate_allocation)
        return 2
    monkeypatch.setattr(runner, "execute_pair", execute)
    assert host.execute_pair(args) == 2
    assert before == (runner.GROUP, runner.validate_matrix, runner.verify_protocol, runner.assess_period)


def test_real_parent_api_preserves_row_identity_and_is_passed_to_core(monkeypatch):
    from memory_tuner import host_capacity_protocol as protocol

    rows, targets, predictions, _ = protocol.planned(SOURCE)
    pairs = protocol.validate_matrix(rows)
    assert (len(rows), len(pairs), len(targets), len(predictions)) == (18, 9, 6, 18)
    assert all(any(row is candidate for candidate in rows)
               for selected in pairs.values() for row in selected)
    # There must be no inherited original protocol path/hash conflicting with
    # the follow-up protocol that execute_pair checks before dispatch.
    assert all(not row.get("protocol_path") and not row.get("protocol_sha256") for row in rows)
    def execute(args, **kwargs):
        assert kwargs["matrix_validator"] is protocol.validate_matrix
        assert kwargs["protocol_verifier"] is protocol.verify_protocol
        assert kwargs["allocation_validator"] is protocol.validate_allocation
        kwargs["allocation_validator"]({"scheduler_record":
            "JobId=123 NumCPUs=64 CPUs/Task=64 AllocTRES=cpu=64,mem=384G,gres/gpu=4"})
        return 0
    monkeypatch.setattr(runner, "execute_pair", execute)
    assert host.execute_pair(SimpleNamespace()) == 0


def test_host_pair_uses_separate_paths_and_unchanged_payload_controls(host_run, monkeypatch):
    from memory_tuner.grpo_raw_evidence import MATCH_FIELDS

    args = host_run.args
    selected = host_run.rows[:2]
    calls, idle_calls = [], []
    original_group = runner.GROUP
    original = args.project_root / "output" / original_group
    original.mkdir(parents=True)
    marker = original / "preserved-original"
    marker.write_text("immutable previous observations")
    def idle(uuids, trace, period, phase, deadline):
        assert uuids == support.UUIDS
        idle_calls.append((period, phase))
        return True
    def payload(command, env, output, allocated, expected, journal, trace, period, deadline):
        row = selected[period - 1]
        calls.append(period)
        assert command == ["bash", str(SOURCE / "run_grpo_instrumented.slurm")]
        assert allocated == support.UUIDS and len(expected) == int(row["gpu_count"])
        assert env["RUN_NAME"] == f"{host.GROUP}/{row['experiment_id']}/attempt-123"
        assert output.parent == args.project_root / "output" / host.GROUP / row["experiment_id"] / "attempt-123"
        assert env["RLVRAM_GPU_UUIDS"] == ",".join(expected)
        assert env["RLVRAM_ALLOCATION_GPU_UUIDS"] == ",".join(allocated)
        assert env["RLVRAM_INVOCATION_ID"] == row["experiment_id"] + "-j123"
        for name, value in {"TOTAL_TRAINING_STEPS": "1", "RESUME_MODE": "disable",
                            "SAVE_FREQ": "-1", "TEST_FREQ": "-1", "VAL_BEFORE_TRAIN": "False",
                            "TRAINING_SEED": "161"}.items():
            assert env[name] == value
        support.period_evidence(output.parent, expected, env["RLVRAM_INVOCATION_ID"],
                                success=True, workers=len(expected))
        trial_path = output.parent / "trial-123.json"
        trial = json.loads(trial_path.read_text())
        trial.update({field: row[field] for field in MATCH_FIELDS})
        trial.update(gpu_count=len(expected), training_seed=int(row["training_seed"]))
        trial_path.write_text(json.dumps(trial))
        return support.execution(True)
    monkeypatch.setattr(runner, "wait_idle", idle)
    monkeypatch.setattr(runner, "run_payload", payload)
    assert host.execute_pair(args) == 0
    assert host_run.calls == ["matrix", "protocol", "allocation"]
    assert calls == [1, 2]
    assert idle_calls == [(1, "pre_idle"), (1, "post_idle"), (2, "pre_idle"), (2, "post_idle")]
    pair = args.project_root / "output" / host.GROUP / "pairs" / selected[0]["pair_id"]
    summary = json.loads((pair / "pair-summary.json").read_text())
    assert summary["validated_outcomes"] == summary["launched_invocations"] == 2
    assert summary["task_gpu_seconds"] == 18
    assert summary["allocated_gpu_seconds_during_invocations"] == 24
    assert runner.GROUP == original_group and marker.read_text() == "immutable previous observations"
    assert list(original.iterdir()) == [marker]
    with pytest.raises(FileExistsError):
        host.execute_pair(args)
    assert calls == [1, 2]


@pytest.mark.parametrize("bad_source", ["wrong_head", "dirty"])
def test_host_still_requires_clean_current_implementation_head(host_run, monkeypatch, bad_source):
    def output(command, **kwargs):
        if "status" in command:
            return " M changed.py\n" if bad_source == "dirty" else ""
        assert command[:3] == ["git", "-C", str(SOURCE)]
        return ("0" * 40 if bad_source == "wrong_head" else support.HEAD) + "\n"
    monkeypatch.setattr(runner.subprocess, "check_output", output)
    monkeypatch.setattr(runner, "allocation_context", lambda: pytest.fail("must not query devices"))
    monkeypatch.setattr(runner, "run_payload", lambda *args: pytest.fail("must not launch"))
    assert host.execute_pair(host_run.args) == 2
    assert host_run.calls == ["matrix"]
    pair = host_run.args.project_root / "output" / host.GROUP / "pairs" / host_run.rows[0]["pair_id"]
    for period in (1, 2):
        record = json.loads((pair / f"period-{period}.json").read_text())
        assert not record["launched"] and "clean, frozen source commit" in record["reason"]


@pytest.mark.parametrize("gate", ["verify_protocol", "validate_allocation"])
def test_host_gate_failure_preserves_both_slots_and_never_retries(host_run, monkeypatch, gate):
    def reject(*args):
        raise ValueError(f"{gate} rejected frozen host condition")
    setattr(host_run.protocol, gate, reject)
    monkeypatch.setattr(runner, "run_payload", lambda *args: pytest.fail("must not launch"))
    assert host.execute_pair(host_run.args) == 2
    pair = host_run.args.project_root / "output" / host.GROUP / "pairs" / host_run.rows[0]["pair_id"]
    for period in (1, 2):
        record = json.loads((pair / f"period-{period}.json").read_text())
        assert record["status"] == "not_launched" and record["launched"] is False
        assert gate in record["reason"]
    events = [json.loads(line) for line in (pair / "ledger.jsonl").read_text().splitlines()]
    assert [event["period"] for event in events if event["event"] == "terminal"] == [1, 2]
    assert not any(event["event"] == "dispatched" for event in events)
    with pytest.raises(FileExistsError):
        host.execute_pair(host_run.args)


def test_host_validator_accepts_last_of_nine_pairs_and_rejects_tenth(host_run, monkeypatch):
    args = host_run.args
    args.index = 8
    monkeypatch.setattr(runner, "wait_idle", lambda *args: False)
    monkeypatch.setattr(runner, "run_payload", lambda *args: pytest.fail("must not launch"))
    assert host.execute_pair(args) == 2
    pair = args.project_root / "output" / host.GROUP / "pairs" / host_run.rows[-1]["pair_id"]
    assert (pair / "pair-summary.json").is_file()
    args.index = 9
    with pytest.raises(ValueError, match=r"outside 0\.\.8"):
        host.execute_pair(args)


def cli_args():
    return ["--matrix", "/frozen/matrix.csv", "--matrix-sha256", "a" * 64,
            "--protocol", "/frozen/protocol.json", "--protocol-sha256", "b" * 64,
            "--source-commit", "c" * 40, "--index", "8", "--project-root", "/artifact"]


def test_host_cli_passes_frozen_arguments_and_preserves_signal_handling(monkeypatch):
    handlers = {}
    monkeypatch.setattr(host.signal, "signal", lambda sig, fn: handlers.setdefault(sig, fn))
    def execute(args):
        assert args.matrix == Path("/frozen/matrix.csv") and args.matrix_sha256 == "a" * 64
        assert args.protocol == Path("/frozen/protocol.json") and args.protocol_sha256 == "b" * 64
        assert args.source_commit == "c" * 40 and args.index == 8 and args.project_root == Path("/artifact")
        return 2
    monkeypatch.setattr(host, "execute_pair", execute)
    assert host.main(cli_args()) == 2
    assert set(handlers) == {signal.SIGTERM, signal.SIGINT}
    with pytest.raises(InterruptedError, match="allocation signal"):
        handlers[signal.SIGTERM](signal.SIGTERM, None)


@pytest.mark.parametrize("omit", range(0, 14, 2))
def test_host_cli_requires_each_frozen_argument(monkeypatch, omit):
    args = cli_args()
    del args[omit:omit + 2]
    monkeypatch.setattr(host, "execute_pair", lambda *args: pytest.fail("must not execute"))
    with pytest.raises(SystemExit) as error:
        host.main(args)
    assert error.value.code == 2


def test_host_wrapper_resources_and_exact_argument_handoff(tmp_path):
    wrapper = SOURCE / "run_host_capacity.slurm"
    text = wrapper.read_text()
    for flag in ("--account=OPEN-35-44", "--partition=qgpu_exp", "--nodes=1", "--ntasks=1",
                 "--gpus=4", "--cpus-per-task=64", "--mem=384G", "--time=00:55:00",
                 "--array=0-8%3", "--no-requeue"):
        assert f"#SBATCH {flag}" in text
    assert "logs/host-capacity-%A_%a.out" in text
    assert "logs/host-capacity-%A_%a.err" in text
    assert "set -euo pipefail" in text
    assert "sbatch " not in text and "scancel " not in text
    subprocess.run(["bash", "-n", str(wrapper)], check=True)
    source = tmp_path / "frozen source"
    source.mkdir()
    project = tmp_path / "output root"
    executable = project / ".venv-analysis/bin/python"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
    executable.chmod(0o700)
    env = dict(os.environ, REVISION_SOURCE_ROOT=str(source), REVISION_PROJECT_ROOT=str(project),
               MATRIX_PATH=str(source / "matrix.csv"), MATRIX_SHA256="a" * 64,
               PROTOCOL_PATH=str(source / "protocol.json"), PROTOCOL_SHA256="b" * 64,
               SOURCE_COMMIT="c" * 40, SLURM_ARRAY_TASK_ID="8")
    # Bash executes a synthetic argument-printing program, not Python or Slurm.
    result = subprocess.run(["bash", str(wrapper)], env=env, text=True,
                            capture_output=True, check=True, timeout=10)
    assert result.stdout.splitlines() == [
        "-m", "memory_tuner.run_host_capacity", "--matrix", env["MATRIX_PATH"],
        "--matrix-sha256", "a" * 64, "--protocol", env["PROTOCOL_PATH"],
        "--protocol-sha256", "b" * 64, "--source-commit", "c" * 40,
        "--index", "8", "--project-root", str(project)]
