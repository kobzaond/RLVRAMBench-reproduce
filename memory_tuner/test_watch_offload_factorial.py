"""CPU-only watcher tests; no real scheduler calls or GPU execution."""
from collections import Counter
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from memory_tuner import capture_allocation_accounting as accounting
from memory_tuner import watch_offload_factorial as watch

COMMAND = watch.command


@pytest.fixture
def scope(tmp_path, monkeypatch):
    project, source = tmp_path / "project", tmp_path / "source"
    project.mkdir()
    source.mkdir()
    monkeypatch.setattr(watch, "PROJECT", project)
    monkeypatch.setattr(watch, "SOURCE", source)
    monkeypatch.setattr(watch, "STATE", project / "output/watch-5069460")
    monkeypatch.setattr(watch, "command", Mock(side_effect=AssertionError("Unexpected external command")))
    monkeypatch.setattr(watch.subprocess, "Popen", Mock(side_effect=AssertionError("Unexpected subprocess")))
    monkeypatch.setattr(watch.subprocess, "run", Mock(side_effect=AssertionError("Unexpected subprocess")))
    return SimpleNamespace(project=project, source=source, state=watch.STATE)


def json_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def sacct(expected=None, state="COMPLETED", *, steps=True):
    lines = []
    for index, display in enumerate(expected or watch.targets()):
        job = str(int(watch.FIRST) + index)
        for step in ("", ".batch", ".extern") if steps else ("",):
            values = [display + step, job + step, step[1:] or "rlvram-offload-factorial",
                      state, "0:0", "2026-09-17T12:00:00", "2026-09-17T12:01:40",
                      "100", "cpu=64,mem=384G,node=1,gres/gpu=4", "acn-test", "None"]
            lines.append("|".join(values))
    return "\n".join(lines) + "\n"


def raw_block(root, index=0):
    block = root / watch.RAW / "blocks" / f"block-{index}"
    summary = dict(block_id=f"block-{index}", pair_id=f"block-{index}",
                   planned_invocations=8, launched_invocations=8)
    json_file(block / "block-summary.json", summary)
    json_file(block / "block.json", {"job_id": str(int(watch.FIRST) + index),
                                    "block_id": f"block-{index}"})
    (block / "ledger.jsonl").write_text(
        json.dumps(dict(event="block_terminal", sequence=25, timestamp_ns=1,
                        monotonic_ns=2, **summary)) + "\n")
    worker = root / watch.RAW / f"experiment-{index}-1/attempt-{int(watch.FIRST) + index}/worker.log"
    worker.parent.mkdir(parents=True, exist_ok=True)
    worker.write_bytes(b"complete raw worker retention\n")
    return block


def outputs(stage, expected=None, outcome="completed"):
    expected = expected or watch.targets()
    result = stage / "results"
    result.mkdir(parents=True)
    slots = []
    for index in range(18):
        for period in range(1, 9):
            claimed = index in expected.values()
            slots.append(dict(
                experiment_id=f"experiment-{index}-{period}", block_id=f"block-{index}",
                block_index=str(index), period=str(period), eligible=claimed,
                status=outcome if claimed else "not_launched",
                validation_errors="[]", cost_validation_errors="[]",
                task_gpu_seconds=40. if claimed else "",
                # Intentionally useless predictions: the gate must never consult them.
                prediction="wrong", label_composition="all_one_label"))
    with (result / "slots.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=slots[0])
        writer.writeheader()
        writer.writerows(slots)
    records = [dict(scheduler_job_id=job, pair_id=f"block-{i}") for job, i in expected.items()]
    json_file(stage / "accounting.json", records)
    json_file(result / "summary.json", dict(
        freeze_sha256=watch.FREEZE, execution_source_commit=watch.COMMIT,
        independent_raw_reconstruction=True, source_and_predictions_verified=True,
        raw_root=str(stage / "raw"), claimed_blocks=len(expected), block_validation_errors=[],
        allocation_accounting=dict(allocations=len(expected), whole_allocation_gpu_seconds=400 * len(expected)),
        statuses=dict(Counter(r["status"] for r in slots))))
    for name in watch.RESULTS - {"slots.csv", "summary.json"}:
        (result / name).write_text("schema\n")
    return stage


def edit_slots(stage, update):
    path = stage / "results/slots.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    update(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def gate_fixture(scope, outcome="completed"):
    stage = scope.state / "first"
    raw_block(stage / "raw")
    outputs(stage, outcome=outcome)
    return stage


@pytest.mark.parametrize("state", sorted(accounting.TERMINAL_STATES))
def test_all_terminal_states_trigger_reconstruction_not_assumed_outcome(scope, state):
    assert watch.terminal_status(sacct(state=state), watch.targets()) == (True, {state: 1})


@pytest.mark.parametrize("text", ["", "pending", "partial", "batch_running"])
def test_missing_pending_and_unfinished_steps_wait(scope, text):
    expected = watch.targets({"array_job_id": "7000000"})
    data = {"": "", "pending": sacct(expected, "PENDING"),
            "partial": sacct(), "batch_running": sacct(expected).replace("batch|COMPLETED", "batch|RUNNING")}[text]
    assert not watch.terminal_status(data, expected)[0]


@pytest.mark.parametrize("damage", ["extra", "duplicate", "bad_fields", "unknown_step", "bad_raw"])
def test_invalid_scheduler_identity_stops(scope, damage):
    data = sacct()
    if damage == "extra":
        data += sacct({"9000000_0": 0})
    elif damage == "duplicate":
        data += data
    elif damage == "bad_fields":
        data = "broken\n"
    elif damage == "unknown_step":
        data = data.replace(".extern", ".123")
    else:
        data = data.replace(f"|{watch.FIRST}|", "|nonnumeric|")
    with pytest.raises(ValueError):
        watch.terminal_status(data, watch.targets())


def test_query_is_local_expanded_and_utc_with_start_date(scope, monkeypatch):
    calls = []
    monkeypatch.setattr(watch, "command", lambda args: calls.append(args) or "")
    watch.query(watch.targets())
    args = calls[0]
    assert args[0] == "sacct" and args[args.index("-j") + 1] == watch.FIRST
    assert "--array" in args and "--local" in args and "-P" in args
    assert "--starttime=2026-09-17T00:00:00" in args
    assert "JobIDRaw%64" in args[-1] and "AllocTRES%512" in args[-1]
    monkeypatch.setenv("SBATCH_ARRAY_INX", "0-100")
    env = watch.environment()
    assert env["TZ"] == "UTC" and env["SLURM_TIME_FORMAT"] == "standard"
    assert all(env[f"{name}_NUM_THREADS"] == "1" for name in ("OMP", "OPENBLAS", "MKL"))
    assert "SBATCH_ARRAY_INX" not in env and env["REVISION_SOURCE_ROOT"] == str(scope.source)


@pytest.mark.parametrize("outcome", ["completed", "memory_failure"])
def test_gate_accepts_any_label_composition_and_ignores_predictions(scope, outcome):
    stage = gate_fixture(scope, outcome)
    watch.check_results(stage, watch.targets(), gate=True)


def test_gate_accepts_mixed_labels(scope):
    stage = gate_fixture(scope)
    edit_slots(stage, lambda rows: rows[0].update(status="memory_failure"))
    watch.check_results(stage, watch.targets(), gate=True)


@pytest.mark.parametrize("change", [
    {"eligible": "False"}, {"status": "timeout"}, {"status": "not_launched"},
    {"status": "infrastructure_invalid"}, {"validation_errors": '["wrong UUID"]'},
    {"validation_errors": '["incomplete cleanup"]'}, {"cost_validation_errors": '["bad timing"]'},
    {"task_gpu_seconds": ""}, {"task_gpu_seconds": "nan"}, {"task_gpu_seconds": "0"},
])
def test_gate_rejects_partial_invalid_or_unknown_cost(scope, change):
    stage = gate_fixture(scope)
    edit_slots(stage, lambda rows: rows[7].update(change))
    with pytest.raises(ValueError, match="execution gate failed"):
        watch.check_results(stage, watch.targets(), gate=True)
    assert (stage / "results/slots.csv").exists()


@pytest.mark.parametrize("change", [
    {"freeze_sha256": "bad"}, {"execution_source_commit": "bad"},
    {"independent_raw_reconstruction": False}, {"source_and_predictions_verified": False},
    {"raw_root": "/wrong"}, {"claimed_blocks": 2}, {"block_validation_errors": ["bad source"]},
    {"allocation_accounting": {"status": "not_supplied"}},
])
def test_gate_requires_provenance_wrapper_and_accounting(scope, change):
    stage = gate_fixture(scope)
    path = stage / "results/summary.json"
    json_file(path, dict(watch.read(path), **change))
    with pytest.raises(ValueError):
        watch.check_results(stage, watch.targets(), gate=True)


@pytest.mark.parametrize("damage", ["missing_slot", "duplicate_period", "wrong_index", "no_terminal",
                                   "wrong_accounting_job", "wrong_accounting_block"])
def test_gate_requires_exact_slots_and_claim_bound_allocation(scope, damage):
    stage = gate_fixture(scope)
    if damage == "missing_slot":
        edit_slots(stage, lambda rows: rows.pop())
    elif damage == "duplicate_period":
        edit_slots(stage, lambda rows: rows[7].update(period="1"))
    elif damage == "wrong_index":
        edit_slots(stage, lambda rows: rows[7].update(block_index="18"))
    elif damage == "no_terminal":
        path = stage / "raw" / watch.RAW / "blocks/block-0/ledger.jsonl"
        path.write_text(path.read_text().replace("block_terminal", "terminal"))
    else:
        item = watch.read(stage / "accounting.json")[0]
        item["scheduler_job_id" if damage == "wrong_accounting_job" else "pair_id"] = "wrong"
        json_file(stage / "accounting.json", [item])
    with pytest.raises(ValueError):
        watch.check_results(stage, watch.targets(), gate=True)


def test_final_retains_invalid_slots_without_outcome_filtering(scope):
    stage = scope.state / "final"
    expected = watch.targets({"array_job_id": "7000000"})
    outputs(stage, expected, "infrastructure_invalid")
    watch.check_results(stage, expected, gate=False)


def test_existing_helper_computes_actual_utc_cost_without_fabrication(scope):
    claims = {watch.FIRST: dict(pair_id="block-0", pair_manifest_sha256="a" * 64,
                               pair_ledger_sha256="b" * 64)}
    records = accounting.capture(sacct(), claims)
    record = accounting.validate_record(records[0])
    assert record["ended_at_epoch"] - record["started_at_epoch"] == 100
    assert record["allocated_gpu_count"] == 4
    assert records[0]["raw_record_provenance"]["timezone"] == "UTC"
    with pytest.raises(ValueError):
        accounting.capture(sacct().replace("|100|", "|99|"), claims)
    assert "c.accounting.capture" in watch.COLLECT and "c.block_claims(raw)" in watch.COLLECT


def test_reconstruction_relocates_full_raw_and_retains_exact_six_outputs(scope, monkeypatch):
    scope.state.mkdir(parents=True)
    raw_block(scope.project)
    before = watch.inventory(scope.project / watch.RAW)
    monkeypatch.setattr(watch, "collect_child", lambda stage: outputs(stage))
    stage = watch.reconstruct("first", sacct())
    assert before == watch.inventory(scope.project / watch.RAW)
    assert before == watch.inventory(stage / "raw" / watch.RAW)
    assert (stage / "sacct.txt").read_text() == sacct()
    assert set(p.name for p in (stage / "results").iterdir()) == watch.RESULTS
    assert watch.read(stage / "snapshot-receipt.json")["pid"] == os.getpid()
    assert watch.reconstruct("first", "not re-queried accounting") == stage
    (stage / "raw" / watch.RAW / "experiment-0-1/attempt-5069460/worker.log").write_text("tampered")
    with pytest.raises(ValueError, match="Changed retained"):
        watch.reconstruct("first", sacct())


@pytest.mark.parametrize("problem", ["partial", "linked", "raw_changes", "child_failure"])
def test_relocation_failure_retained_without_repair(scope, monkeypatch, problem):
    scope.state.mkdir(parents=True)
    raw_block(scope.project)
    monkeypatch.setattr(watch, "collect_child", lambda stage: outputs(stage))
    if problem == "partial":
        (scope.state / "first").mkdir()
    elif problem == "linked":
        (scope.project / watch.RAW / "linked").symlink_to(scope.project / watch.RAW / "blocks", target_is_directory=True)
    elif problem == "raw_changes":
        original = watch.shutil.copytree

        def changed(*args, **kwargs):
            value = original(*args, **kwargs)
            (scope.project / watch.RAW / "new-file").write_text("race")
            return value

        monkeypatch.setattr(watch.shutil, "copytree", changed)
    else:
        monkeypatch.setattr(watch, "collect_child", Mock(side_effect=ValueError("collector failure")))
    with pytest.raises(ValueError):
        watch.reconstruct("first", sacct())
    assert (scope.state / "first").is_dir()
    assert not (scope.state / "first/snapshot-receipt.json").exists()


def test_missing_raw_still_reaches_collector_for_all_planned_slots(scope, monkeypatch):
    scope.state.mkdir(parents=True)
    called = []

    def collect(stage):
        called.append(stage)
        assert (stage / "raw" / watch.RAW).is_dir()
        outputs(stage)
        raise ValueError("accounting missing; six outputs retained")

    monkeypatch.setattr(watch, "collect_child", collect)
    with pytest.raises(ValueError, match="accounting missing"):
        watch.reconstruct("first", sacct())
    assert len(called) == 1 and (scope.state / "first/results/slots.csv").exists()


def prepare_release(scope):
    raw_block(scope.project)
    stage = gate_fixture(scope)
    json_file(stage / "raw-inventory.json", watch.inventory(scope.project / watch.RAW))
    watch.save(stage / "snapshot-receipt.json", files={})
    return stage


def test_release_has_durable_intent_before_single_exact_sbatch(scope, monkeypatch):
    stage = prepare_release(scope)
    monkeypatch.setattr(watch, "verify_source", lambda: None)
    calls = []

    def submit(args):
        intent = watch.bound_record(scope.state / "submit-intent.json")
        assert intent["command"] == args
        assert intent["gate_receipt_sha256"] == watch.digest(stage / "snapshot-receipt.json")
        assert intent["pid"] == os.getpid()
        calls.append(args)
        return "7000000\n"

    monkeypatch.setattr(watch, "command", submit)
    watch.release(stage)
    args = calls[0]
    assert args == ["sbatch", "--parsable", "--array=1-17%3", "--account=OPEN-35-44",
                    "--partition=qgpu", "--nodes=1", "--ntasks=1", "--gpus=4",
                    "--cpus-per-task=64", "--mem=384G", "--time=03:00:00",
                    "--no-requeue", "--export=ALL",
                    str(scope.source / "memory_tuner/run_offload_factorial.slurm")]
    assert watch.bound_record(scope.state / "submit-receipt.json")["indices"] == "1-17%3"
    with pytest.raises(ValueError, match="NEVER retry"):
        watch.release(stage)
    assert len(calls) == 1


@pytest.mark.parametrize("reply", ["", "Submitted batch job 7000000\n", "7000000;foreign\n",
                                 "7000000\n7000001\n", watch.FIRST + "\n", "0\n", " 7000000\n",
                                 subprocess.TimeoutExpired(["sbatch"], 30),
                                 subprocess.CalledProcessError(1, ["sbatch"], output="7000000\n"),
                                 OSError("submission transport failed")])
def test_ambiguous_submission_is_durable_and_never_retried(scope, monkeypatch, reply):
    stage = prepare_release(scope)
    monkeypatch.setattr(watch, "verify_source", lambda: None)
    submit = Mock(side_effect=reply) if isinstance(reply, Exception) else Mock(return_value=reply)
    monkeypatch.setattr(watch, "command", submit)
    with pytest.raises(ValueError, match="Ambiguous"):
        watch.release(stage)
    assert (scope.state / "submit-intent.json").exists()
    assert (scope.state / "submit-response.json").exists()
    assert not (scope.state / "submit-receipt.json").exists()
    with pytest.raises(ValueError, match="intent without receipt"):
        watch.workflow(30)
    assert submit.call_count == 1


def test_changed_claims_prevent_release(scope, monkeypatch):
    stage = prepare_release(scope)
    monkeypatch.setattr(watch, "verify_source", lambda: None)
    raw_block(scope.project, 1)
    with pytest.raises(ValueError, match="changed after"):
        watch.release(stage)
    assert not (scope.state / "submit-intent.json").exists()


@pytest.mark.parametrize("args", [[], ["--once"]])
def test_diagnostic_is_nonmutating(scope, monkeypatch, args):
    monkeypatch.setattr(watch, "query", lambda expected: sacct(state="PENDING"))
    before = watch.inventory(scope.project)
    assert watch.main(args) == 0
    assert watch.inventory(scope.project) == before
    assert not scope.state.exists()


def test_execute_retains_actionable_stop_and_never_auto_resumes(scope, monkeypatch):
    stage = gate_fixture(scope)
    edit_slots(stage, lambda rows: rows[6].update(eligible="False", status="not_launched"))
    monkeypatch.setattr(watch, "query", lambda expected: sacct())
    monkeypatch.setattr(watch, "verify_source", lambda: None)
    monkeypatch.setattr(watch, "reconstruct", lambda name, text: stage)
    assert watch.main(["--execute"]) == 2
    assert "gate failed" in watch.bound_record(scope.state / "STOP.json")["error"]
    assert not (scope.state / "submit-intent.json").exists()
    assert watch.main(["--execute"]) == 2


def test_workflow_waits_on_readonly_errors_and_submits_once_then_reconstructs_144(scope, monkeypatch):
    scope.state.mkdir(parents=True)
    monkeypatch.setattr(watch, "verify_source", lambda: None)
    calls, sleeps = [], []
    queries = iter([subprocess.TimeoutExpired(["sacct"], 30), "PENDING", "PENDING",
                    "COMPLETED", "RUNNING", "COMPLETED"])
    stages = []

    def query(expected):
        calls.append(expected)
        value = next(queries)
        if isinstance(value, Exception):
            raise value
        return sacct(expected, value)

    def reconstruction(name, text):
        stage = scope.state / name
        expected = watch.targets() if name == "first" else watch.targets({"array_job_id": "7000000"})
        if name == "first":
            raw_block(stage / "raw")
        outputs(stage, expected)
        stages.append(name)
        return stage

    submissions = []

    def release(stage):
        submissions.append(stage)
        watch.save(scope.state / "submit-intent.json")
        watch.save(scope.state / "submit-receipt.json", array_job_id="7000000", indices="1-17%3")

    monkeypatch.setattr(watch, "query", query)
    monkeypatch.setattr(watch, "reconstruct", reconstruction)
    monkeypatch.setattr(watch, "release", release)
    monkeypatch.setattr(watch.time, "sleep", sleeps.append)
    watch.workflow(30)
    assert sleeps == [30] * 4
    assert stages == ["first", "final"] and len(submissions) == 1
    assert all(expected == watch.targets() for expected in calls[:4])
    assert all(len(expected) == 18 for expected in calls[4:])
    assert (scope.state / "complete.json").exists()
    watch.workflow(30)
    assert len(submissions) == 1 and len(calls) == 6
    events = [json.loads(line) for line in (scope.state / "events.jsonl").read_text().splitlines()]
    assert len([r for r in events if r["event"] == "scheduler_status"]) == 5  # Repeated pending is sparse.


def test_restart_with_receipt_monitors_without_submission(scope, monkeypatch):
    scope.state.mkdir(parents=True)
    watch.save(scope.state / "submit-intent.json")
    watch.save(scope.state / "submit-receipt.json", array_job_id="7000000", indices="1-17%3")
    expected = watch.targets({"array_job_id": "7000000"})
    stage = outputs(scope.state / "final", expected)
    monkeypatch.setattr(watch, "query", lambda actual: sacct(actual))
    monkeypatch.setattr(watch, "verify_source", lambda: None)
    monkeypatch.setattr(watch, "reconstruct", lambda name, text: stage)
    monkeypatch.setattr(watch, "release", Mock(side_effect=AssertionError("Must not resubmit")))
    watch.workflow(30)
    assert (scope.state / "complete.json").exists()


def test_single_watcher_lock(scope, monkeypatch):
    scope.state.mkdir(parents=True)
    with (scope.state / "watch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            watch.main(["--execute"])
    assert not (scope.state / "events.jsonl").exists()


@pytest.mark.parametrize("change", ["commit", "dirty", "freeze"])
def test_verify_source_rejects_wrong_frozen_identity(scope, monkeypatch, change):
    responses = iter(["a" * 40 if change == "commit" else watch.COMMIT,
                      " M changed" if change == "dirty" else ""])
    monkeypatch.setattr(watch, "command", lambda args: next(responses))
    monkeypatch.setattr(watch, "digest", lambda path: "b" * 64 if change == "freeze" else watch.FREEZE)
    with pytest.raises(ValueError):
        watch.verify_source()


def test_state_cannot_write_outside_project_or_through_link(scope, monkeypatch):
    monkeypatch.setattr(watch, "STATE", scope.project.parent / "outside")
    with pytest.raises(ValueError, match="project-local"):
        watch.main(["--execute"])
    link = scope.project / "link"
    link.symlink_to(scope.source, target_is_directory=True)
    monkeypatch.setattr(watch, "STATE", link / "watch")
    with pytest.raises(ValueError, match="project-local"):
        watch.main(["--execute"])


@pytest.mark.parametrize("interval", ["0", "61", "-1"])
def test_poll_interval_cannot_block_over_sixty_seconds(scope, interval):
    with pytest.raises(SystemExit):
        watch.main(["--execute", "--poll-seconds", interval])


def test_collector_subprocess_is_isolated_frozen_cpu_only_and_polled(scope, monkeypatch):
    stage = scope.state / "first"
    stage.mkdir(parents=True)
    child = Mock(returncode=0)
    child.poll.side_effect = [None, None, 0]
    child.__enter__ = Mock(return_value=child)
    child.__exit__ = Mock(return_value=False)
    launch = Mock(return_value=child)
    sleeps = []
    monkeypatch.setattr(watch.subprocess, "Popen", launch)
    monkeypatch.setattr(watch.time, "sleep", sleeps.append)
    watch.collect_child(stage)
    args = launch.call_args.args[0]
    assert args[:4] == [str(scope.project / ".venv-reproduce/bin/python"), "-I", "-B", "-c"]
    assert args[-4:] == [str(scope.source), str(stage), watch.FREEZE, watch.COMMIT]
    assert launch.call_args.kwargs["cwd"] == scope.source
    assert launch.call_args.kwargs["env"]["TZ"] == "UTC"
    assert sleeps == [1, 1]
    assert (stage / "collector.log").exists()


def test_external_commands_have_thirty_second_timeout(scope, monkeypatch):
    run = Mock(return_value=SimpleNamespace(stdout="reply"))
    monkeypatch.setattr(watch.subprocess, "run", run)
    assert COMMAND(["sacct", "--help"]) == "reply"
    assert run.call_args.kwargs["timeout"] == 30
    assert run.call_args.kwargs["cwd"] == scope.project
    assert run.call_args.kwargs["check"] is True


@pytest.mark.parametrize("failure", ["none", "capture", "validation"])
def test_child_program_uses_existing_capture_and_preserves_unknown_costs(scope, monkeypatch, failure):
    import memory_tuner
    from memory_tuner import collect_offload_factorial as real_collector

    stage = scope.state / "first"
    stage.mkdir(parents=True)
    (stage / "sacct.txt").write_text("bad input" if failure == "capture" else sacct())
    claims = {watch.FIRST: dict(pair_id="block-0", pair_manifest_sha256="a" * 64,
                               pair_ledger_sha256="b" * 64)}
    calls = []

    def collect(raw, source, freeze, commit, records=None):
        assert raw == stage / "raw" and source == scope.source
        assert freeze == watch.FREEZE and commit == watch.COMMIT
        calls.append(records)
        if failure == "validation" and records is not None:
            raise ValueError("accounting node differs")
        # This test exercises the child transaction, not parent GPU validators.
        slots = [dict(experiment_id=f"slot-{i}", status="not_launched",
                      task_gpu_seconds=None) for i in range(144)]
        return dict(slots=slots, summary={"allocation_accounting":
                    {"status": "not_supplied"} if records is None else {"allocations": 1}},
                    **{key: [{"schema": key}] for key in
                       ("configurations", "prediction_scores", "offload_pairs", "block_contrasts")})

    fake = SimpleNamespace(accounting=accounting, block_claims=lambda root: claims,
                           collect=collect, write_results=real_collector.write_results)
    monkeypatch.setattr(memory_tuner, "collect_offload_factorial", fake)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", ["-c", str(scope.source), str(stage), watch.FREEZE, watch.COMMIT])
    if failure == "none":
        exec(watch.COLLECT, {})
        assert calls[0][0]["raw_record_provenance"]["timezone"] == "UTC"
        assert not (stage / "accounting-errors.json").exists()
    else:
        with pytest.raises(SystemExit) as error:
            exec(watch.COLLECT, {})
        assert error.value.code == 2 and calls[-1] is None
        assert watch.read(stage / "accounting-errors.json")["errors"]
    assert {p.name for p in (stage / "results").iterdir()} == watch.RESULTS
    with (stage / "results/slots.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 144 and all(r["task_gpu_seconds"] == "" for r in rows)
    assert (stage / "accounting.json").exists() is (failure != "capture")


def test_background_requires_explicit_execution(scope):
    with pytest.raises(ValueError, match="requires --execute"):
        watch.main(["--background"])
    assert not scope.state.exists()


def test_background_is_detached_and_retains_pid_without_submitting(scope):
    watch.subprocess.Popen.side_effect = None
    watch.subprocess.Popen.return_value = SimpleNamespace(pid=54321)
    assert watch.main(["--execute", "--background"]) == 0
    args, kwargs = watch.subprocess.Popen.call_args
    assert args[0][-3:] == ["--execute", "--poll-seconds", "30"]
    assert "--background" not in args[0] and kwargs["start_new_session"] is True
    assert kwargs["stdin"] == subprocess.DEVNULL
    receipt = watch.bound_record(scope.state / "background-receipt.json")
    assert receipt["child_pid"] == 54321
    assert receipt["controller_sha256"] == watch.CONTROLLER_SHA256
    watch.command.assert_not_called()
    with pytest.raises(FileExistsError):
        watch.main(["--execute", "--background"])
    assert watch.subprocess.Popen.call_count == 1


def test_failed_background_start_retains_intent_without_retry(scope):
    watch.subprocess.Popen.side_effect = OSError("cannot launch")
    with pytest.raises(OSError):
        watch.main(["--execute", "--background"])
    assert (scope.state / "background-intent.json").is_file()
    with pytest.raises(FileExistsError):
        watch.main(["--execute", "--background"])
    assert watch.subprocess.Popen.call_count == 1
