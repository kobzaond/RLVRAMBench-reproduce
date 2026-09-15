"""CPU-only regressions: no Slurm, containers, CUDA, NVML, or training launches."""
import copy
import csv
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
from types import SimpleNamespace

import pytest

from memory_tuner import capture_environment as capture
from memory_tuner import device_contract as devices
from memory_tuner import gpu_monitor as monitor
from memory_tuner import run_matched_gpu as runner


UUIDS = [f"GPU-00000000-0000-0000-0000-{i:012x}" for i in range(1, 5)]
HEAD = "b" * 40


def matrix_rows():
    rows = []
    for dataset in ("gsm8k", "math", "code_heavy_tail"):
        prompt, response, seqs, batch, logprob = {
            "gsm8k": (1024, 2048, 64, 32, 4),
            "math": (512, 512, 128, 64, 8),
            "code_heavy_tail": (1536, 1024, 64, 32, 4),
        }[dataset]
        for level in ("c2", "c3"):
            for seed in (151, 152, 153):
                pair = f"{dataset}-{level}-s{seed}"
                for period, count in enumerate((2, 4), 1):
                    row = dict(
                        pair_id=pair, experiment_id=f"{pair}-g{count}",
                        run_group=runner.GROUP, model=runner.MODEL, algorithm="grpo",
                        dataset=dataset, configuration_level=level, training_seed=seed,
                        period=period, condition=f"gpu{count}", gpu_count=count,
                        gpu_subset_ordinals="0;2" if count == 2 else "0;1;2;3",
                        per_process_timeout_seconds=1500, total_training_steps=1,
                        rollout_tp_size=1, rollout_n=2, gpu_monitor_interval_ms=100,
                        actor_micro_batch=8 if level == "c2" else 16,
                        vllm_gpu_memory_utilization=.6 if level == "c2" else .7,
                        max_prompt_length=prompt, max_response_length=response,
                        max_model_len=prompt + response, max_num_seqs=seqs,
                        train_batch_size=batch, train_max_samples=batch, val_max_samples=batch,
                        rollout_logprob_micro_batch=logprob, ref_logprob_micro_batch=logprob,
                        parameter_offload=False, optimizer_offload=False, free_cache_engine=True,
                        data_dir=f"/frozen/data/{dataset}", allocator_trace_enabled=True,
                        allocator_trace_sync=1, resume_mode="disable", save_freq=-1,
                        test_freq=-1, val_before_train=False,
                        custom_reward_function_path="/frozen/memory_tuner/code_shape_reward.py"
                        if dataset == "code_heavy_tail" else "null",
                        custom_reward_function_name="compute_score",
                    )
                    rows.append({key: str(value) for key, value in row.items()})
    return rows


def new_json(path, value):
    devices.write_new_json(path, value)


def inventory():
    return [dict(uuid=identity, index=i, pci_bus_id=f"0000:{i:02x}:00.0",
                 name="NVIDIA A100-SXM4-40GB", total_mib=40960.,
                 used_mib=0., utilization_pct=0., mig_mode="Disabled")
            for i, identity in enumerate(UUIDS)]


def identity_evidence(directory, expected, invocation, coverage=None, ray=True):
    new_json(directory / "trainer-cuda.json", {
        "requested_uuids": expected, "invocation_id": invocation, "hostname": "test-node",
        "timestamp_ns": 150,
        "pid": 10, "parent_pid": 9, "probe_kind": "exited_subprocess_same_container",
        "cuda_devices": [{"cuda_index": i, "uuid": identity} for i, identity in enumerate(expected)],
    })
    new_json(directory / "probe-exit-idle.json", {
        "invocation_id": invocation, "probe_pid": 10, "pid": 9,
        "probe_sha256": devices.digest(directory / "trainer-cuda.json"),
        "allocation_uuids": UUIDS, "all_allocation_devices_idle": True,
        "observations": [{"gpus": inventory(), "processes": []}],
    })
    new_json(directory / "trainer-exec.json", {
        "invocation_id": invocation, "pid": 9, "no_probe_context_in_trainer": True,
        "cuda_visible_devices": ",".join(expected)})
    if ray:
        new_json(directory / "ray-resources.json", {
            "invocation_id": invocation, "resources": {"GPU": len(expected)},
            "nodes": [{"Alive": True}],
        })
    for rank, identity in enumerate(expected if coverage is None else coverage):
        new_json(directory / f"worker-actor_rollout-{rank}-100.json", {
            "invocation_id": invocation, "hostname": "test-node", "rank": rank,
            "world_size": len(expected), "role": "actor_rollout", "active_uuid": identity,
            "cuda_devices": [{"cuda_index": 0, "uuid": identity}],
        })


def task_trace(attempt, expected, invocation, job="123"):
    new_json(attempt / "gpu-device-map.json", {"uuids": expected, "invocation_id": invocation})
    (attempt / f"gpu-memory-{job}.csv").write_text(
        "".join(f"{stamp}," + ",".join(["100"] * len(expected)) + "\n"
                for stamp in (100, 200)))
    (attempt / f"gpu-telemetry-{job}.csv").write_text(
        "timestamp_ns,step,phase,gpu_index,memory_used_mib,gpu_utilization_pct\n" +
        "".join(f"{stamp},1,actor,{i},100,20\n"
                for stamp in (100, 200) for i in range(len(expected))))


def period_evidence(attempt, expected, invocation, *, success=False, workers=0, code=None):
    attempt.mkdir(parents=True, exist_ok=True)
    job = "123"
    identity_evidence(attempt / "device-evidence", expected, invocation, expected[:workers])
    if success:
        new_json(attempt / "device-evidence/trainer-terminal.json",
                 {"invocation_id": invocation, "timestamp_ns": 200})
    task_trace(attempt, expected, invocation, job)
    (attempt / f"training-{job}.log").write_text(
        "training/global_step:1\n" if success else
        "torch.OutOfMemoryError: CUDA out of memory\n")
    from memory_tuner.grpo_raw_evidence import MATCH_FIELDS
    trial_values = {name: matrix_rows()[0][name] for name in MATCH_FIELDS}
    trial_values.update(
        gpu_count=len(expected), exit_code=(0 if success else 1) if code is None else code,
        model=runner.MODEL, training_seed=151, peak_gpu_memory_mib=100)
    new_json(attempt / f"trial-{job}.json", trial_values)
    new_json(attempt / f"environment-{job}.json", dict(
        gpu_device_contract={"invocation_id": invocation, "requested_uuids": expected},
        software={"cuda_devices": [{"uuid": identity} for identity in expected]},
        repository={"head": HEAD, "dirty": False},
        model={"resolved_revision": runner.MODEL_REVISION}))


def execution(success=False):
    return dict(spawned=True, wrapper_exit_code=0 if success else 1, timed_out=False,
                error=None, sampling_valid=True, contamination_detected=False,
                cleanup={"complete": True}, payload_elapsed_seconds=2.,
                invocation_elapsed_seconds=3.)


@pytest.mark.parametrize("value", ["", "0,1", "GPU-abc", ",".join(UUIDS[:1] * 2),
                                    UUIDS[0] + ", " + UUIDS[1]])
def test_uuid_contract_rejects_ambiguous_empty_duplicate_or_numeric_ids(value):
    with pytest.raises(ValueError):
        devices.parse_uuids(value)


def test_contract_absence_is_legacy_and_empty_is_not_absence(monkeypatch):
    monkeypatch.delenv("RLVRAM_GPU_UUIDS", raising=False)
    assert devices.requested_uuids() is None
    monkeypatch.setenv("RLVRAM_GPU_UUIDS", "")
    with pytest.raises(ValueError):
        devices.requested_uuids()


def test_actual_cuda_check_catches_lost_mask_despite_filtered_host_reports(monkeypatch):
    monkeypatch.setattr(devices, "cuda_inventory",
                        lambda: [{"cuda_index": i, "uuid": u} for i, u in enumerate(UUIDS)])
    with pytest.raises(ValueError, match="Actual CUDA"):
        devices.check_cuda(UUIDS[:2])
    assert len(devices.check_cuda(UUIDS)) == 4


def test_cuda_inventory_uses_real_device_properties_not_requested_mask(monkeypatch):
    monkeypatch.setenv("RLVRAM_GPU_UUIDS", ",".join(UUIDS[:2]))
    queries = []
    def properties(index):
        queries.append(index)
        return SimpleNamespace(uuid=uuid.UUID(UUIDS[index + 2][4:]))
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(
        device_count=lambda: 2, get_device_properties=properties))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    actual = devices.cuda_inventory()
    assert [row["uuid"] for row in actual] == UUIDS[2:]
    assert queries == [0, 1]


def test_metadata_failure_is_not_reported_as_an_oom_or_an_unstarted_slot(tmp_path):
    (tmp_path / "launcher.log").write_text("required paired environment capture failed\n")
    result = runner.assess_period(tmp_path, matrix_rows()[0], execution(), UUIDS[:2],
                                  "invocation", "123", HEAD)
    assert result["status"] == "infrastructure_invalid"
    assert result["failure_stage"] == "metadata_capture"
    assert result["failure_kind"] == "environment_metadata_capture"


def test_online_assessment_checks_shared_full_configuration_fields(tmp_path):
    period_evidence(tmp_path, UUIDS[:2], "invocation", workers=0)
    path = tmp_path / "trial-123.json"
    record = json.loads(path.read_text())
    record["max_num_seqs"] = 999
    path.write_text(json.dumps(record))
    result = runner.assess_period(tmp_path, matrix_rows()[0], execution(), UUIDS[:2],
                                  "invocation", "123", HEAD)
    assert result["status"] == "infrastructure_invalid"
    assert "max_num_seqs" in result["validation_error"]


def test_zero_training_preflight_exercises_nonconsecutive_two_then_four(tmp_path, monkeypatch):
    monkeypatch.setenv("RLVRAM_GPU_UUIDS", ",".join(UUIDS))
    monkeypatch.setenv("RLVRAM_ALLOCATION_GPU_UUIDS", ",".join(UUIDS))
    monkeypatch.setenv("RLVRAM_DEVICE_EVIDENCE_DIR", str(tmp_path))
    monkeypatch.setenv("RLVRAM_INVOCATION_ID", "preflight")
    monkeypatch.setenv("RLVRAM_PREFLIGHT_RAY_ROOT", str(tmp_path / "ray"))
    seen, events = [], []
    def idle(ids):
        assert ids == UUIDS
        events.append("all_four_idle")
        return [{"gpus": inventory(), "processes": []}]
    monkeypatch.setattr(devices, "wait_probe_idle", idle)
    def child(command, *, env, **kwargs):
        assert command[-1] == "--preflight-ray"
        assert kwargs == {"check": True, "timeout": 240}
        selected = devices.parse_uuids(env["RLVRAM_GPU_UUIDS"])
        seen.append(selected)
        events.append(f"child_{len(selected)}_exited")
        assert env["CUDA_VISIBLE_DEVICES"] == ",".join(selected)
        new_json(Path(env["RLVRAM_DEVICE_EVIDENCE_DIR"]) / "preflight-ray.json",
                 {"training_launched": False})
    monkeypatch.setattr(devices.subprocess, "run", child)
    monkeypatch.setattr(devices, "cuda_inventory", lambda: pytest.fail("preflight parent touched CUDA"))
    devices.preflight()
    assert seen == [[UUIDS[0], UUIDS[2]], UUIDS]
    assert events == ["all_four_idle", "child_2_exited", "all_four_idle",
                      "child_4_exited", "all_four_idle"]
    passed = json.loads((tmp_path / "preflight-passed.json").read_text())
    assert not passed["training_launched"]
    for condition in passed["conditions"]:
        directory = tmp_path / f"task-{condition['task_gpu_count']}"
        assert condition["post_exit_idle_sha256"] == devices.digest(directory / "post-child-idle.json")
        record = json.loads((directory / "post-child-idle.json").read_text())
        assert record["all_allocation_devices_idle"] and record["allocation_uuids"] == UUIDS
        assert record["observations"] == condition["post_exit_allocation_observations"]
        assert record["training_launched"] is False


@pytest.mark.parametrize("failed_idle", [2, 3])
def test_preflight_cannot_pass_or_advance_without_post_child_idle(tmp_path, monkeypatch, failed_idle):
    monkeypatch.setenv("RLVRAM_GPU_UUIDS", ",".join(UUIDS))
    monkeypatch.setenv("RLVRAM_ALLOCATION_GPU_UUIDS", ",".join(UUIDS))
    monkeypatch.setenv("RLVRAM_DEVICE_EVIDENCE_DIR", str(tmp_path))
    monkeypatch.setenv("RLVRAM_INVOCATION_ID", "preflight")
    monkeypatch.setenv("RLVRAM_PREFLIGHT_RAY_ROOT", str(tmp_path / "ray"))
    calls, children = [], []
    def idle(ids):
        assert ids == UUIDS
        calls.append(ids)
        if len(calls) == failed_idle:
            raise RuntimeError("CUDA probe did not leave all four allocation GPUs idle")
        return [{"gpus": inventory(), "processes": []}]
    def child(command, *, env, **kwargs):
        children.append(env["CUDA_VISIBLE_DEVICES"])
        new_json(Path(env["RLVRAM_DEVICE_EVIDENCE_DIR"]) / "preflight-ray.json",
                 {"training_launched": False})
    monkeypatch.setattr(devices, "wait_probe_idle", idle)
    monkeypatch.setattr(devices.subprocess, "run", child)
    with pytest.raises(RuntimeError, match="all four allocation GPUs idle"):
        devices.preflight()
    assert len(children) == failed_idle - 1
    assert not (tmp_path / "preflight-passed.json").exists()


@pytest.mark.parametrize("failure", ["exit", "timeout"])
def test_failed_preflight_child_still_requires_allocation_idle(tmp_path, monkeypatch, failure):
    monkeypatch.setenv("RLVRAM_GPU_UUIDS", ",".join(UUIDS))
    monkeypatch.setenv("RLVRAM_ALLOCATION_GPU_UUIDS", ",".join(UUIDS))
    monkeypatch.setenv("RLVRAM_DEVICE_EVIDENCE_DIR", str(tmp_path))
    monkeypatch.setenv("RLVRAM_INVOCATION_ID", "preflight")
    monkeypatch.setenv("RLVRAM_PREFLIGHT_RAY_ROOT", str(tmp_path / "ray"))
    events = []
    def idle(ids):
        assert ids == UUIDS
        events.append("all_four_idle")
        return [{"gpus": inventory(), "processes": []}]
    error = subprocess.CalledProcessError if failure == "exit" else subprocess.TimeoutExpired
    def child(command, **kwargs):
        events.append("child_failed")
        if failure == "exit":
            raise error(1, command)
        raise error(command, 240)
    monkeypatch.setattr(devices, "wait_probe_idle", idle)
    monkeypatch.setattr(devices.subprocess, "run", child)
    with pytest.raises(error):
        devices.preflight()
    assert events == ["all_four_idle", "child_failed", "all_four_idle"]
    assert (tmp_path / "task-2/post-child-idle.json").is_file()
    assert not (tmp_path / "task-4").exists()
    assert not (tmp_path / "preflight-passed.json").exists()


@pytest.mark.parametrize("actual", [UUIDS[:2], [UUIDS[0]], UUIDS])
def test_disposable_import_context_still_requires_actual_cuda_task_devices(
        tmp_path, monkeypatch, actual):
    from memory_tuner import vllm_uuid_compat as compat
    expected = [UUIDS[0], UUIDS[2]]
    monkeypatch.setenv("RLVRAM_GPU_UUIDS", ",".join(expected))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(expected))
    calls = []
    def imports(*, disposable_preflight):
        calls.append(disposable_preflight)
        return {"cuda_initialized_before_imports": False,
                "cuda_initialized_after_imports": True, "training_launched": False}
    monkeypatch.setattr(compat, "import_training_paths", imports)
    monkeypatch.setattr(devices, "cuda_inventory",
                        lambda: [{"cuda_index": i, "uuid": identity}
                                 for i, identity in enumerate(actual)])
    with pytest.raises(ValueError, match="Actual CUDA identities differ"):
        devices.preflight_ray()
    assert calls == [True]


@pytest.mark.skipif(os.environ.get("RLVRAM_TEST_CONTAINER_API") != "1",
                    reason="Set RLVRAM_TEST_CONTAINER_API=1 for CPU-only pinned-image inspection")
def test_real_pinned_container_exposes_uuid_descriptor_without_cuda_initialization():
    script = """
import inspect, json, torch
descriptor = inspect.getattr_static(torch._C._CudaDeviceProperties, 'uuid', None)
assert torch.__version__ == '2.10.0+cu129', torch.__version__
assert descriptor is not None, 'physical UUID descriptor absent'
assert not torch.cuda.is_initialized(), 'CPU inspection initialized CUDA'
print(json.dumps({'torch': torch.__version__, 'uuid_descriptor': True,
                  'cuda_initialized': torch.cuda.is_initialized()}))
"""
    result = subprocess.run(
        ["apptainer", "exec", "--cleanenv",
         "/mnt/proj3/open-35-44/verl/images/verl-vllm018-dev1.sif",
         "python", "-c", script], text=True, capture_output=True, check=True, timeout=120)
    assert json.loads(result.stdout)["uuid_descriptor"] is True


def test_matrix_is_18_pairs_36_rows():
    pairs = runner.validate_matrix(matrix_rows())
    assert len(pairs) == 18


@pytest.mark.parametrize("field,bad", [
    ("gpu_count", "2.9"), ("gpu_count", "3"), ("training_seed", "42"),
    ("gpu_subset_ordinals", "0;0"), ("gpu_subset_ordinals", "0;4"),
    ("gpu_subset_ordinals", "2;0"), ("period", "2"), ("model", "unseen/model"),
    ("total_training_steps", "100"), ("per_process_timeout_seconds", "1200"),
    ("resume_mode", "auto"), ("save_freq", "1"), ("test_freq", "1"),
    ("val_before_train", "True"), ("actor_micro_batch", "16"),
    ("max_prompt_length", "512"), ("dataset", "other"),
])
def test_matrix_rejects_changed_or_unsafe_settings(field, bad):
    rows = matrix_rows()
    rows[0][field] = bad
    with pytest.raises(ValueError):
        runner.validate_matrix(rows)


def test_matrix_rejects_duplicates_and_missing_rows():
    rows = matrix_rows()
    with pytest.raises(ValueError):
        runner.validate_matrix(rows[:-1])
    rows[2:4] = copy.deepcopy(rows[:2])
    with pytest.raises(ValueError):
        runner.validate_matrix(rows)


def test_code_reward_cannot_silently_fall_back():
    rows = matrix_rows()
    next(r for r in rows if r["dataset"] == "code_heavy_tail")["custom_reward_function_path"] = "null"
    with pytest.raises(ValueError, match="shape reward"):
        runner.validate_matrix(rows)


def test_no_retry_claim_and_exclusive_json_survive_new_job_id(tmp_path):
    path = tmp_path / "pair"
    journal = runner.claim_pair(path, {"job_id": "123"})
    journal.emit("dispatched", period=1)
    journal.close()
    with pytest.raises(FileExistsError):
        runner.claim_pair(path, {"job_id": "456"})
    with pytest.raises(FileExistsError):
        devices.write_new_json(path / "pair.json", {"job_id": "456"})
    assert json.loads((path / "pair.json").read_text())["job_id"] == "123"
    assert json.loads((path / "ledger.jsonl").read_text())["event"] == "dispatched"


@pytest.mark.parametrize("workers", [0, 1])
def test_early_oom_remains_eligible_with_partial_worker_initialization(tmp_path, workers):
    expected = [UUIDS[0], UUIDS[2]]
    period_evidence(tmp_path, expected, "invocation", workers=workers)
    result = runner.assess_period(tmp_path, matrix_rows()[0], execution(), expected,
                                  "invocation", "123", HEAD)
    assert result["status"] == "memory_failure"
    assert result["identity_validated"]
    assert not result["worker_coverage_complete"]
    assert len(result["observed_worker_uuids"]) == workers


def test_early_failure_can_precede_ray_resource_record(tmp_path):
    identity_evidence(tmp_path, UUIDS[:2], "invocation", [], ray=False)
    assert runner.validate_identity(tmp_path, UUIDS[:2], "invocation", success=False)[
        "worker_initialization"] == "partial_or_not_reached"


def test_success_requires_complete_worker_device_coverage(tmp_path):
    period_evidence(tmp_path, UUIDS[:2], "invocation", success=True, workers=1)
    result = runner.assess_period(tmp_path, matrix_rows()[0], execution(True), UUIDS[:2],
                                  "invocation", "123", HEAD)
    assert result["status"] == "infrastructure_invalid"
    assert "full worker/device coverage" in result["validation_error"]


def test_complete_success_is_valid(tmp_path):
    period_evidence(tmp_path, UUIDS[:2], "invocation", success=True, workers=2)
    result = runner.assess_period(tmp_path, matrix_rows()[0], execution(True), UUIDS[:2],
                                  "invocation", "123", HEAD)
    assert result["status"] == "completed"
    assert result["worker_coverage_complete"]


@pytest.mark.parametrize("success", [False, True])
def test_any_observed_excluded_worker_invalidates_even_early_failure(tmp_path, success):
    identity_evidence(tmp_path, UUIDS[:2], "invocation", [UUIDS[2]])
    with pytest.raises(ValueError, match="Observed worker"):
        runner.validate_identity(tmp_path, UUIDS[:2], "invocation", success=success)


def test_parent_identity_cannot_be_stale(tmp_path):
    identity_evidence(tmp_path, UUIDS[:2], "older-invocation")
    with pytest.raises(ValueError, match="Parent actual"):
        runner.validate_identity(tmp_path, UUIDS[:2], "current-invocation", success=False)


@pytest.mark.parametrize("field,value", [
    ("contamination_detected", True), ("timed_out", True), ("sampling_valid", False),
    ("error", "allocation_deadline"), ("cleanup", {"complete": False}),
])
def test_oom_text_cannot_hide_invalid_execution(tmp_path, field, value):
    period_evidence(tmp_path, UUIDS[:2], "invocation", workers=0)
    observed = execution()
    observed[field] = value
    result = runner.assess_period(tmp_path, matrix_rows()[0], observed, UUIDS[:2],
                                  "invocation", "123", HEAD)
    assert result["status"] == "infrastructure_invalid"


@pytest.mark.parametrize("text,expected", [
    ("training/global_step:1", True), ("training/global_step:10", False),
    ("training/global_step:100", False), ("training/global_step:1.0", False),
    ("training/global_step:1\ntraining/global_step:2", False), ("no step", False),
])
def test_exact_terminal_step(text, expected):
    assert runner.terminal_step(text) is expected


def test_task_trace_excludes_other_allocation_devices_and_rejects_extra_columns(tmp_path):
    task_trace(tmp_path, UUIDS[:2], "invocation")
    result = runner.validate_trace(tmp_path, UUIDS[:2], "invocation", "123")
    assert result["task_peak_memory_mib"] == 100
    (tmp_path / "gpu-memory-123.csv").write_text("100,100,100,40000\n200,100,100,40000\n")
    with pytest.raises(ValueError, match="wrong device count"):
        runner.validate_trace(tmp_path, UUIDS[:2], "invocation", "123")


@pytest.mark.parametrize("content", ["", "100,1,2\n", "100,nan,2\n200,3,4\n"])
def test_task_trace_must_be_nonempty_and_finite(tmp_path, content):
    task_trace(tmp_path, UUIDS[:2], "invocation")
    (tmp_path / "gpu-memory-123.csv").write_text(content)
    with pytest.raises(ValueError):
        runner.validate_trace(tmp_path, UUIDS[:2], "invocation", "123")


def test_idle_gate_checks_all_four_devices_and_processes():
    sample = {"gpus": inventory(), "processes": []}
    assert runner.is_idle(sample)
    sample["gpus"][3]["used_mib"] = 1500
    assert not runner.is_idle(sample)
    sample["gpus"][3]["used_mib"] = 0
    sample["processes"] = [{"pid": 123, "uuid": UUIDS[3]}]
    assert not runner.is_idle(sample)


def test_task_vs_allocation_cost_is_not_confused():
    assert runner.gpu_costs(10., 2) == {
        "task_gpu_seconds": 20., "allocated_gpu_seconds_during_invocation": 40.}
    assert runner.gpu_costs(10., 4)["task_gpu_seconds"] == 40.


def test_environment_does_not_leak_previous_task_or_ray_override(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setenv("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("ACTOR_MICRO_BATCH", "999")
    monkeypatch.setenv("APPTAINERENV_CUDA_VISIBLE_DEVICES", "all")
    env = runner.clean_environment({"ACTOR_MICRO_BATCH": "8", "RLVRAM_GPU_UUIDS": UUIDS[0]})
    assert env["ACTOR_MICRO_BATCH"] == "8"
    assert "CUDA_VISIBLE_DEVICES" not in env
    assert "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES" not in env
    assert "APPTAINERENV_CUDA_VISIBLE_DEVICES" not in env


def test_capture_container_mask_is_explicit_only_when_opted_in(monkeypatch, tmp_path):
    calls = []
    def fake(command, **kwargs):
        calls.append(command)
        return {"returncode": 0, "stdout": "{}", "stderr": ""}
    monkeypatch.setattr(capture, "command", fake)
    capture.container_software(tmp_path / "image", tmp_path, tmp_path)
    assert not any("CUDA_VISIBLE_DEVICES=" in word for word in calls[-1])
    capture.container_software(tmp_path / "image", tmp_path, tmp_path, UUIDS[:2], tmp_path)
    assert "CUDA_VISIBLE_DEVICES=" + ",".join(UUIDS[:2]) in calls[-1]
    assert "check_cuda(requested_uuids())" in calls[-1][-1]


class FakeFunction:
    def __init__(self, callback):
        self.callback = callback
    def __call__(self, *args):
        return self.callback(*args)


def test_nvml_selection_uses_uuid_handles_not_all_physical_indices(monkeypatch):
    def count(pointer):
        pointer._obj.value = 4
        return 0
    def by_index(index, pointer):
        pointer._obj.value = index + 1
        return 0
    def by_uuid(identity, pointer):
        pointer._obj.value = UUIDS.index(identity.decode()) + 1
        return 0
    def get_uuid(handle, buffer, length):
        buffer.value = UUIDS[handle.value - 1].encode()
        return 0
    def memory(handle, pointer):
        pointer._obj.used = handle.value * 100 * 1024 * 1024
        return 0
    def utilization(handle, pointer):
        pointer._obj.gpu = 10
        return 0
    library = SimpleNamespace(**{name: FakeFunction(fn) for name, fn in {
        "nvmlInit_v2": lambda: 0, "nvmlDeviceGetCount_v2": count,
        "nvmlDeviceGetHandleByIndex_v2": by_index, "nvmlDeviceGetHandleByUUID": by_uuid,
        "nvmlDeviceGetUUID": get_uuid, "nvmlDeviceGetMemoryInfo": memory,
        "nvmlDeviceGetUtilizationRates": utilization,
    }.items()})
    monkeypatch.setattr(monitor.ctypes, "CDLL", lambda name: library)
    assert monitor.Nvml().sample()[0] == [100, 200, 300, 400]
    assert monitor.Nvml([UUIDS[2], UUIDS[0]]).sample()[0] == [300, 100]


@pytest.fixture
def frozen_run(tmp_path, monkeypatch):
    rows = matrix_rows()
    data_sha = {}
    roots = {"gsm8k": "data/gsm8k", "math": "data/math",
             "code_heavy_tail": "data/codecontests/heavy_tail"}
    for dataset, relative in roots.items():
        path = tmp_path / relative
        path.mkdir(parents=True)
        for name in ("train.parquet", "test.parquet"):
            (path / name).write_bytes(b"synthetic data")
            data_sha[relative + "/" + name] = devices.digest(path / name)
        for row in rows:
            if row["dataset"] == dataset:
                row["data_dir"] = str(path)
    matrix = tmp_path / "matrix.csv"
    with matrix.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({
        "protocol_version": "estimation-1.0", "allocation_gpus": 4,
        "planned_pairs": 18, "planned_processes": 36, "model": runner.MODEL,
        "model_revision": runner.MODEL_REVISION, "evaluation_seeds": [151, 152, 153],
        "resource_cap": {"payload_timeout_seconds": 1500, "idle_timeout_seconds": 180,
                         "wall_minutes_per_allocation": 55, "retries": 0, "requeue": False},
        "input_sha256": {"benchmark/estimation/matrix.csv": devices.digest(matrix)},
        "data_sha256": data_sha,
    }))
    outputs = {}
    for name in ("retrospective.json", "fitted_model.json", "predictions.json"):
        path = tmp_path / name
        path.write_text("{}")
        outputs[name] = devices.digest(path)
    (tmp_path / "prediction_freeze.json").write_text(json.dumps({
        "protocol_sha256": devices.digest(protocol), "amendments_sha256": {},
        "effective_input_sha256": {"benchmark/estimation/matrix.csv": devices.digest(matrix)},
        "target_configurations": 12, "predictions": 36,
        "status": "sealed_before_new_standard_grpo_execution",
        "frozen_at_utc": "2026-09-15T00:00:00+00:00", "files_sha256": outputs,
    }))
    monkeypatch.setattr(runner.subprocess, "check_output",
                        lambda command, **kwargs: "" if "status" in command else HEAD + "\n")
    monkeypatch.setattr(runner, "allocation_context", lambda: dict(
        job_id="123", hostname="test-node", gpu_count=4, gpus=inventory(),
        started_at_epoch=time.time(), ends_at_epoch=time.time() + 3300))
    monkeypatch.setattr(runner, "validate_snapshot", lambda path: path)
    args = SimpleNamespace(
        matrix=matrix, matrix_sha256=devices.digest(matrix),
        protocol=protocol, protocol_sha256=devices.digest(protocol),
        source_commit=HEAD, index=0, project_root=tmp_path,
    )
    return args


@pytest.mark.parametrize("first_status", ["oom", "runtime"])
def test_second_period_runs_after_first_failure_if_all_four_idle(
        frozen_run, monkeypatch, first_status):
    calls, idle_calls = [], []
    def idle(uuids, trace, period, phase, deadline):
        idle_calls.append((period, phase, uuids))
        return True
    def fake_run(command, env, output, allocation_uuids, task_uuids,
                 journal, trace, period, deadline):
        calls.append(period)
        assert allocation_uuids == UUIDS
        assert len(task_uuids) == (2 if period == 1 else 4)
        period_evidence(output.parent, task_uuids, env["RLVRAM_INVOCATION_ID"],
                        success=period == 2, workers=0 if period == 1 else 4)
        if first_status == "runtime" and period == 1:
            (output.parent / "training-123.log").write_text("ModuleNotFoundError: bad_module")
        return execution(period == 2)
    monkeypatch.setattr(runner, "wait_idle", idle)
    monkeypatch.setattr(runner, "run_payload", fake_run)
    result = runner.execute_pair(frozen_run)
    assert calls == [1, 2]
    assert [c[:2] for c in idle_calls] == [(1, "pre_idle"), (1, "post_idle"),
                                          (2, "pre_idle"), (2, "post_idle")]
    assert result == (0 if first_status == "oom" else 2)
    pair = frozen_run.project_root / "output/prospective_estimation/pairs/gsm8k-c2-s151"
    events = [json.loads(line) for line in (pair / "ledger.jsonl").read_text().splitlines()]
    assert [e["period"] for e in events if e["event"] == "dispatched"] == [1, 2]
    assert [e["period"] for e in events if e["event"] == "terminal"] == [1, 2]
    summary = json.loads((pair / "pair-summary.json").read_text())
    assert summary["task_gpu_seconds"] == 18
    assert summary["allocated_gpu_seconds_during_invocations"] == 24
    with pytest.raises(FileExistsError):
        runner.execute_pair(frozen_run)


def test_idle_failure_preserves_not_launched_terminal_rows(frozen_run, monkeypatch):
    monkeypatch.setattr(runner, "wait_idle", lambda *args: False)
    monkeypatch.setattr(runner, "run_payload", lambda *args: pytest.fail("must not launch"))
    assert runner.execute_pair(frozen_run) == 2
    pair = frozen_run.project_root / "output/prospective_estimation/pairs/gsm8k-c2-s151"
    for period in (1, 2):
        record = json.loads((pair / f"period-{period}.json").read_text())
        assert record["status"] == "not_launched"
        assert record["launched"] is False


def test_bad_allocation_is_durable_and_cannot_be_retried(frozen_run, monkeypatch):
    def wrong_allocation():
        raise ValueError("Wrong GPU type")
    monkeypatch.setattr(runner, "allocation_context", wrong_allocation)
    assert runner.execute_pair(frozen_run) == 2
    pair = frozen_run.project_root / "output/prospective_estimation/pairs/gsm8k-c2-s151"
    assert json.loads((pair / "pair.json").read_text())["allocation"] is None
    assert json.loads((pair / "period-2.json").read_text())["status"] == "not_launched"
    with pytest.raises(FileExistsError):
        runner.execute_pair(frozen_run)


def test_stale_protocol_produces_both_not_launched_records(frozen_run, monkeypatch):
    frozen_run.protocol.write_text("{}")
    monkeypatch.setattr(runner, "allocation_context",
                        lambda: pytest.fail("invalid protocol must not query GPUs"))
    assert runner.execute_pair(frozen_run) == 2
    pair = frozen_run.project_root / "output/prospective_estimation/pairs/gsm8k-c2-s151"
    assert json.loads((pair / "period-1.json").read_text())["status"] == "not_launched"
    assert json.loads((pair / "period-2.json").read_text())["status"] == "not_launched"
    assert (pair / "ledger.jsonl").is_file()


def test_fixed_timeout_is_not_shortened_to_fit_allocation(frozen_run, monkeypatch):
    monkeypatch.setattr(runner, "allocation_context", lambda: dict(
        job_id="123", hostname="test-node", gpu_count=4, gpus=inventory(),
        started_at_epoch=time.time() - 2000, ends_at_epoch=time.time() + 1300))
    monkeypatch.setattr(runner, "wait_idle", lambda *args: True)
    monkeypatch.setattr(runner, "run_payload", lambda *args: pytest.fail("must not launch"))
    assert runner.execute_pair(frozen_run) == 2
    pair = frozen_run.project_root / "output/prospective_estimation/pairs/gsm8k-c2-s151"
    assert "full fixed-timeout" in json.loads((pair / "period-1.json").read_text())["reason"]


@pytest.mark.parametrize("value", ["0-3", "0,1,2,3"])
def test_scheduler_gpu_range_and_list_are_explicit_not_host_inventory(value):
    assert runner.allocated_identifiers(value) == ["0", "1", "2", "3"]


def test_visible_allocation_does_not_confuse_global_gres_with_nvml_ordinals():
    physical = inventory()
    for index, row in enumerate(physical):
        row["index"] = index
        row["pci_bus_id"] = f"00000000:{index:02x}:00.0"
    assert runner.allocated_identifiers("4,5,6,7") == ["4", "5", "6", "7"]
    assert runner.resolve_visible_allocation(physical, "0,1,2,3") == physical
    with pytest.raises(ValueError, match="visibility"):
        runner.resolve_visible_allocation(physical, "4,5,6,7")
    with pytest.raises(ValueError, match="Four distinct"):
        runner.resolve_visible_allocation(physical + physical, "0,1,2,3")


@pytest.mark.parametrize("value", ["", "0-4", "0,0,1,2", "0-2", "0,1,2,MIG-device"])
def test_ambiguous_scheduler_assignment_is_rejected(value):
    with pytest.raises(ValueError):
        runner.allocated_identifiers(value)


def scheduler_record(job="123", array=None, task=None):
    now = time.time()
    start = runner.datetime.fromtimestamp(now - 10).isoformat(timespec="seconds")
    end = runner.datetime.fromtimestamp(now - 10 + 3300).isoformat(timespec="seconds")
    identity = f"JobId={job}"
    if array is not None:
        identity += f" ArrayJobId={array} ArrayTaskId={task}"
    return (f"{identity} JobName=paired NumCPUs=64 CPUs/Task=64 "
            f"AllocTRES=cpu=64,mem=384G,node=1,gres/gpu=4 StartTime={start} EndTime={end}")


@pytest.mark.parametrize("job,array,task,query", [
    ("123", None, None, "123"),
    ("5054951", "5054951", "17", "5054951_17"),
    ("5054954", "5054951", "4", "5054951_4"),
    ("123", "120", "0", "120_0"),
])
def test_scheduler_identity_uses_exact_element_or_nonarray_job(job, array, task, query):
    env = {"SLURM_JOB_ID": job}
    if array is not None:
        env.update(SLURM_ARRAY_JOB_ID=array, SLURM_ARRAY_TASK_ID=task)
    assert runner.scheduler_job_identity(env) == (job, query, array, task)


@pytest.mark.parametrize("env", [
    {}, {"SLURM_JOB_ID": ""}, {"SLURM_JOB_ID": "0"}, {"SLURM_JOB_ID": "123_4"},
    {"SLURM_JOB_ID": "-1"}, {"SLURM_JOB_ID": "１２３"},
    {"SLURM_JOB_ID": "123", "SLURM_ARRAY_JOB_ID": "120"},
    {"SLURM_JOB_ID": "123", "SLURM_ARRAY_TASK_ID": "0"},
    {"SLURM_JOB_ID": "123", "SLURM_ARRAY_TASK_COUNT": "9"},
    {"SLURM_JOB_ID": "123", "SLURM_ARRAY_JOB_ID": "", "SLURM_ARRAY_TASK_ID": ""},
    {"SLURM_JOB_ID": "123", "SLURM_ARRAY_JOB_ID": "0", "SLURM_ARRAY_TASK_ID": "1"},
    {"SLURM_JOB_ID": "123", "SLURM_ARRAY_JOB_ID": "120", "SLURM_ARRAY_TASK_ID": "-1"},
    {"SLURM_JOB_ID": "123", "SLURM_ARRAY_JOB_ID": "120", "SLURM_ARRAY_TASK_ID": "1-3"},
    {"SLURM_JOB_ID": "123", "SLURM_ARRAY_JOB_ID": "120", "SLURM_ARRAY_TASK_ID": "1%3"},
    {"SLURM_JOB_ID": "123", "SLURM_ARRAY_JOB_ID": " 120", "SLURM_ARRAY_TASK_ID": "1"},
])
def test_scheduler_missing_partial_or_invalid_environment_fails_closed(env):
    with pytest.raises(ValueError):
        runner.scheduler_job_identity(env)


@pytest.mark.parametrize("job,array,task", [
    ("5054951", "5054951", "17"), ("5054954", "5054951", "4"), ("123", None, None),
])
def test_allocation_queries_exact_identity_and_keeps_only_selected_record(
        monkeypatch, job, array, task):
    for key in list(os.environ):
        if key.startswith("SLURM_ARRAY_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("SLURM_JOB_ID", job)
    monkeypatch.setenv("SLURM_JOB_GPUS", "4,5,6,7")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    if array is not None:
        monkeypatch.setenv("SLURM_ARRAY_JOB_ID", array)
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", task)
    selected = scheduler_record(job, array, task)
    response = "\n".join((scheduler_record("999", "5054951", "2"),
                          selected, scheduler_record("888", "5054951", "3"))) + "\n"
    query = f"{array}_{task}" if array is not None else job
    def run(command, **kwargs):
        assert command == ["scontrol", "show", "job", query, "--oneliner"]
        assert kwargs == dict(text=True, capture_output=True, check=True, timeout=10)
        return SimpleNamespace(stdout=response)
    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(runner, "query_gpus", inventory)
    result = runner.allocation_context()
    assert result["job_id"] == job
    assert result["scheduler_query_identifier"] == query
    assert result["scheduler_record"] == selected
    assert "JobId=999" not in result["scheduler_record"] and "JobId=888" not in result["scheduler_record"]
    assert result["scheduler_gpu_ids"] == ["4", "5", "6", "7"]
    assert [gpu["uuid"] for gpu in result["gpus"]] == UUIDS
    assert result["scheduler_cuda_visible_devices"] == "0,1,2,3"
    assert result["gpu_count"] == 4 and result["hardware_confinement_claimed"] is False


@pytest.mark.parametrize("response", [
    "",
    "JobId=999 ArrayJobId=120 ArrayTaskId=2",
    "JobId=123",
    "JobId=123 ArrayJobId=120",
    "JobId=123 ArrayJobId=121 ArrayTaskId=2",
    "JobId=123 ArrayJobId=120 ArrayTaskId=3",
    "JobId=999 ArrayJobId=120 ArrayTaskId=2\nJobId=123 ArrayJobId=120 ArrayTaskId=2",
    "JobId=123 ArrayJobId=120 ArrayTaskId=3\nJobId=123 ArrayJobId=120 ArrayTaskId=2",
    "JobId=123 ArrayJobId=120 ArrayTaskId=2\nJobId=123 ArrayJobId=120 ArrayTaskId=2",
    "JobId=123 ArrayJobId=120 ArrayTaskId=2 JobId=123",
    "JobId=123 ArrayJobId=120 ArrayTaskId=2 ArrayTaskId=2",
    "JobId=123 ArrayJobId=120 ArrayTaskId=2\nunparseable response",
    "JobId=\nJobId=123 ArrayJobId=120 ArrayTaskId=2",
])
def test_scheduler_rejects_missing_mismatched_duplicate_or_ambiguous_records(response):
    with pytest.raises(ValueError):
        runner.select_scheduler_record(response, "123", "120", "2")


def test_nonarray_response_cannot_hide_missing_array_environment():
    with pytest.raises(ValueError, match="array environment"):
        runner.select_scheduler_record("JobId=123 ArrayJobId=120 ArrayTaskId=2", "123")
    with pytest.raises(ValueError, match="exactly one"):
        runner.select_scheduler_record("JobId=123\nJobId=123", "123")
    with pytest.raises(ValueError, match="Partial"):
        runner.select_scheduler_record("JobId=123", "123", "120", None)


def test_allocation_never_borrows_resources_from_another_record(monkeypatch):
    for key in list(os.environ):
        if key.startswith("SLURM_ARRAY_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_JOB_GPUS", "0-3")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setattr(runner, "query_gpus", inventory)
    response = scheduler_record("999") + "\nJobId=123\n"
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=response))
    with pytest.raises(ValueError, match="exactly four GPUs"):
        runner.allocation_context()


def test_ambiguous_scheduler_response_is_durable_without_retry(frozen_run, monkeypatch):
    def ambiguous():
        record = scheduler_record("123")
        runner.select_scheduler_record(record + "\n" + record, "123")
    monkeypatch.setattr(runner, "allocation_context", ambiguous)
    monkeypatch.setattr(runner, "run_payload", lambda *args: pytest.fail("must not launch"))
    assert runner.execute_pair(frozen_run) == 2
    pair = frozen_run.project_root / "output/prospective_estimation/pairs/gsm8k-c2-s151"
    for period in (1, 2):
        record = json.loads((pair / f"period-{period}.json").read_text())
        assert record["status"] == "not_launched" and record["launched"] is False
        assert "exactly one matching scheduler record" in record["reason"]
    with pytest.raises(FileExistsError):
        runner.execute_pair(frozen_run)


def test_failed_worker_identity_gate_cannot_look_like_uninitialized_early_oom(tmp_path):
    identity_evidence(tmp_path, UUIDS[:2], "invocation", [])
    new_json(tmp_path / "worker-error-10.json", {
        "validation_error": "Worker selected an unassigned physical device"})
    with pytest.raises(ValueError, match="worker device gate failed"):
        runner.validate_identity(tmp_path, UUIDS[:2], "invocation", success=False)


def test_timeout_has_terminal_result_and_reaps_cpu_child(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "PAYLOAD_TIMEOUT", .15)
    monkeypatch.setattr(runner, "sample_allocation",
                        lambda uuids: {"gpus": inventory(), "processes": []})
    journal = devices.Journal(tmp_path / "events.jsonl")
    trace = devices.Journal(tmp_path / "allocation.jsonl")
    try:
        result = runner.run_payload(
            [sys.executable, "-c", "import time; time.sleep(10)"], dict(os.environ),
            tmp_path / "child.log", UUIDS, UUIDS[:2], journal, trace, 1,
            time.monotonic() + 120)
    finally:
        journal.close()
        trace.close()
    assert result["timed_out"]
    assert result["wrapper_exit_code"] == 124
    assert result["cleanup"]["complete"]
    assert result["sampling_valid"]


def test_wrapper_has_fixed_allocation_and_no_requeue():
    source = Path(runner.__file__).resolve().parents[1]
    wrapper = (source / "run_matched_gpu.slurm").read_text()
    for flag in ("--account=OPEN-35-44", "--partition=qgpu_exp", "--gpus=4",
                 "--time=00:55:00", "--array=0-17%3", "--no-requeue"):
        assert flag in wrapper
    launcher = (source / "run_grpo_instrumented.slurm").read_text()
    assert '--env "CUDA_VISIBLE_DEVICES=$RLVRAM_GPU_UUIDS"' in launcher
    assert "python -m memory_tuner.device_contract --train" in launcher


def test_probe_is_short_child_then_idle_then_exec_without_parent_cuda(
        tmp_path, monkeypatch):
    monkeypatch.setenv("RLVRAM_GPU_UUIDS", ",".join(UUIDS[:2]))
    monkeypatch.setenv("RLVRAM_ALLOCATION_GPU_UUIDS", ",".join(UUIDS))
    monkeypatch.setenv("RLVRAM_INVOCATION_ID", "probe-test")
    monkeypatch.setenv("RLVRAM_DEVICE_EVIDENCE_DIR", str(tmp_path))
    calls = []
    def child(command, **kwargs):
        assert command[-1] == "--probe-cuda"
        calls.append("child_exited")
        new_json(tmp_path / "trainer-cuda.json", {
            "invocation_id": "probe-test", "pid": 456,
            "requested_uuids": UUIDS[:2],
            "cuda_devices": [{"uuid": u} for u in UUIDS[:2]],
        })
    def idle(allocated):
        assert allocated == UUIDS
        calls.append("all_four_idle")
        return [{"gpus": inventory(), "processes": []}]
    def execute(executable, command):
        calls.append("exec_fresh_trainer")
        assert "--exec-trainer" in command
        assert command[-1] == "trainer.total_training_steps=1"
    monkeypatch.setattr(devices.subprocess, "run", child)
    monkeypatch.setattr(devices, "wait_probe_idle", idle)
    monkeypatch.setattr(devices.os, "execv", execute)
    monkeypatch.setattr(devices, "cuda_inventory", lambda: pytest.fail("parent must not touch CUDA"))
    devices.checked_training(["trainer.total_training_steps=1"])
    assert calls == ["child_exited", "all_four_idle", "exec_fresh_trainer"]
    assert json.loads((tmp_path / "probe-exit-idle.json").read_text())[
        "all_allocation_devices_idle"]


def test_worker_hook_is_after_normal_context_initialization():
    source = Path(runner.__file__).resolve().parents[1]
    worker = (source / "instrumented_python_packages/verl/workers/engine_workers.py").read_text()
    start = worker.index("    def init_model(self):", worker.index("class ActorRolloutRefWorker"))
    end = worker.index("    def compute_ref_log_prob(", start)
    body = worker[start:end]
    assert body.index("aggressive_empty_cache(force_sync=True)") < body.index("record_worker_identity(")


def test_sealed_real_matrix_is_accepted_without_modification():
    source = Path(runner.__file__).resolve().parents[1]
    path = source / "benchmark/estimation/matrix.csv"
    if not path.is_file():
        pytest.skip("prospective frozen matrix not included in this checkout")
    before = devices.digest(path)
    with path.open(newline="") as handle:
        assert len(runner.validate_matrix(list(csv.DictReader(handle)))) == 18
    assert devices.digest(path) == before


def test_execute_pair_default_validators_are_resolved_at_call_time(frozen_run, monkeypatch):
    calls = []
    matrix_validator, protocol_verifier = runner.validate_matrix, runner.verify_protocol
    def matrix(rows):
        calls.append("matrix")
        return matrix_validator(rows)
    def protocol(source, args, rows):
        calls.append("protocol")
        return protocol_verifier(source, args, rows)
    monkeypatch.setattr(runner, "validate_matrix", matrix)
    monkeypatch.setattr(runner, "verify_protocol", protocol)
    monkeypatch.setattr(runner, "wait_idle", lambda *args: False)
    monkeypatch.setattr(runner, "run_payload", lambda *args: pytest.fail("must not launch"))
    assert runner.execute_pair(frozen_run) == 2
    assert calls == ["matrix", "protocol"]
    pair = frozen_run.project_root / "output/prospective_estimation/pairs/gsm8k-c2-s151"
    assert (pair / "period-1.json").is_file() and (pair / "period-2.json").is_file()


@pytest.mark.parametrize("group", ["", "../outside", "/absolute", "a/b", None, 1])
def test_explicit_unsafe_group_does_not_touch_inputs_or_outputs(monkeypatch, group):
    if group is None:
        # None resolves the runtime default rather than becoming a path token.
        monkeypatch.setattr(runner, "GROUP", "../unsafe-default")
    with pytest.raises(ValueError, match="Unsafe output group"):
        runner.execute_pair(SimpleNamespace(), group=group)
