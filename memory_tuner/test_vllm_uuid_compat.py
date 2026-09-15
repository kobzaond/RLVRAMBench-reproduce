"""CPU regressions for UUID/NVML compatibility and sampler cleanup sequencing."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import pytest

from memory_tuner import device_contract as devices
from memory_tuner import run_matched_gpu as runner
from memory_tuner import vllm_uuid_compat as compat


UUIDS = [f"GPU-00000000-0000-0000-0000-{i:012x}" for i in range(1, 5)]
SOURCE = Path(__file__).resolve().parents[1]
IMAGE = "/mnt/proj3/open-35-44/verl/images/verl-vllm018-dev1.sif"


class Nvml:
    """Indices deliberately differ from CUDA logical indices and UUID order."""
    def __init__(self):
        self.indices = {UUIDS[0]: 3, UUIDS[1]: 1, UUIDS[2]: 0, UUIDS[3]: 2}
        self.calls = []
        self.wrong_round_trip = False

    def nvmlInit(self):
        self.calls.append("init")

    def nvmlShutdown(self):
        self.calls.append("shutdown")

    def nvmlDeviceGetHandleByUUID(self, identity):
        self.calls.append(("uuid", identity))
        return identity

    def nvmlDeviceGetIndex(self, handle):
        return self.indices[handle]

    def nvmlDeviceGetHandleByIndex(self, index):
        return next(u for u, i in self.indices.items() if i == index)

    def nvmlDeviceGetUUID(self, handle):
        return (UUIDS[3] if self.wrong_round_trip else handle).encode()


@pytest.fixture
def platform(monkeypatch):
    class Platform:
        device_control_env_var = "CUDA_VISIBLE_DEVICES"

        @classmethod
        def device_id_to_physical_device_id(cls, device_id):
            mask = os.environ.get(cls.device_control_env_var)
            return int(mask.split(",")[device_id]) if mask else device_id

    package = ModuleType("vllm")
    package.__version__ = "0.18.0"
    monkeypatch.setitem(sys.modules, "vllm", package)
    module = SimpleNamespace(Platform=Platform)
    compat.patch_platform(module)
    return Platform


def test_nvml_uuid_lookup_does_not_assume_cuda_or_scheduler_ordinals():
    nvml = Nvml()
    assert compat.uuid_to_nvml_index(UUIDS[0], nvml) == 3
    assert compat.uuid_to_nvml_index(UUIDS[2], nvml) == 0
    assert nvml.calls == ["init", ("uuid", UUIDS[0]), "shutdown",
                          "init", ("uuid", UUIDS[2]), "shutdown"]


def test_nvml_roundtrip_mismatch_fails_closed_and_shuts_down():
    nvml = Nvml()
    nvml.wrong_round_trip = True
    with pytest.raises(ValueError, match="round-trip"):
        compat.uuid_to_nvml_index(UUIDS[0], nvml)
    assert nvml.calls[-1] == "shutdown"


def test_mapping_preserves_requested_mask_and_ray_environment(platform, monkeypatch):
    mask = ",".join([UUIDS[0], UUIDS[2]])
    monkeypatch.setenv("RLVRAM_GPU_UUIDS", mask)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    monkeypatch.setenv("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1")
    nvml = Nvml()
    original = compat.uuid_to_nvml_index
    monkeypatch.setattr(compat, "uuid_to_nvml_index", lambda identity: original(identity, nvml))
    before = dict(os.environ)
    assert platform.device_id_to_physical_device_id(0) == 3
    assert platform.device_id_to_physical_device_id(1) == 0
    assert dict(os.environ) == before
    # Ray assigns a subset of the task UUIDs to a worker; logical index resets.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", UUIDS[2])
    assert platform.device_id_to_physical_device_id(0) == 0


@pytest.mark.parametrize("mask", ["0,2", UUIDS[0] + "," + UUIDS[3],
                                 UUIDS[0] + "," + UUIDS[0], "GPU-short"])
def test_paired_bad_numeric_unassigned_or_duplicate_masks_fail(platform, monkeypatch, mask):
    monkeypatch.setenv("RLVRAM_GPU_UUIDS", ",".join(UUIDS[:2]))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    monkeypatch.setattr(compat, "uuid_to_nvml_index", lambda *a: pytest.fail("invalid mask reached NVML"))
    with pytest.raises(ValueError):
        platform.device_id_to_physical_device_id(0)


@pytest.mark.parametrize("index", [-1, 2, True, "0", 0.0])
def test_logical_index_validation(platform, monkeypatch, index):
    monkeypatch.setenv("RLVRAM_GPU_UUIDS", ",".join(UUIDS[:2]))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(UUIDS[:2]))
    with pytest.raises(ValueError, match="logical"):
        platform.device_id_to_physical_device_id(index)


def test_absent_contract_and_cpu_empty_mask_keep_legacy_behavior(platform, monkeypatch):
    monkeypatch.delenv("RLVRAM_GPU_UUIDS", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,7")
    assert platform.device_id_to_physical_device_id(1) == 7
    monkeypatch.setenv("RLVRAM_GPU_UUIDS", ",".join(UUIDS[:2]))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert platform.device_id_to_physical_device_id(0) == 0


def test_non_cuda_platform_is_not_patched(platform, monkeypatch):
    monkeypatch.setenv("RLVRAM_GPU_UUIDS", UUIDS[0])
    monkeypatch.setenv("OTHER_DEVICES", "5")
    class Other(platform):
        device_control_env_var = "OTHER_DEVICES"
    assert Other.device_id_to_physical_device_id(0) == 5


def test_wrong_pinned_version_is_rejected(platform, monkeypatch):
    class Fresh:
        @classmethod
        def device_id_to_physical_device_id(cls, index):
            return index
    monkeypatch.setattr(sys.modules["vllm"], "__version__", "0.19.0")
    with pytest.raises(RuntimeError, match="pinned"):
        compat.patch_platform(SimpleNamespace(Platform=Fresh))


def test_engine_subprocess_mask_keeps_uuid_tokens_not_nvml_indices(platform, monkeypatch):
    monkeypatch.setenv("RLVRAM_GPU_UUIDS", ",".join(UUIDS))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(UUIDS))
    monkeypatch.setattr(compat, "uuid_to_nvml_index", lambda identity: 99)
    module = SimpleNamespace(get_device_indices=lambda *args: "legacy-mask")
    compat.patch_engine(module)
    function = module.get_device_indices
    compat.patch_engine(module)
    assert module.get_device_indices is function
    before = dict(os.environ)
    assert platform.device_id_to_physical_device_id(0) == 99
    assert module.get_device_indices("CUDA_VISIBLE_DEVICES", 1, 2) == ",".join(UUIDS[2:])
    assert module.get_device_indices("CUDA_VISIBLE_DEVICES", 0, 4, 2) == ",".join(UUIDS[:2])
    assert dict(os.environ) == before
    assert module.get_device_indices("OTHER", 0, 1) == "legacy-mask"
    monkeypatch.delenv("RLVRAM_GPU_UUIDS")
    assert module.get_device_indices("CUDA_VISIBLE_DEVICES", 0, 1) == "legacy-mask"


@pytest.mark.parametrize("rank,world,local", [(-1, 1, None), (1, 4, None),
                                              (0, 0, None), (0, 4, 0),
                                              (True, 1, None), (0, 2, 5)])
def test_engine_invalid_or_out_of_bounds_slices_fail(platform, monkeypatch, rank, world, local):
    monkeypatch.setenv("RLVRAM_GPU_UUIDS", ",".join(UUIDS))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(UUIDS))
    module = SimpleNamespace(get_device_indices=lambda *args: "wrong")
    compat.patch_engine(module)
    with pytest.raises(ValueError, match="slice"):
        module.get_device_indices("CUDA_VISIBLE_DEVICES", rank, world, local)


def test_install_is_opt_in_idempotent_and_does_not_import_frameworks(monkeypatch):
    monkeypatch.delenv("RLVRAM_GPU_UUIDS", raising=False)
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    before = list(sys.meta_path)
    assert compat.install() is False and sys.meta_path == before
    monkeypatch.setenv("RLVRAM_GPU_UUIDS", UUIDS[0])
    monkeypatch.delitem(sys.modules, compat.MODULE, raising=False)
    monkeypatch.delitem(sys.modules, compat.ENGINE_MODULE, raising=False)
    modules = set(sys.modules)
    assert compat.install() and compat.install()
    assert sum(getattr(f, "_rlvram_uuid_bridge_finder", False) for f in sys.meta_path) == 1
    assert not ({"torch", "vllm", "pynvml"} & (set(sys.modules) - modules))


def test_fresh_child_sitecustomize_hooks_vllm_before_first_import(tmp_path):
    package = tmp_path / "vllm"
    (package / "platforms").mkdir(parents=True)
    (package / "__init__.py").write_text('__version__ = "0.18.0"\n')
    (package / "platforms/__init__.py").write_text("")
    (package / "platforms/interface.py").write_text(
        "import os\nclass Platform:\n"
        "    device_control_env_var = 'CUDA_VISIBLE_DEVICES'\n"
        "    @classmethod\n"
        "    def device_id_to_physical_device_id(cls, index):\n"
        "        return int(os.environ[cls.device_control_env_var].split(',')[index])\n")
    (package / "v1/engine").mkdir(parents=True)
    (package / "v1/__init__.py").write_text("")
    (package / "v1/engine/__init__.py").write_text("")
    (package / "v1/engine/utils.py").write_text(
        "from vllm.platforms.interface import Platform\n"
        "def get_device_indices(evar, rank, world, local=None):\n"
        "    return str(Platform.device_id_to_physical_device_id(0))\n")
    script = """
import os, sys
assert "torch" not in sys.modules and "vllm" not in sys.modules
from vllm.platforms.interface import Platform
from memory_tuner import vllm_uuid_compat as bridge
assert getattr(Platform.__dict__["device_id_to_physical_device_id"].__func__, bridge.PATCH_TAG)
bridge.uuid_to_nvml_index = lambda identity: 3
assert Platform.device_id_to_physical_device_id(0) == 3
from vllm.v1.engine.utils import get_device_indices
assert get_device_indices("CUDA_VISIBLE_DEVICES", 0, 1) == os.environ["RLVRAM_GPU_UUIDS"]
assert os.environ["CUDA_VISIBLE_DEVICES"] == os.environ["RLVRAM_GPU_UUIDS"]
assert "torch" not in sys.modules
"""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, (
        SOURCE, SOURCE / "instrumented_python_packages", tmp_path))),
        PYTHONDONTWRITEBYTECODE="1", RLVRAM_GPU_UUIDS=UUIDS[0], CUDA_VISIBLE_DEVICES=UUIDS[0])
    subprocess.run([sys.executable, "-c", script], env=env, check=True, timeout=15)


@pytest.fixture
def runtime_imports(monkeypatch):
    """Simulate import-time state changes without importing Torch or vLLM."""
    monkeypatch.setenv("RLVRAM_GPU_UUIDS", UUIDS[0])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", UUIDS[0])
    state = SimpleNamespace(initialized=False, initialize_on=None, requested=[],
                            query_initializes=False, mask_change_on=None,
                            error_on=None, outside_source=False)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_initialized=lambda: state.initialized)))
    monkeypatch.setattr(compat, "install", lambda: True)
    def import_module(name):
        state.requested.append(name)
        if name == state.initialize_on:
            state.initialized = True
        if name == state.mask_change_on:
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", UUIDS[1])
        if name == state.error_on:
            raise ImportError("ordinary import failed")
        if state.outside_source and name.startswith("verl."):
            return SimpleNamespace(__file__="/wrong-source/verl/worker.py")
        return SimpleNamespace(__file__=str(SOURCE / "instrumented_python_packages"
                                           / (name.replace(".", "/") + ".py")))
    monkeypatch.setattr(importlib, "import_module", import_module)
    def physical_index(index):
        if state.query_initializes:
            state.initialized = True
        return 3
    state.platform = SimpleNamespace(
        device_id_to_physical_device_id=physical_index,
        get_device_uuid=lambda index: UUIDS[0],
        get_device_capability=lambda index: SimpleNamespace(major=8, minor=0))
    state.engine = SimpleNamespace(get_device_indices=lambda *args: UUIDS[0])
    monkeypatch.setitem(sys.modules, "vllm.platforms", SimpleNamespace(current_platform=state.platform))
    monkeypatch.setitem(sys.modules, compat.ENGINE_MODULE, state.engine)
    return state


def test_zero_training_import_gate_queries_actual_target_modules_without_context(runtime_imports):
    result = compat.import_training_paths()
    requested = runtime_imports.requested
    assert "verl.workers.engine_workers" in requested
    assert "verl.workers.rollout.vllm_rollout.vllm_async_server" in requested
    assert "vllm.lora.lora_model" in requested and "vllm.v1.worker.gpu_worker" in requested
    assert result["training_launched"] is False
    assert result["cuda_initialized_after_imports"] is False
    assert result["cuda_initialized_after_nvml_queries"] is False
    assert result["cuda_initializing_imports"] == []
    assert result["uuid_nvml_mapping"][0]["nvml_index"] == 3


@pytest.mark.parametrize("trigger", [
    "verl.workers.engine_workers",
    "verl.workers.rollout.vllm_rollout.vllm_async_server",
    "verl.workers.rollout.vllm_rollout.utils",
    "vllm.lora.lora_model",
    "vllm.v1.worker.gpu_worker",
    compat.ENGINE_MODULE,
])
def test_disposable_import_cuda_init_is_reported_not_training(runtime_imports, capsys, trigger):
    runtime_imports.initialize_on = trigger
    result = compat.import_training_paths(disposable_preflight=True)
    assert result["cuda_initializing_imports"] == [trigger]
    assert result["cuda_initialized_before_imports"] is False
    assert result["cuda_initialized_after_imports"] is True
    assert result["cuda_initialized_after_nvml_queries"] is True
    assert result["training_launched"] is False and result["disposable_preflight"] is True
    assert [entry["module"] for entry in result["module_cuda_states"]] == result["imported_modules"]
    transition = next(entry for entry in result["module_cuda_states"] if entry["initialized_cuda"])
    assert transition == dict(module=trigger, cuda_initialized_before=False,
                              cuda_initialized_after=True, initialized_cuda=True)
    assert result["engine_cuda_mask"] == UUIDS[0]
    assert result["uuid_nvml_mapping"][0]["uuid"] == UUIDS[0]
    assert f"module={trigger}" in capsys.readouterr().out


def test_existing_context_is_not_attributed_to_imports(runtime_imports):
    runtime_imports.initialized = True
    result = compat.import_training_paths(disposable_preflight=True)
    assert result["cuda_initialized_before_imports"] is True
    assert result["cuda_initializing_imports"] == []
    assert all(entry["cuda_initialized_before"] and entry["cuda_initialized_after"]
               for entry in result["module_cuda_states"])
    assert result["training_launched"] is False


def test_import_context_change_requires_explicit_disposable_opt_in(runtime_imports):
    runtime_imports.initialize_on = "verl.workers.engine_workers"
    with pytest.raises(RuntimeError, match="outside a disposable preflight"):
        compat.import_training_paths()


def test_import_transition_is_logged_even_if_the_import_raises(runtime_imports, capsys):
    runtime_imports.initialize_on = runtime_imports.error_on = "verl.workers.engine_workers"
    with pytest.raises(ImportError, match="ordinary import failed"):
        compat.import_training_paths(disposable_preflight=True)
    assert "module=verl.workers.engine_workers" in capsys.readouterr().out


@pytest.mark.parametrize("disposable", [False, True])
def test_nvml_queries_still_must_preserve_post_import_cuda_state(runtime_imports, disposable):
    runtime_imports.query_initializes = True
    with pytest.raises(RuntimeError, match="NVML compatibility queries created"):
        compat.import_training_paths(disposable_preflight=disposable)


@pytest.mark.parametrize("mask", ["0", UUIDS[1], ",".join([UUIDS[0]] * 2)])
def test_preflight_bad_mask_is_rejected_before_imports(runtime_imports, monkeypatch, mask):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    with pytest.raises(ValueError):
        compat.import_training_paths(disposable_preflight=True)
    assert runtime_imports.requested == []


def test_import_cannot_change_the_checked_mask(runtime_imports):
    runtime_imports.mask_change_on = "verl.workers.engine_workers"
    with pytest.raises(ValueError, match="changed the preflight task UUID mask"):
        compat.import_training_paths(disposable_preflight=True)


@pytest.mark.parametrize("bad_evidence", ["uuid", "engine_mask", "capability", "source"])
def test_import_context_permission_does_not_weaken_identity_gates(runtime_imports, bad_evidence):
    runtime_imports.initialize_on = "verl.workers.engine_workers"
    if bad_evidence == "uuid":
        runtime_imports.platform.get_device_uuid = lambda index: UUIDS[1]
    elif bad_evidence == "engine_mask":
        runtime_imports.engine.get_device_indices = lambda *args: "0"
    elif bad_evidence == "capability":
        runtime_imports.platform.get_device_capability = lambda index: SimpleNamespace(major=9, minor=0)
    else:
        runtime_imports.outside_source = True
    with pytest.raises(RuntimeError):
        compat.import_training_paths(disposable_preflight=True)


def test_preflight_wrapper_uses_payload_pythonpath_and_host_sanitizer():
    path = SOURCE / "run_matched_gpu_preflight.slurm"
    text = path.read_text()
    assert ("PREFLIGHT_PYTHONPATH=$REVISION_SOURCE_ROOT:"
            "$REVISION_SOURCE_ROOT/third_party/TransferQueue:"
            "$REVISION_SOURCE_ROOT/instrumented_python_packages:"
            "$PROJECT_VERL_ROOT/python-packages:$REVISION_SOURCE_ROOT/sppo_replay") in text
    assert '--env "PYTHONPATH=$PREFLIGHT_PYTHONPATH"' in text
    assert "os.execvpe(sys.argv[1], sys.argv[1:], clean_environment({}))" in text
    assert "--env \"CUDA_VISIBLE_DEVICES=$PREFLIGHT_UUIDS\"" in text
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        assert f'--env "{name}=1"' in text
    subprocess.run(["bash", "-n", str(path)], check=True)


def test_sampler_query_child_finishes_before_subreaper_cleanup(tmp_path, monkeypatch):
    finished = threading.Event()
    order = []
    def sample(uuids):
        subprocess.run([sys.executable, "-c", "import time; time.sleep(.25)"],
                       check=True, timeout=5)
        order.append("sampler_query_finished")
        finished.set()
        return {"gpus": [], "processes": []}
    cleanup = runner.cleanup_children
    def checked_cleanup(*args):
        assert finished.is_set(), "Cleanup raced the sampler's own subprocess"
        order.append("cleanup")
        return cleanup(*args)
    monkeypatch.setattr(runner, "sample_allocation", sample)
    monkeypatch.setattr(runner, "cleanup_children", checked_cleanup)
    journal = devices.Journal(tmp_path / "ledger.jsonl")
    trace = devices.Journal(tmp_path / "allocation.jsonl")
    try:
        result = runner.run_payload(
            [sys.executable, "-c", "import time; time.sleep(.05)"], dict(os.environ),
            tmp_path / "payload.log", UUIDS, UUIDS[:2], journal, trace, 1,
            time.monotonic() + 120)
    finally:
        journal.close()
        trace.close()
    assert order == ["sampler_query_finished", "cleanup"]
    assert result["sampling_valid"] and result["cleanup"]["complete"]


def test_real_sampler_failure_is_not_suppressed_during_cleanup(tmp_path, monkeypatch):
    def sample(uuids):
        raise RuntimeError("genuine NVML sampling failure")
    monkeypatch.setattr(runner, "sample_allocation", sample)
    journal = devices.Journal(tmp_path / "ledger.jsonl")
    trace = devices.Journal(tmp_path / "allocation.jsonl")
    try:
        result = runner.run_payload([sys.executable, "-c", "pass"], dict(os.environ),
                                    tmp_path / "payload.log", UUIDS, UUIDS[:2], journal, trace, 1,
                                    time.monotonic() + 120)
    finally:
        journal.close()
        trace.close()
    assert not result["sampling_valid"]
    assert "genuine NVML sampling failure" in result["error"]


@pytest.mark.skipif(os.environ.get("RLVRAM_TEST_CONTAINER_API") != "1",
                    reason="Explicit CPU-only pinned-container API check")
def test_real_container_interface_hook_and_nvml_apis_without_cuda():
    script = """
import os, torch
from vllm.platforms.interface import Platform
from vllm.utils.import_utils import import_pynvml
from memory_tuner import vllm_uuid_compat as bridge
assert not torch.cuda.is_initialized()
nvml = import_pynvml()
assert all(hasattr(nvml, name) for name in (
    "nvmlDeviceGetHandleByUUID", "nvmlDeviceGetIndex",
    "nvmlDeviceGetHandleByIndex", "nvmlDeviceGetUUID"))
assert getattr(Platform.__dict__["device_id_to_physical_device_id"].__func__, bridge.PATCH_TAG)
class Fake:
    def nvmlInit(self): pass
    def nvmlShutdown(self): pass
    def nvmlDeviceGetHandleByUUID(self, identity): return identity
    def nvmlDeviceGetIndex(self, handle): return 3
    def nvmlDeviceGetHandleByIndex(self, index):
        assert index == 3
        return os.environ["CUDA_VISIBLE_DEVICES"]
    def nvmlDeviceGetUUID(self, handle): return handle
original = bridge.uuid_to_nvml_index
bridge.uuid_to_nvml_index = lambda identity: original(identity, Fake())
class Cuda(Platform):
    device_control_env_var = "CUDA_VISIBLE_DEVICES"
assert Cuda.device_id_to_physical_device_id(0) == 3
from vllm.v1.engine.utils import get_device_indices
assert getattr(get_device_indices, bridge.ENGINE_PATCH_TAG)
assert get_device_indices("CUDA_VISIBLE_DEVICES", 0, 1) == os.environ["RLVRAM_GPU_UUIDS"]
assert os.environ["CUDA_VISIBLE_DEVICES"] == os.environ["RLVRAM_GPU_UUIDS"]
assert not torch.cuda.is_initialized()
print("Pinned vLLM UUID bridge and NVML APIs passed; no CUDA context")
"""
    subprocess.run([
        "apptainer", "exec", "--cleanenv", "--bind", f"{SOURCE}:{SOURCE}",
        "--env", f"PYTHONPATH={SOURCE}:{SOURCE / 'instrumented_python_packages'}",
        "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", f"RLVRAM_GPU_UUIDS={UUIDS[0]}",
        "--env", f"CUDA_VISIBLE_DEVICES={UUIDS[0]}", IMAGE, "python3", "-c", script],
        check=True, timeout=90)
