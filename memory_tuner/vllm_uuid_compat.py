"""Opt-in vLLM 0.18 UUID-mask compatibility, without creating CUDA contexts.

vLLM's NVML platform converts a CUDA-visible token to int before using an NVML
handle-by-index API. Under the paired protocol that token is a physical UUID.
Resolve that UUID through NVML in THIS process's namespace, with a round-trip
identity check. The returned index is ONLY for vLLM's physical-device lookup;
it must never replace CUDA_VISIBLE_DEVICES or Ray accelerator assignments.

Installing the import hook imports neither vLLM, Torch, CUDA, nor NVML. The
hook also covers fresh Ray/vLLM Python children through opt-in sitecustomize.
With RLVRAM_GPU_UUIDS absent, installation and patched calls are legacy no-ops.
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
from pathlib import Path
import sys

from memory_tuner.device_contract import parse_uuids


MODULE = "vllm.platforms.interface"
ENGINE_MODULE = "vllm.v1.engine.utils"
PATCH_TAG = "_rlvram_uuid_nvml_bridge_v1"
ENGINE_PATCH_TAG = "_rlvram_uuid_engine_mask_v1"


def uuid_to_nvml_index(identity, nvml=None):
    """Context-free NVML lookup; never equate Slurm, CUDA, and NVML ordinals."""
    identity = parse_uuids(identity)[0]
    if nvml is None:
        from vllm.utils.import_utils import import_pynvml

        nvml = import_pynvml()
    nvml.nvmlInit()
    try:
        handle = nvml.nvmlDeviceGetHandleByUUID(identity)
        index = int(nvml.nvmlDeviceGetIndex(handle))
        if index < 0:
            raise ValueError("NVML returned a negative device index")
        indexed = nvml.nvmlDeviceGetHandleByIndex(index)
        actual = nvml.nvmlDeviceGetUUID(indexed)
        if isinstance(actual, bytes):
            actual = actual.decode("ascii")
        if parse_uuids(actual) != [identity]:
            raise ValueError("UUID/NVML-index round-trip identity mismatch")
        return index
    finally:
        nvml.nvmlShutdown()


def patch_platform(module):
    platform = module.Platform
    descriptor = platform.__dict__["device_id_to_physical_device_id"]
    if not isinstance(descriptor, classmethod):
        raise RuntimeError("Unexpected pinned vLLM physical-device API")
    original = descriptor.__func__
    if getattr(original, PATCH_TAG, False):
        return
    version = getattr(sys.modules.get("vllm"), "__version__", None)
    if version != "0.18.0":
        raise RuntimeError(f"UUID bridge requires pinned vLLM 0.18.0, found {version!r}")

    def physical_device(cls, device_id):
        requested = os.environ.get("RLVRAM_GPU_UUIDS")
        if requested is None or cls.device_control_env_var != "CUDA_VISIBLE_DEVICES":
            return original(cls, device_id)
        mask = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        # CPU-only Ray actors may have an empty mask. Preserve the upstream
        # metadata-only behavior; actual GPU workers have independent CUDA gates.
        if not mask:
            return original(cls, device_id)
        allowed, visible = parse_uuids(requested), parse_uuids(mask)
        if not set(visible).issubset(allowed):
            raise ValueError("vLLM UUID mask contains an unassigned task device")
        if not isinstance(device_id, int) or isinstance(device_id, bool) \
                or not 0 <= device_id < len(visible):
            raise ValueError("vLLM logical device index outside the UUID mask")
        return uuid_to_nvml_index(visible[device_id])

    setattr(physical_device, PATCH_TAG, True)
    platform.device_id_to_physical_device_id = classmethod(physical_device)


def patch_engine(module):
    """Engine subprocess masks need UUID tokens, NOT the NVML lookup indices."""
    original = module.get_device_indices
    if getattr(original, ENGINE_PATCH_TAG, False):
        return
    if getattr(sys.modules.get("vllm"), "__version__", None) != "0.18.0":
        raise RuntimeError("Engine UUID mask bridge requires pinned vLLM 0.18.0")

    def get_device_indices(device_control_env_var, local_dp_rank, world_size, local_world_size=None):
        requested = os.environ.get("RLVRAM_GPU_UUIDS")
        if requested is None or device_control_env_var != "CUDA_VISIBLE_DEVICES":
            return original(device_control_env_var, local_dp_rank, world_size, local_world_size)
        allowed = parse_uuids(requested)
        visible = parse_uuids(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
        if not set(visible).issubset(allowed):
            raise ValueError("Engine CUDA mask contains an unassigned task UUID")
        if local_world_size is None:
            local_world_size = world_size
        if (any(not isinstance(n, int) or isinstance(n, bool)
                for n in (local_dp_rank, world_size, local_world_size))
                or local_dp_rank < 0 or world_size <= 0 or local_world_size <= 0):
            raise ValueError("Invalid engine device slice")
        start = local_dp_rank * world_size
        end = start + local_world_size
        if end > len(visible):
            raise ValueError("Engine device slice outside the task UUID mask")
        return ",".join(visible[start:end])

    setattr(get_device_indices, ENGINE_PATCH_TAG, True)
    module.get_device_indices = get_device_indices


class _Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        creator = getattr(self.original, "create_module", None)
        return creator(spec) if creator is not None else None

    def exec_module(self, module):
        self.original.exec_module(module)
        if module.__name__ == MODULE:
            patch_platform(module)
        else:
            patch_engine(module)

    def __getattr__(self, name):
        return getattr(self.original, name)


class _Finder(importlib.abc.MetaPathFinder):
    _rlvram_uuid_bridge_finder = True

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in (MODULE, ENGINE_MODULE) or os.environ.get("RLVRAM_GPU_UUIDS") is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            raise ImportError("Pinned vLLM interface has no executable loader")
        spec.loader = _Loader(spec.loader)
        return spec


def install():
    if os.environ.get("RLVRAM_GPU_UUIDS") is None:
        return False
    parse_uuids(os.environ["RLVRAM_GPU_UUIDS"])
    if MODULE in sys.modules:
        patch_platform(sys.modules[MODULE])
    if ENGINE_MODULE in sys.modules:
        patch_engine(sys.modules[ENGINE_MODULE])
    if not any(getattr(finder, "_rlvram_uuid_bridge_finder", False) for finder in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
    return True


def import_training_paths(*, disposable_preflight=False):
    """Inspect imports and identities without constructing a trainer or model.

    Ordinary framework imports may initialize CUDA. Only the disposable
    preflight child and its assigned Ray actors opt into allowing this; their
    parent checks all allocation GPUs are idle after the child exits.
    """
    import importlib
    import torch

    allowed = parse_uuids(os.environ["RLVRAM_GPU_UUIDS"])
    mask = os.environ["CUDA_VISIBLE_DEVICES"]
    visible = parse_uuids(mask)
    if not set(visible).issubset(allowed):
        raise ValueError("Preflight CUDA mask contains an unassigned task UUID")
    install()
    before = torch.cuda.is_initialized()
    modules = (
        "verl.workers.engine_workers",
        "verl.workers.rollout.vllm_rollout.vllm_async_server",
        "verl.workers.rollout.vllm_rollout.utils",
        "vllm.lora.lora_model",
        "vllm.v1.worker.gpu_worker",
        ENGINE_MODULE,
    )
    module_sources, module_cuda_states = {}, []
    instrumented = Path(__file__).resolve().parents[1] / "instrumented_python_packages/verl"
    for name in modules:
        import_before = torch.cuda.is_initialized()
        try:
            module = importlib.import_module(name)
        finally:
            import_after = torch.cuda.is_initialized()
            module_cuda_states.append(dict(
                module=name, cuda_initialized_before=import_before,
                cuda_initialized_after=import_after,
                initialized_cuda=not import_before and import_after))
            # Keep the transition in the child log even if a later gate fails.
            if not import_before and import_after:
                print(f"RLVRAM_PREFLIGHT_CUDA_INIT module={name} "
                      f"disposable_preflight={disposable_preflight} training_launched=False",
                      flush=True)
        module_sources[name] = str(Path(module.__file__).resolve())
        if name.startswith("verl.") and not Path(module_sources[name]).is_relative_to(instrumented):
            raise RuntimeError("Preflight imported VERL outside the frozen instrumented source")
        if import_after != import_before and not disposable_preflight:
            raise RuntimeError("Runtime imports changed CUDA state outside a disposable preflight")
    after_imports = torch.cuda.is_initialized()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != mask \
            or parse_uuids(os.environ["RLVRAM_GPU_UUIDS"]) != allowed:
        raise ValueError("Runtime imports changed the preflight task UUID mask")
    from vllm.platforms import current_platform
    from vllm.v1.engine.utils import get_device_indices

    engine_mask = get_device_indices("CUDA_VISIBLE_DEVICES", 0, len(visible))
    if engine_mask != mask:
        raise RuntimeError("Engine changed UUID tokens into numeric CUDA masks")
    mappings = []
    for index, identity in enumerate(visible):
        nvml_index = current_platform.device_id_to_physical_device_id(index)
        actual = current_platform.get_device_uuid(index)
        if isinstance(actual, bytes):
            actual = actual.decode("ascii")
        if parse_uuids(actual) != [identity]:
            raise RuntimeError("vLLM's actual NVML UUID differs from the CUDA UUID mask")
        capability = current_platform.get_device_capability(index)
        if capability is None or (capability.major, capability.minor) != (8, 0):
            raise RuntimeError("Pinned A100 vLLM capability differs")
        mappings.append(dict(cuda_logical_index=index, uuid=identity, nvml_index=nvml_index))
    after_queries = torch.cuda.is_initialized()
    if after_queries != after_imports:
        raise RuntimeError("NVML compatibility queries created a CUDA context")
    return dict(imported_modules=list(modules), module_sources=module_sources,
                module_cuda_states=module_cuda_states,
                cuda_initializing_imports=[entry["module"] for entry in module_cuda_states
                                          if entry["initialized_cuda"]],
                disposable_preflight=disposable_preflight,
                uuid_nvml_mapping=mappings,
                engine_cuda_mask=engine_mask,
                cuda_initialized_before_imports=before,
                cuda_initialized_after_imports=after_imports,
                cuda_initialized_after_nvml_queries=after_queries,
                training_launched=False)
