"""Process start-up hooks for the MI250X runs.

Loaded from PYTHONPATH by every Python process in the container, including
Ray and vLLM workers. Two post-import patches are installed:

1. ``verl.utils.debug``: VERL's stage timers also write the current stage to
   a marker file read by the external GPU sampler. Only the trainer driver
   enters these timers. The marker format matches memory_tuner/gpu_monitor.py:
   ``<unix_ns>,<step>,<phase>``.
2. ``verl.utils.vllm.utils``: VERL 0.9.0 loads in-memory LoRA tensors with the
   model's full ``hf_to_vllm_mapper``. vLLM 0.25 expects the *unstacked*
   mapper so that q/k/v and gate/up adapters are packed by the LoRA manager;
   otherwise packed layers receive one fused tensor and ``set_lora`` fails.
   The patch applies the mapper that vLLM's own file-based loader uses. It
   changes adapter loading only, not memory settings.
3. ``verl.utils.tokenizer.chat_template``: the agent loop probes the chat
   template with two consecutive user turns to find removable system-prompt
   and turn-separator tokens (needed for multi-turn tool loops only).
   Mistral's template rejects non-alternating roles; the probe then yields no
   tokens instead of aborting. Single-turn prompts are rendered unchanged.
"""
import os
import sys

_MARKER = os.environ.get("VERL_PHASE_MARKER")


def _patch_marked_timer(module) -> None:
    if not _MARKER:
        return
    import contextlib
    import time

    state = {"step": 0, "stack": []}

    def write(phase: str) -> None:
        temporary = f"{_MARKER}.{os.getpid()}"
        with open(temporary, "w") as handle:
            handle.write(f"{time.time_ns()},{state['step']},{phase}\n")
        os.replace(temporary, _MARKER)

    original = module.marked_timer

    @contextlib.contextmanager
    def marked_timer(name, timing_raw, *args, **kwargs):
        if name == "step":
            state["step"] += 1
        state["stack"].append(name)
        write(name)
        try:
            with original(name, timing_raw, *args, **kwargs):
                yield
        finally:
            state["stack"].pop()
            write(state["stack"][-1] if state["stack"] else "idle")

    module.marked_timer = marked_timer


def _patch_lora_loader(module) -> None:
    hijack = module.VLLMHijack.hijack

    def patched_hijack():
        hijack()
        from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager

        verl_load_adapter = LRUCacheWorkerLoRAManager._load_adapter

        def load_adapter(self, lora_request):
            model = self._adapter_manager.model
            mapper = getattr(model, "hf_to_vllm_mapper", None)
            if mapper is None or not hasattr(mapper, "get_unstacked_mapper"):
                return verl_load_adapter(self, lora_request)
            model.hf_to_vllm_mapper = mapper.get_unstacked_mapper()
            try:
                return verl_load_adapter(self, lora_request)
            finally:
                model.hf_to_vllm_mapper = mapper

        LRUCacheWorkerLoRAManager._load_adapter = load_adapter

    module.VLLMHijack.hijack = staticmethod(patched_hijack)


def _patch_chat_template_probe(module) -> None:
    def tolerant(probe):
        def wrapper(tokenizer, **kwargs):
            try:
                return probe(tokenizer, **kwargs)
            except Exception as error:  # jinja2 TemplateError
                if type(error).__name__ != "TemplateError":
                    raise
                return []
        return wrapper

    module.initialize_system_prompt = tolerant(module.initialize_system_prompt)
    module.initialize_turn_separator = tolerant(module.initialize_turn_separator)


_PATCHES = {"verl.utils.tokenizer.chat_template": _patch_chat_template_probe,
            "verl.utils.debug": _patch_marked_timer, "verl.utils.vllm.utils": _patch_lora_loader}


def _install() -> None:
    import importlib.abc
    import importlib.machinery

    class Loader(importlib.abc.Loader):
        def __init__(self, wrapped, patch):
            self.wrapped = wrapped
            self.patch = patch

        def create_module(self, spec):
            return self.wrapped.create_module(spec)

        def exec_module(self, module):
            self.wrapped.exec_module(module)
            self.patch(module)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            patch = _PATCHES.get(fullname)
            if patch is None:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is not None and spec.loader is not None:
                spec.loader = Loader(spec.loader, patch)
            return spec

    sys.meta_path.insert(0, Finder())


_install()
