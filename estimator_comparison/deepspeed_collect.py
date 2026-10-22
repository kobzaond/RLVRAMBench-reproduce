#!/usr/bin/env python3
"""Call DeepSpeed's published ZeRO model-state memory estimators for every model.

Run in the separate DeepSpeed environment (never needed by run_analyses.py):

    python3.11 -m venv .venv-deepspeed
    .venv-deepspeed/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
    DS_BUILD_OPS=0 .venv-deepspeed/bin/pip install deepspeed
    .venv-deepspeed/bin/python estimator_comparison/deepspeed_collect.py \
        --data benchmark --output estimator_comparison/deepspeed_estimates.json

The output JSON records the exact DeepSpeed/torch versions, the source files
and SHA-256 of the estimator modules, every call's arguments and return value
(bytes), and the text printed by the ``*_all_cold`` reporting functions.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import io
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("DS_ACCELERATOR", "cpu")

from estimator_comparison.estimator import load_architectures  # noqa: E402

GPU_COUNTS = (1, 2, 4)


def param_sets(arch: dict) -> dict:
    layers = int(arch["num_hidden_layers"])
    return {
        # (a) what the estimator assumes: every parameter trainable
        "full": {"total_params": int(arch["floating_checkpoint_elements"]),
                 "largest_layer_params": int(max(arch["largest_transformer_layer_elements"],
                                                 arch["largest_weight_tensor_elements"]))},
        # (b) only the rank-64 all-linear LoRA adapters trainable
        "lora": {"total_params": int(arch["all_linear_layer_adapter_elements"]),
                 "largest_layer_params": int(arch["all_linear_layer_adapter_elements"]) // layers},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "benchmark")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    import deepspeed
    import torch
    from deepspeed.runtime.zero import stage3, stage_1_and_2

    archs = load_architectures(args.data)
    sources = {}
    for name, module, fns in (("stage3", stage3, ("estimate_zero3_model_states_mem_needs",
                                                   "estimate_zero3_model_states_mem_needs_all_cold")),
                              ("stage_1_and_2", stage_1_and_2, ("estimate_zero2_model_states_mem_needs",
                                                                "estimate_zero2_model_states_mem_needs_all_cold"))):
        path = Path(inspect.getsourcefile(module))
        sources[name] = {
            "file": "deepspeed/" + path.relative_to(Path(deepspeed.__file__).parent).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "functions": {fn: {"first_line": inspect.getsourcelines(getattr(module, fn))[1],
                               "source": inspect.getsource(getattr(module, fn)) if not fn.endswith("all_cold") else None}
                          for fn in fns},
        }

    calls = []
    printed = {}
    for model_id in sorted(archs):
        for mode, params in param_sets(archs[model_id]).items():
            for gpus in GPU_COUNTS:
                base = dict(params, num_gpus_per_node=gpus, num_nodes=1)
                for offload, offload_params in ((False, False), (True, False), (True, True)):
                    cpu, gpu, largest = stage3.estimate_zero3_model_states_mem_needs(
                        params["total_params"], params["largest_layer_params"], num_gpus_per_node=gpus, num_nodes=1,
                        cpu_offload=offload, cpu_offload_params=offload_params, zero_init=True,
                        additional_buffer_factor=1.5)
                    calls.append({"model_id": model_id, "trainable": mode, "stage": 3, **base,
                                  "cpu_offload": offload, "cpu_offload_params": offload_params, "zero_init": True,
                                  "gpu_bytes": gpu, "cpu_bytes": cpu, "largest_layer_bytes": largest})
                for offload in (False, True):
                    cpu, gpu = stage_1_and_2.estimate_zero2_model_states_mem_needs(
                        params["total_params"], num_gpus_per_node=gpus, num_nodes=1, cpu_offload=offload,
                        additional_buffer_factor=1.5)
                    calls.append({"model_id": model_id, "trainable": mode, "stage": 2, **base,
                                  "cpu_offload": offload, "cpu_offload_params": None, "zero_init": None,
                                  "gpu_bytes": gpu, "cpu_bytes": cpu, "largest_layer_bytes": None})
                key = f"{model_id}|{mode}|{gpus}"
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    stage3.estimate_zero3_model_states_mem_needs_all_cold(
                        params["total_params"], params["largest_layer_params"], num_gpus_per_node=gpus, num_nodes=1)
                    stage_1_and_2.estimate_zero2_model_states_mem_needs_all_cold(
                        params["total_params"], num_gpus_per_node=gpus, num_nodes=1)
                printed[key] = buf.getvalue()

    record = {
        "deepspeed_version": deepspeed.__version__,
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "install": "DS_BUILD_OPS=0 pip install deepspeed; torch from https://download.pytorch.org/whl/cpu",
        "sources": sources,
        "param_sets": {m: param_sets(a) for m, a in sorted(archs.items())},
        "calls": calls,
        "all_cold_printed_output": printed,
    }
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"DeepSpeed {deepspeed.__version__}: wrote {len(calls)} estimator calls to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
