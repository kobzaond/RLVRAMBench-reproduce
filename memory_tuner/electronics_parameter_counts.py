"""Opt-in metadata evidence for adapter versus full-parameter experiments."""
from collections import Counter
import os
from pathlib import Path
import socket
import time

from memory_tuner.device_contract import write_new_json


def record_parameters(module, *, directory, forward_only, lora_rank, model_path):
    counts = Counter()
    for name, parameter in module.named_parameters():
        elements = parameter.numel()
        counts["total_elements"] += elements
        counts["requires_grad_elements"] += elements if parameter.requires_grad else 0
        counts["adapter_elements"] += elements if "lora_" in name else 0
        counts["bytes_" + str(parameter.dtype)] += elements * parameter.element_size()
    record = dict(counts, forward_only=bool(forward_only), lora_rank=int(lora_rank),
                  model_path=model_path, hostname=socket.gethostname(), pid=os.getpid(),
                  timestamp_ns=time.time_ns(), stage="before_fsdp_wrapping",
                  training_role=not forward_only,
                  scope="Parameter metadata; forward-only does not assert requires_grad=False.")
    role = "reference" if forward_only else "actor"
    write_new_json(Path(directory) / f"{role}-{os.getpid()}.json", record)
