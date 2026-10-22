"""Conservative diagnostic subclasses; a phase marker alone is insufficient."""
from __future__ import annotations

import re


def startup_budget_diagnostics(text):
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    values = re.findall(
        r"Free memory on device[^\n(]*\(([\d.]+)/([\d.]+) GiB\).*?"
        r"desired GPU memory utilization \(([\d.]+),\s*([\d.]+) GiB\)", text)
    return [dict(free_gib=float(free), cuda_capacity_gib=float(capacity),
                 reservation=float(reservation), requested_gib=float(requested))
            for free, capacity, reservation, requested in dict.fromkeys(values)]


def has_gpu_memory_evidence(text):
    """Require an explicit GPU diagnostic, not unrelated earlier CUDA messages."""
    lower = re.sub(r"\x1b\[[0-9;]*m", "", text).lower()
    return bool(
        re.search(r"cuda(?:\s+error)?\s*:?\s*out of memory", lower)
        or ("free memory on device" in lower and "desired gpu memory utilization" in lower)
        or "no available memory for the cache blocks" in lower
        or ("max seq len that can be stored in kv cache" in lower)
    )


def diagnose(text, terminal_phase):
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    lower = text.lower()
    if ("free memory on device" in lower and "desired gpu memory utilization" in lower):
        kind, witnesses = "startup_budget_rejection", ("free memory on device",)
    elif "out of memory" in lower and any(
        marker in lower for marker in ("actor_rollout_update_weights",
                                       "checkpoint_manager.update_weights(",
                                       "actor_wg.update_weights(")
    ):
        kind, witnesses = "weight_sync_oom", ("out of memory", "update_weights(",
                                            "actor_rollout_update_weights")
    elif ("out of memory" in lower and "cuda" in lower and terminal_phase == "actor_update"
          and any(marker in lower for marker in ("loss.backward(", "update_actor(", "update_policy("))):
        kind, witnesses = "actor_update_oom", ("out of memory", "loss.backward(", "update_actor(")
    elif ("out of memory" in lower and "cuda" in lower and terminal_phase == "rollout"
          and any(marker in lower for marker in ("enginecore", "async_llm.py", "generate_sequences("))):
        kind, witnesses = "rollout_oom", ("out of memory", "generate_sequences(")
    elif ("out of memory" in lower and "cuda" in lower and terminal_phase == "initializing"
          and any(marker in lower for marker in ("compile_or_warm_up_model", "engine core initialization failed"))):
        kind, witnesses = "startup_allocation_oom", ("out of memory", "compile_or_warm_up_model")
    else:
        return dict(diagnosis="unclassified", evidence_lines=[],
                    scope="No automatic inference of a GPU root cause from a phase alone")
    selected = [line.strip() for line in text.splitlines()
                if any(marker in line.lower() for marker in witnesses)]
    return dict(diagnosis=kind, evidence_lines=list(dict.fromkeys(selected))[:6],
                scope="Observed diagnostic and operation context; allocation ownership and fragmentation unproven")
