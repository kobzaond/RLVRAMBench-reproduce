"""RLVRAMBench opt-in batch-shape evidence and canonical batch ordering.

Both features are disabled unless their environment variables are set:

- ``RLVRAM_SHAPE_TRACE``: absolute JSONL path. After sequence balancing, the
  trainer appends one record per training step with the prompt token IDs'
  hashes, the per-sequence prompt/response length vectors in dispatch order,
  and derived micro-batch token totals. Records are host-side metadata; no GPU
  operation, synchronization, or allocator call is involved.
- ``RLVRAM_CANONICAL_BATCH_ORDER=1``: before sequence balancing, the sampled
  batch is sorted by (prompt token IDs, rollout index, output index). The
  replay buffer otherwise returns trajectories in asynchronous completion
  order, and the balancer keeps indices in that order within each partition,
  so identical sequence lengths could still form different micro-batches.

The functions below are pure Python so that they can be unit-tested without
Ray, TransferQueue, or vLLM.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import socket
import time
from typing import Any, Iterable, Sequence

SHAPE_TRACE_ENV = "RLVRAM_SHAPE_TRACE"
CANONICAL_ORDER_ENV = "RLVRAM_CANONICAL_BATCH_ORDER"
SCHEMA_VERSION = 1
_KEY = re.compile(r"^(?P<uid>.+)_(?P<session>\d+)_(?P<output>\d+)$")


def trace_path() -> str | None:
    value = os.environ.get(SHAPE_TRACE_ENV, "").strip()
    if not value:
        return None
    if not os.path.isabs(value):
        raise ValueError(f"{SHAPE_TRACE_ENV} must be an absolute path")
    return value


def canonical_order_enabled() -> bool:
    value = os.environ.get(CANONICAL_ORDER_ENV, "").strip()
    if value in ("", "0", "False", "false"):
        return False
    if value in ("1", "True", "true"):
        return True
    raise ValueError(f"{CANONICAL_ORDER_ENV} must be 0/1 or True/False")


def token_lists(value: Any) -> list[list[int]]:
    """Convert a TransferQueue field (jagged, dense, or list) to token lists.

    TransferQueue stores each trajectory unpadded. A batch of equal-length
    entries is returned densely stacked; unequal entries are a jagged nested
    tensor. Padded dense tensors never occur for these fields.
    """
    if isinstance(value, (list, tuple)):
        return [[int(t) for t in (row.tolist() if hasattr(row, "tolist") else row)] for row in value]
    if getattr(value, "is_nested", False):
        return [[int(t) for t in row.tolist()] for row in value.unbind()]
    if hasattr(value, "dim") and value.dim() == 2:
        return [[int(t) for t in row] for row in value.tolist()]
    raise TypeError(f"Unsupported token container: {type(value)!r}")


def parse_key(key: str) -> tuple[int, int]:
    match = _KEY.fullmatch(key)
    if match is None:
        raise ValueError(f"Unexpected trajectory key format: {key!r}")
    return int(match["session"]), int(match["output"])


def canonical_permutation(keys: Sequence[str], prompts: Sequence[Sequence[int]]) -> list[int]:
    """Indices that sort trajectories independently of their random UIDs."""
    if len(keys) != len(prompts):
        raise ValueError("Keys and prompts differ in length")
    decorated = []
    for index, (key, prompt) in enumerate(zip(keys, prompts)):
        session, output = parse_key(key)
        decorated.append((tuple(int(t) for t in prompt), session, output, index))
    decorated.sort()
    return [item[-1] for item in decorated]


def _sha256(value: Any) -> str:
    payload = json.dumps(value, separators=(",", ":"), sort_keys=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def chunk_totals(lengths: Sequence[int], dp_size: int, micro_batch: int | None) -> list[list[int]]:
    """Token totals of contiguous per-rank micro-batches (non-dynamic batching)."""
    if dp_size < 1 or len(lengths) % dp_size:
        raise ValueError("Batch is not evenly divisible across data-parallel ranks")
    per_rank = len(lengths) // dp_size
    ranks = [list(lengths[r * per_rank:(r + 1) * per_rank]) for r in range(dp_size)]
    if not micro_batch:
        return [[sum(rank)] for rank in ranks]
    if per_rank % micro_batch:
        raise ValueError("Per-rank batch is not divisible by the micro-batch size")
    return [[sum(rank[i:i + micro_batch]) for i in range(0, per_rank, micro_batch)] for rank in ranks]


def shape_record(*, keys: Sequence[str], tags: Sequence[dict], prompts: Sequence[Sequence[int]],
                 responses: Sequence[Sequence[int]], step: int, dp_size: int,
                 canonical_order: bool, actor_micro_batch: int | None,
                 log_prob_micro_batch: int | None) -> dict:
    if not (len(keys) == len(tags) == len(prompts) == len(responses)) or not keys:
        raise ValueError("Shape trace inputs differ in length or are empty")
    prompt_lengths = [len(p) for p in prompts]
    response_lengths = [len(r) for r in responses]
    for tag, p, r in zip(tags, prompt_lengths, response_lengths):
        if "prompt_len" in tag and int(tag["prompt_len"]) != p:
            raise ValueError("TransferQueue prompt length tag differs from stored tokens")
        if "response_len" in tag and int(tag["response_len"]) != r:
            raise ValueError("TransferQueue response length tag differs from stored tokens")
    pairs = [[p, r] for p, r in zip(prompt_lengths, response_lengths)]
    sequence_lengths = [p + r for p, r in pairs]
    sessions = [parse_key(k)[0] for k in keys]
    prompt_lists = [[int(t) for t in p] for p in prompts]
    return {
        "schema_version": SCHEMA_VERSION,
        "event": "rlvram_shape_trace",
        "position": "after sequence balancing, before log-probability scoring",
        "invocation_id": os.environ.get("RLVRAM_INVOCATION_ID", ""),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "timestamp_ns": time.time_ns(),
        "step": int(step),
        "sequences": len(keys),
        "dp_size": int(dp_size),
        "canonical_order": bool(canonical_order),
        "padding_entries": sum(bool(t.get("is_padding", False)) for t in tags),
        "dispatch_prompt_tokens_sha256": _sha256(prompt_lists),
        "prompt_multiset_sha256": _sha256(sorted(prompt_lists)),
        "dispatch_length_pairs_sha256": _sha256(pairs),
        "length_multiset_sha256": _sha256(sorted(pairs)),
        "response_lengths_sha256": _sha256(response_lengths),
        "response_tokens_sha256": _sha256([[int(t) for t in r] for r in responses]),
        "dispatch_prompt_lengths": prompt_lengths,
        "dispatch_response_lengths": response_lengths,
        "dispatch_rollout_index": sessions,
        "actor_micro_batch": actor_micro_batch,
        "log_prob_micro_batch": log_prob_micro_batch,
        "actor_micro_batch_tokens": chunk_totals(sequence_lengths, dp_size, actor_micro_batch),
        "log_prob_micro_batch_tokens": chunk_totals(sequence_lengths, dp_size, log_prob_micro_batch),
        "total_tokens": sum(sequence_lengths),
    }


def append_record(path: str, record: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, separators=(",", ":"), sort_keys=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_records(path: str | Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"Blank shape-trace line {line_number}")
        record = json.loads(line)
        if record.get("schema_version") != SCHEMA_VERSION or record.get("event") != "rlvram_shape_trace":
            raise ValueError(f"Invalid shape-trace record at line {line_number}")
        records.append(record)
    return records


def shape_signature(record: dict) -> tuple[str, str]:
    """What must agree across conditions for identical actor tensor shapes."""
    return record["dispatch_prompt_tokens_sha256"], record["dispatch_length_pairs_sha256"]


__all__: Iterable[str] = (
    "SHAPE_TRACE_ENV", "CANONICAL_ORDER_ENV", "trace_path", "canonical_order_enabled",
    "token_lists", "parse_key", "canonical_permutation", "chunk_totals", "shape_record",
    "append_record", "read_records", "shape_signature",
)
