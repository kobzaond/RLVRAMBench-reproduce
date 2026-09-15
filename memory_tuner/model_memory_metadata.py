"""Extract numeric memory-model inputs without loading model weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct

from memory_tuner.resolve_hf_snapshot import resolve_snapshot

MODELS = (
    "Qwen/Qwen2.5-3B-Instruct",
    "microsoft/Phi-4-mini-instruct",
    "ibm-granite/granite-3.3-2b-instruct",
    "Qwen/Qwen2.5-7B-Instruct",
)
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8,
               "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
LAYER = re.compile(r"^model\.layers\.(\d+)\.")


def read_header(path):
    with Path(path).open("rb") as handle:
        size_bytes = handle.read(8)
        if len(size_bytes) != 8:
            raise ValueError("Truncated safetensors size header")
        size = struct.unpack("<Q", size_bytes)[0]
        if not 0 < size <= 20 * 1024 * 1024:
            raise ValueError("Invalid safetensors header size")
        raw = handle.read(size)
    if len(raw) != size:
        raise ValueError("Truncated safetensors JSON header")
    header = json.loads(raw)
    payload_bytes = Path(path).stat().st_size - 8 - size
    intervals = []
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        shape, dtype = tensor["shape"], tensor["dtype"]
        if dtype not in DTYPE_BYTES or not shape or any(
                not isinstance(n, int) or n <= 0 for n in shape):
            raise ValueError(f"Unsupported tensor metadata: {name}")
        start, end = tensor["data_offsets"]
        if not 0 <= start <= end <= payload_bytes:
            raise ValueError(f"Tensor outside payload: {name}")
        if end - start != math.prod(shape) * DTYPE_BYTES[dtype]:
            raise ValueError(f"Tensor shape/size mismatch: {name}")
        intervals.append((start, end))
    intervals.sort()
    if not intervals or intervals[0][0] != 0 or intervals[-1][1] != payload_bytes:
        raise ValueError("Tensor payload coverage differs")
    if any(left[1] != right[0] for left, right in zip(intervals, intervals[1:])):
        raise ValueError("Overlapping or incomplete tensor payload")
    return header, hashlib.sha256(size_bytes + raw).hexdigest()


def extract(snapshot, model_id, rank=64):
    snapshot = Path(snapshot)
    config_path = snapshot / "config.json"
    config = json.loads(config_path.read_text())
    index_path = snapshot / "model.safetensors.index.json"
    expected = (json.loads(index_path.read_text())["weight_map"]
                if index_path.is_file() else None)
    shards = sorted(set(expected.values())) if expected else ["model.safetensors"]
    tensors, header_hashes = {}, {}
    for shard in shards:
        if Path(shard).name != shard:
            raise ValueError("Unsafe checkpoint shard name")
        header, digest = read_header(snapshot / shard)
        header_hashes[shard] = digest
        for name, value in header.items():
            if name == "__metadata__":
                continue
            if name in tensors or (expected is not None and expected.get(name) != shard):
                raise ValueError("Duplicate or incorrectly indexed tensor")
            tensors[name] = value
    if expected is not None and set(tensors) != set(expected):
        raise ValueError("Checkpoint index/header coverage differs")
    floating = {name: value for name, value in tensors.items()
                if value["dtype"] in ("BF16", "F16", "F32", "F64")}
    layer_elements, adapters = {}, 0
    for name, value in floating.items():
        match = LAYER.match(name)
        if not match:
            continue
        layer = int(match[1])
        layer_elements[layer] = layer_elements.get(layer, 0) + math.prod(value["shape"])
        if name.endswith(".weight") and len(value["shape"]) == 2:
            adapters += rank * sum(value["shape"])
    if len(layer_elements) != config["num_hidden_layers"]:
        raise ValueError("Transformer-layer count disagrees with config")
    return {
        "model_id": model_id, "revision": snapshot.name,
        "floating_checkpoint_elements": sum(math.prod(t["shape"]) for t in floating.values()),
        "floating_checkpoint_bytes": sum(
            math.prod(t["shape"]) * DTYPE_BYTES[t["dtype"]] for t in floating.values()),
        "largest_transformer_layer_elements": max(layer_elements.values()),
        "largest_weight_tensor_elements": max(math.prod(t["shape"]) for t in floating.values()),
        "non_transformer_layer_elements": sum(
            math.prod(t["shape"]) for name, t in floating.items() if not LAYER.match(name)),
        "lora_rank": rank,
        "all_linear_layer_adapter_elements": adapters,
        "hidden_size": config["hidden_size"],
        "intermediate_size": config["intermediate_size"],
        "num_hidden_layers": config["num_hidden_layers"],
        "num_attention_heads": config["num_attention_heads"],
        "num_key_value_heads": config.get("num_key_value_heads", config["num_attention_heads"]),
        "head_dim": config.get("head_dim", config["hidden_size"] // config["num_attention_heads"]),
        "vocab_size": config["vocab_size"],
        "tie_word_embeddings": config.get("tie_word_embeddings", False),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "shard_header_sha256": header_hashes,
        "tensor_count": len(tensors),
        "scope": (
            "Numeric architecture/checkpoint-header inputs only; no weight payload. "
            "Adapter count assumes rank-64 all-linear transformer layers, excluding "
            "embeddings/output head. Floating checkpoint elements may include buffers; "
            "no separately allocated reference model is assumed."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {model: extract(resolve_snapshot(args.hf_home, model), model) for model in MODELS}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({model: {"elements": value["floating_checkpoint_elements"],
                             "adapter_elements": value["all_linear_layer_adapter_elements"]}
                      for model, value in result.items()}, indent=2))


if __name__ == "__main__":
    main()
