import json
import struct

import pytest

from memory_tuner.model_memory_metadata import extract, read_header


def tensor_file(path, tensors):
    payload = bytearray()
    header = {}
    for name, shape in tensors.items():
        count = 1
        for dimension in shape:
            count *= dimension
        start = len(payload)
        payload.extend(b"\0" * (count * 2))
        header[name] = {"dtype": "BF16", "shape": shape,
                        "data_offsets": [start, len(payload)]}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


def test_extract_counts_all_linear_adapters_without_output_head(tmp_path):
    config = {
        "hidden_size": 4, "intermediate_size": 8, "num_hidden_layers": 1,
        "num_attention_heads": 2, "num_key_value_heads": 1, "vocab_size": 10,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    tensor_file(tmp_path / "model.safetensors", {
        "model.embed_tokens.weight": [10, 4],
        "model.layers.0.self_attn.q_proj.weight": [4, 4],
        "model.layers.0.input_layernorm.weight": [4],
        "model.layers.0.mlp.up_proj.weight": [8, 4],
        "lm_head.weight": [10, 4],
    })
    result = extract(tmp_path, "test/model", rank=2)
    assert result["floating_checkpoint_elements"] == 132
    assert result["largest_transformer_layer_elements"] == 52
    assert result["largest_weight_tensor_elements"] == 40
    assert result["non_transformer_layer_elements"] == 80
    assert result["all_linear_layer_adapter_elements"] == 2 * (4 + 4 + 8 + 4)
    assert result["head_dim"] == 2


def test_truncated_payload_is_not_valid_metadata(tmp_path):
    path = tmp_path / "model.safetensors"
    tensor_file(path, {"model.layers.0.weight": [4, 4]})
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="outside payload"):
        read_header(path)


def test_index_must_match_actual_tensor_names(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    path = tmp_path / "one.safetensors"
    tensor_file(path, {"actual": [2, 2]})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"different": "one.safetensors"}}))
    with pytest.raises(ValueError, match="incorrectly indexed"):
        extract(tmp_path, "test/model")


@pytest.mark.parametrize("raw", [b"", struct.pack("<Q", 21 * 1024 * 1024),
                                struct.pack("<Q", 20) + b"{}"])
def test_invalid_headers_fail(tmp_path, raw):
    path = tmp_path / "model.safetensors"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        read_header(path)
