"""Pure-Python behaviour of the opt-in VERL batch-shape helpers."""
import importlib.util
import json
import random

import pytest

from ieee_resub import common as C


def load():
    path = C.EXPERIMENT_DIR / "instrumented_python_packages/verl/utils/rlvram_shape_trace.py"
    spec = importlib.util.spec_from_file_location("shape_trace_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = load()


def batch(seed, n_prompts=6, rollouts=2, response=7):
    rng = random.Random(seed)
    prompts = [[rng.randrange(100) for _ in range(rng.randrange(3, 9))] for _ in range(n_prompts)]
    keys, ptoks, rtoks = [], [], []
    for p_index, prompt in enumerate(prompts):
        uid = f"{rng.getrandbits(64):x}-{p_index}"
        for session in range(rollouts):
            keys.append(f"{uid}_{session}_0")
            ptoks.append(prompt)
            rtoks.append([rng.randrange(100) for _ in range(response)])
    order = list(range(len(keys)))
    rng.shuffle(order)  # asynchronous completion order
    return [keys[i] for i in order], [ptoks[i] for i in order], [rtoks[i] for i in order]


def test_token_lists_accepts_lists_dense_and_jagged():
    torch = pytest.importorskip("torch")
    assert S.token_lists([[1, 2], [3]]) == [[1, 2], [3]]
    assert S.token_lists(torch.tensor([[1, 2], [3, 4]])) == [[1, 2], [3, 4]]
    jagged = torch.nested.as_nested_tensor([torch.tensor([1, 2, 3]), torch.tensor([4])], layout=torch.jagged)
    assert S.token_lists(jagged) == [[1, 2, 3], [4]]


def test_parse_key_and_rejects_unknown_format():
    assert S.parse_key("abc-def_1_0") == (1, 0)
    with pytest.raises(ValueError):
        S.parse_key("no-session")


def test_canonical_order_is_independent_of_completion_order_and_uids():
    a_keys, a_prompts, a_resp = batch(1)
    b_keys, b_prompts, b_resp = batch(1)
    # Same prompts, new random UIDs and a different completion order.
    rng = random.Random(99)
    order = list(range(len(b_keys)))
    rng.shuffle(order)
    b_keys = [b_keys[i].replace("-", "x", 1) for i in order]
    b_prompts = [b_prompts[i] for i in order]
    pa = S.canonical_permutation(a_keys, a_prompts)
    pb = S.canonical_permutation(b_keys, b_prompts)
    assert [a_prompts[i] for i in pa] == [b_prompts[i] for i in pb]
    assert [S.parse_key(a_keys[i]) for i in pa] == [S.parse_key(b_keys[i]) for i in pb]


def test_shape_record_hashes_and_microbatch_totals():
    keys, prompts, responses = batch(3)
    perm = S.canonical_permutation(keys, prompts)
    keys, prompts, responses = [keys[i] for i in perm], [prompts[i] for i in perm], [responses[i] for i in perm]
    tags = [dict(prompt_len=len(p), response_len=len(r)) for p, r in zip(prompts, responses)]
    record = S.shape_record(keys=keys, tags=tags, prompts=prompts, responses=responses, step=1, dp_size=2,
                            canonical_order=True, actor_micro_batch=3, log_prob_micro_batch=2)
    assert record["sequences"] == 12 and record["dp_size"] == 2
    assert set(record["dispatch_response_lengths"]) == {7}
    lengths = [len(p) + 7 for p in prompts]
    assert record["actor_micro_batch_tokens"] == [[sum(lengths[0:3]), sum(lengths[3:6])],
                                                  [sum(lengths[6:9]), sum(lengths[9:12])]]
    assert record["total_tokens"] == sum(lengths)
    other = [[t + 1 for t in r] for r in responses]  # different content, identical shapes
    record2 = S.shape_record(keys=keys, tags=tags, prompts=prompts, responses=other, step=1, dp_size=2,
                             canonical_order=True, actor_micro_batch=3, log_prob_micro_batch=2)
    assert S.shape_signature(record) == S.shape_signature(record2)
    assert record["response_tokens_sha256"] != record2["response_tokens_sha256"]


def test_shape_record_rejects_inconsistent_tags_and_indivisible_batches():
    keys, prompts, responses = batch(4)
    bad = [dict(prompt_len=len(p) + 1, response_len=len(r)) for p, r in zip(prompts, responses)]
    with pytest.raises(ValueError):
        S.shape_record(keys=keys, tags=bad, prompts=prompts, responses=responses, step=1, dp_size=2,
                       canonical_order=False, actor_micro_batch=2, log_prob_micro_batch=2)
    with pytest.raises(ValueError):
        S.chunk_totals([1, 2, 3], 2, None)
    with pytest.raises(ValueError):
        S.chunk_totals([1, 2, 3, 4], 2, 3)


def test_environment_flags(monkeypatch, tmp_path):
    monkeypatch.delenv(S.SHAPE_TRACE_ENV, raising=False)
    monkeypatch.delenv(S.CANONICAL_ORDER_ENV, raising=False)
    assert S.trace_path() is None and S.canonical_order_enabled() is False
    monkeypatch.setenv(S.CANONICAL_ORDER_ENV, "0")
    assert S.canonical_order_enabled() is False
    monkeypatch.setenv(S.CANONICAL_ORDER_ENV, "1")
    assert S.canonical_order_enabled() is True
    monkeypatch.setenv(S.SHAPE_TRACE_ENV, "relative.jsonl")
    with pytest.raises(ValueError):
        S.trace_path()
    path = tmp_path / "trace.jsonl"
    monkeypatch.setenv(S.SHAPE_TRACE_ENV, str(path))
    keys, prompts, responses = batch(5)
    tags = [{} for _ in keys]
    record = S.shape_record(keys=keys, tags=tags, prompts=prompts, responses=responses, step=1, dp_size=1,
                            canonical_order=True, actor_micro_batch=None, log_prob_micro_batch=None)
    S.append_record(S.trace_path(), record)
    assert S.read_records(path)[0]["dispatch_prompt_tokens_sha256"] == record["dispatch_prompt_tokens_sha256"]
    path.write_text(json.dumps(dict(schema_version=0)) + "\n")
    with pytest.raises(ValueError):
        S.read_records(path)
