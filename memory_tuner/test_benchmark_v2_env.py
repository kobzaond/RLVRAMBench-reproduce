import csv

import pytest

from memory_tuner.benchmark_v2_env import environment


def write_matrix(path, **updates):
    row = {
        "experiment_id": "v2-test",
        "run_group": "benchmark_v2_test",
        "model": "model",
        "data_dir": "/data",
        "gpu_count": 2,
        "rollout_tp_size": 1,
        "actor_micro_batch": 4,
        "rollout_logprob_micro_batch": 4,
        "ref_logprob_micro_batch": 4,
        "vllm_gpu_memory_utilization": 0.5,
        "parameter_offload": True,
        "optimizer_offload": False,
        "free_cache_engine": True,
        "max_prompt_length": 512,
        "max_response_length": 512,
        "max_model_len": 1024,
        "max_num_seqs": 64,
        "rollout_n": 2,
        "train_batch_size": 32,
        "train_max_samples": 32,
        "val_max_samples": 32,
        "training_seed": 42,
        "total_training_steps": 1,
        "gpu_monitor_interval_ms": 100,
        "algorithm": "grpo",
    }
    row.update(updates)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def test_environment_maps_generic_matrix_row(tmp_path):
    matrix = tmp_path / "matrix.csv"
    write_matrix(matrix)
    values = environment(matrix, 0, expected_gpu_count=2)
    assert values["BENCHMARK_V2_ALGORITHM"] == "grpo"
    assert values["RUN_NAME"] == "benchmark_v2_test/v2-test"
    assert values["GPU_MONITOR_INTERVAL_MS"] == "100"


def test_environment_maps_optional_custom_reward(tmp_path):
    matrix = tmp_path / "matrix.csv"
    write_matrix(
        matrix,
        custom_reward_function_path="/project/code_reward.py",
        custom_reward_function_name="compute_score",
    )
    values = environment(matrix, 0, expected_gpu_count=2)
    assert (
        values["CUSTOM_REWARD_FUNCTION_PATH"]
        == "/project/code_reward.py"
    )
    assert values["CUSTOM_REWARD_FUNCTION_NAME"] == "compute_score"


def test_environment_allows_temporal_phase_overrides(tmp_path):
    matrix = tmp_path / "matrix.csv"
    write_matrix(
        matrix,
        capture_response_lengths=True,
        save_freq=80,
        test_freq=20,
        val_before_train=True,
    )
    values = environment(matrix, 0, expected_gpu_count=2)
    assert values["CAPTURE_RESPONSE_LENGTHS"] == "True"
    assert values["SAVE_FREQ"] == "80"
    assert values["TEST_FREQ"] == "20"
    assert values["VAL_BEFORE_TRAIN"] == "True"


def test_environment_rejects_wrong_gpu_count(tmp_path):
    matrix = tmp_path / "matrix.csv"
    write_matrix(matrix, gpu_count=4)
    with pytest.raises(ValueError, match="requests 4 GPUs"):
        environment(matrix, 0, expected_gpu_count=2)
