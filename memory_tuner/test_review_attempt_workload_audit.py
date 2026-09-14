import json

from memory_tuner.review_attempt_workload_audit import (
    attempt_inventory, extract_workload_log,
)


def specification(**changes):
    return {"train_max_samples": "2", "train_batch_size": "1",
            "max_response_length": "1024", **changes}


def test_tensor_width_clip_ratio_is_not_misreported_as_cap_fraction():
    text = (
        "dataset len: 100\nselected 2 random samples out of 100\nfilter dataset len: 2\n"
        "training/global_step:1 - actor/grad_norm:0.5 - "
        "response_length/mean:50 - response_length/max:64 - response_length/min:36 - "
        "response_length/clip_ratio:0.5 - prompt_length/mean:10 - prompt_length/max:12\n"
    )
    result = extract_workload_log(text, specification())
    assert result["cap_reaching_batches"] == 0
    assert result["selected_before_filter"] == 2
    assert result["filtered_train_examples"] == 0
    assert result["positive_gradient_steps"] == 1
    assert result["response_mean_over_logged_batches"] == 50


def test_logged_maximum_defines_cap_reaching_batch_not_a_response_fraction():
    result = extract_workload_log(
        "training/global_step:1 - response_length/max:64 - response_length/mean:50\n",
        specification(max_response_length="64"),
    )
    assert result["cap_reaching_batches"] == 1
    assert result["batches_with_length_metrics"] == 1
    assert result["original_train_examples"] is None


def test_fixed_order_derived_distinct_count_excludes_dropped_tail():
    text = "dataset len: 5\nfilter dataset len: 5\n" + "\n".join(
        f"training/global_step:{i}" for i in range(1, 6))
    result = extract_workload_log(
        text, specification(train_max_samples="-1", train_batch_size="2"))
    assert result["fixed_order_batches_per_epoch"] == 2
    assert result["fixed_order_distinct_train_examples_derived"] == 4
    assert result["fixed_order_wraps_derived"] == 2


def test_unknown_attempt_is_counted_without_inventing_a_paired_outcome(tmp_path):
    (tmp_path / "memory_tuner").mkdir()
    (tmp_path / "memory_tuner/attempt_exclusions.csv").write_text(
        "job_id,reason,evidence_note\n11,revision_startup_stall_timeout,unknown\n")
    folder = tmp_path / "output/group/slot"
    folder.mkdir(parents=True)
    path = folder / "trial-12.json"
    path.write_text(json.dumps({"exit_code": 0}))
    (folder / "attempt-11").mkdir()
    row = {
        "artifact_path": str(path), "experiment_id": "slot", "job_id": "12",
        "model_family": "model", "dataset": "math", "success": "1",
        "failure_kind": "", "condition": "external", "pair_id": "pair",
    }
    attempts, flow = attempt_inventory(tmp_path, {"instrumentation": [row]})
    assert len(attempts) == 2
    assert flow[0]["retained"] == 1
    assert flow[0]["excluded_no_trial_record"] == 1
    unknown = next(r for r in attempts if r["job_id"] == "11")
    assert unknown["outcome"] == "no_trial_record"
    assert unknown["failure_kind"] == ""


def test_empty_attempt_directory_is_not_required_when_allocation_records_it(tmp_path):
    (tmp_path / "memory_tuner").mkdir()
    (tmp_path / "memory_tuner/attempt_exclusions.csv").write_text(
        "job_id,reason,evidence_note\n11,revision_snapshot_missing_submodule,setup\n")
    folder = tmp_path / "output/group/slot"
    folder.mkdir(parents=True)
    path = folder / "trial-12.json"
    path.write_text(json.dumps({"exit_code": 0}))
    pairs = folder.parent / "pairs/pair"
    pairs.mkdir(parents=True)
    (pairs / "pair-11.json").write_text(json.dumps({
        "job_id": "11", "periods": [{
            "trial": "output/group/slot/attempt-11/trial-11.json",
        }],
    }))
    row = {
        "artifact_path": str(path), "experiment_id": "slot", "job_id": "12",
        "model_family": "model", "dataset": "math", "success": "1",
        "failure_kind": "", "condition": "external", "pair_id": "pair",
    }
    attempts, flow = attempt_inventory(tmp_path, {"instrumentation": [row]})
    assert len(attempts) == 2
    assert flow[0]["excluded_no_trial_record"] == 1
    unknown = next(r for r in attempts if r["job_id"] == "11")
    assert unknown["outcome"] == "no_trial_record"
    assert unknown["evidence_files"] == ""
    assert not (folder / "attempt-11").exists()
