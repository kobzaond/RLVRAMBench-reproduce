import pytest
from memory_tuner import grpo_raw_evidence as evidence

from memory_tuner.grpo_raw_evidence import (
    validate_lifecycle_events, validate_historical_source_horizon,
)


def spec(steps, save):
    return {"experiment_id": "test", "total_training_steps": steps,
            "save_freq": save, "test_freq": 20, "val_before_train": "True"}


def test_historical_horizon_requires_terminal_checkpoint_too():
    events = {"checkpoint": {80, 100}, "validation": {0, 20, 40, 60, 80, 100}}
    validate_lifecycle_events(spec(100, 80), events)
    events["checkpoint"].remove(100)
    with pytest.raises(ValueError, match="terminal checkpoint"):
        validate_lifecycle_events(spec(100, 80), events)


def test_fresh_source_requires_terminal_events_despite_longer_frequencies():
    validate_lifecycle_events(spec(1, 25),
                              {"checkpoint": {1}, "validation": {0, 1}})
    with pytest.raises(ValueError, match="checkpoint"):
        validate_lifecycle_events(spec(1, 25), {"validation": {0, 1}})


def test_revision_horizon_requires_all_cycles():
    events = {"checkpoint": {25, 50, 75, 100},
              "validation": {0, 20, 40, 60, 80, 100}}
    validate_lifecycle_events(spec(100, 25), events)
    events["validation"].remove(60)
    with pytest.raises(ValueError, match="validation"):
        validate_lifecycle_events(spec(100, 25), events)


def test_legacy_source_requires_logged_horizon_without_fabricating_phase_steps():
    log = "'total_training_steps': -1\n 'total_training_steps': 1\ntraining/global_step:1"
    assert validate_historical_source_horizon(log, {}, 1) == (
        "training_log_legacy_zero_phase_counter")
    assert validate_historical_source_horizon(log, {1: 30000}, 1) == (
        "training_log_and_phase")
    with pytest.raises(ValueError, match="one-step"):
        validate_historical_source_horizon(log + "\ntraining/global_step:2", {}, 1)
    with pytest.raises(ValueError, match="multiple steps"):
        validate_historical_source_horizon(log, {1: 30000, 2: 31000}, 1)
    with pytest.raises(ValueError, match="one-step"):
        validate_historical_source_horizon("training/global_step:1", {}, 1)


def test_exact_trial_selection_never_parses_a_corrupt_sibling(tmp_path, monkeypatch):
    directory = tmp_path / "output/panel/slot"
    directory.mkdir(parents=True)
    (directory / "trial-1.json").write_text("corrupt original")
    current = directory / "attempt-2/trial-2.json"
    current.parent.mkdir()
    current.write_text("{}")
    visited = []

    def inspect(path, annotations, *, root):
        assert root == tmp_path
        visited.append(path)
        raise RuntimeError("reached designated current attempt")

    monkeypatch.setattr(evidence, "trial_row", inspect)
    with pytest.raises(RuntimeError, match="designated current"):
        evidence.select_attempt(tmp_path, {"run_group": "panel", "experiment_id": "slot"},
                                exclusions={"1"}, annotations={}, trial_path=current)
    assert visited == [current]
