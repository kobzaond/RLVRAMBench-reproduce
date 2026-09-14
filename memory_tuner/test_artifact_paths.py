import json
from pathlib import Path

import pytest

from memory_tuner.artifact_paths import ORIGINAL_ROOT, load_trial_record, recorded_path


def test_relocation_never_falls_back_to_original_existing_file(tmp_path):
    original = ORIGINAL_ROOT / "output/missing/training.log"
    assert recorded_path(str(original), tmp_path) == tmp_path / "output/missing/training.log"
    assert not recorded_path(str(original), tmp_path).exists()


def test_raw_trial_bytes_remain_immutable(tmp_path):
    path = tmp_path / "trial.json"
    text = json.dumps({"run_log": str(ORIGINAL_ROOT / "output/case/log"),
                       "response_length_trace": ""})
    path.write_text(text)
    trial = load_trial_record(path, tmp_path)
    assert trial["run_log"] == str(tmp_path / "output/case/log")
    assert trial["response_length_trace"] == ""
    assert path.read_text() == text


def test_rejects_unmapped_external_and_parent_paths(tmp_path):
    with pytest.raises(ValueError, match="outside artifact"):
        recorded_path("/arbitrary/external/record", tmp_path)
    with pytest.raises(ValueError, match="escapes artifact"):
        recorded_path("../record", tmp_path)
    with pytest.raises(ValueError, match="escapes artifact"):
        recorded_path(str(tmp_path / "../record"), tmp_path)
