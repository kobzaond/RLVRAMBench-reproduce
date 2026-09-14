import json
import math

import pytest

from memory_tuner.scientific_revision_results import number, render
from memory_tuner.standard_grpo_paper_analysis import json_safe


def test_missing_measurements_are_not_rendered_as_zero():
    assert number(math.nan) == "—"
    assert number(None) == "—"
    assert number(0) == "0"
    observed = json_safe({"values": [math.nan, math.inf, -math.inf, 0.0]})
    assert observed == {"values": [None, None, None, 0.0]}
    json.dumps(observed, allow_nan=False)


def test_incomplete_revision_cannot_generate_results():
    with pytest.raises(ValueError, match="incomplete"):
        render({"processes": 149, "all_frozen_rows_and_provenance_validated": False})
