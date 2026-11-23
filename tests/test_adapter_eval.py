import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import nusantara_adapter as recipe
from adapter_eval import composite, parse_scales


def test_scales_must_be_distinct_strengths_up_to_full():
    assert parse_scales("1,0.75,0.5") == [1.0, 0.75, 0.5]
    for bad in ("", "0", "1.5", "0.5,0.5", "-0.2"):
        with pytest.raises(ValueError):
            parse_scales(bad)


def test_composite_matches_the_training_runs_selection_score():
    cells = {"ind_to_jav": {"after": {"chrf_pp": 60.0}}, "jav_to_ind": {"after": {"chrf_pp": 70.0}},
             "senti_jav": {"after": {"accuracy": 90.0}}, "triage_jav": {"after": {"both_labels_correct": 80.0}}}
    # Three families: translation (mean of 60 and 70), sentiment, triage. Each counts once.
    assert composite(recipe, cells) == recipe.selection_score(cells, "after", "composite") == \
        pytest.approx((65.0 + 90.0 + 80.0) / 3)
