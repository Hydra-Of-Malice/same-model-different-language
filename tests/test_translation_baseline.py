import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import nusantara_adapter as recipe
from translation_baseline import CODES, directions, language_pair


def test_baseline_directions_follow_the_adapter_runs_within_nllb_coverage():
    assert directions(recipe, regional=False) == [("ind", "jav"), ("jav", "ind"), ("eng", "jav"), ("jav", "eng")]
    regional = directions(recipe, regional=True)
    assert ("ind", "sun") in regional and ("bug", "ind") in regional
    assert not {code for pair in regional for code in pair} & {"mad", "bbc", "nij"}
    assert {code for pair in regional for code in pair} <= set(CODES)


def test_language_codes_come_from_the_test_key():
    assert language_pair("ind_to_jav") == ("ind_Latn", "jav_Latn")
    assert language_pair("jav_to_ind_tweets") == ("jav_Latn", "ind_Latn")
