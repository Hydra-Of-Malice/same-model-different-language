import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from adapter_safety_check import (capability_change, filter_prompts, pin_provider, safety_gates, scaled,
                                  split_by_preamble)

from langsafe.config import load_spec
from langsafe.eval.stats import mcnemar_exact, paired_diff_ci


def test_only_judges_without_a_provider_get_one_pinned():
    spec = pin_provider(load_spec(), "deepinfra")
    assert [judge["model"] for judge in spec["judging"]["judges"]] == ["CohereLabs/aya-expanse-32b:cohere",
                                                                     "Qwen/Qwen3-32B:deepinfra"]
    assert pin_provider(spec, "other") == spec


def test_prompts_are_filtered_to_the_requested_languages():
    lines = [json.dumps({"id": i, "lang": lang}) for i, lang in enumerate(["jv", "en", "th", "jv"])] + [""]
    kept = filter_prompts(lines, {"jv", "en"})
    assert [json.loads(line)["id"] for line in kept] == [0, 1, 3]
    with pytest.raises(ValueError, match="missing languages"):
        filter_prompts(lines, {"jv", "id"})


def test_preamble_conditions_never_share_a_table():
    rows = [{"id": "a", "preamble": "default"}, {"id": "a", "preamble": "none"}, {"id": "b", "preamble": "none"}]
    groups = split_by_preamble(rows)
    assert {key: [row["id"] for row in group] for key, group in groups.items()} == {"default": ["a"],
                                                                                   "none": ["a", "b"]}
    with pytest.raises(ValueError, match="Duplicate prompt IDs"):
        split_by_preamble([*rows, {"id": "b", "preamble": "none"}])


def test_gates_fail_on_more_refusal_or_a_significant_safety_loss():
    comparison = {
        "jv": {"over_refusal": {"delta": 0.021}, "content_safe": {"delta": -0.03, "ci": [-0.07, 0.01]}},
        "en": {"over_refusal": {"delta": 0.02}, "content_safe": {"delta": -0.05, "ci": [-0.09, -0.01]}},
    }
    assert safety_gates(comparison, 0.02) == {
        "jv": {"over_refusal_within_limit": False, "content_safe_not_significantly_lower": True},
        "en": {"over_refusal_within_limit": True, "content_safe_not_significantly_lower": False},
    }


def test_capability_change_is_paired_per_language():
    parent = {"languages": {"id": {"per_item": [1, 1, 1, 0]}, "en": {"per_item": [1, 0]}}}
    adapter = {"languages": {"id": {"per_item": [1, 0, 1, 0]}, "en": {"per_item": [1, 1]}}}
    change = capability_change(parent, adapter, 0.01, paired_diff_ci, mcnemar_exact)
    assert change["id"]["delta"] == -0.25 and change["id"]["drop_within_limit"] is False
    assert change["en"]["delta"] == 0.5 and change["en"]["drop_within_limit"] is True
    assert change["id"]["n"] == 4 and change["id"]["parent"] == 0.75 and change["id"]["adapter"] == 0.5


def test_scaling_an_adapter_touches_only_the_b_matrices():
    tensors = {"layers.0.q_proj.lora_A.weight": 2.0, "layers.0.q_proj.lora_B.weight": 4.0, "other": 3.0}
    assert scaled(tensors, 0.5) == {"layers.0.q_proj.lora_A.weight": 2.0, "layers.0.q_proj.lora_B.weight": 2.0,
                                    "other": 3.0}
