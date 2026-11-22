import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import nusantara_adapter as recipe
from javanese_tools import build_prompt


def test_tool_prompts_are_the_training_prompts():
    row = {"id": "5", "label": "positive", **{code: f"{code} text" for code in recipe.LANGUAGES}}
    assert build_prompt("translate", " ind text ", source="ind", target="jav").prompt == \
        recipe.translation(row, "ind", "jav").prompt
    assert build_prompt("sentiment", "jav text", language="jav").prompt == recipe.sentiment(row, "jav").prompt
    assert build_prompt("triage", "jav text", language="jav").prompt == recipe.triage(row, "jav").prompt
    assert build_prompt("identify", "sun text", among=("ind", "jav", "sun")).prompt == \
        recipe.identification(row, "sun", ("ind", "jav", "sun")).prompt
    paragraph = [{"id": "1", "text": "Bal-balan iku rame.", "label": "sports"}]
    for task in ("topic", "emotion"):
        assert build_prompt(task, "Bal-balan iku rame.").prompt == \
            recipe.paragraph_examples([{**paragraph[0], "label": "sports" if task == "topic" else "happy"}], task)[0].prompt


def test_speech_level_prompts_match_training_for_every_level_and_direction():
    rows = [{"id": "u0", "group": "9", "level": 0, "jav": "Kowe mangan apa?", "ind": "Kamu makan apa?", "eng": ""},
            {"id": "u1", "group": "9", "level": 2, "jav": "Sampeyan nedha menapa?", "ind": "Kamu makan apa?", "eng": ""},
            {"id": "u2", "group": "9", "level": 3, "jav": "Panjenengan dhahar menapa?", "ind": "Kamu makan apa?",
             "eng": ""}]
    trained = {example.key: example for example in recipe.level_examples(rows)}
    assert build_prompt("level", "Kowe mangan apa?").prompt == \
        next(e for e in recipe.level_examples(rows[:1]) if e.key == "level_jav").prompt
    assert build_prompt("translate", "Kamu makan apa?", level="krama-alus").prompt == trained["ind_to_krama_alus"].prompt
    assert build_prompt("translate", "Kamu makan apa?", level="ngoko").prompt == trained["ind_to_ngoko"].prompt
    assert build_prompt("rewrite", "Kowe mangan apa?", level="krama").prompt == trained["ngoko_to_krama"].prompt
    assert build_prompt("rewrite", "Sampeyan nedha menapa?", level="ngoko").prompt == trained["krama_to_ngoko"].prompt


def test_tool_rejects_requests_the_adapter_was_not_trained_for():
    with pytest.raises(ValueError, match="Ngoko and Krama only"):
        build_prompt("rewrite", "Kowe mangan apa?", level="krama-alus")
    with pytest.raises(ValueError, match="Indonesian-to-Javanese"):
        build_prompt("translate", "How are you?", source="eng", target="jav", level="krama")
    with pytest.raises(ValueError, match="Give some text"):
        build_prompt("sentiment", "   ")
