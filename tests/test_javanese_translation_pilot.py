import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from javanese_translation_pilot import encode, partition, select_holdout


def test_parallel_examples_with_a_shared_reference_cannot_cross_splits():
    train = [{"id": "1", "ind": "Selamat pagi!", "jav": "Sugeng enjing!"}]
    test = [{"id": "2", "ind": "Selamat subuh.", "jav": " SUGENG   ENJING. "}]
    with pytest.raises(ValueError, match="overlap"):
        partition(train, test)


def test_partition_keeps_both_translation_directions_for_each_source_row():
    train = [{"id": "1", "ind": "Apa kabar?", "jav": "Piye kabare?"}]
    test = [{"id": "2", "ind": "Sampai nanti!", "jav": "Nganti mengko!"}]
    train_pairs, test_pairs = partition(train, test)
    assert [(p.row_id, p.target, p.reference) for p in train_pairs] == [
        ("1", "jav", "Piye kabare?"),
        ("1", "ind", "Apa kabar?"),
    ]
    assert {(p.row_id, p.reference) for p in test_pairs} == {
        ("2", "Nganti mengko!"),
        ("2", "Sampai nanti!"),
    }


def test_supervised_loss_masks_all_prompt_tokens_and_keeps_the_response():
    class CharacterTokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is False
            result = "system/user:" + messages[0]["content"] + "/assistant:"
            return result if add_generation_prompt else result + messages[1]["content"] + "#"

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return list(map(ord, text))

    tok = CharacterTokenizer()
    encoded = encode(tok, "Translate", "Piye?", max_length=64)
    prompt_length = len("system/user:Translate/assistant:")
    assert encoded["labels"] == [-100] * prompt_length + list(map(ord, "Piye?#"))
    assert encoded["input_ids"][-6:] == list(map(ord, "Piye?#"))
    with pytest.raises(ValueError, match="length"):
        encode(tok, "Translate", "Piye?", max_length=20)


def test_followup_evaluation_excludes_every_previously_scored_row():
    rows = [{"id": str(i)} for i in range(4)]
    selected = select_holdout(rows, 2, excluded_ids={"1", "3"})
    assert {row["id"] for row in selected} == {"0", "2"}
    with pytest.raises(ValueError, match="available"):
        select_holdout(rows, 3, excluded_ids={"1", "3"})
