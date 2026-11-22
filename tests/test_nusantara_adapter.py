import json
import random
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from nusantara_adapter import (
    LABELS,
    LANGUAGES,
    LEVELS,
    Example,
    build,
    choice_cell,
    collate,
    encode,
    guard_splits,
    join_labels,
    length_batches,
    level_examples,
    mcnemar,
    parse_pairs,
    paragraph_examples,
    parse_triage,
    predicted_label,
    response_loss,
    prior_holdout_ids,
    schedule,
    selection_score,
    split_levels,
    subsample,
    triage,
    tweet_examples,
    unseen,
)


def row(index, **text):
    return {"id": str(index), "label": "neutral",
            **{code: text.get(code, f"{code} sentence {index}") for code in LANGUAGES}}


def test_sentiment_labels_join_through_the_indonesian_sentence():
    parallel = [{column: f"{column} {i}" for column, _ in LANGUAGES.values()} for i in range(2)]
    parallel[0]["indonesian"], parallel[1]["indonesian"] = " Bagus sekali! ", "Biasa saja."
    labelled = [{"text": "biasa saja", "label": "neutral"}, {"text": "Bagus sekali", "label": "positive"}]
    rows = join_labels(parallel, labelled)
    assert [(r["id"], r["label"], r["ind"]) for r in rows] == [("0", "positive", "Bagus sekali!"),
                                                             ("1", "neutral", "Biasa saja.")]
    with pytest.raises(ValueError, match="no sentiment label"):
        join_labels(parallel, labelled[:1])
    with pytest.raises(ValueError, match="two sentiment labels"):
        join_labels(parallel, [*labelled, {"text": "BIASA SAJA", "label": "negative"}])


def test_split_guard_drops_rows_that_share_a_sentence_with_a_later_split():
    train = [row(1), row(2, sun="Wilujeng enjing!")]
    valid = [row(3), row(4, nij="  SALAMAT hanjewu ")]
    test = [row(5, sun="wilujeng enjing"), row(6, nij="Salamat hanjewu.")]
    kept_train, kept_valid, dropped = guard_splits(train, valid, test)
    assert [r["id"] for r in kept_train] == ["1"]
    assert [r["id"] for r in kept_valid] == ["3"]
    assert dropped == {"train_rows_dropped": 1, "valid_rows_dropped": 1}
    with pytest.raises(ValueError, match="unique row IDs"):
        guard_splits([row(1), row(1)], valid, test)


def test_build_makes_every_requested_task_for_each_row():
    examples = build([row(7)], parse_pairs("ind-jav"), ("jav",), ("ind", "jav"))
    assert [(e.key, e.task, e.reference) for e in examples] == [
        ("ind_to_jav", "text", "jav sentence 7"),
        ("jav_to_ind", "text", "ind sentence 7"),
        ("senti_jav", "choice", "neutral"),
        ("lid_ind", "choice", "Indonesian"),
        ("lid_jav", "choice", "Javanese"),
    ]
    assert examples[2].choices == LABELS and examples[3].choices == ("Indonesian", "Javanese")
    assert examples[0].prompt.startswith("Translate the following Indonesian text into Javanese.")
    assert "one of: Indonesian, Javanese." in examples[3].prompt
    with pytest.raises(ValueError, match="Unknown language pair"):
        parse_pairs("ind-xyz")
    with pytest.raises(ValueError, match="listed twice"):
        parse_pairs("ind-jav,jav-ind")


def test_subsample_uses_one_nested_draw_of_rows_for_every_key():
    examples = build([row(i) for i in range(20)], parse_pairs("ind-jav,ind-sun"), ("jav",), ())
    picked = subsample(examples, 5, {"ind_to_jav", "senti_jav"}, 12)
    ids = {key: {e.row_id for e in picked if e.key == key} for key in {e.key for e in picked}}
    assert len(ids["ind_to_jav"]) == len(ids["senti_jav"]) == 12
    assert len(ids["ind_to_sun"]) == len(ids["jav_to_ind"]) == 5
    assert ids["ind_to_sun"] == ids["jav_to_ind"] < ids["ind_to_jav"] == ids["senti_jav"]
    assert len(subsample(examples, 50)) == len(examples)


def test_earlier_pilot_rows_are_two_disjoint_draws_of_120():
    test = [{"id": str(i)} for i in range(400)]
    scored = prior_holdout_ids(test)
    assert len(scored) == 240
    assert scored == prior_holdout_ids(test)


def test_training_batches_pad_on_the_right_and_mask_the_padding():
    class CharacterTokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            result = "user:" + messages[0]["content"] + "/assistant:"
            return result if add_generation_prompt else result + messages[1]["content"] + "#"

        def encode(self, text, *, add_special_tokens):
            return list(map(ord, text))

    tok = CharacterTokenizer()
    short, long = encode(tok, "a", "b", max_length=64), encode(tok, "a", "bcd", max_length=64)
    batch = collate([short, long], pad_id=0)
    assert batch["input_ids"][0][-2:] == [0, 0] and batch["labels"][0][-2:] == [-100, -100]
    assert batch["attention_mask"] == [[1] * len(short["input_ids"]) + [0, 0], [1] * len(long["input_ids"])]
    assert batch["labels"][1][-4:] == list(map(ord, "bcd#"))
    with pytest.raises(ValueError, match="maximum length"):
        encode(tok, "a", "b", max_length=5)


def test_length_batches_cover_every_example_once():
    lengths = [random.Random(3).randint(5, 90) for _ in range(203)]
    batches = length_batches(lengths, 8, random.Random(1))
    assert sorted(i for batch in batches for i in batch) == list(range(203))
    assert max(map(len, batches)) == 8
    spread = [max(lengths[i] for i in batch) - min(lengths[i] for i in batch) for batch in batches]
    assert sum(spread) / len(spread) < 10
    capped = length_batches(lengths, 8, random.Random(1), max_tokens=200)
    assert sorted(i for batch in capped for i in batch) == list(range(203))
    assert all(len(batch) * max(lengths[i] for i in batch) <= 200 for batch in capped)
    assert len(capped) > len(batches)


def test_schedule_warms_up_then_decays_to_a_tenth():
    assert schedule(0, 100, 10) == pytest.approx(0.1)
    assert schedule(9, 100, 10) == pytest.approx(1.0)
    assert schedule(10, 100, 10) == pytest.approx(1.0)
    assert schedule(100, 100, 10) == pytest.approx(0.1)
    assert schedule(0, 100, 0) == pytest.approx(1.0)
    values = [schedule(step, 100, 10) for step in range(10, 101)]
    assert values == sorted(values, reverse=True)


def test_an_answer_counts_only_when_it_names_exactly_one_label():
    assert predicted_label("Positive.", LABELS) == "positive"
    assert predicted_label("The sentiment is NEGATIVE", LABELS) == "negative"
    assert predicted_label("positive or negative", LABELS) is None
    assert predicted_label("Sentimen: positif", LABELS) is None
    assert predicted_label("Toba Batak", ("Toba Batak", "Javanese")) == "Toba Batak"
    assert predicted_label("Krama Alus", LEVELS) == "Krama Alus"
    assert predicted_label("ngoko", LEVELS) == "Ngoko"
    assert predicted_label("Ngoko Alus or Krama", LEVELS) is None


def level_row(index, group, level, jav):
    return {"id": f"u{index}", "group": str(group), "level": level, "jav": jav,
            "ind": f"kalimat {group}", "eng": f"sentence {group}"}


def test_speech_level_groups_stay_in_one_split_and_shared_sentences_are_dropped():
    rows = [level_row(2 * g + level // 2, g, level, f"ukara {g} tingkat {level}")
            for g in range(200) for level in (0, 2)]
    splits = split_levels(rows)
    groups = {name: {row["group"] for row in part} for name, part in splits.items()}
    assert not (groups["train"] & groups["valid"] or groups["train"] & groups["test"]
                or groups["valid"] & groups["test"])
    assert sum(map(len, splits.values())) == 400
    assert 20 <= len(groups["test"]) <= 60 and len(groups["train"]) > len(groups["test"]) > len(groups["valid"])
    leaked = splits["train"][0]["jav"]
    assert leaked not in {row["jav"] for row in split_levels(rows, held_out={leaked})["train"]}
    assert split_levels(rows) == splits


def test_speech_level_examples_cover_classification_translation_and_rewriting():
    rows = [level_row(0, 1, 0, "Kowe mangan apa?"), level_row(1, 1, 2, "Sampeyan nedha menapa?"),
            level_row(2, 2, 3, "Panjenengan dhahar menapa?")]
    examples = level_examples(rows)
    assert Counter(e.key for e in examples) == {"level_jav": 3, "ind_to_ngoko": 1, "ind_to_krama": 1,
                                                "ind_to_krama_alus": 1, "ngoko_to_krama": 1, "krama_to_ngoko": 1}
    rewrite = next(e for e in examples if e.key == "ngoko_to_krama")
    assert (rewrite.source, rewrite.reference, rewrite.row_id) == ("Kowe mangan apa?", "Sampeyan nedha menapa?", "u0")
    assert "at the Krama speech level" in rewrite.prompt and rewrite.task == "text"
    level = next(e for e in examples if e.key == "level_jav" and e.row_id == "u2")
    assert level.reference == "Krama Alus" and level.choices == LEVELS
    controlled = next(e for e in examples if e.key == "ind_to_krama_alus")
    assert controlled.prompt.endswith("\n\nkalimat 2") and controlled.reference == "Panjenengan dhahar menapa?"


def test_choice_metrics_compare_arms_on_the_same_items():
    examples = [Example("senti_jav", str(i), "p", label, "s", LABELS) for i, label in
                enumerate(["positive", "negative", "neutral", "positive"])]
    before = [("positive", False), ("positive", False), ("unsure", False), ("negative", False)]
    after = [("positive", False), ("negative", False), ("neutral", False), ("negative", False)]
    cell = choice_cell(examples, {"before": before, "after": after})
    assert cell["before"]["accuracy"] == 25.0 and cell["after"]["accuracy"] == 75.0
    assert cell["before"]["no_single_label"] == 1
    assert cell["accuracy_delta"] == 50.0
    assert cell["mcnemar_p"] == 0.5
    assert cell["after"]["macro_f1"] == pytest.approx(100 * (2 / 3 + 2 / 3 + 1) / 3)


def test_mcnemar_is_exact_and_two_sided():
    assert mcnemar([True, False], [True, False]) == 1.0
    assert mcnemar([False] * 6, [True] * 6) == pytest.approx(2 / 64)
    assert mcnemar([False] * 5 + [True], [True] * 5 + [False]) == pytest.approx(14 / 64)


def test_checkpoint_selection_prefers_the_javanese_directions():
    cells = {"ind_to_jav": {"after": {"chrf_pp": 50.0}}, "jav_to_ind": {"after": {"chrf_pp": 60.0}},
             "ind_to_sun": {"after": {"chrf_pp": 10.0}}, "senti_jav": {"after": {"accuracy": 90.0}}}
    assert selection_score(cells) == 55.0
    assert selection_score({"ind_to_sun": {"after": {"chrf_pp": 10.0}}}) == 10.0
    with pytest.raises(ValueError, match="translation key"):
        selection_score({"senti_jav": {"after": {"accuracy": 90.0}}})


def test_triage_asks_for_one_json_reply_and_tolerates_only_a_code_fence():
    example = triage({**row(3, jav="Apik tenan!", ind="Bagus sekali!"), "label": "positive"}, "jav")
    assert example.key == "triage_jav" and example.task == "text" and example.source == "Apik tenan!"
    assert json.loads(example.reference) == {"language": "Javanese", "sentiment": "positive",
                                             "indonesian": "Bagus sekali!"}
    assert parse_triage(example.reference) == json.loads(example.reference)
    assert parse_triage("```json\n" + example.reference + "\n```") == json.loads(example.reference)
    assert parse_triage('{"language": "Javanese", "sentiment": "positive"}') is None
    assert parse_triage('{"language": "Javanese", "sentiment": 1, "indonesian": "x"}') is None
    assert parse_triage("Here is the JSON: " + example.reference) is None
    assert parse_triage("[1, 2]") is None


def test_nusa_writes_examples_keep_tweets_apart_and_drop_held_out_text():
    tweets = tweet_examples([{"id": "indolem-train-1", "ind_text": " lagi apa? ", "tgt_text": "lagi ngapa?"}])
    assert [(e.key, e.row_id, e.source, e.reference) for e in tweets] == [
        ("ind_to_jav_tweets", "t:indolem-train-1", "lagi apa?", "lagi ngapa?"),
        ("jav_to_ind_tweets", "t:indolem-train-1", "lagi ngapa?", "lagi apa?"),
    ]
    topic = paragraph_examples([{"id": "7", "text": "Bal-balan iku rame.", "label": "sports"}], "topic")[0]
    assert (topic.key, topic.row_id, topic.reference, topic.task) == ("topic_jav", "p:7", "sports", "choice")
    assert "one of: food & beverages, sports" in topic.prompt
    assert predicted_label("Food & Beverages", topic.choices) == "food & beverages"
    with pytest.raises(ValueError, match="Unknown emotion label"):
        paragraph_examples([{"id": "7", "text": "x", "label": "sports"}], "emotion")
    assert unseen(tweets, {"lagi ngapa"}) == []
    assert unseen(tweets, {"liyane"}) == tweets
    assert unseen([topic], {"sports"}) == [topic] and unseen([topic], {"bal balan iku rame"}) == []


def test_response_loss_matches_the_usual_loss_without_prompt_logits():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)

    class Body(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(11, 6)

        def forward(self, input_ids, attention_mask, use_cache):
            return type("Output", (), {"last_hidden_state": self.embed(input_ids) * attention_mask[..., None]})()

    class LanguageModel(torch.nn.Module):
        logit_scale = 0.5

        def __init__(self):
            super().__init__()
            self.model, self.lm_head, self.calls = Body(), torch.nn.Linear(6, 11, bias=False), []
            self.lm_head.register_forward_hook(lambda module, inputs, output: self.calls.append(inputs[0].shape))

    lm = LanguageModel()
    batch = {"input_ids": torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 0, 0]]),
             "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]),
             "labels": torch.tensor([[-100, -100, 3, 4, 5], [-100, 7, 8, -100, -100]])}
    loss = response_loss(lm, batch)
    hidden = lm.model(batch["input_ids"], batch["attention_mask"], False).last_hidden_state
    full = lm.lm_head(hidden) * lm.logit_scale
    expected = torch.nn.functional.cross_entropy(full[:, :-1].reshape(-1, 11), batch["labels"][:, 1:].reshape(-1),
                                                 ignore_index=-100)
    assert loss.item() == pytest.approx(expected.item(), rel=1e-6)
    assert lm.calls[0] == (5, 6)
