# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "torch==2.8.0",
#   "transformers==5.22.1",
#   "accelerate==1.15.0",
#   "peft==0.21.2",
#   "huggingface-hub==2.4.0",
#   "sacrebleu==2.5.1",
# ]
# ///
"""Multi-task LoRA adapter for Javanese and neighbouring Indonesian languages on Tiny Aya.

Trains on the official NusaX train split (translation in both directions, sentiment and language identification) and,
on request, on Javanese speech levels from Unggah-Ungguh. Picks the epoch on the official validation split, and scores
the parent checkpoint and the adapter on identical test items. The parent is the same loaded model with the adapter
switched off.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import random
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

BASE = "HydraOfMalice/tiny-aya-earth-langsafe-circuit-merge-0p5"
BASE_REVISION = "ef47ebee4b9307bf23d728a171efd2b74e1dc789"
DATA_REVISION = "75b0be5d982348332509f296e10d9bf9cb66d60f"
DATA_URL = f"https://raw.githubusercontent.com/IndoNLP/nusax/{DATA_REVISION}/datasets"
DATA_SHA256 = {
    "mt/train.csv": "cf4276730de04f0cb3369c48b67d454f4aae52646315f19eb71b908f5dfaae0e",
    "mt/valid.csv": "fd148ce931676460bb89ce94205f5cea3fb0fbd1fc4b561a016dc25ceb8e049a",
    "mt/test.csv": "0082161de28a0a17a85fc6e02fc07a2e2514d8dd7f344ead95a88aa86dc706cc",
    "sentiment/indonesian/train.csv": "698c4802111085c14fbe35122957e1145338d307bcc295764026a23095a32984",
    "sentiment/indonesian/valid.csv": "4049c4a53017e08db72c72e1907345bfdfc19385481c58f54fe4637073d813f9",
    "sentiment/indonesian/test.csv": "43b612ed6fde972c752ab9fa8b2ce290d9ffa06b176b732fcef2a947cb94a161",
}
# NusaX column, English name. Indonesian is the pivot for every regional pair.
LANGUAGES = {
    "ind": ("indonesian", "Indonesian"), "jav": ("javanese", "Javanese"), "eng": ("english", "English"),
    "sun": ("sundanese", "Sundanese"), "ban": ("balinese", "Balinese"), "mad": ("madurese", "Madurese"),
    "min": ("minangkabau", "Minangkabau"), "bjn": ("banjarese", "Banjarese"), "ace": ("acehnese", "Acehnese"),
    "bug": ("buginese", "Buginese"), "bbc": ("toba_batak", "Toba Batak"), "nij": ("ngaju", "Ngaju"),
}
REGIONAL = ("sun", "ban", "mad", "min", "bjn", "ace", "bug", "bbc", "nij")
LABELS = ("positive", "neutral", "negative")
# Unggah-Ungguh (CC BY-NC 4.0): one meaning written at two to four Javanese speech levels, labelled 0 to 3.
LEVELS_REVISION = "20b8bba6a39bb4614cd5be0d65a261978d7f0def"
LEVELS_URL = ("https://huggingface.co/datasets/JavaneseHonorifics/Unggah-Ungguh/resolve/"
              f"{LEVELS_REVISION}/data/translation/train-00000-of-00001.csv")
LEVELS_SHA256 = "fadfdfe3473a5d2bb52112d93828e1bb762ed6d1d38dc3577d2928a7612f2e6c"
LEVELS = ("Ngoko", "Ngoko Alus", "Krama", "Krama Alus")
# NusaWrites (Apache-2.0): NusaTranslation is tweets translated from Indonesian by native speakers, NusaParagraph is
# paragraphs written in Javanese by native speakers. Both come with official splits.
WRITES_REVISION = "b337f511185cd44c9c57ff906ecef1af6165adbc"
WRITES_URL = f"https://raw.githubusercontent.com/IndoNLP/nusa-writes/{WRITES_REVISION}/data"
WRITES_SHA256 = {
    "nusa_kalimat-mt-jav-train.csv": "ecc3b6843ac72db58a620ed01cd5022caab28b4cc3f6968ad51d5a370dfe1160",
    "nusa_kalimat-mt-jav-valid.csv": "50bf70d54ff4f290e113d422e5acb1963312f02f8225f527e4d34d43a76b410b",
    "nusa_kalimat-mt-jav-test.csv": "88d7faaf96ffd4101a4819c1a91520eadacab62aa76c33705f60dd6e63bae753",
    "nusa_alinea-topic-jav-train.csv": "b7d4ec9f62a0c8c6eb6b6d29686e48777fd98984d77ac30cf09894e21c672fe8",
    "nusa_alinea-topic-jav-valid.csv": "5673327a5f636f5f8fb188cc5e549b6896babfa1ef0651d05ae07404710cea25",
    "nusa_alinea-topic-jav-test.csv": "92ae880038fa39ff6fdd1ba7b7e4ab33619009db6f3d9a5ba19fb639d21d8251",
    "nusa_alinea-emot-jav-train.csv": "d17f8035c6f430296a2e1f4e72c47743514ac7a9aa1abb21a1ddc13cfff97f65",
    "nusa_alinea-emot-jav-valid.csv": "ad9856f29ca150db85097d5c5cce9e9547b2028698ed13f2e3ce40ce2c14f27f",
    "nusa_alinea-emot-jav-test.csv": "8a84f425cdb3b223be449bc89f4149f52791e82b8b2e2aec38ce0d7e031c638a",
}
TOPICS = ("food & beverages", "sports", "leisures", "religion", "culture & heritage", "slice of life", "technology",
          "business")
EMOTIONS = ("happy", "sad", "angry", "fear", "surprise", "disgusted", "shame")
CORE = ("ind_to_jav", "jav_to_ind")
CONTROL = "eng_to_ind"
SEED = 20261122
MAX_LENGTH = 1024
NEW_TOKENS = {"text": 256, "choice": 8}


@dataclass(frozen=True)
class Example:
    key: str
    row_id: str
    prompt: str
    reference: str
    source: str
    choices: tuple[str, ...] = ()

    @property
    def task(self) -> str:
        return "choice" if self.choices else "text"


def normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def fetch(url: str, local: Path | None) -> bytes:
    if local is not None:
        return local.read_bytes()
    for attempt in range(4):
        try:
            with urlopen(url, timeout=30) as response:
                return response.read()
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def read_csv(relative: str, data_dir: Path | None, hashes: dict) -> list[dict]:
    raw = fetch(f"{DATA_URL}/{relative}", data_dir / relative if data_dir else None)
    hashes[relative] = hashlib.sha256(raw).hexdigest()
    if hashes[relative] != DATA_SHA256[relative]:
        raise ValueError(f"{relative} differs from NusaX revision {DATA_REVISION}")
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))


def join_labels(parallel: list[dict], labelled: list[dict]) -> list[dict]:
    """Attach the NusaX-senti label to each parallel row through its Indonesian sentence."""
    labels = {}
    for row in labelled:
        key = normalize(row["text"])
        if labels.setdefault(key, row["label"]) != row["label"]:
            raise ValueError("One Indonesian sentence carries two sentiment labels")
    rows = []
    for index, row in enumerate(parallel):
        label = labels.get(normalize(row["indonesian"]))
        if label not in LABELS:
            raise ValueError(f"Parallel row {index} has no sentiment label")
        rows.append({"id": str(index), "label": label,
                     **{code: row[column].strip() for code, (column, _) in LANGUAGES.items()}})
    return rows


def load_split(split: str, data_dir: Path | None, hashes: dict) -> list[dict]:
    return join_labels(read_csv(f"mt/{split}.csv", data_dir, hashes),
                       read_csv(f"sentiment/indonesian/{split}.csv", data_dir, hashes))


def guard_splits(train: list[dict], valid: list[dict], test: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """Drop training and validation rows that share a normalized sentence, in any language, with a later split."""
    def keys(rows):
        return {(code, normalize(row[code])) for row in rows for code in LANGUAGES}

    for name, rows in (("train", train), ("valid", valid), ("test", test)):
        if not rows or len({row["id"] for row in rows}) != len(rows):
            raise ValueError(f"{name} requires non-empty, unique row IDs")
        if any(not normalize(row[code]) for row in rows for code in LANGUAGES):
            raise ValueError(f"{name} contains an empty sentence")
    test_keys = keys(test)
    kept_valid = [row for row in valid if keys([row]).isdisjoint(test_keys)]
    later = test_keys | keys(kept_valid)
    kept_train = [row for row in train if keys([row]).isdisjoint(later)]
    if not kept_train or not kept_valid:
        raise ValueError("Split guard removed every training or validation row")
    return kept_train, kept_valid, {"train_rows_dropped": len(train) - len(kept_train),
                                    "valid_rows_dropped": len(valid) - len(kept_valid)}


def load_levels(local: Path | None, hashes: dict) -> list[dict]:
    raw = fetch(LEVELS_URL, local)
    hashes["unggah-ungguh/translation.csv"] = hashlib.sha256(raw).hexdigest()
    if hashes["unggah-ungguh/translation.csv"] != LEVELS_SHA256:
        raise ValueError(f"The speech-level file differs from Unggah-Ungguh revision {LEVELS_REVISION}")
    rows = [{"id": f"u{row['index']}", "group": row["group"], "level": int(row["label"]),
             "jav": row["javanese sentence"].strip(), "ind": row["indonesian sentence"].strip(),
             "eng": row["english sentence"].strip()}
            for row in csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))]
    if len({row["id"] for row in rows}) != len(rows) or any(not 0 <= row["level"] < len(LEVELS) for row in rows):
        raise ValueError("Speech-level rows need unique IDs and a level from 0 to 3")
    return rows


def split_levels(rows: list[dict], held_out: set[str] = frozenset()) -> dict[str, list[dict]]:
    """Split speech-level rows by sentence group: two tenths test, one tenth validation, the rest training.

    A group is one meaning at several levels, so its rows stay together. Training and validation rows that share a
    normalized sentence with a later split, or with `held_out`, are dropped.
    """
    splits: dict[str, list[dict]] = {"train": [], "valid": [], "test": []}
    for row in rows:
        bucket = int(hashlib.sha256(f"unggah-ungguh:{row['group']}".encode()).hexdigest(), 16) % 10
        splits["test" if bucket < 2 else "valid" if bucket == 2 else "train"].append(row)

    def sentences(group):
        return {normalize(row[code]) for row in group for code in ("jav", "ind", "eng")}

    later = sentences(splits["test"]) | held_out
    splits["valid"] = [row for row in splits["valid"] if sentences([row]).isdisjoint(later)]
    later |= sentences(splits["valid"])
    splits["train"] = [row for row in splits["train"] if sentences([row]).isdisjoint(later)]
    if not all(splits.values()):
        raise ValueError("A speech-level split is empty")
    return splits


def level_examples(rows: list[dict]) -> list[Example]:
    """Level classification, level-controlled translation from Indonesian, and Ngoko–Krama rewriting."""
    names = ", ".join(LEVELS)
    examples, groups = [], {}
    for row in rows:
        level = LEVELS[row["level"]]
        groups.setdefault(row["group"], {})[level] = row
        examples.append(Example("level_jav", row["id"],
                                "Identify the speech level of the following Javanese sentence. Answer with the level "
                                f"name only, one of: {names}.\n\n{row['jav']}", level, row["jav"], LEVELS))
        examples.append(Example(f"ind_to_{level.lower().replace(' ', '_')}", row["id"],
                                f"Translate the following Indonesian text into Javanese at the {level} speech level. "
                                f"Return only the translation.\n\n{row['ind']}", row["jav"], row["ind"]))
    for group in groups.values():
        for source, target in (("Ngoko", "Krama"), ("Krama", "Ngoko")):
            if source in group and target in group:
                examples.append(Example(f"{source.lower()}_to_{target.lower()}", group[source]["id"],
                                        f"Rewrite the following Javanese sentence at the {target} speech level. "
                                        f"Return only the rewritten sentence.\n\n{group[source]['jav']}",
                                        group[target]["jav"], group[source]["jav"]))
    return examples


def load_writes(name: str, split: str, local: Path | None, hashes: dict) -> list[dict]:
    file = f"{name}-jav-{split}.csv"
    raw = fetch(f"{WRITES_URL}/{file}", local / file if local else None)
    hashes[f"nusa-writes/{file}"] = hashlib.sha256(raw).hexdigest()
    if hashes[f"nusa-writes/{file}"] != WRITES_SHA256[file]:
        raise ValueError(f"{file} differs from NusaWrites revision {WRITES_REVISION}")
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))


def tweet_examples(rows: list[dict]) -> list[Example]:
    """Both translation directions for NusaTranslation rows. Keys end in `_tweets` to keep them apart from NusaX."""
    examples = []
    for row in rows:
        pair = {"id": f"t:{row['id']}", "ind": row["ind_text"].strip(), "jav": row["tgt_text"].strip()}
        for source, target in (("ind", "jav"), ("jav", "ind")):
            example = translation(pair, source, target)
            examples.append(Example(f"{example.key}_tweets", example.row_id, example.prompt, example.reference,
                                    example.source))
    return examples


def paragraph_examples(rows: list[dict], task: str) -> list[Example]:
    choices = TOPICS if task == "topic" else EMOTIONS
    names, examples = ", ".join(choices), []
    for row in rows:
        if row["label"] not in choices:
            raise ValueError(f"Unknown {task} label {row['label']!r}")
        text = row["text"].strip()
        examples.append(Example(f"{task}_jav", f"p:{row['id']}",
                                f"Classify the {task} of the following Javanese paragraph. Answer with the {task} only, "
                                f"one of: {names}.\n\n{text}", row["label"], text, choices))
    return examples


def unseen(examples: list[Example], held_out: set[str]) -> list[Example]:
    """Drop examples whose source or reference text also appears in a validation or test split."""
    return [example for example in examples
            if normalize(example.source) not in held_out
            and (example.choices or normalize(example.reference) not in held_out)]


def translation(row: dict, source: str, target: str) -> Example:
    prompt = (f"Translate the following {LANGUAGES[source][1]} text into {LANGUAGES[target][1]}. "
              f"Return only the translation.\n\n{row[source]}")
    return Example(f"{source}_to_{target}", row["id"], prompt, row[target], row[source])


def sentiment(row: dict, language: str) -> Example:
    prompt = (f"Classify the sentiment of the following {LANGUAGES[language][1]} text. "
              f"Answer with exactly one word: positive, neutral or negative.\n\n{row[language]}")
    return Example(f"senti_{language}", row["id"], prompt, row["label"], row[language], LABELS)


def identification(row: dict, language: str, choices: tuple[str, ...]) -> Example:
    names = ", ".join(LANGUAGES[code][1] for code in choices)
    prompt = (f"Identify the language of the following text. Answer with the language name only, one of: {names}."
              f"\n\n{row[language]}")
    return Example(f"lid_{language}", row["id"], prompt, LANGUAGES[language][1], row[language],
                   tuple(LANGUAGES[code][1] for code in choices))


def triage(row: dict, language: str) -> Example:
    """One structured reply for a customer message: its language, its sentiment and an Indonesian translation."""
    prompt = ('Read the following customer message. Reply with JSON only, using the keys "language", "sentiment" '
              f'(positive, neutral or negative) and "indonesian" (the message in Indonesian).\n\n{row[language]}')
    reference = json.dumps({"language": LANGUAGES[language][1], "sentiment": row["label"], "indonesian": row["ind"]},
                           ensure_ascii=False)
    return Example(f"triage_{language}", row["id"], prompt, reference, row[language])


def build(rows: list[dict], directions: list[tuple[str, str]], senti: tuple[str, ...],
          lid: tuple[str, ...], triaged: tuple[str, ...] = ()) -> list[Example]:
    examples = [translation(row, source, target) for row in rows for source, target in directions]
    examples += [sentiment(row, language) for row in rows for language in senti]
    examples += [identification(row, language, lid) for row in rows for language in lid]
    examples += [triage(row, language) for row in rows for language in triaged]
    return examples


def parse_pairs(text: str) -> list[tuple[str, str]]:
    directions = []
    for item in filter(None, (part.strip() for part in text.split(","))):
        left, _, right = item.partition("-")
        if left not in LANGUAGES or right not in LANGUAGES or left == right:
            raise ValueError(f"Unknown language pair {item!r}")
        directions += [(left, right), (right, left)]
    if len(set(directions)) != len(directions):
        raise ValueError("A language pair is listed twice")
    return directions


def parse_languages(text: str) -> tuple[str, ...]:
    codes = tuple(filter(None, (part.strip() for part in text.split(","))))
    if set(codes) - set(LANGUAGES) or len(set(codes)) != len(codes):
        raise ValueError(f"Unknown or repeated language in {text!r}")
    return codes


def prior_holdout_ids(test: list[dict]) -> set[str]:
    """Test rows the two earlier pilot runs scored, rebuilt from their seeded draws."""
    first = random.Random(SEED).sample(test, 120)
    seen = {row["id"] for row in first}
    second = random.Random(SEED).sample([row for row in test if row["id"] not in seen], 120)
    return seen | {row["id"] for row in second}


def subsample(examples: list[Example], rows: int, wide: set[str] = frozenset(), wide_rows: int = 0) -> list[Example]:
    """Keep one seeded draw of `rows` row IDs for every key, widened to `wide_rows` IDs for the keys in `wide`."""
    order = sorted({example.row_id for example in examples}, key=lambda row_id: (len(row_id), row_id))
    random.Random(SEED).shuffle(order)
    narrow, broad = set(order[:rows]), set(order[:max(rows, wide_rows)])
    return [example for example in examples if example.row_id in (broad if example.key in wide else narrow)]


def chat_prefix(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                         add_generation_prompt=True)


def encode(tokenizer, prompt: str, answer: str, max_length: int = MAX_LENGTH) -> dict:
    prefix = tokenizer.encode(chat_prefix(tokenizer, prompt), add_special_tokens=False)
    messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(prefix) >= len(ids) or ids[:len(prefix)] != prefix:
        raise ValueError("Rendered assistant response does not extend the generation prefix")
    if len(ids) > max_length:
        raise ValueError(f"Training example exceeds the maximum length of {max_length}")
    return {"input_ids": ids, "labels": [-100] * len(prefix) + ids[len(prefix):]}


def collate(batch: list[dict], pad_id: int) -> dict:
    width = max(len(item["input_ids"]) for item in batch)
    padded = {"input_ids": [], "labels": [], "attention_mask": []}
    for item in batch:
        gap = width - len(item["input_ids"])
        padded["input_ids"].append(item["input_ids"] + [pad_id] * gap)
        padded["labels"].append(item["labels"] + [-100] * gap)
        padded["attention_mask"].append([1] * len(item["input_ids"]) + [0] * gap)
    return padded


def length_batches(lengths: list[int], batch_size: int, rng: random.Random, max_tokens: int = 0) -> list[list[int]]:
    """Shuffled batches of similar length, so padding stays small.

    A batch holds at most `batch_size` examples and, when `max_tokens` is set, at most that many padded tokens.
    """
    order = list(range(len(lengths)))
    rng.shuffle(order)
    batches = []
    for start in range(0, len(order), batch_size * 64):
        batch = []
        for index in sorted(order[start:start + batch_size * 64], key=lengths.__getitem__):
            if batch and (len(batch) == batch_size or 0 < max_tokens < (len(batch) + 1) * lengths[index]):
                batches.append(batch)
                batch = []
            batch.append(index)
        batches.append(batch)
    rng.shuffle(batches)
    return batches


def response_loss(language_model, batch: dict):
    """Mean cross-entropy over response tokens, with logits computed for those positions only.

    The vocabulary has 262,144 entries, so logits for every prompt position cost far more memory than the layers do.
    """
    import torch

    hidden = language_model.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                  use_cache=False).last_hidden_state
    targets = batch["labels"][:, 1:]
    keep = targets != -100
    logits = language_model.lm_head(hidden[:, :-1][keep]) * language_model.logit_scale
    return torch.nn.functional.cross_entropy(logits.float(), targets[keep])


def schedule(step: int, total: int, warmup: int) -> float:
    """Linear warm-up, then cosine decay to one tenth of the peak."""
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def generate(model, tokenizer, examples: list[Example], batch_size: int) -> list[tuple[str, bool]]:
    import torch

    model.eval()
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    prompts = [chat_prefix(tokenizer, example.prompt) for example in examples]
    order = sorted(range(len(examples)), key=lambda i: (NEW_TOKENS[examples[i].task], -len(prompts[i])))
    outputs: list[tuple[str, bool] | None] = [None] * len(examples)
    with torch.inference_mode():
        start = 0
        while start < len(order):
            limit = NEW_TOKENS[examples[order[start]].task]
            group = [i for i in order[start:start + batch_size] if NEW_TOKENS[examples[i].task] == limit]
            tokens = tokenizer([prompts[i] for i in group], padding=True, return_tensors="pt",
                               add_special_tokens=False).to(model.device)
            result = model.generate(**tokens, max_new_tokens=limit, do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id, use_cache=True)
            for i, ids in zip(group, result[:, tokens.input_ids.shape[1]:].tolist()):
                reached_limit = len(ids) == limit and not (eos_ids & set(ids))
                outputs[i] = (tokenizer.decode(ids, skip_special_tokens=True).strip(), reached_limit)
            start += len(group)
    return outputs


def predicted_label(text: str, choices: tuple[str, ...]) -> str | None:
    """The single choice named in the answer, or None when the answer names none or several.

    A choice inside a longer named choice does not count: "Krama Alus" names one level, not two.
    """
    answer = f" {normalize(text)} "
    named = [choice for choice in choices if f" {normalize(choice)} " in answer]
    found = [choice for choice in named
             if not any(choice != other and f" {normalize(choice)} " in f" {normalize(other)} " for other in named)]
    return found[0] if len(found) == 1 else None


def mcnemar(before: list[bool], after: list[bool]) -> float:
    """Exact two-sided McNemar p-value on paired correctness."""
    gained = sum(new and not old for old, new in zip(before, after))
    lost = sum(old and not new for old, new in zip(before, after))
    n, k = gained + lost, min(gained, lost)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def bootstrap(deltas: list[float]) -> list[float]:
    rng = random.Random(SEED)
    means = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(2000))
    return [means[49], means[1949]]


def translation_cell(examples: list[Example], arms: dict[str, list[tuple[str, bool]]]) -> dict:
    from sacrebleu.metrics import BLEU, CHRF

    chrf = CHRF(word_order=2)
    refs = [example.reference for example in examples]
    cell, sentence = {"n": len(examples)}, {}
    for arm, outputs in arms.items():
        text = [output[0] for output in outputs]
        words = [normalize(value) for value in text]
        sentence[arm] = [chrf.sentence_score(hyp, [ref]).score for hyp, ref in zip(text, refs)]
        echo = [chrf.sentence_score(hyp, [example.source]).score for hyp, example in zip(text, examples)]
        cell[arm] = {
            "chrf_pp": chrf.corpus_score(text, [refs]).score,
            "bleu": BLEU().corpus_score(text, [refs]).score,
            "empty": sum(not value for value in words),
            "exact_reference": sum(value == normalize(ref) for value, ref in zip(words, refs)),
            "avoidable_source_echo": sum(
                value == normalize(example.source) and value != normalize(example.reference)
                for value, example in zip(words, examples)
            ),
            "closer_to_source_than_reference": sum(src > ref for src, ref in zip(echo, sentence[arm])),
            "generation_limit": sum(output[1] for output in outputs),
        }
    if {"before", "after"} <= set(arms):
        deltas = [new - old for old, new in zip(sentence["before"], sentence["after"])]
        cell["corpus_chrf_pp_delta"] = cell["after"]["chrf_pp"] - cell["before"]["chrf_pp"]
        cell["paired_mean_sentence_chrf_pp_delta"] = sum(deltas) / len(deltas)
        cell["paired_sentence_delta_95_ci"] = bootstrap(deltas)
    return cell


def choice_cell(examples: list[Example], arms: dict[str, list[tuple[str, bool]]]) -> dict:
    refs, choices = [example.reference for example in examples], examples[0].choices
    cell, correct = {"n": len(examples)}, {}
    for arm, outputs in arms.items():
        labels = [predicted_label(output[0], choices) for output in outputs]
        correct[arm] = [label == ref for label, ref in zip(labels, refs)]
        f1 = []
        for choice in sorted(set(refs)):
            hit = sum(label == choice == ref for label, ref in zip(labels, refs))
            predicted, actual = labels.count(choice), refs.count(choice)
            f1.append(2 * hit / (predicted + actual) if predicted + actual else 0.0)
        cell[arm] = {"accuracy": 100 * sum(correct[arm]) / len(refs), "macro_f1": 100 * sum(f1) / len(f1),
                     "no_single_label": labels.count(None)}
    if {"before", "after"} <= set(arms):
        deltas = [100.0 * (new - old) for old, new in zip(correct["before"], correct["after"])]
        cell["accuracy_delta"] = sum(deltas) / len(deltas)
        cell["accuracy_delta_95_ci"] = bootstrap(deltas)
        cell["mcnemar_p"] = mcnemar(correct["before"], correct["after"])
    return cell


def parse_triage(text: str) -> dict | None:
    """The reply as a JSON object with three string fields, or None. One Markdown code fence is tolerated."""
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text.strip(), flags=re.S | re.I)
    try:
        reply = json.loads(fenced.group(1) if fenced else text)
    except json.JSONDecodeError:
        return None
    fields = ("language", "sentiment", "indonesian")
    if not isinstance(reply, dict) or any(not isinstance(reply.get(field), str) for field in fields):
        return None
    return {field: reply[field] for field in fields}


def triage_cell(examples: list[Example], arms: dict[str, list[tuple[str, bool]]]) -> dict:
    from sacrebleu.metrics import CHRF

    chrf = CHRF(word_order=2)
    truth = [json.loads(example.reference) for example in examples]
    refs = [item["indonesian"] for item in truth]
    cell, correct, sentence = {"n": len(examples)}, {}, {}
    for arm, outputs in arms.items():
        replies = [parse_triage(output[0]) for output in outputs]
        language = [bool(reply) and normalize(reply["language"]) == normalize(item["language"])
                    for reply, item in zip(replies, truth)]
        feeling = [bool(reply) and normalize(reply["sentiment"]) == item["sentiment"]
                   for reply, item in zip(replies, truth)]
        text = [reply["indonesian"] if reply else "" for reply in replies]
        correct[arm] = [a and b for a, b in zip(language, feeling)]
        sentence[arm] = [chrf.sentence_score(hyp, [ref]).score for hyp, ref in zip(text, refs)]
        cell[arm] = {"valid_json": 100 * sum(map(bool, replies)) / len(replies),
                     "language_accuracy": 100 * sum(language) / len(replies),
                     "sentiment_accuracy": 100 * sum(feeling) / len(replies),
                     "both_labels_correct": 100 * sum(correct[arm]) / len(replies),
                     "chrf_pp": chrf.corpus_score(text, [refs]).score,
                     "generation_limit": sum(output[1] for output in outputs)}
    if {"before", "after"} <= set(arms):
        labels = [100.0 * (new - old) for old, new in zip(correct["before"], correct["after"])]
        deltas = [new - old for old, new in zip(sentence["before"], sentence["after"])]
        cell["both_labels_correct_delta"] = sum(labels) / len(labels)
        cell["both_labels_correct_delta_95_ci"] = bootstrap(labels)
        cell["mcnemar_p"] = mcnemar(correct["before"], correct["after"])
        cell["paired_mean_sentence_chrf_pp_delta"] = sum(deltas) / len(deltas)
        cell["paired_sentence_delta_95_ci"] = bootstrap(deltas)
    return cell


def evaluate(examples: list[Example], arms: dict[str, list[tuple[str, bool]]],
             only_rows: set[str] | None = None) -> dict:
    """Per-key metrics for each arm. `only_rows` restricts every key to those row IDs."""
    results = {}
    for key in dict.fromkeys(example.key for example in examples):
        indices = [i for i, example in enumerate(examples)
                   if example.key == key and (only_rows is None or example.row_id in only_rows)]
        if not indices:
            continue
        subset = [examples[i] for i in indices]
        picked = {arm: [outputs[i] for i in indices] for arm, outputs in arms.items()}
        cell = choice_cell if subset[0].choices else triage_cell if key.startswith("triage_") else translation_cell
        results[key] = cell(subset, picked)
    return results


def headline(arm: dict) -> float:
    """The one number that summarises a cell: accuracy for choices, both labels for triage, chrF++ otherwise."""
    for name in ("accuracy", "both_labels_correct", "chrf_pp"):
        if name in arm:
            return arm[name]
    raise KeyError("Cell has no headline metric")


def selection_score(cells: dict, arm: str = "after") -> float:
    """Mean validation chrF++ over the Javanese directions, or over every translation key without them."""
    keys = [key for key in CORE if key in cells] or [key for key in cells if "_to_" in key]
    if not keys:
        raise ValueError("Checkpoint selection needs at least one translation key")
    return sum(cells[key][arm]["chrf_pp"] for key in keys) / len(keys)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default="ind-jav,eng-jav", help="Translation pairs, trained in both directions")
    parser.add_argument("--regional", action="store_true",
                        help="Add every NusaX regional language: translation with Indonesian, sentiment and "
                             "language identification")
    parser.add_argument("--senti", default="jav", help="Languages for sentiment classification")
    parser.add_argument("--lid", default="", help="Languages for language identification")
    parser.add_argument("--triage", default="", help="Languages for structured message triage: language, sentiment "
                                                     "and an Indonesian translation in one JSON reply")
    parser.add_argument("--levels", action="store_true",
                        help="Add Javanese speech levels: classification, level-controlled translation and "
                             "Ngoko-Krama rewriting")
    parser.add_argument("--levels-file", type=Path, help="Local copy of the pinned Unggah-Ungguh translation CSV")
    parser.add_argument("--tweets", action="store_true",
                        help="Add NusaTranslation: Indonesian-Javanese tweets translated by native speakers")
    parser.add_argument("--paragraphs", action="store_true",
                        help="Add NusaParagraph: topic and emotion of paragraphs written in Javanese")
    parser.add_argument("--writes-dir", type=Path, help="Local copy of the pinned NusaWrites `data` directory")
    parser.add_argument("--base", default=BASE, help="Starting checkpoint on the Hub")
    parser.add_argument("--base-revision", default=BASE_REVISION, help="Full Hub commit hash of that checkpoint")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=int, help="LoRA alpha. Defaults to twice the rank")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--rslora", action="store_true")
    parser.add_argument("--modules", choices=("attention", "all"), default="all",
                        help="attention: q_proj and v_proj. all: every attention and MLP projection")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=float, default=0.05, help="Share of optimizer steps spent warming up")
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--batch-tokens", type=int, default=6000,
                        help="Most padded tokens in one training batch. 0 removes the limit")
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--train-rows", type=int, help="Use only the first N guarded training rows")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32, help="Generation batch size")
    parser.add_argument("--eval-rows", type=int, default=400, help="Test rows for the Javanese keys and the control")
    parser.add_argument("--eval-rows-other", type=int, default=200, help="Test rows for every other key")
    parser.add_argument("--valid-rows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=Path("runs/pilot/nusantara-adapter"))
    parser.add_argument("--data-dir", type=Path, help="Local copy of the pinned NusaX `datasets` directory")
    parser.add_argument("--private-repo", help="Private Hugging Face model repository for adapters and metrics")
    parser.add_argument("--run-name", default="run", help="Folder for this run inside the private repository")
    parser.add_argument("--compare-adapter", metavar="REPO@SHA", help="Also score an earlier adapter on the test rows")
    parser.add_argument("--scales", default="", help="Also score the adapter at these strengths below 1, e.g. 0.5,0.75")
    parser.add_argument("--device", choices=("cuda", "mps"), default="cuda", help="mps is for small local checks")
    parser.add_argument("--data-only", action="store_true", help="Validate and fingerprint splits without weights")
    args = parser.parse_args()
    positive = (args.epochs, args.rank, args.train_batch_size, args.accumulation, args.batch_size, args.eval_rows,
                args.eval_rows_other, args.valid_rows)
    if min(positive) < 1 or not 0 <= args.warmup < 1:
        parser.error("counts must be positive and warmup must be in [0, 1)")
    prior_repo, _, prior_revision = (args.compare_adapter or "").partition("@")
    if args.compare_adapter and (not prior_repo or not re.fullmatch(r"[0-9a-f]{40}", prior_revision)):
        parser.error("--compare-adapter requires REPO@SHA with a full Hub commit hash")
    if not re.fullmatch(r"[0-9a-f]{40}", args.base_revision):
        parser.error("--base-revision must be a full Hub commit hash")
    if args.compare_adapter and (args.base, args.base_revision) != (BASE, BASE_REVISION):
        parser.error("--compare-adapter only fits the default checkpoint, which the earlier adapters were trained on")

    directions = parse_pairs(args.pairs)
    senti, lid, triaged = parse_languages(args.senti), parse_languages(args.lid), parse_languages(args.triage)
    if args.regional:
        directions += [pair for code in REGIONAL for pair in (("ind", code), (code, "ind"))
                       if pair not in directions]
        senti = tuple(dict.fromkeys((*senti, "jav", "ind", *REGIONAL)))
        lid = tuple(dict.fromkeys((*lid, "ind", "jav", *REGIONAL)))
        triaged = tuple(dict.fromkeys((*triaged, "jav", *REGIONAL)))
    if "ind" in triaged:
        parser.error("Triage translates into Indonesian, so Indonesian cannot be a triage language")
    if (CONTROL.split("_to_")[0], CONTROL.split("_to_")[1]) in directions:
        parser.error("English to Indonesian is the untrained control and cannot be a training direction")
    if len(lid) == 1:
        parser.error("Language identification needs at least two languages")
    scales = [float(value) for value in filter(None, args.scales.split(","))]
    if any(not 0 < scale < 1 for scale in scales) or len(set(scales)) != len(scales):
        parser.error("--scales takes distinct strengths between 0 and 1")

    hashes: dict[str, str] = {}
    splits = {split: load_split(split, args.data_dir, hashes) for split in ("train", "valid", "test")}
    train_rows, valid_rows, dropped = guard_splits(splits["train"], splits["valid"], splits["test"])
    if args.train_rows:
        train_rows = train_rows[:args.train_rows]
    test_rows = splits["test"]
    scored_before = prior_holdout_ids(test_rows)
    javanese_keys = {key for key in (*(f"{a}_to_{b}" for a, b in directions), "senti_jav", "lid_jav", "triage_jav",
                                     CONTROL) if "jav" in key or key == CONTROL}
    levels = None
    if args.levels:
        def sentences(rows):
            return {normalize(row[code]) for row in rows for code in ("ind", "jav", "eng")}

        levels = split_levels(load_levels(args.levels_file, hashes), sentences(test_rows) | sentences(valid_rows))
        level_test = sentences(levels["test"]) | sentences(levels["valid"])
        kept = [row for row in train_rows if sentences([row]).isdisjoint(level_test)]
        dropped["train_rows_dropped_for_speech_level_overlap"] = len(train_rows) - len(kept)
        train_rows = kept
    train = build(train_rows, directions, senti, lid, triaged)
    valid = subsample(build(valid_rows, [pair for pair in directions if "_to_".join(pair) in CORE] or directions,
                            tuple(code for code in senti if code == "jav"), ()), args.valid_rows)
    test = subsample(build(test_rows, [*directions, ("eng", "ind")], senti, lid, triaged), args.eval_rows_other,
                     javanese_keys, args.eval_rows)
    if levels:
        train += level_examples(levels["train"][:4 * args.train_rows] if args.train_rows else levels["train"])
        valid += subsample(level_examples(levels["valid"]), args.valid_rows)
        test += subsample(level_examples(levels["test"]),
                          len(levels["test"]) if args.eval_rows >= len(test_rows) else args.eval_rows)
    writes = {}
    for name, wanted, make in (("nusa_kalimat-mt", args.tweets, tweet_examples),
                               ("nusa_alinea-topic", args.paragraphs, lambda rows: paragraph_examples(rows, "topic")),
                               ("nusa_alinea-emot", args.paragraphs, lambda rows: paragraph_examples(rows, "emotion"))):
        if wanted:
            writes[name] = {split: make(load_writes(name, split, args.writes_dir, hashes))
                            for split in ("train", "valid", "test")}
    if writes:   # NusaWrites reuses text across its tasks and splits, so guard every source against every held-out text
        extra_valid = [example for parts in writes.values() for example in subsample(parts["valid"], args.valid_rows)]
        extra_test = [example for parts in writes.values() for example in subsample(parts["test"], args.eval_rows)]
        held_out = {normalize(text) for parts in writes.values() for split in ("valid", "test")
                    for example in parts[split] for text in (example.source, example.reference)}
        held_out |= {normalize(text) for example in (*valid, *test) for text in (example.source, example.reference)}
        extra_test = unseen(extra_test, {normalize(e.source) for e in valid + extra_valid})
        extra_train = [example for parts in writes.values()
                       for example in (parts["train"][:8 * args.train_rows] if args.train_rows else parts["train"])]
        kept = unseen(extra_train, held_out)
        dropped["nusa_writes_train_examples_dropped"] = len(extra_train) - len(kept)
        before = len(train)
        train = unseen(train, {normalize(text) for example in (*extra_valid, *extra_test)
                               for text in (example.source, example.reference)})
        dropped["train_examples_dropped_for_nusa_writes_overlap"] = before - len(train)
        train, valid, test = train + kept, valid + extra_valid, test + extra_test
    provenance = {
        "source": DATA_URL, "revision": DATA_REVISION, "sha256": hashes, **dropped,
        "train_rows": len(train_rows), "valid_rows": len(valid_rows), "test_rows": len(test_rows),
        "speech_levels": {"source": LEVELS_URL, "revision": LEVELS_REVISION,
                          "rows": {split: len(rows) for split, rows in levels.items()}} if levels else None,
        "nusa_writes": {"source": WRITES_URL, "revision": WRITES_REVISION, "files": sorted(writes)} if writes else None,
        "test_rows_scored_by_earlier_pilots": len(scored_before),
        "train_examples": dict(Counter(example.key for example in train)),
        "valid_examples": dict(Counter(example.key for example in valid)),
        "test_examples": dict(Counter(example.key for example in test)),
    }
    print(json.dumps({"data": provenance}), flush=True)
    if args.data_only:
        return

    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from peft import LoraConfig, get_peft_model
    from peft.helpers import rescale_adapter_scale
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    if args.device == "cuda" and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        raise RuntimeError("Training requires a CUDA GPU with BF16 support")
    if args.output.exists():
        raise FileExistsError(f"Choose a fresh output path: {args.output}")
    api = HfApi() if args.private_repo else None
    if api:
        api.create_repo(args.private_repo, repo_type="model", private=True, exist_ok=True)
        if not api.model_info(args.private_repo).private:
            raise RuntimeError("Adapter outputs require a private model repository")

    started = time.monotonic()
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base, revision=args.base_revision)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.base, revision=args.base_revision, dtype=dtype,
                                                device_map={"": 0 if args.device == "cuda" else "mps"},
                                                attn_implementation="sdpa")
    modules = ["q_proj", "v_proj"] if args.modules == "attention" else [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    alpha = args.alpha or 2 * args.rank
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=args.rank, lora_alpha=alpha,
                                            lora_dropout=args.dropout, use_rslora=args.rslora,
                                            target_modules=modules, revision=args.base_revision))
    checkpointing = args.gradient_checkpointing
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)

    encoded, too_long = [], Counter()
    for example in train:
        try:
            encoded.append(encode(tokenizer, example.prompt, example.reference))
        except ValueError as error:
            if "maximum length" not in str(error):
                raise
            too_long[example.key] += 1
    lengths = [len(item["input_ids"]) for item in encoded]
    planned = len(length_batches(lengths, args.train_batch_size, random.Random(args.seed), args.batch_tokens))
    total_steps = math.ceil(planned / args.accumulation) * args.epochs   # later epochs can differ by a few batches
    warmup_steps = int(total_steps * args.warmup)
    print(json.dumps({"trainable_parameters": sum(param.numel() for param in trainable),
                      "training_examples": len(encoded), "skipped_too_long": dict(too_long),
                      "mean_tokens": sum(lengths) / len(lengths), "max_tokens": max(lengths),
                      "optimizer_steps": total_steps}), flush=True)

    rng = random.Random(args.seed)
    language_model = model.get_base_model()   # the same modules, adapter layers included
    history, best, step = [], None, 0
    for epoch in range(args.epochs):
        model.train()
        batches = length_batches(lengths, args.train_batch_size, rng, args.batch_tokens)
        losses = []
        for index, batch in enumerate(batches):
            tensors = {name: torch.tensor(values, device=model.device)
                       for name, values in collate([encoded[i] for i in batch], tokenizer.pad_token_id).items()}
            for attempt in range(2):
                try:
                    loss = response_loss(language_model, tensors)
                    (loss / args.accumulation).backward()
                    break
                except torch.OutOfMemoryError:
                    if checkpointing or attempt:
                        raise
                    loss = None   # drop the failed graph, keep gradients already accumulated from earlier batches
                    torch.cuda.empty_cache()
                    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                    checkpointing = True
                    print("Out of GPU memory: continuing with gradient checkpointing.", flush=True)
            if not loss.isfinite().item():
                raise RuntimeError(f"Non-finite training loss in epoch {epoch + 1}, batch {index + 1}")
            losses.append(loss.item())
            if (index + 1) % args.accumulation == 0 or index + 1 == len(batches):
                for group in optimizer.param_groups:
                    group["lr"] = args.learning_rate * schedule(step, total_steps, warmup_steps)
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step == 1 or step % 25 == 0:
                    recent = losses[-25 * args.accumulation:]
                    print(f"Epoch {epoch + 1}, step {step}/{total_steps}, mean response loss "
                          f"{sum(recent) / len(recent):.4f}", flush=True)
        cells = evaluate(valid, {"after": generate(model, tokenizer, valid, args.batch_size)})
        score = selection_score(cells)
        history.append({"epoch": epoch + 1, "mean_response_loss": sum(losses) / len(losses),
                        "minutes": (time.monotonic() - started) / 60,
                        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2 ** 30 if args.device == "cuda" else None,
                        "validation_score": score,
                        "validation": {key: headline(cell["after"]) for key, cell in cells.items()}})
        print(json.dumps(history[-1]), flush=True)
        if best is None or score > best[0]:
            best = (score, epoch + 1, {name: value.detach().cpu().clone()
                                       for name, value in model.state_dict().items() if "lora_" in name})
    model.load_state_dict(best[2], strict=False)
    if checkpointing:
        model.gradient_checkpointing_disable()
    training_minutes = (time.monotonic() - started) / 60

    args.output.mkdir(parents=True)
    model.save_pretrained(args.output)
    card = (
        f"---\nlicense: cc-by-nc-4.0\nbase_model: {args.base}\npipeline_tag: translation\n---\n\n"
        "# Private Javanese and regional-language adapter\n\n"
        "Research adapter for translation, classification and structured message triage in Javanese and "
        "neighbouring Indonesian languages. Safety and general capability have yet to be assessed.\n\n"
        f"Pinned starting checkpoint: {args.base}@{args.base_revision}. Training used the native-translated "
        "[NusaX](https://github.com/IndoNLP/nusax) corpus (CC BY-SA 4.0), "
        f"revision `{DATA_REVISION}`, reformatted into instructions. "
        + ("It also used the [Unggah-Ungguh](https://huggingface.co/datasets/JavaneseHonorifics/Unggah-Ungguh) "
           f"speech-level corpus (CC BY-NC 4.0), revision `{LEVELS_REVISION}`. " if levels else "")
        + ("It also used [NusaWrites](https://github.com/IndoNLP/nusa-writes) (Apache-2.0), "
           f"revision `{WRITES_REVISION}`. " if writes else "")
        + "The adapter inherits the model's non-commercial license and acceptable use policy.\n"
    )
    (args.output / "README.md").write_text(card, encoding="utf-8")
    if api:
        api.upload_folder(repo_id=args.private_repo, repo_type="model", folder_path=args.output,
                          path_in_repo=args.run_name,
                          allow_patterns=["adapter_config.json", "adapter_model.safetensors", "README.md"])
        print(f"Uploaded the epoch-{best[1]} adapter to {args.private_repo}/{args.run_name}.", flush=True)

    print(f"Evaluating the adapter and the parent on {len(test)} test examples.", flush=True)
    arms = {"after": generate(model, tokenizer, test, args.batch_size)}
    with model.disable_adapter():
        arms["before"] = generate(model, tokenizer, test, args.batch_size)
    for scale in scales:   # a weaker adapter trades task gain for closeness to the parent, with no further training
        with rescale_adapter_scale(model, scale):
            arms[f"scale_{scale:g}"] = generate(model, tokenizer, test, args.batch_size)
    if prior_repo:   # last, and optional: a missing earlier adapter must not cost the results above
        try:
            files = [hf_hub_download(prior_repo, name, revision=prior_revision)
                     for name in ("adapter_config.json", "adapter_model.safetensors")]
            model.load_adapter(str(Path(files[0]).parent), adapter_name="prior")
            model.set_adapter("prior")
            arms["prior"] = generate(model, tokenizer, test, args.batch_size)
        except Exception as error:   # noqa: BLE001
            print(f"Skipping the earlier adapter: {type(error).__name__}: {error}", flush=True)
            arms.pop("prior", None)
            prior_repo = prior_revision = None
        model.set_adapter("default")
    fresh = {row["id"] for row in test_rows} - scored_before
    metrics = {
        "base": args.base, "base_revision": args.base_revision, "data": provenance, "seed": args.seed,
        "runtime": {"device": torch.cuda.get_device_name() if args.device == "cuda" else "mps",
                    "precision": str(dtype).removeprefix("torch."),
                    "minutes": {"training": training_minutes, "total": (time.monotonic() - started) / 60},
                    "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2 ** 30 if args.device == "cuda" else None,
                    "dependencies": {name: importlib.metadata.version(name)
                                     for name in ("torch", "transformers", "accelerate", "peft", "sacrebleu")}},
        "training": {"epochs": args.epochs, "selected_epoch": best[1], "optimizer_steps": total_steps,
                     "examples": len(encoded), "skipped_too_long": dict(too_long), "rank": args.rank, "alpha": alpha,
                     "rslora": args.rslora, "dropout": args.dropout, "target_modules": modules,
                     "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
                     "warmup_steps": warmup_steps, "batch_size": args.train_batch_size,
                     "batch_tokens": args.batch_tokens,
                     "gradient_accumulation": args.accumulation, "gradient_checkpointing": checkpointing,
                     "history": history},
        "generation": {"preamble": "default", "do_sample": False, "max_new_tokens": NEW_TOKENS,
                       "batch_size": args.batch_size},
        "prior_adapter": {"repo": prior_repo, "revision": prior_revision} if prior_repo else None,
        "test": evaluate(test, arms),
        "test_rows_not_scored_by_earlier_pilots": evaluate(test, arms, only_rows=fresh),
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for i, example in enumerate(test):
            handle.write(json.dumps({"key": example.key, "row_id": example.row_id, "reference": example.reference,
                                     **{arm: outputs[i][0] for arm, outputs in arms.items()}},
                                    ensure_ascii=False) + "\n")
    commit = None
    if api:
        upload = api.upload_folder(repo_id=args.private_repo, repo_type="model", folder_path=args.output,
                                   path_in_repo=args.run_name,
                                   allow_patterns=["metrics.json", "predictions.jsonl"])
        commit = upload.oid
    summary = {key: {arm: round(headline(cell[arm]), 2) for arm in arms} for key, cell in metrics["test"].items()}
    print(json.dumps({"summary": summary, "selected_epoch": best[1], "private_repo": args.private_repo,
                      "run_name": args.run_name, "artifact_revision": commit,
                      "minutes": metrics["runtime"]["minutes"]}), flush=True)


if __name__ == "__main__":
    main()
