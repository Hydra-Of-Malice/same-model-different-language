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
"""Private LoRA pilot for benign Indonesian–Javanese translation on Tiny Aya."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

BASE = "HydraOfMalice/tiny-aya-earth-langsafe-circuit-merge-0p5"
BASE_REVISION = "ef47ebee4b9307bf23d728a171efd2b74e1dc789"
DATA_REVISION = "75b0be5d982348332509f296e10d9bf9cb66d60f"
DATA_URL = f"https://raw.githubusercontent.com/IndoNLP/nusax/{DATA_REVISION}/datasets/mt"
DATA_SHA256 = {
    "train": "cf4276730de04f0cb3369c48b67d454f4aae52646315f19eb71b908f5dfaae0e",
    "test": "0082161de28a0a17a85fc6e02fc07a2e2514d8dd7f344ead95a88aa86dc706cc",
}
LANGUAGES = {"ind": "Indonesian", "jav": "Javanese", "eng": "English"}
DIRECTIONS = (("ind", "jav"), ("jav", "ind"))
CONTROL = ("eng", "ind")
SEED = 20261122
MAX_LENGTH = 640
MAX_NEW_TOKENS = 96
RANK = 8
LEARNING_RATE = 1e-4
ACCUMULATION = 4


@dataclass(frozen=True)
class Pair:
    row_id: str
    source: str
    target: str
    sentence: str
    reference: str

    @property
    def prompt(self) -> str:
        return (f"Translate the following {LANGUAGES[self.source]} text into {LANGUAGES[self.target]}. "
                f"Return only the translation.\n\n{self.sentence}")


def normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def row_keys(row: dict) -> set[tuple[str, str]]:
    return {(lang, normalize(row[lang])) for lang in ("ind", "jav", "eng") if row.get(lang)}


def partition(train: list[dict], test: list[dict]) -> tuple[list[Pair], list[Pair]]:
    for split, rows in (("train", train), ("test", test)):
        ids = [row["id"] for row in rows]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError(f"{split} requires non-empty, unique row IDs")
        if any(not normalize(row[lang]) for row in rows for lang in ("ind", "jav")):
            raise ValueError(f"{split} contains an empty source or reference")
    if {key for row in train for key in row_keys(row)} & {key for row in test for key in row_keys(row)}:
        raise ValueError("Normalized text overlap across train and test")

    def pairs(rows, directions):
        return [Pair(row["id"], source, target, row[source].strip(), row[target].strip())
                for row in rows for source, target in directions]

    test_directions = (*DIRECTIONS, CONTROL) if all(row.get("eng", "").strip() for row in test) else DIRECTIONS
    return pairs(train, DIRECTIONS), pairs(test, test_directions)


def select_holdout(rows: list[dict], count: int, excluded_ids: set[str]) -> list[dict]:
    available = [row for row in rows if row["id"] not in excluded_ids]
    if count > len(available):
        raise ValueError(f"Requested {count} holdout rows. Only {len(available)} are available.")
    return random.Random(SEED).sample(available, count)


def load_split(split: str, data_dir: Path | None) -> tuple[list[dict], str]:
    if data_dir is None:
        for attempt in range(3):
            try:
                with urlopen(f"{DATA_URL}/{split}.csv", timeout=30) as response:
                    raw = response.read()
                break
            except OSError:
                if attempt == 2:
                    raise
    else:
        raw = (data_dir / f"{split}.csv").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != DATA_SHA256[split]:
        raise ValueError(f"{split}.csv differs from NusaX-MT revision {DATA_REVISION}")
    rows = [{"id": str(index), "ind": row["indonesian"], "jav": row["javanese"], "eng": row["english"]}
            for index, row in enumerate(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))]
    return rows, digest


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


def generate(model, tokenizer, pairs: list[Pair], batch_size: int) -> list[tuple[str, bool]]:
    import torch

    model.eval()
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            prompts = [chat_prefix(tokenizer, pair.prompt) for pair in pairs[start:start + batch_size]]
            tokens = tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False).to(model.device)
            result = model.generate(**tokens, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id, use_cache=True)
            for ids in result[:, tokens.input_ids.shape[1]:].tolist():
                reached_limit = len(ids) == MAX_NEW_TOKENS and not (eos_ids & set(ids))
                outputs.append((tokenizer.decode(ids, skip_special_tokens=True).strip(), reached_limit))
    return outputs


def evaluate(pairs: list[Pair], before: list[tuple[str, bool]], after: list[tuple[str, bool]]) -> dict:
    from sacrebleu.metrics import CHRF

    metric = CHRF(word_order=2)
    results = {}
    for source, target in dict.fromkeys((pair.source, pair.target) for pair in pairs):
        indices = [i for i, pair in enumerate(pairs) if (pair.source, pair.target) == (source, target)]
        refs = [pairs[i].reference for i in indices]
        cell = {"n": len(indices)}
        scores = []
        for label, outputs in (("before", before), ("after", after)):
            text = [outputs[i][0] for i in indices]
            words = [normalize(sentence) for sentence in text]
            cell[label] = {
                "chrf_pp": metric.corpus_score(text, [refs]).score,
                "empty": sum(not value for value in words),
                "exact_reference": sum(value == normalize(ref) for value, ref in zip(words, refs)),
                "avoidable_source_echo": sum(
                    value == normalize(pairs[i].sentence) and value != normalize(pairs[i].reference)
                    for i, value in zip(indices, words)
                ),
                "generation_limit": sum(outputs[i][1] for i in indices),
            }
            scores.append([metric.sentence_score(hyp, [ref]).score for hyp, ref in zip(text, refs)])
        deltas = [new - old for old, new in zip(*scores)]
        rng = random.Random(SEED)
        bootstrap = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(2000))
        cell["paired_mean_sentence_chrf_pp_delta"] = sum(deltas) / len(deltas)
        cell["paired_sentence_delta_95_ci"] = [bootstrap[49], bootstrap[1949]]
        cell["corpus_chrf_pp_delta"] = cell["after"]["chrf_pp"] - cell["before"]["chrf_pp"]
        results[f"{source}_to_{target}"] = cell
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--eval-rows", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=4, help="Generation batch size")
    parser.add_argument("--output", type=Path, default=Path("runs/pilot/javanese-translation"))
    parser.add_argument("--data-dir", type=Path, help="Local copy of the pinned train.csv and test.csv files")
    parser.add_argument("--private-repo", help="Private Hugging Face model repository for the adapter and metrics")
    parser.add_argument("--compare-adapter", metavar="REPO@SHA",
                        help="Compare with an earlier adapter on test rows it did not score")
    parser.add_argument("--english-replay", action="store_true",
                        help="Replace every fifth training example with English-to-Indonesian translation")
    parser.add_argument("--data-only", action="store_true", help="Validate and fingerprint splits without loading weights")
    args = parser.parse_args()
    if min(args.steps, args.eval_rows, args.batch_size) < 1:
        parser.error("steps, eval-rows and batch-size must be positive")

    train_rows, train_sha = load_split("train", args.data_dir)
    test_rows, test_sha = load_split("test", args.data_dir)
    test_keys = {key for row in test_rows for key in row_keys(row)}
    filtered = [row for row in train_rows if row_keys(row).isdisjoint(test_keys)]
    train, _ = partition(filtered, test_rows)
    if any(not row["eng"].strip() for row in test_rows):
        raise ValueError("NusaX-MT holdout is missing an English control sentence")
    prior_repo, prior_revision = None, None
    excluded_ids = set()
    if args.compare_adapter:
        from huggingface_hub import hf_hub_download

        prior_repo, _, prior_revision = args.compare_adapter.partition("@")
        if not prior_repo or not re.fullmatch(r"[0-9a-f]{40}", prior_revision):
            parser.error("--compare-adapter requires REPO@SHA with a full Hub commit hash")
        prior_path = hf_hub_download(prior_repo, "pilot_metrics.json", revision=prior_revision)
        prior = json.loads(Path(prior_path).read_text(encoding="utf-8"))
        if (prior["base"] != BASE or prior["base_revision"] != BASE_REVISION
                or prior["data"]["test_sha256"] != test_sha):
            raise ValueError("Previous adapter was evaluated against a different base or test set")
        excluded_ids.update(prior["data"]["evaluated_holdout_row_ids"])
        excluded_ids.update(prior["data"].get("excluded_previously_scored_holdout_row_ids", []))
        if excluded_ids - {row["id"] for row in test_rows}:
            raise ValueError("Previous adapter recorded unknown test row IDs")
    selected = select_holdout(test_rows, args.eval_rows, excluded_ids)
    _, test = partition(filtered, selected)
    provenance = {
        "source": DATA_URL,
        "revision": DATA_REVISION,
        "train_sha256": train_sha,
        "test_sha256": test_sha,
        "train_rows": len(filtered),
        "excluded_train_rows_exact_overlap": len(train_rows) - len(filtered),
        "holdout_rows_total": len(test_rows),
        "evaluated_holdout_row_ids": [row["id"] for row in selected],
        "excluded_previously_scored_holdout_row_ids": sorted(excluded_ids),
    }
    print(json.dumps({"data": provenance, "evaluation_pairs": dict(Counter(f"{p.source}_to_{p.target}" for p in test))}),
          flush=True)
    if args.data_only:
        return

    import torch
    from huggingface_hub import HfApi
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This pilot requires a CUDA GPU with BF16 support")
    if args.output.exists():
        raise FileExistsError(f"Choose a fresh output path: {args.output}")
    api = HfApi() if args.private_repo else None
    if api:
        api.create_repo(args.private_repo, repo_type="model", private=True, exist_ok=True)
        if not api.model_info(args.private_repo).private:
            raise RuntimeError("Pilot outputs require a private model repository")

    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(BASE, revision=BASE_REVISION)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, revision=BASE_REVISION, dtype=torch.bfloat16,
                                                device_map={"": 0}, attn_implementation="sdpa")
    print(f"Loaded {BASE}@{BASE_REVISION}. Evaluating baseline on {len(test)} examples.", flush=True)
    before = generate(model, tokenizer, test, args.batch_size)
    prior_outputs = None
    if prior_repo:
        print(f"Evaluating earlier adapter {prior_repo}@{prior_revision} on {len(test)} examples.", flush=True)
        prior_model = PeftModel.from_pretrained(model, prior_repo, revision=prior_revision)
        prior_outputs = generate(prior_model, tokenizer, test, args.batch_size)
        model = prior_model.unload()
        del prior_model
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=RANK, lora_alpha=2 * RANK,
                                            target_modules=["q_proj", "v_proj"], revision=BASE_REVISION))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    trainable = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=0.0)
    encoded = [encode(tokenizer, pair.prompt, pair.reference) for pair in train]
    replay = {}
    if args.english_replay:
        for row in filtered:
            pair = Pair(row["id"], *CONTROL, row["eng"].strip(), row["ind"].strip())
            replay[pair.row_id] = encode(tokenizer, pair.prompt, pair.reference)
    rng = random.Random(SEED)
    order = list(range(len(encoded)))
    rng.shuffle(order)
    losses = []
    for step in range(args.steps):
        losses_for_step = []
        for offset in range(ACCUMULATION):
            i = (step * ACCUMULATION + offset) % len(order)
            if i == 0 and step > 0:
                rng.shuffle(order)
            example = encoded[order[i]]
            if replay and (step * ACCUMULATION + offset + 1) % 5 == 0:
                example = replay[train[order[i]].row_id]
            batch = {name: torch.tensor([values], device=model.device) for name, values in example.items()}
            loss = model(**batch, use_cache=False).loss
            if not loss.isfinite().item():
                raise RuntimeError(f"Non-finite training loss at step {step + 1}")
            (loss / ACCUMULATION).backward()
            losses_for_step.append(loss.item())
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(sum(losses_for_step) / ACCUMULATION)
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == args.steps:
            print(f"Step {step + 1}/{args.steps}, mean response loss {losses[-1]:.4f}", flush=True)

    model.gradient_checkpointing_disable()
    args.output.mkdir(parents=True)
    model.save_pretrained(args.output)
    print(f"Evaluating trained adapter on the same {len(test)} examples.", flush=True)
    after = generate(model, tokenizer, test, args.batch_size)
    metrics = {
        "base": BASE,
        "base_revision": BASE_REVISION,
        "data": provenance,
        "seed": SEED,
        "runtime": {"gpu": torch.cuda.get_device_name(), "precision": "bfloat16",
                    "dependencies": {name: importlib.metadata.version(name)
                                     for name in ("torch", "transformers", "accelerate", "peft", "sacrebleu")}},
        "training": {"steps": args.steps, "examples_seen": args.steps * ACCUMULATION, "rank": RANK,
                     "gradient_accumulation": ACCUMULATION, "learning_rate": LEARNING_RATE,
                     "english_replay_examples": (args.steps * ACCUMULATION) // 5 if replay else 0,
                     "mean_response_loss_by_step": losses},
        "generation": {"preamble": "default", "do_sample": False, "max_new_tokens": MAX_NEW_TOKENS,
                       "batch_size": args.batch_size},
        "translation": evaluate(test, before, after),
    }
    if prior_outputs is not None:
        metrics["comparison"] = {"repo": prior_repo, "revision": prior_revision,
                                 "prior_vs_base": evaluate(test, before, prior_outputs),
                                 "trained_vs_prior": evaluate(test, prior_outputs, after)}
    (args.output / "pilot_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (args.output / "README.md").write_text(
        f"---\nlicense: cc-by-nc-4.0\nbase_model: {BASE}\npipeline_tag: translation\n---\n\n"
        "# Private Javanese translation pilot\n\n"
        "Research adapter for Indonesian–Javanese translation. Safety and general capability have yet to be assessed.\n\n"
        f"Pinned starting checkpoint: {BASE}@{BASE_REVISION}. Training used the native-translated "
        "[NusaX-MT](https://huggingface.co/datasets/indonlp/NusaX-MT) corpus (CC BY-SA 4.0), "
        f"revision `{DATA_REVISION}`. The corpus was reformatted into translation instructions. "
        "The adapter inherits the model's non-commercial license and acceptable use policy.\n",
        encoding="utf-8",
    )
    commit = None
    if api:
        upload = api.upload_folder(repo_id=args.private_repo, repo_type="model", folder_path=args.output,
                                   allow_patterns=["adapter_config.json", "adapter_model.safetensors",
                                                   "pilot_metrics.json", "README.md"])
        commit = upload.oid
    print(json.dumps({"translation": metrics["translation"], "comparison": metrics.get("comparison"),
                      "private_repo": args.private_repo, "artifact_revision": commit}), flush=True)


if __name__ == "__main__":
    main()
