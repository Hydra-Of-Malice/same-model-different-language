# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "torch==2.8.0",
#   "transformers==5.22.1",
#   "huggingface-hub==2.4.0",
#   "sacrebleu==2.5.1",
#   "sentencepiece>=0.2",
#   "protobuf>=4",
# ]
# ///
"""Score NLLB-200 on the translation test items the adapter runs use, as an external baseline.

NLLB is used as released, with no fine-tuning on these corpora, so it is a zero-shot reference and the adapters are
not. The test items and the metrics come from scripts/nusantara_adapter.py, fetched at a pinned commit of this
repository, so both sides are scored by the same code on the same rows.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

CODE = "Hydra-Of-Malice/same-model-different-language"
MODEL = "facebook/nllb-200-3.3B"
MODEL_REVISION = "1a07f7d195896b2114afcb79b7b57ab512e7b43e"
# NLLB-200 has no Madurese, Toba Batak or Ngaju.
CODES = {"ind": "ind_Latn", "jav": "jav_Latn", "eng": "eng_Latn", "sun": "sun_Latn", "ban": "ban_Latn",
         "min": "min_Latn", "bjn": "bjn_Latn", "ace": "ace_Latn", "bug": "bug_Latn"}
MAX_NEW_TOKENS = 256


def directions(recipe, regional: bool) -> list[tuple[str, str]]:
    """The adapter runs' translation directions, limited to languages NLLB-200 covers."""
    pairs = recipe.parse_pairs("ind-jav,eng-jav")
    if regional:
        pairs += [pair for code in recipe.REGIONAL if code in CODES for pair in (("ind", code), (code, "ind"))]
    return pairs


def language_pair(key: str) -> tuple[str, str]:
    source, target = key.removesuffix("_tweets").split("_to_")
    return CODES[source], CODES[target]


def fetch_recipe(revision: str, directory: Path):
    url = f"https://raw.githubusercontent.com/{CODE}/{revision}/scripts/nusantara_adapter.py"
    with urlopen(url, timeout=60) as response:
        (directory / "nusantara_adapter.py").write_bytes(response.read())
    spec = importlib.util.spec_from_file_location("nusantara_adapter", directory / "nusantara_adapter.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["nusantara_adapter"] = module
    spec.loader.exec_module(module)
    return module


def translate(model, tokenizer, examples: list, batch_size: int, num_beams: int) -> list[tuple[str, bool]]:
    import torch

    outputs: list[tuple[str, bool] | None] = [None] * len(examples)
    order = sorted(range(len(examples)), key=lambda i: (examples[i].key, -len(examples[i].source)))
    step = max(1, batch_size // num_beams)
    with torch.inference_mode():
        start = 0
        while start < len(order):
            key = examples[order[start]].key
            group = [i for i in order[start:start + step] if examples[i].key == key]
            source, target = language_pair(key)
            tokenizer.src_lang = source
            tokens = tokenizer([examples[i].source for i in group], padding=True, truncation=True, max_length=512,
                               return_tensors="pt").to(model.device)
            result = model.generate(**tokens, forced_bos_token_id=tokenizer.convert_tokens_to_ids(target),
                                    max_new_tokens=MAX_NEW_TOKENS, num_beams=num_beams, do_sample=False)
            for i, ids in zip(group, result):
                text = tokenizer.decode(ids, skip_special_tokens=True).strip()
                outputs[i] = (text, int((ids != tokenizer.pad_token_id).sum()) >= MAX_NEW_TOKENS)
            start += len(group)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-revision", required=True, help="Commit of this repository to take the test items from")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--tweets", action="store_true", help="Also score the NusaTranslation tweets")
    parser.add_argument("--regional", action="store_true", help="Also score the other NusaX languages NLLB covers")
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-rows", type=int, default=400)
    parser.add_argument("--eval-rows-other", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("runs/pilot/translation-baseline"))
    parser.add_argument("--private-repo", help="Private Hugging Face model repository for the aggregate metrics")
    parser.add_argument("--run-name", default="baseline-nllb", help="Folder for this run inside that repository")
    args = parser.parse_args()
    for name in ("code_revision", "model_revision"):
        if not re.fullmatch(r"[0-9a-f]{40}", getattr(args, name)):
            parser.error(f"--{name.replace('_', '-')} must be a full commit hash")
    if args.output.exists():
        parser.error(f"Choose a fresh output path: {args.output}")

    import torch
    from huggingface_hub import HfApi
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    api = HfApi() if args.private_repo else None
    if api and not api.model_info(args.private_repo).private:
        raise RuntimeError("Baseline outputs require a private model repository")
    with tempfile.TemporaryDirectory() as directory:
        recipe = fetch_recipe(args.code_revision, Path(directory))
    settings = SimpleNamespace(data_dir=None, train_rows=None, levels=False, levels_file=None, tweets=args.tweets,
                               paragraphs=False, writes_dir=None, valid_rows=100, eval_rows=args.eval_rows,
                               eval_rows_other=args.eval_rows_other, repeat_javanese=1)
    data = recipe.assemble(settings, directions(recipe, args.regional), (), (), ())
    test = [example for example in data["test"] if "_to_" in example.key]
    print(json.dumps({"test_examples": {key: sum(e.key == key for e in test)
                                        for key in dict.fromkeys(e.key for e in test)}}), flush=True)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, revision=args.model_revision,
                                                  dtype=torch.float32).to(device).eval()
    arms = {"greedy": translate(model, tokenizer, test, args.batch_size, 1)}
    if args.num_beams > 1:
        arms[f"beam_{args.num_beams}"] = translate(model, tokenizer, test, args.batch_size, args.num_beams)
    metrics = {"model": args.model, "model_revision": args.model_revision,
               "code": {"repository": CODE, "revision": args.code_revision},
               "data": {key: data["provenance"][key] for key in ("source", "revision", "sha256", "nusa_writes")},
               "generation": {"max_new_tokens": MAX_NEW_TOKENS, "num_beams": [1, args.num_beams],
                              "precision": "float32", "device": device},
               "test": recipe.evaluate(test, arms)}
    args.output.mkdir(parents=True)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    commit = None
    if api:
        commit = api.upload_file(path_or_fileobj=args.output / "metrics.json", repo_id=args.private_repo,
                                 path_in_repo=f"{args.run_name}/metrics.json", repo_type="model").oid
    print(json.dumps({"summary": {key: {arm: {"chrf_pp": round(cell[arm]["chrf_pp"], 2),
                                              "bleu": round(cell[arm]["bleu"], 2)} for arm in arms}
                                  for key, cell in metrics["test"].items()}, "artifact_revision": commit}), flush=True)


if __name__ == "__main__":
    main()
