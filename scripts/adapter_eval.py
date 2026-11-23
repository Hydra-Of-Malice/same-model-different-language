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
"""Score a trained adapter on the adapter test items at one or more strengths, against its parent.

Training runs score the adapter at full strength and at the strengths named when they were launched. This scores an
existing adapter at any strength afterwards, for example the strength a safety check cleared. The test items and the
metrics come from scripts/nusantara_adapter.py, fetched at a pinned commit of this repository, so the numbers are
comparable with the training runs'. Each strength gets its own paired comparison with the parent.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

CODE = "Hydra-Of-Malice/same-model-different-language"


def fetch_recipe(revision: str, directory: Path):
    url = f"https://raw.githubusercontent.com/{CODE}/{revision}/scripts/nusantara_adapter.py"
    with urlopen(url, timeout=60) as response:
        (directory / "nusantara_adapter.py").write_bytes(response.read())
    spec = importlib.util.spec_from_file_location("nusantara_adapter", directory / "nusantara_adapter.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["nusantara_adapter"] = module
    spec.loader.exec_module(module)
    return module


def parse_scales(text: str) -> list[float]:
    scales = [float(value) for value in filter(None, text.split(","))]
    if not scales or any(not 0 < scale <= 1 for scale in scales) or len(set(scales)) != len(scales):
        raise ValueError("--scales takes distinct strengths in (0, 1]")
    return scales


def composite(recipe, cells: dict) -> float:
    """Mean over task families of each family's mean headline metric, as the training runs select epochs."""
    return recipe.selection_score(cells, "after", "composite")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-revision", required=True, help="Commit of this repository to take the recipe from")
    parser.add_argument("--adapter-repo", required=True)
    parser.add_argument("--adapter-revision", required=True, help="Full Hub commit hash of the adapter repository")
    parser.add_argument("--adapter-path", default="", help="Folder of the adapter inside the repository")
    parser.add_argument("--base", help="Parent checkpoint. Defaults to the recipe's")
    parser.add_argument("--base-revision", help="Full Hub commit hash of the parent. Defaults to the recipe's")
    parser.add_argument("--scales", default="1,0.75,0.5", help="Adapter strengths to score, e.g. 1,0.5")
    parser.add_argument("--triage", default="", help="As in nusantara_adapter.py")
    parser.add_argument("--lid", default="", help="As in nusantara_adapter.py")
    parser.add_argument("--senti", default="jav", help="As in nusantara_adapter.py")
    parser.add_argument("--levels", action="store_true")
    parser.add_argument("--tweets", action="store_true")
    parser.add_argument("--paragraphs", action="store_true")
    parser.add_argument("--regional", action="store_true")
    parser.add_argument("--eval-rows", type=int, default=400)
    parser.add_argument("--eval-rows-other", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("runs/adapter-eval"))
    parser.add_argument("--private-repo", help="Private Hugging Face model repository for the metrics")
    parser.add_argument("--run-name", default="adapter-eval", help="Folder for this run inside that repository")
    args = parser.parse_args()
    for name in ("code_revision", "adapter_revision"):
        if not re.fullmatch(r"[0-9a-f]{40}", getattr(args, name)):
            parser.error(f"--{name.replace('_', '-')} must be a full commit hash")
    try:
        scales = parse_scales(args.scales)
    except ValueError as error:
        parser.error(str(error))
    if args.output.exists():
        parser.error(f"Choose a fresh output path: {args.output}")

    with tempfile.TemporaryDirectory() as directory:
        recipe = fetch_recipe(args.code_revision, Path(directory))
    base, base_revision = args.base or recipe.BASE, args.base_revision or recipe.BASE_REVISION
    directions = recipe.parse_pairs("ind-jav,eng-jav")
    senti, lid, triaged = (recipe.parse_languages(text) for text in (args.senti, args.lid, args.triage))
    if args.regional:
        directions += [pair for code in recipe.REGIONAL for pair in (("ind", code), (code, "ind"))
                       if pair not in directions]
        senti = tuple(dict.fromkeys((*senti, "jav", "ind", *recipe.REGIONAL)))
        lid = tuple(dict.fromkeys((*lid, "ind", "jav", *recipe.REGIONAL)))
        triaged = tuple(dict.fromkeys((*triaged, "jav", *recipe.REGIONAL)))
    settings = SimpleNamespace(data_dir=None, train_rows=None, levels=args.levels, levels_file=None,
                               tweets=args.tweets, paragraphs=args.paragraphs, writes_dir=None, valid_rows=100,
                               eval_rows=args.eval_rows, eval_rows_other=args.eval_rows_other, repeat_javanese=1)
    data = recipe.assemble(settings, directions, senti, lid, triaged)
    test = data["test"]
    print(json.dumps({"test_examples": len(test)}), flush=True)

    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from peft import PeftModel
    from peft.helpers import rescale_adapter_scale
    from transformers import AutoModelForCausalLM, AutoTokenizer

    api = HfApi() if args.private_repo else None
    if api and not api.model_info(args.private_repo).private:
        raise RuntimeError("Evaluation outputs require a private model repository")
    started = time.monotonic()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(base, revision=base_revision)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, revision=base_revision, dtype=dtype, device_map={"": device},
                                                 attn_implementation="sdpa")
    prefix = f"{args.adapter_path.strip('/')}/" if args.adapter_path else ""
    files = [hf_hub_download(args.adapter_repo, prefix + name, revision=args.adapter_revision)
             for name in ("adapter_config.json", "adapter_model.safetensors")]
    model = PeftModel.from_pretrained(model, str(Path(files[0]).parent)).eval()

    with model.disable_adapter():
        parent = recipe.generate(model, tokenizer, test, args.batch_size)
    by_scale, outputs = {}, {}
    for scale in scales:
        print(f"Scoring the adapter at strength {scale:g}.", flush=True)
        if scale == 1:
            outputs[scale] = recipe.generate(model, tokenizer, test, args.batch_size)
        else:
            with rescale_adapter_scale(model, scale):
                outputs[scale] = recipe.generate(model, tokenizer, test, args.batch_size)
        cells = recipe.evaluate(test, {"before": parent, "after": outputs[scale]})
        by_scale[f"{scale:g}"] = {"composite": composite(recipe, cells), "test": cells}

    metrics = {"code": {"repository": CODE, "revision": args.code_revision},
               "base": base, "base_revision": base_revision,
               "adapter": {"repo": args.adapter_repo, "revision": args.adapter_revision, "path": args.adapter_path},
               "data": data["provenance"],
               "generation": {"preamble": "default", "do_sample": False, "max_new_tokens": recipe.NEW_TOKENS,
                              "batch_size": args.batch_size, "precision": str(dtype).removeprefix("torch."),
                              "device": torch.cuda.get_device_name() if device == "cuda" else device},
               "minutes": (time.monotonic() - started) / 60,
               "parent_composite": recipe.selection_score(
                   {key: {"after": cell["before"]} for key, cell in by_scale[f"{scales[0]:g}"]["test"].items()},
                   "after", "composite"),
               "scales": by_scale}
    args.output.mkdir(parents=True)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for i, example in enumerate(test):
            handle.write(json.dumps({"key": example.key, "row_id": example.row_id, "reference": example.reference,
                                     "before": parent[i][0],
                                     **{f"scale_{scale:g}": outputs[scale][i][0] for scale in scales}},
                                    ensure_ascii=False) + "\n")
    commit = None
    if api:
        commit = api.upload_folder(repo_id=args.private_repo, repo_type="model", folder_path=args.output,
                                   path_in_repo=args.run_name,
                                   allow_patterns=["metrics.json", "predictions.jsonl"]).oid
    print(json.dumps({"summary": {scale: {"composite": round(entry["composite"], 2),
                                          **{key: round(recipe.headline(cell["after"]), 2)
                                             for key, cell in entry["test"].items()}}
                                  for scale, entry in by_scale.items()},
                      "parent_composite": round(metrics["parent_composite"], 2), "artifact_revision": commit}),
          flush=True)


if __name__ == "__main__":
    main()
