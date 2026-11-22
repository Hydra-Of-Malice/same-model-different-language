"""Use a trained Javanese adapter from the command line: translate, classify, rewrite and triage.

Every command builds its prompt with the functions the adapter was trained with, so the wording cannot drift. Runs on
CUDA, Apple MPS or CPU. Research use only: read docs/FOLLOWUP_USE_CASES.md for what has and has not been checked.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nusantara_adapter as recipe  # noqa: E402

LEVEL_NAMES = {name.lower().replace(" ", "-"): index for index, name in enumerate(recipe.LEVELS)}


def build_prompt(command: str, text: str, *, source: str = "ind", target: str = "jav", level: str | None = None,
                 language: str = "jav", among: tuple[str, ...] = ("ind", "jav")) -> recipe.Example:
    """The training-time example for one request. Only its prompt and choices are used."""
    text = text.strip()
    if not text:
        raise ValueError("Give some text")
    if command == "translate" and level is None:
        return recipe.translation({"id": "0", source: text, target: ""}, source, target)
    if command in {"translate", "level", "rewrite"}:
        row = {"id": "u0", "group": "0", "level": LEVEL_NAMES[level or "ngoko"], "jav": text, "ind": text, "eng": ""}
        if command == "rewrite":
            if level not in {"ngoko", "krama"}:
                raise ValueError("Rewriting is trained between Ngoko and Krama only")
            other = {**row, "id": "u1", "level": LEVEL_NAMES["krama" if level == "ngoko" else "ngoko"]}
            wanted = f"{recipe.LEVELS[other['level']].lower()}_to_{level}"
            return next(e for e in recipe.level_examples([other, {**row, "jav": ""}]) if e.key == wanted)
        if command == "translate" and (source, target) != ("ind", "jav"):
            raise ValueError("A speech level applies to Indonesian-to-Javanese translation only")
        wanted = "level_jav" if command == "level" else f"ind_to_{level.replace('-', '_')}"
        return next(e for e in recipe.level_examples([row]) if e.key == wanted)
    if command == "sentiment":
        return recipe.sentiment({"id": "0", "label": "neutral", language: text}, language)
    if command == "identify":
        return recipe.identification({"id": "0", among[0]: text}, among[0], among)
    if command == "triage":
        return recipe.triage({"id": "0", "label": "neutral", "ind": "", language: text}, language)
    if command in {"topic", "emotion"}:
        label = (recipe.TOPICS if command == "topic" else recipe.EMOTIONS)[0]
        return recipe.paragraph_examples([{"id": "0", "text": text, "label": label}], command)[0]
    raise ValueError(f"Unknown command {command!r}")


def load(base: str, base_revision: str, adapter: str | None, adapter_revision: str | None, adapter_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = {"cuda": torch.bfloat16, "mps": torch.float16, "cpu": torch.float32}[device]
    tokenizer = AutoTokenizer.from_pretrained(base, revision=base_revision)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(base, revision=base_revision, dtype=dtype, device_map={"": device})
    if adapter:
        from huggingface_hub import hf_hub_download
        from peft import PeftModel

        folder = Path(adapter)
        if not folder.is_dir():
            prefix = f"{adapter_path.strip('/')}/" if adapter_path else ""
            files = [hf_hub_download(adapter, prefix + name, revision=adapter_revision)
                     for name in ("adapter_config.json", "adapter_model.safetensors")]
            folder = Path(files[0]).parent
        model = PeftModel.from_pretrained(model, str(folder))
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=recipe.BASE)
    parser.add_argument("--base-revision", default=recipe.BASE_REVISION)
    parser.add_argument("--adapter", help="Adapter folder, or a Hugging Face repository holding one. Omit for the parent")
    parser.add_argument("--adapter-revision", help="Hub commit of the adapter repository")
    parser.add_argument("--adapter-path", default="", help="Folder of the adapter inside the repository")
    parser.add_argument("--show-prompt", action="store_true", help="Print the prompt and stop, without loading weights")
    commands = parser.add_subparsers(dest="command", required=True)
    codes = sorted(recipe.LANGUAGES)
    translate = commands.add_parser("translate", help="Translate between Indonesian, English, Javanese and neighbours")
    translate.add_argument("--source", choices=codes, default="ind")
    translate.add_argument("--target", choices=codes, default="jav")
    translate.add_argument("--level", choices=sorted(LEVEL_NAMES), help="Javanese speech level of the translation")
    commands.add_parser("level", help="Name the speech level of a Javanese sentence")
    rewrite = commands.add_parser("rewrite", help="Rewrite a Javanese sentence between Ngoko and Krama")
    rewrite.add_argument("--level", choices=("ngoko", "krama"), required=True, help="Level to rewrite into")
    for name, text in (("sentiment", "Positive, neutral or negative"),
                       ("triage", "Language, sentiment and an Indonesian translation as JSON")):
        commands.add_parser(name, help=text).add_argument("--language", choices=codes, default="jav")
    identify = commands.add_parser("identify", help="Name the language of a text")
    identify.add_argument("--among", default="ind,jav", help="Comma-separated language codes to choose from")
    commands.add_parser("topic", help="Topic of a Javanese paragraph")
    commands.add_parser("emotion", help="Emotion of a Javanese paragraph")
    for command in commands.choices.values():
        command.add_argument("text", nargs="?", help="Text to work on. Reads standard input when omitted")
    args = parser.parse_args()

    text = args.text if args.text is not None else sys.stdin.read()
    options = {key: getattr(args, key) for key in ("source", "target", "level", "language") if hasattr(args, key)}
    if hasattr(args, "among"):
        options["among"] = recipe.parse_languages(args.among)
    try:
        example = build_prompt(args.command, text, **options)
    except ValueError as error:
        parser.error(str(error))
    if args.show_prompt:
        print(example.prompt)
        return
    model, tokenizer = load(args.base, args.base_revision, args.adapter, args.adapter_revision, args.adapter_path)
    answer, cut = recipe.generate(model, tokenizer, [example], 1)[0]
    print(answer)
    if cut:
        print("The answer reached the length limit and may be cut off.", file=sys.stderr)


if __name__ == "__main__":
    main()
