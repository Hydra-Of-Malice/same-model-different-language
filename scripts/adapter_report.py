"""Compare adapter runs: one Markdown table of test metrics per key, and one of validation scores per epoch.

Reads the `metrics.json` that scripts/nusantara_adapter.py writes for each run, from local folders or from the private
Hugging Face repository the runs were uploaded to. Prints aggregates only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HEADLINES = ("accuracy", "both_labels_correct", "chrf_pp")


def headline(arm: dict) -> tuple[str, float]:
    for name in HEADLINES:
        if name in arm:
            return name, arm[name]
    raise KeyError("Cell has no headline metric")


INTERVALS = {"accuracy": "accuracy_delta_95_ci", "both_labels_correct": "both_labels_correct_delta_95_ci",
             "chrf_pp": "paired_sentence_delta_95_ci"}


def interval(cell: dict, metric: str) -> str:
    """The paired 95% interval of the adapter's change from the parent. For chrF++ it is the mean sentence change."""
    if INTERVALS[metric] not in cell:
        return ""
    low, high = cell[INTERVALS[metric]]
    return f" [{low:+.1f}, {high:+.1f}]"


def comparison_table(runs: dict[str, dict], section: str = "test", arm: str = "after") -> str:
    """Rows are test keys. Each run shows its parent score once per distinct parent, then its own score and change."""
    keys = list(dict.fromkeys(key for metrics in runs.values() for key in metrics[section]))
    lines = ["| Key | Metric | n | " + " | ".join(f"Parent of {name} | {name}" for name in runs) + " |",
             "|---|---|---:|" + "---:|---:|" * len(runs)]
    for key in keys:
        cells, metric, count = [], "", ""
        for metrics in runs.values():
            cell = metrics[section].get(key)
            if not cell or arm not in cell:
                cells += ["—", "—"]
                continue
            metric, after = headline(cell[arm])
            count = str(cell["n"])
            before = headline(cell["before"])[1] if "before" in cell else None
            change = (f" ({after - before:+.1f}{interval(cell, metric) if arm == 'after' else ''})"
                      if before is not None else "")
            cells += [f"{before:.1f}" if before is not None else "—", f"{after:.1f}{change}"]
        lines.append(f"| `{key}` | {metric.replace('_pp', '++').replace('_', ' ')} | {count} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def training_table(runs: dict[str, dict]) -> str:
    lines = ["| Run | Base | Examples | Rank | Modules | Learning rate | Validation score by epoch | Chosen | Minutes |",
             "|---|---|---:|---:|---|---:|---|---:|---:|"]
    for name, metrics in runs.items():
        training = metrics["training"]
        scores = ", ".join(f"{epoch['validation_score']:.2f}" for epoch in training["history"])
        lines.append(f"| {name} | {metrics['base'].split('/')[-1]} | {training['examples']:,} | {training['rank']} | "
                     f"{len(training['target_modules'])} | {training['learning_rate']:g} | {scores} | "
                     f"{training['selected_epoch']} | {metrics['runtime']['minutes']['total']:.0f} |")
    return "\n".join(lines)


def load(run: str, repo: str | None, revision: str | None) -> dict:
    path = Path(run) / "metrics.json"
    if not path.exists():
        if not repo:
            raise FileNotFoundError(f"No {path}. Pass --repo to read the run from the Hub")
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(repo, f"{run}/metrics.json", revision=revision))
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="Run folders, or run names inside --repo")
    parser.add_argument("--repo", help="Hugging Face model repository holding the runs")
    parser.add_argument("--revision", help="Hub commit of that repository")
    parser.add_argument("--section", choices=("test", "test_rows_not_scored_by_earlier_pilots"), default="test")
    parser.add_argument("--arm", default="after", help="Arm to show beside the parent, e.g. after, scale_0.5 or prior")
    args = parser.parse_args()
    runs = {Path(run).name: load(run, args.repo, args.revision) for run in args.runs}
    print(training_table(runs), end="\n\n")
    print(comparison_table(runs, args.section, args.arm))


if __name__ == "__main__":
    main()
