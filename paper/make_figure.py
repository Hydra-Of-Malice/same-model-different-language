"""Draw the paper's adapter figure: change from the parent on tasks (left) and on safety and capability (right).

Reads the `metrics.json` and `safety_check.json` that scripts/nusantara_adapter.py and
scripts/adapter_safety_check.py write for each adapter, and writes `adapter_tradeoff.pdf` next to this file.

    python paper/make_figure.py --adapter suite-r128 RUN/metrics.json SAFETY/safety_check.json \
                                --adapter suite-r64-lr2e-4 RUN2/metrics.json SAFETY2/safety_check.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Validated categorical slots 1-3 (blue, orange, aqua) on a white page, with a distinct marker per series so the
# figure still reads in greyscale.
COLOURS = ("#2a78d6", "#eb6834", "#1baf7a")
MARKERS = ("o", "s", "D")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"

TASKS = (("ind_to_jav", "Indonesian → Javanese"), ("jav_to_ind", "Javanese → Indonesian"),
         ("eng_to_jav", "English → Javanese"), ("ind_to_jav_tweets", "Tweets → Javanese"),
         ("ngoko_to_krama", "Ngoko → Krama rewrite"), ("level_jav", "Speech level"),
         ("senti_jav", "Sentiment"), ("topic_jav", "Paragraph topic"), ("emotion_jav", "Paragraph emotion"),
         ("eng_to_ind", "English → Indonesian (control)"))
SAFETY = (("default", "jv", "content_safe", "Javanese content-safe"),
          ("none", "jv", "content_safe", "Javanese content-safe, no preamble"),
          ("default", "en", "content_safe", "English content-safe"),
          ("none", "en", "content_safe", "English content-safe, no preamble"),
          ("none", "jv", "harmful_refusal", "Javanese harmful refusal, no preamble"),
          ("default", "jv", "over_refusal", "Javanese over-refusal"))
CAPABILITY = (("id", "Indonesian Global-MMLU-Lite"), ("en", "English Global-MMLU-Lite"))


def task_change(cell: dict) -> tuple[float, float, float]:
    """Paired change from the parent and its 95% interval: mean sentence chrF++ or accuracy, in points."""
    if "paired_mean_sentence_chrf_pp_delta" in cell:
        low, high = cell["paired_sentence_delta_95_ci"]
        return cell["paired_mean_sentence_chrf_pp_delta"], low, high
    low, high = cell["accuracy_delta_95_ci"]
    return cell["accuracy_delta"], low, high


def safety_rows(report: dict) -> list[tuple[float, float, float]]:
    rows = []
    for preamble, language, metric, _ in SAFETY:
        change = report["safety"][preamble]["adapter_minus_parent"][language][metric]
        rows.append((100 * change["delta"], 100 * change["ci"][0], 100 * change["ci"][1]))
    for language, _ in CAPABILITY:
        change = report["capability"][language]
        rows.append((100 * change["delta"], 100 * change["ci"][0], 100 * change["ci"][1]))
    return rows


def panel(axis, labels: list[str], series: dict[str, list[tuple[float, float, float]]], title: str) -> None:
    count = len(series)
    offsets = [(i - (count - 1) / 2) * 0.22 for i in range(count)]
    for index, (name, rows) in enumerate(series.items()):
        ys = [position + offsets[index] for position in range(len(rows))]
        xs = [row[0] for row in rows]
        errors = [[row[0] - row[1] for row in rows], [row[2] - row[0] for row in rows]]
        axis.errorbar(xs, ys, xerr=errors, fmt=MARKERS[index], color=COLOURS[index], markersize=4.2,
                      elinewidth=1.0, capsize=0, markeredgecolor="white", markeredgewidth=0.6, label=name, zorder=3)
    axis.axvline(0, color=MUTED, linewidth=0.8, zorder=1)
    axis.set_yticks(range(len(labels)), labels)
    axis.invert_yaxis()
    axis.set_title(title, fontsize=8.5, color=INK, loc="left")
    axis.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    axis.tick_params(axis="both", labelsize=7.5, colors=MUTED, length=0)
    for label in axis.get_yticklabels():
        label.set_color(INK)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(GRID)
    axis.set_xlabel("Change from parent (points, paired 95% CI)", fontsize=7.5, color=MUTED)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", nargs=3, action="append", required=True,
                        metavar=("NAME", "METRICS_JSON", "SAFETY_JSON"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("adapter_tradeoff.pdf"))
    args = parser.parse_args()
    if len(args.adapter) > len(COLOURS):
        parser.error(f"At most {len(COLOURS)} adapters")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "pdf.fonttype": 42})
    tasks, safety = {}, {}
    for name, metrics_path, safety_path in args.adapter:
        test = json.loads(Path(metrics_path).read_text(encoding="utf-8"))["test"]
        tasks[name] = [task_change(test[key]) for key, _ in TASKS]
        safety[name] = safety_rows(json.loads(Path(safety_path).read_text(encoding="utf-8")))

    figure, (left, right) = plt.subplots(1, 2, figsize=(6.3, 2.9), gridspec_kw={"wspace": 1.05})
    panel(left, [label for _, label in TASKS], tasks, "Tasks (chrF++ or accuracy)")
    panel(right, [label for *_, label in SAFETY] + [label for _, label in CAPABILITY], safety,
          "Safety and capability (rate)")
    handles, names = left.get_legend_handles_labels()
    figure.legend(handles, names, loc="upper center", ncol=len(names), frameon=False, fontsize=7.5,
                  bbox_to_anchor=(0.5, 1.03), handletextpad=0.3, columnspacing=1.4)
    figure.savefig(args.output, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
