#!/usr/bin/env python3
"""Compare two report.json files built from the same answers by different judges: rates, gaps and repair verdicts.

    python scripts/compare_reports.py runs/experiments/full/earth/report.json \
        runs/experiments/full/earth-router-report/report.json --labels "8B pair,32B pair" --out docs/judge_checks/x.md

Only rates, intervals, counts and verdicts are read and written: no prompt, answer or judge text.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

METRICS = (('harmful_refusal', 'harmful refused'), ('content_safe', 'harmful answers safe'),
           ('over_refusal', 'benign refused'))


def pct(value) -> str:
    return '–' if value is None else f'{100 * value:.1f}'


def cell(table: dict, lang: str, metric: str):
    return table.get(lang, {}).get(metric, {}).get('rate')


def compare(a: dict, b: dict, labels: tuple[str, str], tags: list[str], condition: str) -> str:
    lines = [f'# {labels[1]} against {labels[0]}, on the same answers', '',
             f'Rates in percent, `{condition}` preamble. Each cell is {labels[0]} → {labels[1]}.', '']
    langs = list(a['models'][tags[0]]['conditions'][condition])
    for metric, title in METRICS:
        lines += [f'## {title.capitalize()}', '', '| Model | ' + ' | '.join(langs) + ' |',
                  '|---|' + '---|' * len(langs)]
        for tag in tags:
            ta = a['models'].get(tag, {}).get('conditions', {}).get(condition, {})
            tb = b['models'].get(tag, {}).get('conditions', {}).get(condition, {})
            lines.append(f'| {tag} | ' + ' | '.join(f'{pct(cell(ta, l, metric))} → {pct(cell(tb, l, metric))}'
                                                     for l in langs) + ' |')
        lines.append('')
    lines += ['## Drift gap confirmed (Global vs Earth, primary metric)', '',
              '| Language | ' + ' | '.join(labels) + ' |', '|---|---|---|']
    ga = a['gap_confirmation']['conditions'].get(condition, {})
    gb = b['gap_confirmation']['conditions'].get(condition, {})
    for lang in langs:
        fa, fb = ga.get(lang, {}), gb.get(lang, {})
        lines.append(f"| {lang} | {pct(fa.get('delta'))} pts, {'confirmed' if fa.get('confirmed') else 'not confirmed'}"
                     f" | {pct(fb.get('delta'))} pts, {'confirmed' if fb.get('confirmed') else 'not confirmed'} |")
    lines += ['', '## Overall', '', f'| | {labels[0]} | {labels[1]} |', '|---|---|---|',
              f"| Acceptance candidate | {a.get('acceptance_candidate')} | {b.get('acceptance_candidate')} |",
              f"| Empirical repair success | {a.get('empirical_repair_success')} | {b.get('empirical_repair_success')} |",
              f"| Validated | {a.get('validated')} | {b.get('validated')} |",
              f"| Validation gaps | {len(a.get('validation_gaps', []))} | {len(b.get('validation_gaps', []))} |", '']
    return '\n'.join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('first', type=Path)
    parser.add_argument('second', type=Path)
    parser.add_argument('--labels', default='first,second')
    parser.add_argument('--tags', default='global,earth,earth-circuit_merge-0p5')
    parser.add_argument('--condition', default='default')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    a, b = (json.loads(p.read_text(encoding='utf-8')) for p in (args.first, args.second))
    text = compare(a, b, tuple(args.labels.split(',')), args.tags.split(','), args.condition)
    if args.out:
        args.out.write_text(text, encoding='utf-8', newline='\n')
    sys.stdout.reconfigure(encoding='utf-8')  # the Windows console's code page has no arrow
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
