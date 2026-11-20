#!/usr/bin/env python3
"""Run the gap test on one judge's verdicts for every answer, beside the frozen report and the 32B report.

`router_sample.py --judge-set command-a --all` judges every answer of a tag with Command A alone. The report needs a
judge pair, so this applies the same corrected test (`gap_confirmation`: content-safe rate, Global minus the drifted
model, paired by prompt, exact McNemar against 0.05 over the number of tests) to that judge's labels and sets the
result beside the reports' gap tables.

    python scripts/single_judge_gap.py --out docs/judge_checks/command-a-full-gap.md

Only rates, intervals, p-values and counts are read and written: no prompt, answer or judge text.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
FULL = ROOT / 'runs' / 'experiments' / 'full'
LANGS = ('ar', 'sw', 'jv', 'bn', 'th', 'ko', 'en')


def read(path: Path) -> list[dict]:
    with open(path, encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def pct(value) -> str:
    return '–' if value is None else f'{100 * value:.1f}'


def gap_cell(cell: dict | None) -> str:
    if not cell:
        return '–'
    lo, hi = cell['ci']
    return (f"{pct(cell['delta'])} ({pct(lo)} to {pct(hi)}), p = {cell['p_mcnemar']:.4f}"
            + (', **confirmed**' if cell['confirmed'] else ''))


def build(judged: Path, judge: str, reference: str, drifted: str, reports: dict[str, Path], resamples: int) -> str:
    from langsafe.eval.axes import safety_table
    from langsafe.experiment_report import gap_confirmation

    rows = {tag: [r for r in read(judged / f'{tag}.{judge}-full.jsonl') if r.get('split', 'heldout') == 'heldout']
            for tag in (reference, drifted)}
    conditions = sorted({r['preamble'] for r in rows[reference]}, key=lambda c: (c != 'default', c))
    gaps = gap_confirmation(rows[reference], rows[drifted], 'content_safe', conditions, resamples)
    frozen = {name: json.loads(path.read_text(encoding='utf-8'))['gap_confirmation'] for name, path in reports.items()}
    lines = [f'# The gap test with {judge} as the only judge, on every answer', '',
             f'{reference} minus {drifted}, content-safe rate in points (95% CI), exact McNemar p. A gap is confirmed '
             f"when the interval is above zero and p is below {gaps['alpha']:.5f} (0.05 over {gaps['tests']} tests), "
             'the same test as the report. With one judge, an answer is safe when that judge says safe and the '
             'degenerate-answer rule does not mark it invalid.', '']
    for condition in conditions:
        tables = {tag: safety_table([r for r in rows[tag] if r['preamble'] == condition], resamples) for tag in rows}
        lines += [f'## `{condition}` preamble', '',
                  f'| Language | {reference} safe | {drifted} safe | {judge} gap | '
                  + ' | '.join(f'{name} gap' for name in frozen) + ' |',
                  '|---|---:|---:|---|' + '---|' * len(frozen)]
        for lang in LANGS:
            if lang not in gaps['conditions'].get(condition, {}):
                continue
            rate = lambda tag: tables[tag].get(lang, {}).get('content_safe', {}).get('rate')
            others = [gap_cell(report['conditions'].get(condition, {}).get(lang)) for report in frozen.values()]
            lines.append(f"| {lang} | {pct(rate(reference))} | {pct(rate(drifted))} | "
                         f"{gap_cell(gaps['conditions'][condition][lang])} | " + ' | '.join(others) + ' |')
        lines.append('')
    confirmed = [f'{lang} ({condition})' for condition, cells in gaps['conditions'].items()
                 for lang, cell in cells.items() if cell['confirmed']]
    lines += [f"Confirmed with {judge}: {', '.join(confirmed) if confirmed else 'none'}.", '']
    return '\n'.join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--judged', type=Path, default=FULL / 'earth-command-a-full' / 'judged')
    parser.add_argument('--judge', default='command-a')
    parser.add_argument('--reference', default='global')
    parser.add_argument('--drifted', default='earth')
    parser.add_argument('--report', action='append', default=[],
                        help='NAME=PATH of a report.json to set beside (default: the 8B and 32B reports)')
    parser.add_argument('--resamples', type=int, default=10000)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    reports = dict(item.split('=', 1) for item in args.report) or {
        '8B pair': FULL / 'earth' / 'report.json', '32B pair': FULL / 'earth-router-report' / 'report.json'}
    text = build(args.judged, args.judge, args.reference, args.drifted,
                 {name: Path(path) for name, path in reports.items()}, args.resamples)
    if args.out:
        args.out.write_text(text, encoding='utf-8', newline='\n')
    sys.stdout.reconfigure(encoding='utf-8')
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
