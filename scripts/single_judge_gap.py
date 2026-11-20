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


def comparison(judged: Path, judge: str, reference: str, other: str, resamples: int) -> str:
    """The same test for a comparison model. It is outside the report's tests, so a pass is never called confirmed."""
    from langsafe.experiment_report import gap_confirmation

    rows = {tag: read(judged / f'{tag}.{judge}-full.jsonl') for tag in (reference, other)}
    conditions = sorted({r['preamble'] for r in rows[reference]}, key=lambda c: (c != 'default', c))
    gaps = gap_confirmation(rows[reference], rows[other], 'content_safe', conditions, resamples)
    lines = [f'## Comparison model: {reference} minus {other}', '',
             f'{other} is a comparison, outside the report\'s tests, so no gap of it is called confirmed. "Would pass" '
             f'marks a cell whose interval is above zero and whose p is below the same {gaps["alpha"]:.5f}.', '',
             '| Language | ' + ' | '.join(f'`{c}` preamble' for c in conditions) + ' |', '|---|' + '---|' * len(conditions)]
    for lang in LANGS:
        cells = [gaps['conditions'].get(c, {}).get(lang) for c in conditions]
        lines.append(f'| {lang} | ' + ' | '.join(
            gap_cell({**cell, 'confirmed': False}) + (', would pass' if cell and cell['confirmed'] else '')
            if cell else '–' for cell in cells) + ' |')
    return '\n'.join(lines + [''])


def repair(judged: Path, judge: str, reference: str, drifted: str, repaired: str, target: str, frozen: Path,
           resamples: int) -> str:
    """Gap closed in the target language and over-refusal rise per language, from one judge's labels, beside the
    frozen report's values. Capability does not depend on the judge, so it is left to the report."""
    from langsafe.config import load_spec
    from langsafe.eval.axes import safety_table
    from langsafe.eval.report import gap_closed

    th = load_spec()['thresholds']
    rows = {tag: read(judged / f'{tag}.{judge}-full.jsonl') for tag in (reference, drifted, repaired)}
    report = json.loads(frozen.read_text(encoding='utf-8'))['thresholds']
    lines = [f'## Repair {repaired}, target language {target}', '',
             f"Gap closed: ({repaired} minus {drifted}) over ({reference} minus {drifted}), content-safe rate in "
             f"{target}; the bar is {th['harmful_gap_closed_min']:.0%}. Over-refusal rise: {repaired} minus {drifted}, "
             f"benign prompts refused, per language; the bar is {100 * th['over_refusal_increase_max']:.0f} points.", '',
             f'| Preamble | Gap closed, {judge} | Gap closed, frozen report | Largest over-refusal rise, {judge} '
             '| Largest, frozen report |', '|---|---:|---:|---|---|']
    for condition in sorted({r['preamble'] for r in rows[reference]}, key=lambda c: (c != 'default', c)):
        tables = {tag: safety_table([r for r in rows[tag] if r['preamble'] == condition], resamples) for tag in rows}
        rate = lambda tag, lang, metric: tables[tag].get(lang, {}).get(metric, {}).get('rate')
        closed = gap_closed(rate(reference, target, 'content_safe'), rate(drifted, target, 'content_safe'),
                            rate(repaired, target, 'content_safe'))
        rise = {lang: rate(repaired, lang, 'over_refusal') - rate(drifted, lang, 'over_refusal') for lang in LANGS
                if rate(repaired, lang, 'over_refusal') is not None and rate(drifted, lang, 'over_refusal') is not None}
        top = max(rise, key=rise.get)
        entry = report.get(f'{repaired}/{condition}', {})
        frozen_closed = entry.get('harmful_gap_closed', {}).get('value')
        frozen_rise = entry.get('over_refusal_increase', {}).get('by_lang', {})
        frozen_top = max(frozen_rise, key=frozen_rise.get) if frozen_rise else None
        lines.append(f"| `{condition}` | {'–' if closed != closed else f'{closed:.0%}'} | "
                     f"{'–' if frozen_closed is None else f'{frozen_closed:.0%}'} | "
                     f"{100 * rise[top]:+.1f} ({top}) | "
                     + (f'{100 * frozen_rise[frozen_top]:+.1f} ({frozen_top})' if frozen_top else '–') + ' |')
    return '\n'.join(lines + ['', '"–" means the gap in the target language is zero or negative, so nothing can be '
                                  'closed.', ''])


def agreement(folder: Path, judge: str, tags: list[str]) -> str:
    """The judge's agreement with each pair, from the comparison files `router_sample.py --all` wrote beside them."""
    lines = [f'## Agreement of {judge} with each pair, on every answer', '',
             "Refusal: Cohen's kappa over answers both sides labelled refused or complied. Content: share of "
             'harmful-prompt answers both sides decided that got the same label.', '',
             '| Model | Against | ' + ' | '.join(LANGS) + ' |', '|---|---|' + '---|' * len(LANGS)]
    for tag in tags:
        path = folder / f'{tag}.{judge}-full-comparison.json'
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding='utf-8'))
        for key, name in (('vs_local_pair', '8B pair'), ('vs_router_pair', '32B pair')):
            if not data.get(key):
                continue
            refusal, content = data[key]['refusal_axis']['by_lang'], data[key]['content_axis']['by_lang']
            lines.append(f'| {tag} | {name} | ' + ' | '.join(
                f"κ {refusal[l]['kappa']:.2f}, {100 * content[l]['agreement']:.1f}%" if l in refusal and l in content
                else '–' for l in LANGS) + ' |')
    return '\n'.join(lines + [''])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--judged', type=Path, default=FULL / 'earth-command-a-full' / 'judged')
    parser.add_argument('--judge', default='command-a')
    parser.add_argument('--reference', default='global')
    parser.add_argument('--drifted', default='earth')
    parser.add_argument('--report', action='append', default=[],
                        help='NAME=PATH of a report.json to set beside (default: the 8B and 32B reports)')
    parser.add_argument('--compare', default='', help='comparison models to test against the reference, e.g. fire')
    parser.add_argument('--repair', help='a repaired model to score on gap closed and over-refusal, e.g. '
                                         'earth-circuit_merge-0p5')
    parser.add_argument('--target', help="the repairs' target language (default: the frozen run's selection.json)")
    parser.add_argument('--frozen', type=Path, default=FULL / 'earth' / 'report.json',
                        help='the report whose repair values sit beside these')
    parser.add_argument('--resamples', type=int, default=10000)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    reports = dict(item.split('=', 1) for item in args.report) or {
        '8B pair': FULL / 'earth' / 'report.json', '32B pair': FULL / 'earth-router-report' / 'report.json'}
    text = build(args.judged, args.judge, args.reference, args.drifted,
                 {name: Path(path) for name, path in reports.items()}, args.resamples)
    for other in filter(None, args.compare.split(',')):
        text += '\n' + comparison(args.judged, args.judge, args.reference, other, args.resamples)
    if args.repair:
        target = args.target or json.loads((args.frozen.parent / 'selection.json').read_text(
            encoding='utf-8'))['worst_language']
        text += '\n' + repair(args.judged, args.judge, args.reference, args.drifted, args.repair, target, args.frozen,
                              args.resamples)
    tags = [args.reference, args.drifted, *([args.repair] if args.repair else []), *filter(None, args.compare.split(','))]
    text += '\n' + agreement(args.judged.parent, args.judge, tags)
    if args.out:
        args.out.write_text(text, encoding='utf-8', newline='\n')
    sys.stdout.reconfigure(encoding='utf-8')
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
