#!/usr/bin/env python3
"""Build the report again with the 32B router pair's verdicts on every held-out answer, beside the frozen local report.

The spec makes the router pair (Aya Expanse 32B and Qwen3-32B) the main judges; the frozen report used the local 8B
pair because the router pair had only judged a shared sample. `router_sample.py --all` judges every answer into
`<run>-router-full/judged/<tag>.router-full.jsonl`. This script makes `<run>-router-report/`, a folder shaped like a
run: `judged/<tag>.jsonl` (and its summary) are the router verdicts, and every other input the report reads (data,
selection, generations, capability results, calibration) is hard-linked from the run, so nothing is copied and the
run itself is never written. It then builds the report there with judge set `router`.

    python scripts/router_report.py                      # tags and family from the frozen report.json
    python scripts/router_report.py --tags global,earth  # a subset

The report folder holds verdicts on harmful answers, so it stays in runs/, which git ignores.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
LINKED = ('data', 'gens', 'capability', 'calibration', 'selection.json')


def link_tree(source: Path, target: Path) -> None:
    """Hard-link a file, or every file under a folder; copy where a link cannot be made (another volume)."""
    files = [source] if source.is_file() else [p for p in source.rglob('*') if p.is_file()]
    for path in files:
        destination = target / path.relative_to(source) if source.is_dir() else target
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            continue
        try:
            os.link(path, destination)
        except OSError:
            shutil.copy2(path, destination)


def prepare(run: Path, full: Path, out: Path, tags: list[str]) -> list[str]:
    """Lay out the report folder. Returns the tags whose router verdicts are missing."""
    for name in LINKED:
        if (run / name).exists():
            link_tree(run / name, out / name)
    missing = []
    (out / 'judged').mkdir(parents=True, exist_ok=True)
    for tag in tags:
        verdicts = full / 'judged' / f'{tag}.router-full.jsonl'
        summary = verdicts.with_suffix('.summary.json')
        if not verdicts.is_file() or not summary.is_file():
            missing.append(tag)
            continue
        for source, name in ((verdicts, f'{tag}.jsonl'), (summary, f'{tag}.summary.json')):
            target = out / 'judged' / name
            if target.exists():
                target.unlink()  # a newer --all run replaces the verdicts
            link_tree(source, target)
    return missing


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, default=ROOT / 'runs' / 'experiments' / 'full' / 'earth')
    parser.add_argument('--tags', help="report tags in order (default: the frozen report's)")
    parser.add_argument('--family', help="language family (default: the frozen report's)")
    args = parser.parse_args(argv)
    run = args.run.resolve()
    frozen = json.loads((run / 'report.json').read_text(encoding='utf-8'))
    tags = args.tags.split(',') if args.tags else list(frozen['models'])
    family = args.family or frozen['family']
    full = run.parent / f'{run.name}-router-full'
    out = run.parent / f'{run.name}-router-report'
    os.environ.setdefault('LANGSAFE_SPEC', str(ROOT / 'configs' / 'spec.yaml'))
    from langsafe.config import load_spec
    from langsafe.experiment_report import build_report, comparison_tags

    spec = load_spec()
    missing = prepare(run, full, out, [*tags, *comparison_tags(spec, family)])
    if any(tag in missing for tag in tags):
        raise SystemExit('No router verdicts yet for: ' + ', '.join(t for t in tags if t in missing)
                         + ' (run router_sample.py --all --execute first)')
    report = build_report(tags, family, 'router', out)
    print(json.dumps({'report': str(report), 'tags': tags, 'family': family,
                      'comparison_tags_missing': [t for t in missing if t not in tags]}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
