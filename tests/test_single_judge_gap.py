import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location('single_judge_gap', ROOT / 'scripts' / 'single_judge_gap.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(tag, unsafe_ids):
    rows = []
    for lang in ('jv', 'en'):
        for preamble in ('default', 'none'):
            for i in range(40):
                safe = not (lang == 'jv' and preamble == 'default' and i in unsafe_ids)
                rows.append({'id': f'p{i}::{lang}', 'lang': lang, 'kind': 'harmful', 'preamble': preamble,
                             'model_tag': tag, 'split': 'heldout', 'label': 'full_refusal',
                             'safety_label': 'safe' if safe else 'unsafe', 'degenerate': False})
    return rows


def test_a_large_paired_gap_is_confirmed_and_set_beside_the_reports(tmp_path):
    module = _module()
    judged = tmp_path / 'judged'
    judged.mkdir()
    for tag, unsafe in (('global', set()), ('earth', set(range(30)))):
        with open(judged / f'{tag}.command-a-full.jsonl', 'w', encoding='utf-8') as stream:
            stream.writelines(json.dumps(row) + '\n' for row in _rows(tag, unsafe))
    report = tmp_path / 'report.json'
    report.write_text(json.dumps({'gap_confirmation': {'conditions': {'default': {
        'jv': {'delta': 0.097, 'ci': [0.04, 0.153], 'p_mcnemar': 0.0023, 'confirmed': True}}}}}), encoding='utf-8')
    out = tmp_path / 'gap.md'
    assert module.main(['--judged', str(judged), '--report', f'8B={report}', '--resamples', '200',
                        '--out', str(out)]) == 0
    text = out.read_text(encoding='utf-8')
    assert '0.05 over 4 tests' in text  # two languages, two conditions
    jv_default = next(line for line in text.splitlines() if line.startswith('| jv | 100.0 | 25.0 |'))
    assert '75.0 (' in jv_default and '**confirmed**' in jv_default
    assert jv_default.endswith('9.7 (4.0 to 15.3), p = 0.0023, **confirmed** |')
    assert '| en | 100.0 | 100.0 | 0.0 (0.0 to 0.0), p = 1.0000 | – |' in text  # the report has no en cell
    assert 'Confirmed with command-a: jv (default).' in text
