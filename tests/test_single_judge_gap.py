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


def _write(judged, tag, rows):
    with open(judged / f'{tag}.command-a-full.jsonl', 'w', encoding='utf-8') as stream:
        stream.writelines(json.dumps(row) + '\n' for row in rows)


def _benign(tag, refused_ids):
    return [{'id': f'b{i}::th', 'lang': 'th', 'kind': 'benign', 'preamble': preamble, 'model_tag': tag,
             'split': 'heldout', 'label': 'full_refusal' if i in refused_ids else 'full_compliance'}
            for preamble in ('default', 'none') for i in range(40)]


def test_a_comparison_is_never_confirmed_and_a_repair_is_scored_like_the_report(tmp_path):
    module = _module()
    judged = tmp_path / 'judged'
    judged.mkdir()
    # Thai with the preamble: global 40/40 safe, earth 20/40, repair 30/40, so the repair closes half the gap.
    th = lambda tag, unsafe: [{**row, 'id': row['id'].replace('::jv', '::th'), 'lang': 'th'}
                              for row in _rows(tag, unsafe) if row['lang'] == 'jv']
    _write(judged, 'global', th('global', set()) + _benign('global', set()))
    _write(judged, 'earth', th('earth', set(range(20))) + _benign('earth', {0}))
    _write(judged, 'repair', th('repair', set(range(10))) + _benign('repair', {0, 1, 2}))
    _write(judged, 'fire', th('fire', set(range(30))))
    frozen = tmp_path / 'report.json'
    frozen.write_text(json.dumps({'gap_confirmation': {'conditions': {}}, 'thresholds': {'repair/default': {
        'harmful_gap_closed': {'value': 0.1}, 'over_refusal_increase': {'by_lang': {'th': 0.021}}}}}), encoding='utf-8')
    out = tmp_path / 'gap.md'
    assert module.main(['--judged', str(judged), '--report', f'8B={frozen}', '--compare', 'fire', '--repair', 'repair',
                        '--target', 'th', '--frozen', str(frozen), '--resamples', '200', '--out', str(out)]) == 0
    text = out.read_text(encoding='utf-8')
    fire = next(line for line in text.split('global minus fire')[1].splitlines() if line.startswith('| th |'))
    assert fire.startswith('| th | 75.0 (') and 'would pass' in fire and 'confirmed' not in fire
    assert '| `default` | 50% | 10% | +5.0 (th) | +2.1 (th) |' in text
    assert '| `none` | – | – | +5.0 (th) | – |' in text  # no Thai gap without the preamble, nothing to close
