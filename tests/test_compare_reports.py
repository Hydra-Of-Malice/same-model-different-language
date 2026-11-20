import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location('compare_reports', ROOT / 'scripts' / 'compare_reports.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report(refusal, delta, confirmed, candidate):
    cell = {'harmful_refusal': {'rate': refusal}, 'content_safe': {'rate': 0.5}, 'over_refusal': {'rate': None}}
    return {'models': {'global': {'conditions': {'default': {'jv': cell}}}},
            'gap_confirmation': {'conditions': {'default': {'jv': {'delta': delta, 'confirmed': confirmed}}}},
            'acceptance_candidate': candidate, 'empirical_repair_success': False, 'validated': False,
            'validation_gaps': ['x'] if confirmed else []}


def test_compare_writes_both_sides_of_every_cell(tmp_path):
    module = _module()
    a, b = tmp_path / 'a.json', tmp_path / 'b.json'
    a.write_text(json.dumps(_report(0.25, 0.097, True, 'm')), encoding='utf-8')
    b.write_text(json.dumps(_report(0.5, 0.091, False, 'm')), encoding='utf-8')
    out = tmp_path / 'out.md'
    assert module.main([str(a), str(b), '--labels', '8B,32B', '--tags', 'global', '--out', str(out)]) == 0
    text = out.read_text(encoding='utf-8')
    assert '| global | 25.0 → 50.0 |' in text
    assert '| global | 50.0 → 50.0 |' in text
    assert '| global | – → – |' in text  # a missing rate is a dash, not an error
    assert '| jv | 9.7 pts, confirmed | 9.1 pts, not confirmed |' in text
    assert '| Validation gaps | 1 | 0 |' in text
