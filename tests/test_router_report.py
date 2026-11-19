import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location('router_report', ROOT / 'scripts' / 'router_report.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _snapshot(folder):
    return {str(p.relative_to(folder)): p.read_bytes() for p in folder.rglob('*') if p.is_file()}


def test_prepare_lays_out_a_run_shaped_folder_without_touching_the_run(tmp_path):
    module = _module()
    run, full, out = tmp_path / 'earth', tmp_path / 'earth-router-full', tmp_path / 'earth-router-report'
    for name, text in (('data/manifest.json', '{}'), ('data/heldout.jsonl', ''), ('gens/global.jsonl', 'g'),
                       ('capability/global.json', '{}'), ('calibration/global.local.json', '{}'),
                       ('selection.json', '{}'), ('judged/global.local.jsonl', 'local')):
        (run / name).parent.mkdir(parents=True, exist_ok=True)
        (run / name).write_text(text, encoding='utf-8')
    (full / 'judged').mkdir(parents=True)
    (full / 'judged' / 'global.router-full.jsonl').write_text('router v1', encoding='utf-8')
    (full / 'judged' / 'global.router-full.summary.json').write_text('{"v": 1}', encoding='utf-8')
    before = _snapshot(run)

    assert module.prepare(run, full, out, ['global', 'earth']) == ['earth']
    assert (out / 'judged' / 'global.jsonl').read_text(encoding='utf-8') == 'router v1'
    assert (out / 'judged' / 'global.summary.json').read_text(encoding='utf-8') == '{"v": 1}'
    for name in ('data/manifest.json', 'gens/global.jsonl', 'capability/global.json', 'calibration/global.local.json',
                 'selection.json'):
        assert (out / name).read_bytes() == (run / name).read_bytes()
    assert not (out / 'judged' / 'global.local.jsonl').exists()  # the local verdicts are not carried over

    # A newer --all run replaces the verdicts; the run itself is never written.
    (full / 'judged' / 'global.router-full.jsonl').unlink()
    (full / 'judged' / 'global.router-full.jsonl').write_text('router v2', encoding='utf-8')
    assert module.prepare(run, full, out, ['global']) == []
    assert (out / 'judged' / 'global.jsonl').read_text(encoding='utf-8') == 'router v2'
    assert _snapshot(run) == before
