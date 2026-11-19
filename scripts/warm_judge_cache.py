#!/usr/bin/env python3
"""Fill one router judge's verdicts into a judge cache ahead of a `router_sample.py --all` run.

`judge_generations` asks the judges one after the other, so while the slower judge works through a tag the faster
one waits. This sends one judge's requests (both rubrics, every answer of the given tags) straight into the cache the
full run reads, with the same messages and parameters, so the full run finds them as cache hits. It writes no
verdict file and prints only counts.

    python scripts/warm_judge_cache.py --judge second-judge --tags global,earth --workers 32 --execute
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, default=ROOT / 'runs' / 'experiments' / 'full' / 'earth')
    parser.add_argument('--tags', required=True)
    parser.add_argument('--judge', required=True, help="a judge name from the spec's router pair")
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--retries', type=int, default=10)
    parser.add_argument('--chunk', type=int, default=2000)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    run = args.run.resolve()
    out = run.parent / f'{run.name}-router-full'
    os.environ['LANGSAFE_RUNS_DIR'] = str(out)
    os.environ.setdefault('LANGSAFE_SPEC', str(ROOT / 'configs' / 'spec.yaml'))
    from langsafe import llm_client
    from langsafe.config import load_spec
    from langsafe.data.records import read_prompts
    from langsafe.judge.llm_judge import LLMJudge

    spec = load_spec()
    cfg = next(c for c in spec['judging']['judges'] if c['name'] == args.judge)
    prompts = {p.id: p for p in read_prompts(run / 'data' / 'heldout.jsonl')}
    refusal, content = {}, {}
    for tag in args.tags.split(','):
        with open(run / 'gens' / f'{tag}.jsonl', encoding='utf-8') as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    item = (prompts[row['id']].text, row['response'], row['lang'])
                    refusal[item] = None
                    if row['kind'] == 'harmful':
                        content[item] = None
    print(json.dumps({'judge': args.judge, 'model': cfg['model'], 'refusal_items': len(refusal),
                      'content_items': len(content)}), flush=True)
    if not args.execute:
        return 0
    if not os.environ.get('HF_TOKEN'):
        from huggingface_hub import get_token
        token = get_token()
        if not token:
            raise SystemExit('No Hugging Face login on this machine; run `hf auth login` yourself first')
        os.environ['HF_TOKEN'] = token

    usage = defaultdict(int)
    lock = threading.Lock()
    original = llm_client.ChatClient.__init__

    def __init__(self, *a, **k):
        k.setdefault('workers', args.workers)
        k.setdefault('max_retries', args.retries)
        original(self, *a, **k)
        create = self.client.chat.completions.create

        def counted(**request):
            response = create(**request)
            u = getattr(response, 'usage', None)
            with lock:
                usage['calls'] += 1
                usage['prompt_tokens'] += getattr(u, 'prompt_tokens', 0) or 0
                usage['completion_tokens'] += getattr(u, 'completion_tokens', 0) or 0
            return response

        self.client.chat.completions.create = counted

    llm_client.ChatClient.__init__ = __init__
    judge = LLMJudge(cfg['name'], cfg['base_url'], cfg['model'])
    start = time.time()
    try:
        for rubric, items, call in (('refusal', list(refusal), judge.judge_many),
                                    ('content', list(content), judge.judge_safety_many)):
            for i in range(0, len(items), args.chunk):
                call(items[i:i + args.chunk])
                print(json.dumps({'rubric': rubric, 'done': min(i + args.chunk, len(items)), 'of': len(items),
                                  'billed_calls': usage['calls'], 'minutes': round((time.time() - start) / 60, 1)}),
                      flush=True)
    finally:
        judge.close()
        with open(out / 'usage.jsonl', 'a', encoding='utf-8') as stream:
            stream.write(json.dumps({'warm_cache': args.judge, 'tags': args.tags,
                                     'usage': {cfg['model']: dict(usage)}}) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
