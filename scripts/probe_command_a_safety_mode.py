"""Is Command A's default (no safety_mode sent) the same as safety_mode=CONTEXTUAL? Counts only, no text printed.

Judges 40 Global answers (20 harmful, 20 benign, both rubrics) with safety_mode set, into an empty cache, and compares the raw outputs with the ones the full run got without the field.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs' / 'experiments' / 'full' / 'earth'
JUDGED = ROOT / 'runs' / 'experiments' / 'full' / 'earth-command-a-full' / 'judged' / 'global.command-a-full.jsonl'
mode = sys.argv[1]  # CONTEXTUAL or STRICT; the router passes it to Cohere, which rejects OFF for this model
os.environ['LANGSAFE_RUNS_DIR'] = str(ROOT / 'runs' / 'experiments' / 'full' / f'command-a-mode-probe-{mode.lower()}')  # an empty cache, so every call reaches the provider
os.environ['LANGSAFE_SPEC'] = str(ROOT / 'configs' / 'spec.yaml')
sys.path.insert(0, str(ROOT / 'src'))
from huggingface_hub import get_token
os.environ['HF_TOKEN'] = get_token()
from langsafe import llm_client
from langsafe.data.records import read_prompts
from langsafe.judge.llm_judge import LLMJudge

original = llm_client.ChatClient.__init__


def __init__(self, *a, **k):
    original(self, *a, **k)
    create = self.client.chat.completions.create
    self.client.chat.completions.create = lambda **r: create(**r, extra_body={'safety_mode': mode})


llm_client.ChatClient.__init__ = __init__
prompts = {p.id: p for p in read_prompts(RUN / 'data' / 'heldout.jsonl')}
earlier = {}
with open(JUDGED, encoding='utf-8') as stream:
    for line in stream:
        row = json.loads(line)
        earlier[(row['id'], row['preamble'])] = row
picked = {'harmful': [], 'benign': []}
with open(RUN / 'gens' / 'global.jsonl', encoding='utf-8') as stream:
    for line in stream:
        row = json.loads(line)
        bucket = picked[row['kind']]
        if len(bucket) < 20 and row['lang'] != 'en' and (row['id'], row['preamble']) in earlier:
            bucket.append(row)
rows = picked['harmful'] + picked['benign']
judge = LLMJudge('command-a', 'https://router.huggingface.co/v1', 'CohereLabs/c4ai-command-a-03-2025:cohere')
try:
    items = [(prompts[r['id']].text, r['response'], r['lang']) for r in rows]
    refusal = judge.judge_many(items)
    content = judge.judge_safety_many(items[:20])
finally:
    judge.close()
same_raw = sum(v.raw == earlier[(r['id'], r['preamble'])]['verdicts']['command-a']['raw'] for r, v in zip(rows, refusal))
same_label = sum(v.label == earlier[(r['id'], r['preamble'])]['judges']['command-a'] for r, v in zip(rows, refusal))
same_safety = sum(v.label == earlier[(r['id'], r['preamble'])]['safety_votes']['command-a']
                  for r, v in zip(rows[:20], content))
print(json.dumps({'mode': mode, 'refusal_items': len(rows), 'identical_raw_output': same_raw,
                  'same_refusal_label': same_label, 'content_items': 20, 'same_content_label': same_safety}))
