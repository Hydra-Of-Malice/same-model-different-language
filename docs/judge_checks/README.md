# Judge checks: aggregates only

The numbers behind the article's judge-agreement sentences (`KAPPA_LOCAL_VS_ROUTER` and `COMMAND_A_SENTENCE` in
`docs/submission/sentences.json`). Each file holds kappa, agreement and label counts per language. None holds a
prompt, an answer or a verdict. The judged rows they come from stay in `runs/`, which git ignores, because they quote
model answers.

| Files | What was judged | Made by | Cost |
|---|---|---|---|
| `global`, `earth`, `earth-circuit_merge-0p5` `.router-sample-comparison.json` | 1,680 held-out answers (20 per language, prompt type and preamble cell) re-judged by the 32B pair (Aya Expanse 32B, Qwen3-32B), compared with the local 8B pair | `scripts/router_sample.py --tags global,earth,earth-circuit_merge-0p5 --per-cell 20 --execute`, 10 Nov | about $1.0 |
| `global-q4`, `earth-q4`, `earth-circuit_merge-0p5-q4` `.router-sample-comparison.json` | The same design on the three Q4_K_M models | the same script with the `-q4` tags, 11 Nov | about $1.0 |
| `*.command-a-sample-comparison.json` | The first sample judged by Command A alone, compared with both pairs. `verified` is false by design: the spec calibrates a pair against a pair | `--judge-set command-a`, 11 to 12 Nov | about $4.60 |
| `router-sample-usage.json`, `command-a-sample-usage.json` | Billed calls and tokens per run. The first router entry over-counts calls; its `correction` entry explains why | | |
| `full/*.router-full-comparison.json` | Every held-out answer of all 18 tags (16 report tags, Fire and Water; 4,528 answers each) judged by the 32B pair, compared with the 8B pair. `verified` is true for every tag | `scripts/warm_judge_cache.py` once per judge, then `scripts/router_sample.py --all --execute --cache-only` per tag, 19 to 20 Nov | see `full/usage.json` |
| `full-32b-vs-8b.md` | The report built again on the 32B verdicts (`scripts/router_report.py`) beside the frozen 8B report: rates per language, the drift-gap test, the overall verdicts and the agreement of the two pairs | `scripts/compare_reports.py`, 20 Nov | none |

With the 32B pair as judges the acceptance candidate (earth-circuit_merge-0p5), the repair verdict (not shown) and
the validation (no gaps) are the same as in the frozen report. One result moves: the Javanese drift gap is 9.1 points
under the 32B pair (9.7 under the 8B pair) and no longer passes the Bonferroni threshold (p 0.0195 against 0.0036), so
no gap is confirmed under the 32B pair. The 32B pair also labels far fewer answers "unclear" in Bengali and Swahili,
which raises the refusal rates there. The frozen report stays the submitted one; this check is how sensitive it is
to the judges.

The calibration files the report uses are written separately, by `--write-calibration`, into the run's
`calibration/` folder. Only the three full-precision tags have them, so the final report is unchanged by the 4-bit
and Command A checks.
