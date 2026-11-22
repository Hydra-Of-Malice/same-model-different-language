# Practical uses and the Javanese follow-up

The project supports offline multilingual model and runtime evaluation. A private LoRA pilot improved Indonesian–Javanese translation in both directions on NusaX-MT and exposed a small decline on an English-to-Indonesian control. These results apply to benchmark translation from the released circuit-merge checkpoint. [Project README](../README.md) · [Released checkpoint](https://huggingface.co/HydraOfMalice/tiny-aya-earth-langsafe-circuit-merge-0p5)

## What the existing evidence supports

On harmful Javanese prompts in the historical holdout, with the default preamble and FP16 weights, the local 8B pair estimated a Global–Earth content-safe gap of 9.7 percentage points (95% CI 4.0–15.3, p = 0.0023). Command A estimated 10.8 points (p = 0.0013). The 32B pair estimated 9.1 points (p = 0.0195), above the prespecified significance cutoff of 0.00357. The local pair classified one of 176 answers as unsafe for each model. Earth had 10 more degenerate and seven more disputed answers. On benign Javanese prompts, 7.5% of Earth's responses were degenerate versus 0.7% of Global's. The blind rereads used model readers, and calibration by native speakers remains open. [Run findings](PROGRESS.md#7-november-2026) · [Full judge comparison](judge_checks/full-32b-vs-8b.md) · [Command A comparison](judge_checks/command-a-full-gap.md)

The historical Javanese safety prompts came from MultiJail translations. The benign prompts were translated by Tiny Aya Global from English XSTest. Script and format checks cannot establish meaning or distinguish Latin-script Javanese from Indonesian. General capability was measured using Indonesian as a proxy. The frozen repair run targeted Thai, where the gap missed the corrected cutoff, and no repair passed the acceptance thresholds. [Dataset card](submission/filled/dataset_card.md#translation) · [Results and limitations](../article/README.md#711-limitations)

## Use cases in priority order

| Priority | Use case | Evidence and next check |
|---|---|---|
| Available | Offline multilingual release checks | Compare two checkpoints on paired prompts and surface language-specific changes in invalid responses, refusal, or capability. Calibrate model judges against blind native review before relying on code-mixed or disputed cells. The results explorer can share aggregate findings with reviewers. [Project README](../README.md#what-we-found) · [Review workflow](REVIEW_WORKFLOW.md) · [Results explorer](../space/README.md) |
| Available | Local inference and compression research | Compare memory and throughput alongside per-language quality on the same checkpoint. Measure FP16, NF4, and GGUF separately, since runtime and precision can both affect answers. [Quantization results](../article/README.md#78-does-it-survive-4-bit) |
| First pilot | Indonesian–Javanese translation for internal localization and human-reviewed drafting | Measure both directions explicitly. Use a native reviewer to check faithfulness, register, and unintended Indonesian carryover before using outputs in a workflow. [NusaX-MT card](https://huggingface.co/datasets/indonlp/NusaX-MT) |
| Next | Source-grounded Javanese summarization, extraction, and document QA | Use short, licensed source documents and known answers. Score source support, language, completeness, and abstention when a document lacks the answer. Test separately because translation scores leave these tasks open. [NusaWrites](https://github.com/IndoNLP/nusa-writes) |
| Later | Register-controlled rewriting and educational or cultural materials | Validate separately with native-authored Ngoko, Krama, and code-mixed prompts. Start with offline previews and human review. Public-facing advice, autonomous actions, moderation enforcement, and high-stakes settings require task-specific evaluation and licensing. [Dataset scope](submission/filled/dataset_card.md#known-gaps) · [Model scope and licence](submission/filled/model_card.md#intended-use) |

Tiny Aya weights carry CC BY-NC 4.0 and the Cohere Labs Acceptable Use Policy. The released checkpoint's card limits it to research and documents the applicable Lexsi deployment terms. Offline research can evaluate output-language checks, empty or repetitive answers, source echoes, and referral to a human when quality checks fail. [Released model card](https://huggingface.co/HydraOfMalice/tiny-aya-earth-langsafe-circuit-merge-0p5) · [Cohere policy](https://docs.cohere.com/docs/c4ai-acceptable-use-policy)

## Where Hugging Face credits help

- Use capped GPU Jobs to train adapters and run paired release checks on the intended hardware. Use matching runtimes to isolate precision effects when revisiting NF4 and GGUF. [Jobs quickstart](https://huggingface.co/docs/hub/en/jobs-quickstart) · [Jobs pricing](https://huggingface.co/docs/hub/jobs-pricing)
- Use Inference Providers if a targeted new judge comparison needs a third opinion. Reuse the full-run judge aggregates for historical questions. Cache repeat judgments and pin the provider before paying for new evaluations. [Historical judge checks](judge_checks/README.md) · [Inference Providers pricing](https://huggingface.co/docs/inference-providers/pricing)
- The existing aggregate results explorer targets the free CPU tier. GPU hosting is relevant only if a separate, evaluated workflow needs live inference and the license permits that use. [Results explorer](../space/README.md) · [Hugging Face pricing](https://huggingface.co/pricing)

## Pilot and evaluation order

1. Train one private LoRA adapter from the existing `circuit-merge-0p5` checkpoint on Hugging Face Jobs. Use the NusaX-MT official training split in both Indonesian–Javanese directions. The dataset authors report native speakers wrote and verified the translations. The underlying sentences are Indonesian comments and reviews, so this is a narrow translation test. Keep the official validation and test groups separate from training, including all directions and language variants of each parallel item. Record data revision, split fingerprints, and decoding settings. NusaX-MT is CC BY-SA 4.0. Use FLORES-200 only for evaluation under its Hugging Face access terms. [NusaX source and license](https://github.com/IndoNLP/nusax) · [Official splits](https://github.com/IndoNLP/nusax/tree/main/datasets/mt) · [FLORES terms](https://huggingface.co/datasets/facebook/flores)
2. Compare the parent checkpoint and adapter on identical official test items and decoding settings. Report chrF++ by direction, empty outputs, source echoes, generation-limit hits, and the English-to-Indonesian control. Use a paired bootstrap over sentence-level chrF++ changes. chrF++ measures overlap with reference translations, so a native review should assess meaning and register. A public test split leaves pretraining contamination possible. [SacreBLEU](https://github.com/mjpost/sacrebleu#chrf--chrf)
3. If the translation gain is credible, freeze a new native-authored evaluation by topic and document family. Include no-adapter, prompt-only, and adapter conditions with the default safety preamble held fixed. Check native task completion, invalid answers, over-refusal, and unsafe-output regressions separately. Evaluate the intended runtime and any quantized artifact separately. [Generation settings](../src/langsafe/gen/generate.py#L19) · [Existing review workflow](REVIEW_WORKFLOW.md) · [Quantization results](../article/README.md#78-does-it-survive-4-bit)

## First pilot results

The job completed 80 optimizer steps with a rank-8 adapter and evaluated 120 rows per direction at BF16 on one NVIDIA A10G. The default Tiny Aya preamble, checkpoint revision, and greedy decoding were identical for both arms. GPU runtime was 10 minutes 15 seconds, about $0.17 at the listed $1.00 per hour. The adapter and aggregate metrics are in a private Hub repository. [Job](https://huggingface.co/jobs/PDGamerSG/6b0201034fdc5ab3c55e3bb8) · [Private adapter and metrics](https://huggingface.co/PDGamerSG/tiny-aya-javanese-translation-pilot-v1/tree/c49da652d879a4596236226c5967411c0177cc9c) · [Jobs pricing](https://huggingface.co/docs/hub/jobs-pricing)

| Direction | Parent chrF++ | Adapter chrF++ | Corpus change | Mean sentence change, 95% CI |
|---|---:|---:|---:|---:|
| Indonesian → Javanese | 44.17 | 53.57 | +9.41 | +9.42 [7.86, 11.11] |
| Javanese → Indonesian | 54.94 | 64.66 | +9.72 | +10.93 [8.85, 12.97] |
| English → Indonesian control | 47.97 | 47.09 | −0.88 | −1.26 [−2.28, −0.30] |

Indonesian → Javanese generation-limit hits fell from 13 to four and avoidable source echoes fell from two to zero. Javanese → Indonesian source echoes increased from zero to three. All outputs contained at least one word character. Five Indonesian → Javanese reference continuations exceed the 96-token generation limit when turn markers are included. The limit is shared by both arms. A separate five-character shingle audit at Jaccard 0.8 flagged one English near-match across the official splits. Its test row was outside the first evaluation sample, and its training row was outside the first run's training sample. Safety, over-refusal, and general capability still require separate evaluation. [Private aggregate metrics](https://huggingface.co/PDGamerSG/tiny-aya-javanese-translation-pilot-v1/blob/c49da652d879a4596236226c5967411c0177cc9c/pilot_metrics.json)

## Follow-up results on a second holdout sample

The follow-up trained a fresh adapter from the same checkpoint and replaced 64 of 320 training examples with English → Indonesian from the same training split. It tested both adapters and the parent on 120 rows outside the first evaluation sample. This run took 14 minutes 6 seconds. Both jobs together cost about $0.41 at list price, estimated from GPU runtime. [Follow-up job](https://huggingface.co/jobs/PDGamerSG/6b0204e6efaf5e14f4b66454) · [Private adapter and metrics](https://huggingface.co/PDGamerSG/tiny-aya-javanese-translation-pilot-v2/tree/271039d85b82f83a89ff3eaae63849a9f3bd8c74)

| Direction | Parent chrF++ | First adapter | Adapter with English examples | First adapter change from parent |
|---|---:|---:|---:|---:|
| Indonesian → Javanese | 45.10 | 54.64 | 52.18 | +9.54 |
| Javanese → Indonesian | 56.78 | 64.84 | 62.89 | +8.06 |
| English → Indonesian control | 49.90 | 48.47 | 48.59 | −1.43 |

The first adapter had the largest Javanese gains on both holdout samples. On the second sample it scored 9.54 and 8.06 corpus points above the parent in the two Javanese directions. The corresponding mean sentence changes were +10.56 [8.70, 12.60] and +8.87 [6.63, 11.17]. Adding English examples gained just 0.12 corpus points on the control versus the first adapter, with a mean sentence change of +0.25 [−0.99, 1.49]. The English-mix adapter lost 2.47 and 1.94 corpus points in the Javanese directions versus the first adapter. It also produced six avoidable Javanese source echoes versus three from the first adapter and one from the parent. Use the first adapter for further offline Javanese translation evaluation. Check native translation quality and safety regressions before drawing broader conclusions. [Private aggregate metrics](https://huggingface.co/PDGamerSG/tiny-aya-javanese-translation-pilot-v2/blob/271039d85b82f83a89ff3eaae63849a9f3bd8c74/pilot_metrics.json)

## Local checkpoint verification

The 6.7 GB parent checkpoint was downloaded with fixed byte ranges, assembled at the expected size, and loaded locally through Transformers on Apple MPS. The private first adapter attached through PEFT. A four-row smoke sample per direction reproduced the cloud result directionally:

| Direction | Parent chrF++ | Adapter chrF++ | Change |
|---|---:|---:|---:|
| Indonesian → Javanese | 52.53 | 64.41 | +11.88 |
| Javanese → Indonesian | 52.48 | 67.50 | +15.02 |
| English → Indonesian control | 50.65 | 50.09 | −0.56 |

The local MPS runtime became memory-bound during an attempted 120-row, 360-pair pass and that pass was stopped. The full held-out comparison remains the completed A10G result above. The local checkpoint is retained under the ignored cache/huggingface/ directory for another run on a CUDA host or with a smaller MPS sample.

## Use-case survey

**22 November 2026.** We ranked what a 3B on-device model could do for Javanese and its neighbours by three things: evidence of need, open data to train and test on, and whether a LoRA adapter can plausibly deliver it. Sizes, licences and splits below were read from the dataset cards or files. Published scores are as their papers report them.

| Rank | Use case | Evidence of a gap | Open data | In this repo |
|---|---|---|---|---|
| 1 | Translation between Indonesian, English and Javanese | Strong systems score about 9 chrF++ lower into Javanese than out of it on FLORES-200 (NLLB-200 MoE: 49.8 against 59.0). The Tiny Aya report puts Earth at 0.09 chrF for English to Javanese, against 0.22 for Global. [NLLB metrics](https://dl.fbaipublicfiles.com/large_objects/nllb/models/nllb_200_moe_54b/metrics.csv) · [Tiny Aya report](https://arxiv.org/abs/2603.11510) | NusaX-MT (500 / 100 / 400 rows, 12 languages). NusaTranslation (6,600 / 849 / 2,000 Indonesian–Javanese tweets, translated by native speakers) | Trained and tested: `--pairs`, `--tweets` |
| 2 | Speech levels: Ngoko, Ngoko Alus, Krama, Krama Alus | GPT-4o classifies the four levels at 53.5%, a fine-tuned DistilBERT at 95.65%. [Unggah-Ungguh paper](https://arxiv.org/abs/2502.20864) | Unggah-Ungguh: 2,009 sentences in 746 meaning groups, CC BY-NC 4.0, no official split | Trained and tested: `--levels` |
| 3 | Sentiment, topic and emotion of regional-language text | Small encoders already do this well, so the value is one model for many tasks | NusaX-senti (same sentences and splits as NusaX-MT). NusaParagraph: paragraphs written in Javanese, 8 topics and 7 emotions | Trained and tested: `--senti`, `--paragraphs` |
| 4 | Customer-message triage: language, sentiment and an Indonesian translation in one JSON reply | Community Tiny Aya adapters already target structured extraction in other languages. [Hub listing](https://huggingface.co/models?other=base_model:adapter:CohereLabs/tiny-aya-global) | Built from NusaX rows, which are product and service comments | Trained and tested: `--triage` |
| 5 | Telling Javanese from Indonesian and other regional languages | Mixing the two is normal on social media. GlotLID over-predicts Indonesian (precision about 0.71). [GlotLID](https://huggingface.co/cis-lmu/glotlid) | NusaX as a 12-way set. IJELID has token-level labels, CC BY-NC-SA | Trained and tested as sentence-level identification: `--lid`, `--regional` |
| 6 | Other regional languages: Sundanese, Balinese, Madurese, Minangkabau, Banjarese, Acehnese, Buginese, Toba Batak, Ngaju | NusaMT-7B improved Indonesian to Balinese from 14.85 (NLLB-3.3B) to 22.95 spBLEU with LoRA. [NusaMT-7B](https://arxiv.org/abs/2410.07830) | NusaX-MT and NusaX-senti | Trained and tested: `--regional` |
| 7 | Reading comprehension and document questions | No training set written in Javanese. Large ones are machine-translated | Belebele Javanese, 900 questions, test only | Not yet. Evaluation only is possible |
| 8 | Voice-assistant intents | Demand evidence is thin | MASSIVE `jv-ID`, 11,514 / 2,033 / 2,974, CC BY 4.0 | Not yet |
| 9 | Chat, cultural knowledge, Javanese script, summarisation | IndoMMLU: GPT-3.5 scores 39.3% on local languages and culture against 62.2% on Indonesian. [IndoMMLU](https://huggingface.co/datasets/indolem/IndoMMLU) | Human-written instruction data is tiny (247 Javanese rows in the Aya Dataset). Script and summarisation sets are evaluation-sized or machine-made | Out of scope for a LoRA adapter on this data |

Three things the survey ruled out as training data: FLORES-derived sets (SIB-200, Belebele), because training on them spoils FLORES as a test; machine-generated Javanese (Gatra, Centhini, translated Alpaca); and the SEA-HELM Javanese translation and sentiment sets, which are sampled from the NusaX test split.

## Expanded adapter

[`scripts/nusantara_adapter.py`](../scripts/nusantara_adapter.py) replaces the pilot script for further runs. It trains one adapter on several tasks, picks the epoch on validation data, and scores the parent and the adapter on identical test items. The parent is the same loaded model with the adapter switched off.

| Flag | Task | Source and split | Scored by |
|---|---|---|---|
| `--pairs ind-jav,eng-jav` | Translation, both directions | NusaX-MT official split | chrF++, BLEU, source echoes, limit hits |
| `--tweets` | Translation of informal tweets | NusaTranslation official split | Same, on keys ending in `_tweets` |
| `--senti jav` | Sentiment, three classes | NusaX-senti labels joined to the same rows | Accuracy, macro-F1, exact McNemar |
| `--triage jav` | One JSON reply: language, sentiment, Indonesian translation | NusaX rows | Valid JSON, both labels right, chrF++ of the translation |
| `--levels` | Speech-level classification, level-controlled translation from Indonesian, Ngoko–Krama rewriting | Unggah-Ungguh, split by meaning group: 1,398 / 176 / 429 sentences | Accuracy and chrF++ |
| `--paragraphs` | Topic and emotion of native paragraphs | NusaParagraph official split | Accuracy, macro-F1 |
| `--lid ind,jav` | Language identification | NusaX rows | Accuracy |
| `--regional` | All of the NusaX tasks for nine more languages | NusaX official split | As above, on 200 test rows per key |

English to Indonesian stays untrained in every run and is scored as the control.

**Leakage guards.** A training or validation row is dropped when its normalized text, in any language, also appears in a later split of any loaded source. NusaWrites reuses text across its tasks, and the guard drops 202 of its training examples. The two pilot runs scored 240 of the 400 NusaX test rows, so every report also gives the metrics on the 160 rows nothing has scored before.

**Recipe.** The pilot trained 1.9 million parameters on 320 examples. The new defaults follow the strongest evidence we found, a study that fine-tuned Tiny Aya Global for translation: rank 64 on all seven projections, where rank 4 scored 8 to 14 COMET points lower. Learning rate 1e-4 with 5% warm-up and cosine decay, weight decay 0.01, dropout 0.05, batches capped at 6,000 padded tokens, and the epoch chosen by mean validation chrF++ on the two Indonesian–Javanese directions. `--scales 0.5,0.75` also scores the adapter at reduced strength, which costs no training and is the cheapest lever if the safety check below fails. [AfriScience-MT](https://arxiv.org/abs/2605.29741) · [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673) · [PEFT adapter scaling](https://huggingface.co/docs/peft/package_reference/helpers)

Published reference points on the NusaX-MT test split, SacreBLEU, for models fine-tuned on the same 500 training rows: IndoBARTv2 39.97 into Javanese and 50.64 into Indonesian. The script reports BLEU beside chrF++ so the two can be set side by side. [NusaX paper](https://arxiv.org/abs/2205.15960)

## Safety check for an adapter

Benign fine-tuning can weaken refusals: one study of 0.6B to 4B models found adversarial compliance rising up to four-fold after 1,000 benign LoRA examples. So no adapter is described as usable until [`scripts/adapter_safety_check.py`](../scripts/adapter_safety_check.py) has run on it. [Hawkins et al.](https://arxiv.org/abs/2606.28843)

The script fetches this repository at a pinned commit and runs the frozen pipeline on the published held-out prompts for Javanese and English, in both preamble conditions: it generates with the parent and with the adapter on the same GPU, judges both with the router pair (Aya Expanse 32B on Cohere, Qwen3-32B pinned to DeepInfra), and scores Global-MMLU-Lite in Indonesian and English. It reports paired differences and applies the thresholds from the specification: over-refusal up by at most 2 points, capability down by at most 1 point, and no language significantly less content-safe. Answers and judge text stay on the job's disk. Only aggregates are uploaded, to a private repository.

Indonesian has no safety prompts in this project, so Indonesian is covered for capability only.

## Run plan

All runs start from the same pinned checkpoint and upload to one private repository, one folder per run. Prices are the listed Hugging Face Jobs rates on 22 November: L40S (48 GB) $1.80 per hour, A10G large $1.50.

| Run | Flags | Training examples | Question it answers |
|---|---|---:|---|
| `probe` | `--train-rows 40 --epochs 1 --eval-rows 8` | about 500 | Does the script run on CUDA, and how much memory does a batch need? |
| `core-r64` | defaults, `--epochs 8`, `--compare-adapter`, `--scales 0.5,0.75` | 2,500 | How far do the recipe and the full split move the pilot's numbers? |
| `core-attention-r8` | `--modules attention --rank 8 --epochs 8` | 2,500 | How much of that is capacity, how much is training length? |
| `suite-r64` | `--triage jav --levels --tweets --paragraphs --lid ind,jav --epochs 3` | about 26,000 | Does one adapter serve every Javanese use case, and does the extra data help or hurt NusaX translation? |
| `regional-r64` | `--regional --epochs 3` | about 27,000 | Do the other nine languages gain, and does mixing them cost Javanese? |
| safety check | on the adapter chosen by validation score | — | Is the chosen adapter still as safe and as capable as its parent? |

Expected total is under $20 of the available credits, including about $2 of judge calls.

## Reproduction

Run the split check without a GPU:

```bash
python scripts/javanese_translation_pilot.py --data-only
```

With `hf auth login` completed, export an authenticated `HF_TOKEN` in your shell. Set `HF_PILOT_REPO` to a private model repository name in your account, such as `your-user/tiny-aya-javanese-translation-pilot`. The job creates the repository as private when needed and checks its visibility before writing:

```bash
hf jobs uv run --flavor a10g-small --timeout 60m --secrets HF_TOKEN --detach \
  scripts/javanese_translation_pilot.py --steps 80 --eval-rows 120 --private-repo "$HF_PILOT_REPO"
```

For the follow-up, use a separate private destination and add `--english-replay --compare-adapter REPO@SHA`, where `REPO@SHA` pins the first adapter and its metrics. The metrics identify the scored test rows. The training script excludes them before selecting the follow-up sample.

The published training and test splits have 500 and 400 rows, respectively. The script checks both file hashes and checks all three languages for exact normalized text overlap across the splits. It saves the adapter and aggregate metrics. The existing model loader accepts a PEFT directory or Hub adapter ID for further research evaluation. Pin the starting checkpoint and adapter by downloading their recorded revisions before using that loader. [Training script](../scripts/javanese_translation_pilot.py) · [Model loader](../src/langsafe/gen/model.py#L71) · [Hugging Face Jobs quickstart](https://huggingface.co/docs/hub/en/jobs-quickstart)

### Expanded adapter and safety check

Check every split and guard without a GPU:

```bash
python scripts/nusantara_adapter.py --data-only --triage jav --levels --tweets --paragraphs --lid ind,jav
```

With `hf auth login` completed, start a run on Hugging Face Jobs. `--secrets HF_TOKEN` passes the login to the job, which creates the destination as a private repository when needed and refuses to write to a public one:

```bash
hf jobs uv run --flavor l40sx1 --timeout 3h --secrets HF_TOKEN --detach --name suite-r64 \
  scripts/nusantara_adapter.py --triage jav --levels --tweets --paragraphs --lid ind,jav --epochs 3 \
  --private-repo "$HF_ADAPTER_REPO" --run-name suite-r64
```

Each run writes `adapter_model.safetensors`, `metrics.json` and `predictions.jsonl` to its own folder of that repository. The predictions are answers to benign translation and classification prompts, kept for native review.

Then check the chosen adapter against its parent. `REVISION` is the full commit hash of the adapter repository:

```bash
hf jobs uv run --flavor l40sx1 --timeout 3h --secrets HF_TOKEN --detach --name suite-r64-safety \
  scripts/adapter_safety_check.py --adapter-repo "$HF_ADAPTER_REPO" --adapter-revision REVISION \
  --adapter-path suite-r64 --private-repo "$HF_ADAPTER_REPO" --run-name suite-r64
```

The token needs permission to call Inference Providers, since the judges run on the router. Add `--limit 4` for a ten-minute trial of the whole path.

### Using a trained adapter

[`scripts/javanese_tools.py`](../scripts/javanese_tools.py) runs one request through a checkpoint and an adapter on CUDA, Apple MPS or CPU. It builds each prompt with the training functions, so the wording matches what the adapter saw. Export a token that can read the private repository first, for example `export HF_TOKEN="$(hf auth token)"`.

```bash
python scripts/javanese_tools.py --adapter "$HF_ADAPTER_REPO" --adapter-path suite-r64 \
  translate --source ind --target jav --level krama "Kamu makan apa hari ini?"
```

The other commands are `level`, `rewrite --level krama`, `sentiment`, `triage`, `identify --among ind,jav,sun`, `topic` and `emotion`. `--show-prompt` prints the prompt without loading weights. Leave `--adapter` out to see the parent's answer.

Compare runs as Markdown tables, from aggregates only:

```bash
python scripts/adapter_report.py core-r64 suite-r64 --repo "$HF_ADAPTER_REPO"
```

## Source search log

**22 November 2026, second pass.** Three wider searches covered use cases and datasets, fine-tuning recipes for Tiny Aya and other small models, and this repository's own evaluation path. Dataset sizes, licences and file layouts were read from the Hub API, dataset cards and the pinned files. Gated sets (FLORES+, BOUQuET, SEA-HELM) were listed but not opened. Paper figures were read from HTML renderings and have not been re-derived. The Hub listing of models built on Tiny Aya showed 388 derived models on that date.

**22 November 2026.** This was a bounded review of project evidence and primary dataset, model, and metric documentation. Search terms included `Javanese`, `invalid`, `degenerate`, `native`, `NusaX`, `NusaWrites`, `FLORES`, and `chrF++`. Project searches covered the specification, reports, cards, and generation and split code. Public-source checks covered the dataset maintainers' documentation and SacreBLEU. NusaParagraph offers native-written passages. The Gatra-1 card describes a dataset that is 98% GPT-4-generated with minimal human curation. The gated FLORES raw card returned HTTP 401. The public access form displayed the evaluation-only terms explicitly. [Specification](../SPEC.md) · [NusaParagraph](https://huggingface.co/datasets/indonlp/nusaparagraph_topic) · [Gatra-1](https://huggingface.co/datasets/afrizalha/Gatra-1-Javanese) · [FLORES dataset](https://huggingface.co/datasets/facebook/flores)
