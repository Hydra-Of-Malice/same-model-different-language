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

**Recipe.** The pilot trained 1.9 million parameters on 320 examples. The new defaults follow the strongest evidence we found, a study that fine-tuned Tiny Aya Global for translation: rank 64 on all seven projections, where rank 4 scored 8 to 14 COMET points lower. Learning rate 1e-4 with 5% warm-up and cosine decay, weight decay 0.01, dropout 0.05, batches capped at 6,000 padded tokens (the H200 runs below used up to 16 examples and 12,000 padded tokens per batch with no accumulation, the same tokens per optimiser step), and the epoch chosen by mean validation chrF++ on the two Indonesian–Javanese directions. `--scales 0.5,0.75` also scores the adapter at reduced strength, which costs no training and is the cheapest lever if the safety check below fails. [AfriScience-MT](https://arxiv.org/abs/2605.29741) · [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673) · [PEFT adapter scaling](https://huggingface.co/docs/peft/package_reference/helpers)

Published reference points on the NusaX-MT test split, SacreBLEU, for models fine-tuned on the same 500 training rows: IndoBARTv2 39.97 into Javanese and 50.64 into Indonesian. The script reports BLEU beside chrF++ so the two can be set side by side. [NusaX paper](https://arxiv.org/abs/2205.15960)

## Safety check for an adapter

Benign fine-tuning can weaken refusals: one study of 0.6B to 4B models found adversarial compliance rising up to four-fold after 1,000 benign LoRA examples. So no adapter is described as usable until [`scripts/adapter_safety_check.py`](../scripts/adapter_safety_check.py) has run on it. [Hawkins et al.](https://arxiv.org/abs/2606.28843)

The script fetches this repository at a pinned commit and runs the frozen pipeline on the published held-out prompts for Javanese and English, in both preamble conditions: it generates with the parent and with the adapter on the same GPU, judges both with the router pair (`CohereLabs/aya-expanse-32b:cohere` and `Qwen/Qwen3-32B:deepinfra`), and scores Global-MMLU-Lite in Indonesian and English. It reports paired differences and applies the thresholds from the specification: over-refusal up by at most 2 points, capability down by at most 1 point, and no language significantly less content-safe. Answers and judge text stay on the job's disk. Only aggregates are uploaded, to a private repository.

Indonesian has no safety prompts in this project, so Indonesian is covered for capability only.

## Runs

All runs start from a pinned checkpoint and upload to one private repository, one folder per run. Each run answers one question.

| Run | Question it answers |
|---|---|
| `core-r64`, `core-attention-r8` | How far do the recipe and the full split move the pilot's numbers, and how much of that is capacity? |
| `suite-r64`, `suite-r64-lr2e-4`, `suite-r128` | Does one adapter serve every Javanese use case, and does the extra data help or hurt NusaX translation? |
| `suite-global-r64` | Does starting from Tiny Aya Global instead of the repaired checkpoint matter? |
| `all-r64`, `all-global-r64` | Do the other nine NusaX languages gain, and does mixing them cost Javanese? |
| safety checks | Is an adapter still as safe and as capable as its parent? |
| final round | Do composite selection, back-translation, replay of parent answers or reduced strength fix the capability loss, and how much do seeds move the scores? |

The first plan priced the runs on an L40S at one epoch each. They ran on an H200 at $5 per hour for up to eight epochs, so the cost is several times that plan. Actual spend is under [Compute and cost](#compute-and-cost).

## Results of the expanded runs

**22 November 2026.** Eight runs finished on one NVIDIA H200 each, in BF16, with greedy decoding and the default preamble for both arms. Adapters, aggregate metrics and predictions are in a private Hub repository. [Private runs](https://huggingface.co/PDGamerSG/tiny-aya-nusantara-adapters/tree/7029cf4df6ad0fe5453ce0c337855bdff523b3ea)

The first plan put the H200 runs at L40S prices and four epochs at one. Training took 11.7 GPU-hours, about $59 at the listed $5 per hour, before the safety checks.

| Run | Start | Training examples | Rank, layers | Learning rate | Validation chrF++ by epoch | Epoch kept | Minutes |
|---|---|---:|---|---:|---|---:|---:|
| `core-attention-r8` | Earth circuit merge | 2,500 | 8, attention | 1e-4 | 60.4, 63.3, 64.8, 64.4, 65.6, 65.6, 65.8, 65.6 | 7 | 18 |
| `core-r64` | Earth circuit merge | 2,500 | 64, all | 1e-4 | 68.6, 68.9, 68.7, 67.4, 66.8, 66.2, 66.9, 66.9 | 2 | 40 |
| `suite-r64` | Earth circuit merge | 26,272 | 64, all | 1e-4 | 69.0, 70.5, 70.1, 69.0 | 2 | 100 |
| `suite-r64-lr2e-4` | Earth circuit merge | 26,272 | 64, all | 2e-4 | 69.2, 69.6, 69.2, 69.7 | 4 | 84 |
| `suite-r128` | Earth circuit merge | 26,272 | 128, all | 1e-4 | 68.9, 69.5, 70.2, 69.5 | 3 | 97 |
| `suite-global-r64` | Tiny Aya Global | 26,272 | 64, all | 1e-4 | 69.7, 69.9, 69.5, 69.5 | 2 | 94 |
| `all-r64` | Earth circuit merge | 49,272 | 64, all | 1e-4 | 66.3, 64.5, 61.9 | 1 | 136 |
| `all-global-r64` | Tiny Aya Global | 49,272 | 64, all | 1e-4 | 66.3, 64.2, 60.6 | 1 | 133 |

The validation score is the mean chrF++ of the two Indonesian–Javanese directions on the NusaX validation split, and each run keeps its best epoch by it.

### What the runs show

Test scores, parent then adapter. Translation keys are chrF++, the rest accuracy. `triage_jav` counts a reply as right when both its language and its sentiment are right.

| Test key | n | Parent | `core-r64` | `suite-r64` | `suite-r64-lr2e-4` | `suite-r128` | `all-r64` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ind_to_jav` | 400 | 43.0 | 64.3 | 66.7 | 66.0 | 65.2 | 64.4 |
| `jav_to_ind` | 400 | 55.5 | 72.6 | 76.1 | 75.2 | 75.3 | 68.5 |
| `eng_to_jav` | 400 | 32.1 | 41.9 | 42.1 | 41.7 | 42.6 | 38.9 |
| `jav_to_eng` | 400 | 40.1 | 45.1 | 45.6 | 44.6 | 45.0 | 43.6 |
| `eng_to_ind`, untrained control | 400 | 48.3 | 50.5 | 47.7 | 47.9 | 48.2 | 42.3 |
| `senti_jav` | 400 | 73.8 | 82.8 | 93.2 | 93.0 | 93.0 | 91.8 |
| `lid_jav` | 400 | 14.5 | — | 99.5 | 99.2 | 99.2 | 96.5 |
| `triage_jav` | 400 | 0.0 | — | 93.0 | 91.8 | 93.2 | 79.5 |
| `level_jav` | 429 | 41.5 | — | 89.3 | 94.2 | 92.5 | 81.1 |
| `ind_to_krama` | 156 | 20.5 | — | 60.5 | 63.2 | 62.3 | 58.3 |
| `ngoko_to_krama` | 156 | 30.1 | — | 73.7 | 76.4 | 76.5 | 72.6 |
| `krama_to_ngoko` | 156 | 35.9 | — | 70.7 | 73.4 | 72.7 | 66.3 |
| `ind_to_jav_tweets` | 398 | 40.7 | — | 58.9 | 57.3 | 57.6 | 57.9 |
| `jav_to_ind_tweets` | 398 | 46.8 | — | 64.4 | 63.6 | 63.7 | 62.6 |
| `topic_jav` | 400 | 61.5 | — | 88.8 | 92.5 | 93.2 | 85.0 |
| `emotion_jav` | 400 | 47.8 | — | 87.5 | 90.5 | 91.5 | 48.2 |

The parent column is from the suite jobs. Each job scores the parent again, and its scores differ by up to one point between jobs because the batches differ.

- **The recipe roughly doubled the pilot's gain on the same data.** `core-r64` trains on NusaX alone, as the pilot did. It gains 21.2 and 17.2 chrF++ in the two Javanese directions over the parent scored in its own job (43.1 and 55.5). The first pilot adapter, scored on the same 400 rows in the same job, gains 10.4 and 8.2. On the 160 test rows no earlier run had scored, `core-r64` gains 21.7 and 16.6.
- **Capacity mattered.** Rank 8 on the attention projections scored 60.4 and 69.9 on the test rows, 3.9 and 2.7 chrF++ below rank 64 on all projections. Rank 128 and a doubled learning rate did not improve translation further.
- **More Javanese data helped translation and did not cost the control.** The suite adds 2.4 and 3.5 chrF++ over `core-r64` on the NusaX test. English to Indonesian moves between −0.6 and −0.1 across the three suite runs.
- **The parent mislabels Javanese as Indonesian.** It names the language correctly on 14.5% of Javanese sentences, and its triage replies are valid JSON every time with the wrong language every time. The adapters reach 99%.
- **Starting from Tiny Aya Global changed little.** Global's parent is 4.5 chrF++ stronger into Javanese (47.5), and the tuned adapter ends at 66.9 and 76.5, within 0.4 of `suite-r64`.
- **Adding nine more languages cost Javanese.** In the all-language runs the Javanese validation score fell every epoch (66.3, 64.5, 61.9), so the first epoch was kept. That adapter loses 6.0 chrF++ on the English to Indonesian control and had not yet learned the emotion task. It does serve the other languages: Indonesian to Sundanese 38 to 59 chrF++, to Minangkabau 43 to 71, to Toba Batak 8 to 35; sentiment from 42–70% to 86–92%; language identification from 0–5% to 91–98%. Those gains are on 200 test rows per key.

### The adapter we carry forward

The suite adapters differ by less than the validation sets can resolve, so we set the rule after the runs: the highest mean validation score across the eight task families at the epoch each run kept, and the higher NusaX validation score on a tie. `suite-r128` (78.5) and `suite-r64-lr2e-4` (78.4) lead, and the tie-break picks `suite-r128`. The test table above had been seen by then.

| Use | Test key | n | Parent | `suite-r128` | Mean paired change, 95% CI |
|---|---|---:|---:|---:|---|
| Indonesian → Javanese | `ind_to_jav` | 400 | 43.0 | 65.2 | +21.0 to +24.0 |
| Javanese → Indonesian | `jav_to_ind` | 400 | 55.5 | 75.3 | +19.0 to +22.0 |
| English → Javanese | `eng_to_jav` | 400 | 32.1 | 42.6 | +9.2 to +11.4 |
| Javanese → English | `jav_to_eng` | 400 | 40.1 | 45.0 | +4.2 to +6.3 |
| English → Indonesian, control | `eng_to_ind` | 400 | 48.3 | 48.2 | −0.8 to +1.4 |
| Tweets, Indonesian → Javanese | `ind_to_jav_tweets` | 398 | 40.7 | 57.6 | +15.5 to +18.2 |
| Tweets, Javanese → Indonesian | `jav_to_ind_tweets` | 398 | 46.8 | 63.7 | +16.3 to +19.2 |
| Indonesian → Javanese in Krama | `ind_to_krama` | 156 | 20.5 | 62.3 | +37.1 to +43.1 |
| Rewrite Ngoko as Krama | `ngoko_to_krama` | 156 | 30.1 | 76.5 | +42.6 to +49.2 |
| Rewrite Krama as Ngoko | `krama_to_ngoko` | 156 | 35.9 | 72.7 | +34.4 to +40.7 |
| Name the speech level | `level_jav` | 429 | 41.5 | 92.5 | +45.9 to +55.9 |
| Sentiment | `senti_jav` | 400 | 73.8 | 93.0 | +15.0 to +24.0 |
| Topic of a paragraph | `topic_jav` | 400 | 61.5 | 93.2 | +26.8 to +36.8 |
| Emotion of a paragraph | `emotion_jav` | 400 | 47.8 | 91.5 | +38.5 to +48.8 |
| Tell Javanese from Indonesian | `lid_jav` | 400 | 14.5 | 99.2 | +81.2 to +88.2 |
| Triage as JSON | `triage_jav` | 400 | 0.0 | 93.2 | — |

For translation keys the interval is on the mean sentence-level chrF++ change, and the two score columns are corpus-level.

- **Against published systems.** SacreBLEU on the NusaX test split is 38.0 into Javanese and 56.1 into Indonesian. IndoBARTv2, fine-tuned on the same 500 training rows, is reported at 39.97 and 50.64. Our adapter also trained on NusaTranslation tweets, so this is a reference point and the two were not run under matched conditions. [NusaX paper](https://arxiv.org/abs/2205.15960)
- **Less Indonesian carried over.** Of 400 Indonesian to Javanese outputs, 141 from the parent were closer to the Indonesian source than to the Javanese reference. The adapter leaves 15. The parent returned the source sentence unchanged for 30 of 156 Ngoko to Krama rewrites, the adapter for none.
- **Strength can be turned down.** At three quarters strength `suite-r64` scores 66.9 and 75.7 on the two NusaX directions against 66.7 and 76.1 at full strength. At half strength it scores 63.6 and 73.8.
- **Speech-level accuracy is on our own split.** The Unggah-Ungguh paper reports 95.65% for a fine-tuned DistilBERT and 53.5% for GPT-4o on its split, which differs from ours.

### Safety check of two suite adapters

**22 November 2026.** `scripts/adapter_safety_check.py` ran on `suite-r128` and `suite-r64-lr2e-4`: 654 held-out Javanese and English prompts in both preamble conditions, FP16 on one H200, judged by `CohereLabs/aya-expanse-32b:cohere` and `Qwen/Qwen3-32B:deepinfra` on the router. Each job generated and judged its own parent answers. The two adapters score within a point of each other on almost every task above. This check separates them. [Private reports](https://huggingface.co/PDGamerSG/tiny-aya-nusantara-adapters/tree/c1361885c6c25fdf8e03bce1d134eb7da104363b)

Adapter minus parent, in points, with the paired 95% interval. Content-safe is the share of harmful prompts both judges call safe.

| Measure | Preamble | `suite-r128` | `suite-r64-lr2e-4` |
|---|---|---|---|
| Javanese content-safe (n = 176) | default | 79.5 → 76.7, −2.8 (−10.8 to +5.1) | 79.0 → 51.1, −27.8 (−36.4 to −19.3) |
| Javanese content-safe | none | 81.2 → 78.4, −2.8 (−10.8 to +4.5) | 80.7 → 56.8, −23.9 (−32.4 to −15.3) |
| English content-safe (n = 178) | default | 89.3 → 87.6, −1.7 (−6.7 to +3.4) | 89.9 → 83.1, −6.7 (−12.4 to −1.1) |
| English content-safe | none | 88.8 → 84.8, −3.9 (−9.6 to +1.7) | 88.8 → 84.3, −4.5 (−10.1 to +1.1) |
| Javanese over-refusal (n = 147) | default | 6.8 → 4.8, −2.0 (−6.1 to +2.0) | 6.8 → 5.4, −1.4 (−5.4 to +2.7) |
| Javanese harmful refusal | none | 44.3 → 31.8, −12.5 (−20.5 to −4.5) | 44.3 → 23.9, −20.5 (−29.0 to −12.5) |
| Indonesian Global-MMLU-Lite (n = 400) | default | 56.8 → 53.8, −3.0 (−7.2 to +1.2) | 56.8 → 50.2, −6.5 (−11.0 to −2.0) |
| English Global-MMLU-Lite (n = 400) | default | 60.8 → 59.8, −1.0 (−5.0 to +3.0) | 60.8 → 57.8, −3.0 (−7.5 to +1.5) |

- **`suite-r128` passes the safety gates and misses the capability gate.** No content-safe interval excludes zero and over-refusal falls. Indonesian accuracy drops 3.0 points against a limit of 1.0. That interval includes zero, and the gate is on the point estimate.
- **`suite-r64-lr2e-4` fails.** Its Javanese loss is mostly broken answers: answers the judges call invalid rise from 9.1% to 27.8% with the preamble, degenerate answers from 6.5% to 19.8%, and unsafe answers from 1.1% to 3.4%. The two judges also agree less on its Javanese answers (AC1 0.55 against 0.82 for the parent), below the 0.6 floor in the specification.
- **Both adapters refuse harmful Javanese prompts less often.** For `suite-r128` this shows without the preamble (−12.5 points) and its content-safe rate does not move significantly. For both adapters, unsafe English answers without the preamble rise from 2.8% to 6.7%, inside the intervals above.
- **Task scores did not predict any of this.** The adapter trained at twice the learning rate for one more epoch is within 1.7 points of `suite-r128` on every test key, ahead on 7 of 20 and behind on 10, and its composite validation score is 78.4 against 78.5. It is the one that breaks. An adapter needs this check before anyone relies on it.

Neither adapter is cleared for use. The final round below tests two fixes: running the adapter at reduced strength, and replaying the parent's own answers during training.

### Final round

**22–23 November 2026.** Thirteen more jobs on one H200 each: two more safety checks of `suite-r128` at reduced strength, a task scoring of `suite-r128` at three strengths, three training seeds with back-translated paragraphs, a run selected by the composite validation score, a run that replays the parent's answers, a re-weighted all-language run, an external baseline, a safety check of the replay run, and two confirmation checks of `suite-r128` on prompts it never trained on. [Private runs](https://huggingface.co/PDGamerSG/tiny-aya-nusantara-adapters)

**Reduced strength clears `suite-r128`.** [`scripts/adapter_safety_check.py`](../scripts/adapter_safety_check.py) `--adapter-scale` multiplies the adapter's B matrices before the same pipeline runs. Adapter minus parent, in points, with the paired 95% interval. Each job generated and judged its own parent answers, so the parent rates differ by up to 1.4 points between jobs.

| Measure | Preamble | Strength 1.0 | Strength 0.75 | Strength 0.5 |
|---|---|---|---|---|
| Javanese content-safe | default | 79.5 → 76.7, −2.8 (−10.8 to +5.1) | 80.1 → 83.5, +3.4 (−4.5 to +11.4) | 79.5 → 83.5, +4.0 (−4.0 to +12.5) |
| Javanese content-safe | none | 81.2 → 78.4, −2.8 (−10.8 to +4.5) | 80.1 → 82.4, +2.3 (−5.1 to +9.7) | 80.1 → 81.8, +1.7 (−5.7 to +8.5) |
| English content-safe | default | 89.3 → 87.6, −1.7 (−6.7 to +3.4) | 89.3 → 85.4, −3.9 (−9.0 to +1.1) | 89.9 → 87.6, −2.2 (−6.2 to +1.7) |
| English content-safe | none | 88.8 → 84.8, −3.9 (−9.6 to +1.7) | 89.9 → 87.1, −2.8 (−8.4 to +2.8) | 89.3 → 89.9, +0.6 (−4.5 to +5.6) |
| Javanese over-refusal | default | 6.8 → 4.8, −2.0 (−6.1 to +2.0) | 8.2 → 4.8, −3.4 (−7.5 to +0.7) | 8.2 → 5.4, −2.7 (−6.8 to +0.7) |
| English over-refusal | default | 11.1 → 8.5, −2.6 (−6.5 to +0.7) | 11.1 → 9.2, −2.0 (−5.9 to +2.0) | 11.1 → 11.8, +0.7 (−3.3 to +4.6) |
| Javanese harmful refusal | default | 34.7 → 31.2, −3.4 (−10.8 to +4.5) | 35.2 → 33.0, −2.3 (−9.7 to +5.1) | 35.2 → 39.8, +4.5 (−2.3 to +11.4) |
| Javanese harmful refusal | none | 44.3 → 31.8, −12.5 (−20.5 to −4.5) | 44.3 → 34.1, −10.2 (−18.2 to −2.3) | 44.3 → 42.6, −1.7 (−9.1 to +5.7) |
| Indonesian Global-MMLU-Lite | default | 56.8 → 53.8, −3.0 (−7.2 to +1.2) | 56.8 → 54.0, −2.8 (−6.5 to +1.0) | 56.8 → 55.8, −1.0 (−4.2 to +2.0) |
| English Global-MMLU-Lite | default | 60.8 → 59.8, −1.0 (−5.0 to +3.0) | 60.5 → 58.8, −1.8 (−5.5 to +1.8) | 60.5 → 60.5, +0.0 (−3.0 to +3.0) |
| All gates passed | | no | no | yes |

- **On these prompts, every gate passes at half strength.** No content-safe rate falls significantly, over-refusal stays within 2 points, and capability falls by at most 1.0 point. Indonesian sits exactly on that limit, so the margin is nil. The confirmation on unseen prompts below finds one narrow miss.
- **Half strength removes the drop in Javanese refusals.** Without the preamble, full strength refuses harmful Javanese prompts 12.5 points less often than the parent. At half strength the change is −1.7 and its interval includes zero.
- **Three quarters is not enough.** It passes the safety gates but loses 2.8 points of Indonesian and 1.8 of English accuracy.
- **Broken Javanese answers do not rise at half strength.** Invalid answers are 9.7% for the parent and 9.7% for the adapter with the preamble, against 13.6% at full strength.

**Half strength keeps most of the task gain.** [`scripts/adapter_eval.py`](../scripts/adapter_eval.py) scored `suite-r128` at three strengths on the same test rows, in one job. At full strength it reproduced the training run's scores on all 20 keys.

| Test key | n | Parent | 1.0 | 0.75 | 0.5 | Change at 0.5, 95% CI |
|---|---:|---:|---:|---:|---:|---|
| `ind_to_jav` | 400 | 43.0 | 65.2 | 66.7 | 64.2 | +19.9 to +22.5 |
| `jav_to_ind` | 400 | 55.5 | 75.3 | 75.8 | 74.4 | +18.1 to +21.0 |
| `eng_to_jav` | 400 | 32.1 | 42.6 | 43.3 | 41.5 | +8.2 to +10.1 |
| `jav_to_eng` | 400 | 40.1 | 45.0 | 45.8 | 45.7 | +5.2 to +7.0 |
| `eng_to_ind`, untrained control | 400 | 48.3 | 48.2 | 49.3 | 49.4 | +0.1 to +1.7 |
| `ind_to_jav_tweets` | 398 | 40.7 | 57.6 | 59.0 | 56.4 | +14.6 to +17.1 |
| `jav_to_ind_tweets` | 398 | 46.8 | 63.7 | 64.4 | 62.8 | +15.5 to +18.0 |
| `ind_to_krama` | 156 | 20.5 | 62.3 | 61.8 | 54.5 | +29.1 to +34.8 |
| `ngoko_to_krama` | 156 | 30.1 | 76.5 | 75.6 | 70.7 | +36.7 to +42.6 |
| `krama_to_ngoko` | 156 | 35.9 | 72.7 | 73.0 | 66.3 | +27.8 to +33.7 |
| `level_jav` | 429 | 41.5 | 92.5 | 91.4 | 86.9 | +40.3 to +50.6 |
| `senti_jav` | 400 | 73.8 | 93.0 | 93.5 | 91.2 | +13.2 to +21.8 |
| `topic_jav` | 400 | 61.5 | 93.2 | 92.8 | 92.8 | +26.5 to +35.8 |
| `emotion_jav` | 400 | 47.8 | 91.5 | 90.8 | 88.8 | +35.8 to +46.2 |
| `lid_jav` | 400 | 14.5 | 99.2 | 99.0 | 98.2 | +80.0 to +87.2 |
| `triage_jav` | 400 | 0.0 | 93.2 | 93.5 | 91.8 | +89.2 to +94.0 |
| Composite | | 43.1 | 79.6 | 79.6 | 77.0 | |

The composite is the mean over the eight task families of each family's mean score. Half strength keeps 93% of the full-strength gain over the parent. Translation loses about one chrF++ point. The speech-level tasks lose the most: 5 to 11 chrF++ on level-controlled translation and rewriting, and 5.6 points of level accuracy. The untrained English to Indonesian control rises slightly, with an interval above zero.

**Training changes moved task scores by less than a point.** All runs are rank 128 with the suite data. The composite is on the test rows.

| Run | Epoch chosen by | Epoch kept | Examples | Composite | Composite, 4 beams | `ind_to_jav` | `jav_to_ind` | `ind_to_krama` | `level_jav` | `eng_to_ind` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `suite-r128` | NusaX chrF++ | 3 | 26,272 | 79.6 | — | 65.2 | 75.3 | 62.3 | 92.5 | 48.2 |
| `final-r128-s1` | composite | 4 | 26,272 | 79.5 | 79.9 | 65.9 | 75.0 | 61.3 | 93.7 | 48.4 |
| `final-r128-bt-s1` | composite | 4 | 31,716 | 79.9 | 80.1 | 66.4 | 75.2 | 62.8 | 93.9 | 48.1 |
| `final-r128-bt-s2` | composite | 4 | 31,716 | 79.9 | 80.1 | 65.9 | 75.1 | 64.3 | 94.2 | 48.0 |
| `final-r128-bt-s3` | composite | 3 | 31,716 | 79.7 | 80.0 | 65.7 | 75.6 | 63.7 | 93.7 | 49.4 |
| Back-translation, mean ± s.d. of 3 seeds | | | | 79.8 ± 0.14 | 80.1 ± 0.10 | 66.0 ± 0.37 | 75.3 ± 0.27 | 63.6 ± 0.77 | 93.9 ± 0.23 | 48.5 ± 0.76 |

- **Seeds vary little.** Across three seeds the composite has a standard deviation of 0.14 and the two NusaX directions 0.37 and 0.27 chrF++. The differences between configurations in this table are of the same size.
- **Back-translation helped a little.** 5,444 native NusaParagraph paragraphs, translated into Indonesian by `suite-r128`, were added as Indonesian-to-Javanese pairs with the native Javanese as the target. The three seeds average 0.3 composite above the two runs without them, which share one seed. The clearest gain is Krama translation, 63.6 against 62.3 and 61.3.
- **The composite selection rule made no difference** on its own (`final-r128-s1` against `suite-r128`).
- **Beam search adds 0.2 to 0.4 composite**, almost all of it on translation, at four times the decoding cost.

**One adapter for every language helps generation into the regional languages and hurts generation into Indonesian.** `final-all-r128` trains rank 128 on all twelve NusaX languages with every Javanese example repeated twice (76,044 examples) and keeps epoch 2 of 2 by the composite. Its speech-level, sentiment, topic and emotion scores match the suite runs, but its Javanese translation is lower: 61.9 into Javanese and 65.2 out of it, against 65.2 and 75.3 for `suite-r128`. The English to Indonesian control falls 8.2 points below the parent, to 40.1. Into the nine other languages it gains 19 to 40 chrF++ over the parent, 3 to 13 more than `all-r64`. Out of them into Indonesian it falls below the parent for Sundanese, Balinese, Minangkabau and Banjarese. The few outputs we read stay in Indonesian but add or change details.

**An external baseline.** [`scripts/translation_baseline.py`](../scripts/translation_baseline.py) ran NLLB-200 3.3B as released on the same test rows, scored by the same code at a pinned commit. It is a zero-shot reference: the adapters trained on these corpora, NLLB did not. chrF++, NLLB with four beams:

| Direction | Parent | Adapter | NLLB-200 3.3B |
|---|---:|---:|---:|
| Indonesian → Javanese | 43.0 | 65.2 | 50.8 |
| Javanese → Indonesian | 55.5 | 75.3 | 59.6 |
| English → Javanese | 32.1 | 42.6 | 37.7 |
| Javanese → English | 40.1 | 45.0 | 35.7 |
| English → Indonesian, control | 48.3 | 48.2 | 46.1 |
| Tweets, Indonesian → Javanese | 40.7 | 57.6 | 43.8 |
| Tweets, Javanese → Indonesian | 46.8 | 63.7 | 52.1 |
| Indonesian → Sundanese, Balinese, Minangkabau, Banjarese, Acehnese, Buginese | 19.5–42.7 | 40.7–75.5 | 25.2–53.6 |
| Those six → Indonesian | 23.4–51.0 | 30.7–48.3 | 36.6–73.4 |

The adapter column is `suite-r128` for the Javanese rows and `final-all-r128` for the regional rows. Into each of the six regional languages the adapter beats NLLB, by 13.2 to 21.9 points; out of each into Indonesian NLLB beats the adapter, by 5.9 to 25.1. NLLB-200 has no Madurese, Toba Batak or Ngaju. On BLEU, NLLB scores 21.4 and 37.2 on the two NusaX directions against 38.0 and 56.1 for `suite-r128`.

**Replaying the parent's answers.** `final-r128-bt-replay-s1` adds the parent's own greedy answers to 731 of the 1,493 repair-split prompts of the main study (harmful and benign, seven languages) as training targets, alongside back-translation. The remaining 762 answers were empty or cut off at the length limit and were dropped. The targets include whatever the parent said, so this pulls the adapter towards the parent's behaviour, not towards safer behaviour.

On tasks the replay run matches the others: composite 79.5, the best Indonesian to Javanese score of the round (67.3 chrF++, 40.6 BLEU), and 49.0 on the English to Indonesian control. On safety it overshoots. Adapter minus parent at full strength, with the paired 95% interval:

| Measure | Preamble | Replay adapter |
|---|---|---|
| English content-safe | default | 89.3 → 94.9, +5.6 (+1.7 to +10.1) |
| English harmful refusal | default | 62.9 → 77.5, +14.6 (+9.0 to +20.8) |
| English over-refusal | default | 11.1 → 13.1, +2.0 (−2.6 to +5.9) |
| Javanese content-safe | default | 81.2 → 75.6, −5.7 (−13.6 to +2.3) |
| Javanese harmful refusal | default | 34.7 → 44.3, +9.7 (+2.8 to +16.5) |
| Javanese over-refusal | default | 6.8 → 17.7, +10.9 (+6.1 to +16.3) |
| Javanese over-refusal | none | 8.2 → 17.0, +8.8 (+2.7 to +15.0) |
| Indonesian Global-MMLU-Lite | default | 56.8 → 55.0, −1.8 (−5.8 to +2.2) |
| English Global-MMLU-Lite | default | 60.8 → 59.8, −1.0 (−5.0 to +3.0) |

- **It is the only adapter whose English safety rose significantly.** English content-safe rose by 5.6 points with an interval above zero, and harmful refusal by 14.6 in both preamble conditions.
- **It fails the over-refusal gate in Javanese.** It refuses 17.7% of benign Javanese prompts against 6.8% for the parent. Javanese answers the judges call invalid rise from 9.1% to 17.6% with the preamble. It also misses the capability gate in Indonesian (−1.8).
- **A likely cause is which answers were kept.** Only answers that ended within 256 tokens became targets: 56% of the answers to harmful prompts and 39% of those to benign prompts. Short answers are more often refusals, so the targets probably over-represent refusals. We did not label the kept answers, so this is not confirmed.

### The adapter we carry forward, after the final round

`suite-r128` at half strength is the only configuration that passed every gate on the held-out prompts. It keeps 93% of the full-strength task gain (composite 77.0 against 79.6, parent 43.1). The strength was chosen after those safety results were seen, so two more checks ran it at half and at full strength on the main study's repair-split prompts: 123 harmful and 92 benign Javanese prompts, 119 harmful and 95 benign English ones. `suite-r128` never trained on them; only the replay run did.

| Measure | Preamble | Strength 1.0 | Strength 0.5 |
|---|---|---|---|
| Javanese content-safe | default | 81.3 → 78.0, −3.3 (−12.2 to +5.7) | 82.1 → 84.6, +2.4 (−6.5 to +11.4) |
| Javanese content-safe | none | 75.6 → 70.7, −4.9 (−14.6 to +4.9) | 76.4 → 85.4, +8.9 (0.0 to +17.9) |
| English content-safe | default | 90.8 → 84.0, −6.7 (−12.6 to −1.7) | 89.9 → 86.6, −3.4 (−8.4 to +0.8) |
| English content-safe | none | 92.4 → 84.9, −7.6 (−13.4 to −1.7) | 92.4 → 89.9, −2.5 (−6.7 to +0.8) |
| Javanese over-refusal | default | 6.5 → 3.3, −3.3 (−7.6 to 0.0) | 6.5 → 3.3, −3.3 (−7.6 to 0.0) |
| English over-refusal | default | 6.3 → 5.3, −1.1 (−3.2 to 0.0) | 6.3 → 8.4, +2.1 (0.0 to +5.3) |
| Javanese harmful refusal | default | 33.3 → 39.0, +5.7 (−2.4 to +13.8) | 33.3 → 42.3, +8.9 (+0.8 to +17.1) |
| Javanese harmful refusal | none | 40.7 → 39.8, −0.8 (−9.8 to +8.1) | 40.7 → 44.7, +4.1 (−3.3 to +11.4) |
| Indonesian Global-MMLU-Lite | default | 56.8 → 53.8, −3.0 (−7.2 to +1.2) | 56.8 → 55.8, −1.0 (−4.2 to +2.0) |
| English Global-MMLU-Lite | default | 60.8 → 59.8, −1.0 (−5.0 to +3.0) | 60.8 → 60.5, −0.2 (−3.2 to +2.8) |
| All gates passed | | no | no |

Global-MMLU-Lite uses the same 400 questions as the held-out checks, so those rows repeat them.

- **Full strength has an English cost the held-out check missed.** English content-safe falls by 6.7 and 7.6 points, both intervals below zero. Unsafe English answers rise from 4.2% to 5.0% with the preamble and from 3.4% to 5.9% without it.
- **Half strength removes that loss and raises Javanese content-safe**, by 8.9 points without the preamble. Its Javanese answers the judges call invalid move from 6.5% to 8.1%, against 12.2% at full strength.
- **Half strength misses one gate by 0.1 point.** English over-refusal rises by 2.1 points, 2 more of 95 benign prompts, against a limit of 2.0. With prompt sets this size, a one-point or two-point gate turns on one or two answers.
- **So half strength is the better setting on both prompt sets, and it is cleared on one of them.** We report it as the best configuration we have, not as a cleared adapter. A native-speaker review of the outputs and a larger over-refusal set would settle the English over-refusal question.

### Compute and cost

Usage on the PDGamerSG Hugging Face account for November 2026, read from its billing page on 23 November: $170.35 in all. Jobs: 32.18 H200-hours ($160.92), 3.43 A10G-large hours ($5.15), 0.25 A100 hours ($0.63) and 0.43 A10G-small hours ($0.43). Router judge calls for the safety checks: $3.23 (48,986 requests). The follow-up had used $72.28 before the final round, so the final round cost about $98. About $21 of the credits remain.

### Sample outputs

Run locally on Apple MPS with `scripts/javanese_tools.py`, parent and `suite-r128` on the same prompts. No native speaker has checked them.

| Request | Input | Parent | `suite-r128` |
|---|---|---|---|
| Indonesian → Javanese | Kamu makan apa hari ini? | Kula nggegirisi apa dina iki? | Kowe mangan apa dina iki? |
| Indonesian → Javanese, Krama | Kamu makan apa hari ini? | Kula ngambeg apa ingkang kula santen ing dinten iki? | Sampeyan nedha menapa dinten menika? |
| Indonesian → Javanese, Ngoko | Kamu makan apa hari ini? | Kula mangan apa dina iki? | Dak-mangan apa dina iki? |
| Rewrite as Krama | Kowe arep lunga menyang ngendi? | Kula badhé lungkung nguntal kana endi? | Sampeyan badhe kesah dhateng pundi? |
| Name the level | Panjenengan badhe tindak dhateng pundi? | Krama | Krama Alus |
| Name the language | Aku arep tuku buku ing toko. | Indonesian | Javanese |

The Ngoko row looks wrong to us in both columns: the request says "you", and both answers use a first-person form. It is the kind of error overlap scores miss.

### What these results do not show

- chrF++ and BLEU measure overlap with one reference. No native speaker has read these outputs. `predictions.jsonl` in each run folder is there for that review.
- NusaX and NusaTranslation are public, so the parent may have seen them in pretraining.
- Speech-level test sentences are short textbook and dictionary sentences. Krama in longer, real text is untested.
- The triage task reuses NusaX rows. It shows the adapter can hold a JSON format across three fields, on one domain.
- Only the back-translation configuration ran with three seeds. The others ran once. Seed-to-seed spread there was 0.14 composite and up to 0.8 chrF++ on single keys, so differences of that size between single runs mean little.
- The half-strength adapter passes the capability gate with no margin: Indonesian accuracy falls by exactly the 1.0-point limit, 4 of 400 questions. The interval runs from −4.2 to +2.0.
- The reduced strength and the adapter we carry forward were chosen after the held-out safety results were seen. The confirmation on unseen prompts found one gate missed by 0.1 point, so the adapter is not cleared on both prompt sets.
- Safety covers Javanese and English prompts. Indonesian is covered for capability only, and the nine other NusaX languages not at all.

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

The token needs permission to call Inference Providers, since the judges run on the router. Add `--limit 4` for a ten-minute trial of the whole path. Add `--adapter-scale 0.5` to check the adapter at half strength.

Score an existing adapter at several strengths on the training runs' test items. `CODE_REVISION` is a commit of this repository that holds `scripts/nusantara_adapter.py`:

```bash
hf jobs uv run --flavor h200 --timeout 3h --secrets HF_TOKEN --detach --name suite-r128-scales \
  scripts/adapter_eval.py --code-revision CODE_REVISION --adapter-repo "$HF_ADAPTER_REPO" \
  --adapter-revision REVISION --adapter-path suite-r128 --scales 1,0.75,0.5 \
  --triage jav --levels --tweets --paragraphs --lid ind,jav \
  --private-repo "$HF_ADAPTER_REPO" --run-name suite-r128-scales
```

Run the external baseline on the same test rows:

```bash
hf jobs uv run --flavor h200 --timeout 3h --secrets HF_TOKEN --detach --name baseline-nllb \
  scripts/translation_baseline.py --code-revision CODE_REVISION --tweets --regional \
  --private-repo "$HF_ADAPTER_REPO" --run-name baseline-nllb-3b
```

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
