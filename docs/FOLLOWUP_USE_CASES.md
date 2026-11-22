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

## Source search log

**22 November 2026.** This was a bounded review of project evidence and primary dataset, model, and metric documentation. Search terms included `Javanese`, `invalid`, `degenerate`, `native`, `NusaX`, `NusaWrites`, `FLORES`, and `chrF++`. Project searches covered the specification, reports, cards, and generation and split code. Public-source checks covered the dataset maintainers' documentation and SacreBLEU. NusaParagraph offers native-written passages. The Gatra-1 card describes a dataset that is 98% GPT-4-generated with minimal human curation. The gated FLORES raw card returned HTTP 401. The public access form displayed the evaluation-only terms explicitly. [Specification](../SPEC.md) · [NusaParagraph](https://huggingface.co/datasets/indonlp/nusaparagraph_topic) · [Gatra-1](https://huggingface.co/datasets/afrizalha/Gatra-1-Javanese) · [FLORES dataset](https://huggingface.co/datasets/facebook/flores)
