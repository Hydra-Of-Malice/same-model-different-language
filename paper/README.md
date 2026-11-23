# Paper: Same Model, Different Language, Different Safety

EMNLP 2023 write-up (final, non-anonymous version) of this project: the per-language safety study of Tiny Aya
Earth, the training-free repairs, and the Javanese task-adapter follow-up. The main text is cut to 6 pages and
follows the hackathon's recommended write-up sections; everything that did not fit is in the appendices.

| File | What it is |
|---|---|
| `main.tex` | The paper. Main text (sections 1 to 6), Limitations, Ethics Statement, Acknowledgements, references, appendices A to L |
| `custom.bib` | Bibliography. Every entry was checked against its arXiv abstract page, ACL Anthology export or Crossref record |
| `EMNLP2023.sty` | Official EMNLP 2023 style (`emnlp2023.sty`, renamed because the template loads `EMNLP2023` and Overleaf is case-sensitive) |
| `acl_natbib.bst` | Official bibliography style |
| `adapter_tradeoff.pdf` | Figure 1 (Appendix G), included by `main.tex` |
| `make_figure.py` | Draws `adapter_tradeoff.pdf` |
| `main.pdf` | Last local build |

Current build: 29 pages in all. The main text (sections 1 to 6) ends at the bottom of page 6, and Limitations starts
on page 7. Limitations and the Ethics Statement do not count toward the page limit. The EMNLP template itself states only the review limit (8 pages for long papers); the 6-page target
is the hackathon's.

## Structure

| Part | Contents | Hackathon write-up item |
|---|---|---|
| Box on page 1 ("For judges: one minute") | Claim, tracks and problem statements, headline before and after, base model and method, content note | Title and claim, tracks, headline, model and method, content warning, judges box |
| Abstract | | Abstract |
| 1 Introduction | Why this problem, questions and approach (pre-registered thesis), related work | Problem and approach, use case, research questions, related work |
| 2 Lexsi Kits and Pipeline | The four kits, features used, how they connect | Repos used, how they connect |
| 3 Experimental Setup | Models and licence, data and splits, judges and how we checked them, metrics and why, pre-registered thresholds | Models, datasets, metrics, judge check, method |
| 4 Results | 4.1 gap, 4.2 harm versus breakdown (attack success rate), 4.3 repairs, 4.4 circuits, 4.5 4-bit | Results, before and after, CIs, findings, quantised model |
| 5 Extension | Javanese task adapter: setup, task gains, safety check, two fixes | Adapter follow-up (an extension, no problem number) |
| 6 Conclusion | Conclusion, what did not work, cost, with more credits | What didn't work, cost, with more credits |
| Appendix A | Tracks, artifacts, documentation and reproduction, work done outside the hackathon, team | Artifacts, reproduction, declaration, teammates |
| Appendix B | Kit features, omissions and gaps (Tables 4 and 5) | Repos in detail |
| Appendix C | Models, datasets, licences, CuratorKIT filtering, translation (Tables 6 to 8) | Models and datasets in detail |
| Appendix D | Thesis, development run, planning checks, amendments | Pre-registered hypotheses |
| Appendix E | Judge checks, regional comparisons, circuit details, main-study tables | Judge check in detail, extra tables |
| Appendices F and G | Adapter data, recipe, ablations, safety check, samples, final round, Figure 1 | Ablations, seeds, sample outputs |
| Appendices H to L | Prompt templates, hyperparameters, compute and ROI, with-more-credits table, reproducibility checklist and commands | Prompts, judge rubric, settings, cost, reproduction |

## Compile

**Locally with Tectonic** (what produced `main.pdf`):

```bash
cd paper
tectonic main.tex
```

Tectonic runs BibTeX itself and writes only `main.pdf`. It is XeTeX-based and also defines `\pdfoutput`, so
`main.tex` has one guarded line before the style is loaded,
`\ifdefined\XeTeXversion\PassOptionsToPackage{xetex}{hyperref}\fi`, that picks the right hyperref driver. It does
nothing under pdfLaTeX. The two "Wrong driver" hyperref warnings and the `inputenc` warning in the log are harmless.
The build has no undefined references or citations and no overfull boxes.

Layout choices: paragraph columns in tables use the `L{width}` column type (ragged right, from the `array` package),
so narrow cells do not stretch word spacing; the appendix raises the float limits (`\topfraction`,
`\floatpagefraction` and the counters after `\appendix`) so that tables share pages with text instead of sitting
alone on half-empty pages; and the artifact and reproducibility lists are set ragged right because of their long
URLs.

**Locally with pdfLaTeX:**

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

or `latexmk -pdf main.tex`. Delete the `.aux`, `.bbl`, `.blg`, `.log` and `.out` files afterwards if you do not want
them in the repository.

**On Overleaf:** create a blank project, upload `main.tex`, `custom.bib`, `EMNLP2023.sty`, `acl_natbib.bst` and
`adapter_tradeoff.pdf` (the figure in Appendix G; the build fails without it), set the compiler to pdfLaTeX (Menu,
Settings) and the main document to `main.tex`. Keep the file name `EMNLP2023.sty` exactly; Overleaf is
case-sensitive.

## Final version

The file is set up as the final, non-anonymous version:

- `\usepackage{EMNLP2023}` without the `review` option, so the author block shows and there are no line numbers.
- `\camerareadytrue`, so everything the `\anon{...}` macro would hide is shown: our Hugging Face and GitHub repository
  names, the commit hashes of our own repositories (pinned revisions, Appendix L), the release URLs in the
  reproducibility checklist, and the Acknowledgements section.
- Three authors, in this order, each with an e-mail address: Aditya Arnav, Pallab Das and Sriram. No affiliations
  were given, so the author block has none.

An anonymous review version would need more than the two switches above: Appendix A (artifact URLs, team, declaration) and the pull requests in Table 5 identify us without the `\anon{...}` macro.

## Still to fill in

Nothing. `grep -n '\\pending{\|PLACEHOLDER' main.tex` finds nothing. The author contributions
paragraph was removed, because the split of work among the three authors is not recorded anywhere; EMNLP does not
require one. To add it back, put a `\paragraph{Author contributions.}` after the Team paragraph in Appendix A.

## Where the numbers come from

Every number in the paper is copied from these files; none is re-derived or rounded differently.

| Paper part | Source |
|---|---|
| Sections 1 to 4, 6; Appendices B to E (main study) | `article/README.md`, `README.md`, `SPEC.md`, `configs/spec.yaml`, `docs/submission/abstract.md` |
| Judge sensitivity (32B pair, Command A), and the dates of the full re-judging | `docs/judge_checks/full-32b-vs-8b.md`, `docs/judge_checks/command-a-full-gap.md`, `docs/judge_checks/README.md` |
| Section 5 and Appendices F and G | `docs/FOLLOWUP_USE_CASES.md` (survey, expanded adapter, safety check, results, sample outputs, caveats) |
| Kits, licences, with more credits, reproduction, declaration, team (Appendices A to C, K) | `article/README.md` sections 2 to 6, 8 to 10 and 12, `README.md`, `docs/submission/checklist.md` (hackathon dates) |
| Cost of serving (Appendix J) | `docs/submission/roi.md` and `article/README.md` section 7.9 |
| Method details, prompts, pins and hashes | `scripts/nusantara_adapter.py`, `scripts/adapter_safety_check.py`, `scripts/translation_baseline.py`, `scripts/javanese_tools.py`, `src/langsafe/judge/llm_judge.py` |
| Follow-up spend ($170.35 in November: 32.18 H200-hours of Jobs, smaller jobs and router judge calls) | `docs/FOLLOWUP_USE_CASES.md`, "Compute and cost", read from the account's billing page on 23 November |
| Author block (three authors, e-mails, no affiliations) | Supplied by the authors |

## Open checks before submission

- Appendix A states that nothing was done before or outside the hackathon (1 to 24 November 2026), and that the 32B
  and Command A re-judging (19 and 20 November) and the adapter study (22 and 23 November) were done in the second
  round, after the 15 November first-round deadline.
- The 0.09 and 0.22 chrF figures for English to Javanese are quoted from the follow-up notes' reading of the Tiny Aya
  report and were not re-checked against the report.
- No native speaker has reviewed the translated prompts, the judges' labels or the adapter outputs; the paper says so
  in the main text, the Limitations and the Ethics Statement.
