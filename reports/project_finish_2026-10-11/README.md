# PES evaluation review pack

Prepared 11 October 2026 for Arvind C R's M.Tech Project Phase 2 evaluation; corrected 16 October 2026.

> **Author approval required — AI-tool declaration.** On 16 October 2026 the thesis's "Declaration on the Use of AI Tools" was redrafted by an AI agent to name the agentic tools the repository records (Claude Code agent teams, Codex, Grok CLI, Muse CLI, Antigravity/Gemini, `agy`, `squad`, Kimi, Jev, ChatGPT Pro, Gemini Deep Think) and agents launching cloud jobs. It is a draft. Arvind C R must check it against the actual workflow and PES policy wording, and approve or correct it, before any submission.

> **Provisional numbers.** The v2 same-stack GRPO-vs-PPO figures (thesis Table 6.2) use 7 of 10 PPO seeds; the exposure-matched PPO arm is pending. See `validation/samestack_v2/README.md` for the update path.


Start with the [corrected thesis](thesis/Thesis_Report_ArvindCR_2026-10-11_REVIEW.pdf),
the [12-slide defense](defense/TinkerRL_PES_Defense_2026-10-11.pptx), and the
[reviewer guide](REVIEWER_GUIDE.md). Rehearse with [defense preparation](DEFENSE_PREP.md)
and the [full speaking brief](defense/DEFENSE_BRIEF.md).

This pack presents the implemented measurement system, individual contribution,
controlled comparisons, and bounded benchmark evidence. It includes a Colab reanalysis of retained numeric records, no new model
training or benchmark results and makes no prediction about the awarded grade.

## What changed

16 October 2026, after a publication-readiness audit (interpretations changed; no stored measurement changed):

- Superseded the unsaturated GRPO G8 − PPO gap (+0.050; 200-token cap, cold critic, PPO below its start, Holm sign-flip p = 0.19) with the v2 rerun: +0.0076 [−0.0034, +0.0185], historical unadjusted ±0.02 sensitivity check (TOST p = 0.016), without a justified practical margin or confirmatory equivalence claim; G8 − G2 +0.0079, Holm p = 0.14. Provisional. Thesis §6.5.5 (new Table 6.2), Table 1.1, §9.3, abstract, appendices A and E, deck slides 4, 5, 6 (notes), 10 and 12, brief, viva answers and reviewer guide. Analysis scripts and outputs: `validation/samestack_v2/`.
- Restated the ZVF U-shape and 92.3% escalation share as largely following from the binomial identity, prompt-difficulty heterogeneity and reward level; nine model labels, not six; +0.71 marked unrecomputable; ρ = 0.27 marked withdrawn wherever it appears (−0.37, p = 0.36, on the eight non-simulated cells); the "anti-herding" δ_div analysis and the +0.1224 residual withdrawn as sign-guaranteed; contemporaneous prior work cited (arXiv 2605.21125, 2605.07689).
- Decomposed the 17× stack exhibit (checkpoint change alone 64%), declared the DAPO pair non-comparable, showed the 0.034 band is not equivalence, corrected the registry count to 28, described the revised `stackdiff` with its actual CLI output, and stated the `stackdiff` limits; noted the in-progress registry revision without evaluating it.
- Corrected five citations (Hochlehnert, Cobbe, Lightman claim removed, Nimmaturi and Tan locations verified against the PDFs), added Stiennon's arXiv identifier, marked the capstone survey unpublished.
- Redrafted the AI-tool declaration (author approval required; see the note above).

11 October 2026:

- Incorporated the five numerical and provenance corrections from the 4 October draft.
- Corrected the Jensen argument, undefined kappa, bootstrap identity, layer-overlap
  reference, historical mediation interpretation and clustered E12 uncertainty.
- Added an objective-to-evidence map and a working registry example to make the
  individual contribution and implemented system easier to examine.
- Distinguished 945 passing checks from 22 context-only occurrences in Appendix J.
  The passing checks include 34 calculations from printed counts alone.
- Narrowed decontamination, prediction, and serving-defect claims to match the evidence.
- Corrected the defense's unsaturated study budget to GRPO 8 prompts times 8
  completions and PPO 64 prompts times 1 completion.
- Added tested statistical edge-case fixes without altering stored experimental results.
- Reproduced the source audit through Colab CLI: 17 finite correlation pairs, nine
  synthetic cells, and mixed outcome definitions. Withdrew the predictive interpretation
  and included the frozen inputs, replay script, result and execution receipt.
- Bound the current files to a fresh manifest. Historical October releases remain intact.

The PDF is a corrected evaluation review copy. The historical identified submission,
certificate and institutional workflow remain described in [SUBMISSION.md](../../SUBMISSION.md).
The author still needs to approve the claims, attribution and AI declaration, obtain
required signatures or certificates, confirm portal requirements, and submit the
requested files. These local build commands have not uploaded anything.

## Use the offline demo

From the repository root, or the root of the extracted review ZIP:

```sh
python3 submission/demo/run_demo.py --self-test
python3 submission/demo/run_demo.py
```

Open `submission/demo/output/demo_report.html`. The demo separates four illustrative
reward groups from an exact-byte and arithmetic check of a recorded artifact. It is
not a training run or a new benchmark score.

## Verify the archive

```sh
python3 reports/project_finish_2026-10-11/verify_review_pack.py
```

The verifier checks every manifest-listed file. Source links in the guides use the
repository layout. Selected supporting documents are included in the ZIP; the thesis's
full historical source map still includes evidence outside this review package.
No hidden datasets, checkpoints, provider logs, credentials, or full execution history
are bundled. The pack is sufficient for reading the documents and running the offline
demo, not for reproducing every research run.

The [validation summary](validation/SUMMARY.md) records the software checks.
[Evaluator corrections](EVALUATOR_REVIEW.md) records the scientific and presentation fixes.
[Acceptance review](ACCEPTANCE.md) records the final review scope and remaining limits.
[WORKFLOW.md](WORKFLOW.md) and [decisions.tsv](decisions.tsv) record the pstack workflow.

## Rebuild the current documents

From the full repository checkout with its locked development environment:

```sh
uv run --no-sync python tools/audit_thesis_numbers.py --check \
  --thesis-dir reports/project_finish_2026-10-11/thesis
python3 reports/project_finish_2026-10-11/thesis/build_thesis.py
python3 tools/build_pes_review_pack.py
```

The thesis build uses Pandoc and Tectonic with cached resources and untrusted mode.
It needs an existing compatible TeX cache. Rebuilding changes PDF hashes, so inspect
the new PDF and refresh its quality receipt before assembling a new review archive.
The included deck is editable. The historical deck and its source builder remain separate.
