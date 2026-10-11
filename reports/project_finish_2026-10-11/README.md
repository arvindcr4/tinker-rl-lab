# PES evaluation review pack

Prepared 11 October 2026 for Arvind C R's M.Tech Project Phase 2 evaluation.

Start with the [corrected thesis](thesis/Thesis_Report_ArvindCR_2026-10-11_REVIEW.pdf),
the [12-slide defense](defense/TinkerRL_PES_Defense_2026-10-11.pptx), and the
[reviewer guide](REVIEWER_GUIDE.md). Rehearse with [defense preparation](DEFENSE_PREP.md)
and the [full speaking brief](defense/DEFENSE_BRIEF.md).

This pack presents the implemented measurement system, individual contribution,
controlled comparisons, and bounded benchmark evidence. It includes a Colab reanalysis of retained numeric records, no new model
training or benchmark results and makes no prediction about the awarded grade.

## What changed

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
