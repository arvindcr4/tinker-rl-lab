# Phase 2 submission guide

**Current deliverable:** Arvind C R's M.Tech Project Phase 2 thesis, UE20CS972,
PES University. Guide: Ramesh Prakash Guledgudd. This is an **identified academic
submission**, not an anonymous conference package.

## What to submit

- [Thesis PDF](outputs/PES_Phase2_Third_Review_2026-09-24/thesis/Thesis_Report_ArvindCR.pdf)
- [Canonical source and build instructions](outputs/PES_Phase2_Third_Review_2026-09-24/thesis/README.md)
- `dist/TinkerRL_Phase2_Submission.zip` — generated review package
- `dist/TinkerRL_Phase2_Submission.zip.sha256` — archive checksum
- [Verification record](outputs/PES_Phase2_Third_Review_2026-09-24/thesis_review_2026-09-27/submission_validation.md)

The directory date identifies the original review folder. The thesis incorporates
the 2 October follow-up and the submission-preparation corrections. Chapter
Markdown, `frontmatter.tex`, `references.bib`, and `build_thesis.py` are the
sources; `thesis_master.tex` and the PDF are generated.

**Do not upload the old deck or an older bundle as this submission.**
`FINAL_HANDOFF.md`, `submission/contents/`, `submission/mtech_phase1/`, and the
September review slides record earlier deliverables. They are not kept in sync
with this thesis. A presentation is not included in the current package.

## Build the package

From the repository root, with Python 3.11+:

```bash
# Install the locked development environment if needed.
uv sync --locked --extra dev
# Install pandoc and tectonic using your OS package manager.
# Example on macOS: brew install pandoc tectonic
uv run --no-sync python platform_modal/scripts/build_university_submission.py
```

This runs the selected evidence checks and demo tests, rebuilds every used
figure, builds the thesis, and writes the ZIP only if all required inputs and
commands succeed. A failed build leaves the previous submission ZIP and thesis
PDF intact. Tectonic can download TeX packages on its first run; no training,
provider inference, cloud allocation, or paid experiment is launched.

The ZIP uses a fixed file order and timestamps. Packaging the same input bytes
and base revision produces identical ZIP bytes; PDF compilation itself is not
promised to be byte-identical across TeX versions. `MANIFEST.json` records every
selected file's SHA-256 and size. The git revision is the **base revision**, not
a claim that the working tree was clean. Current modified files are included.

## Verify without GPU, credentials, or network

```bash
uv run --no-sync python tools/check_thesis_evidence.py
uv run --no-sync python submission/demo/run_demo.py
uv run --no-sync python submission/demo/run_demo.py --self-test
uv run --no-sync python platform_modal/scripts/build_university_submission.py \
  --verify dist/TinkerRL_Phase2_Submission.zip
```

The evidence checker and demo use only the Python standard library. After
extracting the ZIP, run these commands from its root with `python3` instead of
`uv run --no-sync python`; no ML dependency installation is needed. Verify the
extracted files with `sha256sum -c SHA256SUMS` (Linux) or
`shasum -a 256 -c SHA256SUMS` (macOS).

To rebuild the thesis from an extracted ZIP, install Pandoc and Tectonic, then
run the two commands in the thesis README with `python3` instead of
`uv run --no-sync python`. The builders are standard-library scripts. Do not run
`uv sync` inside this compact package: the full installable research code is
not included. Regenerating the ZIP with its base git revision uses the full
repository checkout.

**Check scope:** the evidence checker recomputes selected arithmetic from stored
records. It does not rerun native graders, authenticate upstream provenance,
validate all thesis claims, or reproduce training. The demo has four explicitly
synthetic reward groups and a separately hash-checked recorded artifact. It
proves mechanism and internal consistency, not model improvement. The historical
19 September 11/11 ledger receipt is not a current all-results validation.

## Evidence and limits

| Evidence | Location | Interpretation |
|---|---|---|
| M1b unsaturated same-stack rerun | `platform_hybrid/experiments/results/samestack_gsm8k_cot{,_full}.json` | Five paired seeds; GRPO G8 minus untuned-critic PPO is +5.0 pp. Paired t-test p=0.016; exact sign-flip p=0.0625. Not superiority over tuned PPO. |
| Trained/base small-scale pairs | `outputs/e1_e14_small_scale_2026-09-26/E*/paired.json` | Same serving engine, per-lane comparison. No significant primary lane difference; no cross-lane aggregate. E12 rubric items are clustered by application. |
| Six replacement scopes | `outputs/finish_pending_2026-09-27/E*/result.json` | E1, E2, E5, E6, E9, E13. Attempted coverage is not the same as successful native grading. |
| Full source map | Thesis Appendix G | Other evidence remains in the repository, not all in this compact ZIP. |

The E12 receipt uses the shared E3/E7/E12 generator's staged rounding: it
subtracts four-decimal arm means to obtain -0.0332. The direct per-item
difference is -5/151 = -0.0331125828. The checker verifies the original rounding
rule and reports the raw difference separately. The receipt is unchanged; this
distinction does not affect the three-decimal thesis table or the conclusion.

These limits are retained, not marked complete by packaging:

- **E1:** the new 1/190 result stays separate from the earlier 4/110. Serving
  context and reconstructed source collection differ.
- **E6:** 90/812 is a lower bound. 108 judge-dependent tasks remain ungraded
  because the judge accounts lacked credit. Regrading needs authorized access
  and the native inputs; packaging does not silently fill those results.
- **Original benchmark contracts:** E3 (SDAB private bundle), E7 (BinaryAudit
  private payload), E8 (LifeSciBench package), E10 (AgentHarm private tasks),
  E12 (AppBench deployment), and E14 (FrontierMath hosted evaluation) have
  external-closure records. E13's OpenReward held-out games are also blocked,
  without a separate closure record because its BALROG replacement continued.
  These require provider access, not a code change. Public substitutes are not
  the original scores.
- **Claims:** saturated arithmetic nulls do not establish equivalence. ZVF is
  regime-dependent. Missing, simulated, and measured results stay distinct.
- **Runtime:** `uv.lock` pins the supported development environment.
  `requirements.txt` records a different, broader research environment; it is
  not proof that every historical experiment used today's installed versions.

No weights, full datasets, hidden answer keys, cloud credentials, local caches,
or provider logs are bundled. They require their own licenses, accounts, or
large downloads. Some original execution sources and the historical Tinker
sampler route are no longer available. The ZIP is a **thesis review package**,
not a self-contained reproduction of every research run. Use the complete
repository and each study's receipts for broader reproduction.

## Before the actual submission — author checklist

- [ ] Read the rebuilt PDF and approve the claims, authorship, and AI-tool declaration.
- [ ] Confirm the certificate wording with the guide. Obtain required guide,
      Head of Department, and examiner signatures; no approval is implied by
      the prepared certificate page.
- [ ] Confirm declaration date, degree/course details, and the existing student
      signature. The package contains identifying details and a signature image.
- [ ] Obtain any required institutional similarity/plagiarism certificate.
- [ ] Confirm portal deadline, permitted formats, filenames, size/page limits,
      and whether a revised presentation is required.
- [ ] Upload only the portal-requested files and keep its submission receipt.

**Nothing has been uploaded or submitted by these build commands.**

## Ownership and license

[PROJECT_HISTORY.md](PROJECT_HISTORY.md) and the
[Semester 4 provenance record](platform_hybrid/sem%204%20work/PROVENANCE.md)
separate the inherited Group 6 work from the individual continuation. The root
[CITATION.cff](CITATION.cff) is the historical Semester 3 group citation, not a
new citation for this thesis. Cite this report using its title page and author.
Repository code is [Apache-2.0](LICENSE); third-party data, models, benchmark
assets, and institutional images retain their own rights. No publication,
acceptance, or institutional approval is asserted here.
