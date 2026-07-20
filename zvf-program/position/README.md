# ZVF Program — Pillar 4: MIN-REPORT-RL (Position Paper)

This pillar argues that RL post-training of LLMs is reported as if a three-letter algorithm
label (PPO/GRPO/DPO and the GRPO family: DAPO, GSPO, Dr.GRPO, M-GRPO) fixes the experiment,
when in fact the unreported *stack* — backend, sampler, precision, reference/KL handling, loss
form, LoRA config, reward parser, group-size schedule — co-determines the result. An
under-specified same-label comparison spanned 84.4% to 5.0% last-10 reward
(~17×), but also changed the base checkpoint, so it is not a backend effect
estimate. It proposes **MIN-REPORT-RL**, an eight-item standard: seven
run-manifest fields plus pass@k evaluation, alongside a controlled
single-stack **reproducibility audit** (re-implement DAPO/GSPO/Dr.GRPO/M-GRPO in one trainer
and report which claimed gains survive). It builds directly on the ZVF Program v1 audit
(stack-conditioning thesis; ZVF/GU triage; the clean 82.0%→83.3%, p=0.26 held-out control).

## Files
- `min_report_rl.tex` — standalone position-paper draft using the canonical shared
  bibliography at `../../platform_hybrid/paper/references.bib`.
- `CHECKLIST.md` — copy-pasteable author checklist + fillable appendix template
  (seven manifest fields plus pass@k, with good/bad examples).
- `README.md` — this file.

## Target venues
- **Primary:** NeurIPS / ICML **Position Track**.
- **Secondary / companion:** **MLRC** (ML Reproducibility Challenge) and **ICLR Reproducibility
  Track** for the audit results once runs exist.

## Execution status and remaining external work

- The audit now has a frozen 8-seed contract in
  `../audit/preregistration.json` and a fail-closed aggregator in
  `../audit/aggregate_audit.py`; no full-scale survival verdict is claimed.
- The registry schema, manifest generator/verifier, and entry-level stackdiff
  are executable under `../../platform_hybrid/registry/`.
- Stable arXiv/venue identifiers still require publication.
- TRL / verl / OpenRLHF issue or PR submission is an external write. Drafts are
  in `ADOPTION_PACK.md`; a maintainer should review and submit them.
- The controlled DAPO/GSPO/Dr.GRPO/AERO audit still requires accelerator time.
  M-GRPO is correctly separated into an agentic stratum rather than presented
  as a minimal arithmetic hook.
- When promoting to a venue template, swap the `article` preamble for `neurips_2026.sty`;
  retain `natbib` and the shared canonical bibliography.
