# ZVF Program — Pillar 4: MIN-REPORT-RL (Position Paper)

This pillar argues that RL post-training of LLMs is reported as if a three-letter algorithm
label (PPO/GRPO/DPO and the GRPO family: DAPO, GSPO, Dr.GRPO, MAD-GRPO) fixes the experiment,
when in fact the unreported *stack* — backend, sampler, precision, reference/KL handling, loss
form, LoRA config, reward parser, group-size schedule — co-determines the result and can flip
a comparison by more than the claimed algorithmic effect (a matched-config backend swap moved
last-10 reward 84.4% → 5.0%, ~17×, with no visible knob change). It proposes **MIN-REPORT-RL**,
a 7-item minimum-reportable-stack where every item is a documented flip lever, plus a controlled
single-stack **reproducibility audit** (re-implement DAPO/GSPO/Dr.GRPO/MAD-GRPO in one trainer
and report which claimed gains survive). It builds directly on the ZVF Program v1 audit
(stack-conditioning thesis; ZVF/GU triage; the clean 82.0%→83.3%, p=0.26 held-out control).

## Files
- `min_report_rl.tex` — standalone position-paper draft (compiles with plain `article`; TODO
  `\cite{}` keys render in red; bibliography is a TODO stub — no fabricated authors/years).
- `CHECKLIST.md` — copy-pasteable author checklist + fillable appendix template (the 7 items,
  each with why-it-flips, good/bad examples).
- `README.md` — this file.

## Target venues
- **Primary:** NeurIPS / ICML **Position Track**.
- **Secondary / companion:** **MLRC** (ML Reproducibility Challenge) and **ICLR Reproducibility
  Track** for the audit results once runs exist.

## What's left to do
- **Fill the audit tables** (`tab:audit`, `tab:audit_telemetry` in the .tex) from the audit
  corpus once the controlled DAPO/GSPO/Dr.GRPO/MAD-GRPO runs complete — all numeric cells are
  `[TODO]` placeholders.
- **Replace every `[cite:KEY]` / `[TODO]` reference** with real citations + a real `.bib`
  (the v1 audit paper, Henderson et al., and each GRPO variant's source paper). No fake
  authors/years were invented.
- **Finalize author block / affiliations.**
- **File the TRL / verl / OpenRLHF issues+PRs** (adoption section) and publish the shared
  MIN-REPORT-RL JSON emitter + schema; link them in the .tex.
- **Pre-register** the audit's seed count `S`, survival thresholds, and minimum detectable
  effect before scoring controlled runs.
- When promoting to a venue template, swap the `article` preamble for `neurips_2026.sty` +
  `natbib` and replace the TODO-cite shim / `thebibliography` stub.
