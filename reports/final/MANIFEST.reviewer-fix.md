# Reviewer-Fix Build Manifest

Build date: 2026-04-22
Scope: Absorbed hostile-reviewer critique into the capstone report and the
NeurIPS paper. All algorithm-level claims, evaluator-harness claims,
training-on-test exposure, and statistical-power claims have been narrowed.

## What changed

### `reports/final/capstone_final_report.tex` (109 pages)

1. **Abstract** — rescoped to "systems-oriented empirical audit of when a
   critic-free, group-relative RL loop receives a usable reward signal".
   Explicit algorithm-fidelity and evaluator-harness caveats.
2. **§1.5 Thesis statement** — narrowed to surviving-claims framing.
3. **§3.3 Implementation fidelity of the training runner** (new, with
   Table `tab:grpo-fidelity`) — records which canonical GRPO components are
   implemented, partial, or absent in the central Tinker runner.
4. **§6 Results narrative (both copies of the "stark contrast" paragraph)**
   — narrowed to evaluator-harness framing: HumanEval sandbox removes Python
   built-ins; tool-use reward is schema-only; MATH gives boxed-answer partial
   credit; MATH-500 / HumanEval prompt pools come from the `test` split.
5. **A1 / A2 / A3 paragraphs** — ZVF/GU n=2 caveat, `P(usable)`
   necessary-not-sufficient first-order diagnostic, identifiability framing.
6. **Limitations chapter** — seven new `\section*` subsections:
   Algorithm-Fidelity Caveat, Evaluator-Harness Validity, Training-on-Test
   for Probe Tasks, Benchmark Contamination, Missing Canonical Baselines
   (SFT-only, best-of-N, DAPO, Dr.GRPO, REINFORCE++, RLOO), Statistical Power
   and Paired Testing, Diagnostic Scope of ZVF/GU.
7. **Conclusion** — rewritten to the surviving-claims version.
8. **Bibliography** — added `liu2025rlzvp` entry.

### `paper/main.tex` (55 pages, NeurIPS-style)

Mirror edits applied to shared section files with both author and anonymous
variants kept in sync:

- `paper/sections/abstract.tex` + `abstract_anon.tex` — systems-audit framing
  and algorithm-fidelity disclaimer, closing evaluator-harness caveat block.
- `paper/sections/intro.tex` + `intro_anon.tex` — three-scoping-choice
  preamble (algorithm fidelity / evaluator scope / probe-vs-generalization).
- `paper/sections/conclusion.tex` + `conclusion_anon.tex` — no algorithm-level
  claim about GRPO-the-paper survives; surviving claim is reward-contrast
  amplification on this runner.

## Reviewer objections addressed

| Reviewer objection | Where addressed |
|---|---|
| "Runner is not canonical GRPO (no ratio, no clip, no reference KL)" | Capstone §3.3 + Table, Limitations, paper abstract + intro + conclusion |
| "HumanEval sandbox removes `__builtins__`" | Capstone §6 narrative, Limitations, paper abstract + intro + conclusion |
| "Tool-use reward is JSON schema, not execution" | Same |
| "MATH reward gives boxed-answer credit, not reasoning" | Same |
| "MATH-500 and HumanEval from `test` split as training prompts" | Capstone §6, Limitations, paper intro scoping choice (iii) |
| "Held-out GSM8K test is underpowered; paired test preferred" | Limitations / Statistical Power |
| "P(usable) = iid-binary idealization, not a theorem" | Capstone A2 paragraph |
| "ZVF n=2 positives ≠ predictive claim" | Capstone A1, Limitations / Diagnostic Scope |
| "Missing SFT-only / best-of-N / DAPO / Dr.GRPO baselines" | Limitations / Missing Canonical Baselines |
| "Contamination of public benchmarks" | Limitations / Benchmark Contamination |
| "Framework-effect claim is observational, not causal" | A3 identifiability framing |

## Build commands

```bash
# Paper
cd paper && pdflatex -interaction=nonstopmode main.tex \
  && pdflatex -interaction=nonstopmode main.tex

# Capstone
cd reports/final && pdflatex -interaction=nonstopmode capstone_final_report.tex \
  && bibtex capstone_final_report \
  && pdflatex -interaction=nonstopmode capstone_final_report.tex \
  && pdflatex -interaction=nonstopmode capstone_final_report.tex
```

Zero BibTeX warnings after `liu2025rlzvp` entry was added.

## Submission bundles (regenerated)

- `reports/final/capstone_final_report_submission.zip` (986 KB)
- `reports/final/capstone_complete_submission_all_files.zip` (6.6 MB,
  with refreshed `overleaf_document.zip` and `final_report.pdf`)

## Verification

See `SHA256SUMS.reviewer-fix.txt` — all 14 artifacts verified OK via
`shasum -a 256 -c`.
