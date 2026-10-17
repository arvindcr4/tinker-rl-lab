# NeurIPS Paper Checklist: stack-provenance paper

Paper: a machine-readable **stack-provenance schema plus a `stackdiff` tool** (claim C).
It argues that GRPO-family results are labelled by algorithm, but the stack (model,
harness, reward, template, decoding, batch, loss, optimizer) drives the outcome. It has
two case studies: (A) same-stack GRPO vs PPO, and (B) the Zero-Variance Fraction.
LaTeX entry point: `platform_hybrid/paper/paper_P6_registry.tex`. Artifact:
`platform_hybrid/registry/`.

This file answers the checklist honestly for that paper as of 2026-10-16. It replaces
`NEURIPS_CHECKLIST_FINAL.md`, which audits the April 2026 compendium. Answers are
**Yes / Partial / No / NA**. "Partial" means the answer becomes Yes after the stated fix.

| # | Item | Answer |
|---|---|---|
| 1 | Claims | Partial |
| 2 | Limitations | Yes |
| 3 | Theory | NA (B: Partial) |
| 4 | Reproducibility | Partial |
| 5 | Open access to data and code | Yes |
| 6 | Experimental details | Partial |
| 7 | Statistical significance | Partial |
| 8 | Compute | Partial |
| 9 | Code of ethics | Yes |
| 10 | Broader impacts | Yes |
| 11 | Safeguards | NA |
| 12 | Licences | Partial |
| 13 | New assets | Partial |
| 14 | Crowdsourcing / human subjects | NA |
| 15 | IRB | NA |
| 16 | LLM usage | Partial (until the declaration is rewritten) |

---

## 1. Claims: Partial

Do the abstract and introduction reflect the contributions and scope?

- **C (main).** The core claims (a schema exists, `stackdiff` flags risky
  cross-stack comparisons, and the numbers in the exhibits match raw logs) are supported.
  Three framings still overreach and must change before submission:
  - **(a) The "17× Tinker vs TRL swing".** Base→Instruct on the same Tinker harness
    already accounts for 64% of the gap. Reward function, prompt, template, decoding,
    batch, data, loss and optimizer also differ. The "matched" YAML describes neither run.
    Reframe it as a decomposition, not a backend effect.
  - **(b) The "DAPO" pair.** It differs in model, task, G and steps. The 0.00 open score is
    true by construction, and the Tinker "dapo" run is not DAPO. Drop the pair or rebuild it
    matched. The registry entry is being corrected (`tests/test_registry_stackdiff.py::test_tinker_dapo_entry_is_corrected`).
  - **(c) The "0.034 band" equivalence.** It fails TOST at ±0.05 to ±0.15 when unpaired.
    Rerun it with a pre-set margin, ≥5 seeds and ≥100 steps.
- **Novelty.** The idea that implementation details drive RL outcomes is known (Henderson
  et al. 2018; Engstrom et al. 2020). The contribution is the machine-readable schema and
  diff tool. Cite arXiv 2604.23747 (framework bugs).
- **A (case study).** State the result as the v2 null (TOST-equivalent at ±0.02, n=7),
  not as v1's +0.050. The thesis (`reports/public_revision_2026-10-04/thesis/`, ch06 §6.5.5,
  ch10) still cites v1.
- **B (case study).** State that ZVF ≈ mean_i h_G(p_i), a binomial identity, so it is
  not an independent diagnostic. ρ=0.27 is still asserted in
  `platform_hybrid/paper/sections/p2_abstract.tex:15`, `p2_results_intro.tex:9` and
  `p7_ushape.tex:94`, and the "controller trace" claim must go because no controller ran.
  Prior work already covers this ground: Advantage Collapse Rate (2605.21125), Gradient
  Starvation (2605.07689), DAPO/GRESO and DARA (2610.00574).

## 2. Limitations: Yes

Evidence: `LIMITATIONS_AND_IMPACT.md`; thesis ch06 §6.5.5 and ch10 (ZVF predictive
validity unestablished, ρ CI [−0.37, 0.88]). The paper must also state that the
registry entries were partly hand-written from YAML rather than generated from run
logs.

## 3. Theory, assumptions and proofs: NA (B: Partial)

C makes no theorems. For B, the binomial identity ZVF = E[p^G + (1−p)^G] holds by
definition, and the paper should present it as such. The gradient link is supported
only at toy scale (`platform_hybrid/paper/sections/p7_abstract.tex`), and there are no
formal proofs.

## 4. Experimental result reproducibility: Partial

- `REPRODUCE.md` now has sections for C (Part C: registry CLI), A (Part A:
  analysis-only replay of v1 and v2 plus full reruns, Modal and Colab) and B (Part B:
  ZVF recompute from stored reward tensors and the W&B audit). Every offline command was
  run with a timeout on 2026-10-16 and printed the values shown.
- `smoke_test.sh`: 6/6 checks pass offline in 67 s with the venv active. The full
  `pytest` suite passes (954 pass, 3 skip per the audit), and
  `tests/test_registry_stackdiff.py` passes 27/27.
- Gaps: (i) the v2 per-run JSONs live on the Modal volume `tinkerrl-results`, not in the
  repo, so commit `samestack_gsm8k_v2s/*.json` or mirror them to HF. (ii) The C exhibits'
  matched rerun (audit plan step 3: Base vs Instruct × two harnesses, then TRL vs Tinker
  with matched checkpoint, reward, prompt, decoding and batch) has not been run.
  (iii) corr(‖∇‖, p(1−p)) = +0.71 cannot be recomputed because only the scalar was stored.

## 5. Open access to data and code: Yes

- Code: the GitHub repository is public under Apache-2.0. The anonymized export for
  review is built by `tools/anonymize_export.py` into `dist/anon_export/`.
- Results: `platform_hybrid/experiments/results/` (same-stack v1, ZVF tensors, W&B
  snapshot `tinker_wandb_registry/wandb_runs.jsonl`), plus the public HF datasets
  `arvindcr4/tinker-rl-bench-wandb` and `arvindcr4/tinker-rl-bench-checkpoints`.
- Weights, all public on HF with model cards (2026-10-16):
  - A: `arvindcr4/samestack-v2-qwen2.5-1.5b-gsm8k-lora` (27 adapters).
  - B: `arvindcr4/zvf-grpo-qwen3.5-4b-gsm8k-s123` (made public 2026-10-16).
  - E1 GRPO-family confirmatory runs: `arvindcr4/tinker-rl-lab-e1-*` (45 repos).
- Data: GSM8K is public. No new dataset is released.

## 6. Experimental setting / details: Partial

- A: the script header and results config give everything (LR 1e-5, LoRA r16, 30 steps,
  64 gens/step, G, token cap 200 or 512, critic set-up). They are repeated in the HF card.
- C: every registry entry records its stack fields. As of the 2026-10-16 uncommitted
  working tree, `stackdiff` compares checkpoint, reward, prompt/chat template, dataset,
  eval protocol, loss, batch and optimizer fields, and it reports field coverage. However,
  `stackdiff --all` still puts 32 of the 34 same-label pairs at R5 and marks 26 of 34 as
  underspecified, so the levels barely discriminate yet. The paper must give the R1–R5
  definitions and the coverage threshold (0.8).
- B: configs depend on the regime and are scattered across cohorts. Collect them into one
  table.

## 7. Statistical significance: Partial

- A v1: n=5 seeds; paired t p=0.016, but the exact sign-flip p is 0.0625 (Holm 0.19);
  200 fixed test items with no item-level CI. v2: n=7/10 seeds; paired-t 95% CIs, exact
  sign-flip p, Holm across 3 pre-registered contrasts, and TOST ±0.02 are all reported.
  The minimum detectable effect is 0.015 at n=7. Three PPO seeds and the `ppo_8x8` arm
  are missing (~10 A10G-h).
- B: 17 finite cells, CI spans 0; secondary p-values are uncorrected.
- C: the exhibit numbers have no uncertainty attached. (c) needs paired TOST with a
  pre-set margin.
- The paper must state the unit of analysis (seed), the CI method (paired t or
  percentile bootstrap), and that error bars are 95% CIs, not SE.

## 8. Compute resources: Partial

- A: v1 was ~3.5 A10G-h (15 runs); v2 has ~19.5 A10G-h done (27 runs × ~42–48 min) out
  of ~29 A10G-h for all 40 runs. These figures are in `REPRODUCE.md` Part A.2.
- B: Tinker sampling, Qwen3-8B, 200×8 samples per seed (< $1 per seed). The W&B
  `zvf-audit` project covers 368 runs, but its total compute is not reported.
- C: registry tooling runs on CPU in < 0.1 s per command.
- Missing: total compute including failed and preliminary runs, and Tinker spend for the
  B cohorts.

## 9. Code of ethics: Yes

Evidence: `platform_hybrid/paper/ethics_statement_anon.tex`. No human subjects and no
personal data. The released models are small LoRA adapters trained on math or
tool-calling tasks.

## 10. Broader impacts: Yes

Evidence: `LIMITATIONS_AND_IMPACT.md`. The positive impact is fewer misattributed RL
results. The negative impact is minimal; a registry could lend false authority to
entries that are wrong, and the paper should say entries are only as good as their
provenance.

## 11. Safeguards: NA

There is no high-risk release: LoRA adapters on ≤8B models (plus one 35B-A3B smoke
adapter) trained on GSM8K or tool calling, and a JSON schema.

## 12. Licences: Partial

- Code: Apache-2.0 (`LICENSE`). The README's "see component dirs" note needs
  reconciling. Some components (e.g. vendored trainers) may carry their own licences,
  and those must be listed.
- Base models: Qwen2.5-1.5B-Instruct, Qwen3-8B, Qwen3.5-4B and Qwen3.6-35B-A3B are all
  Apache-2.0 (checked on HF 2026-10-16), and each HF card states Apache-2.0 to match.
- Data: GSM8K is MIT. Salesforce/xlam-function-calling-60k (pavlov smoke adapters only)
  is CC-BY-4.0.
- To do: name the GSM8K and Qwen licences in the paper. Older HF repos still have mixed
  MIT/Apache/no licence and are not cited by this paper.

## 13. New assets: Partial

- `stackdiff` and the schema are documented in `platform_hybrid/registry/README.md` and
  the CLI `--help`, with usage in `REPRODUCE.md` Part C. Unit tests exist
  (`tests/test_registry_stackdiff.py`, 27 pass) but are not yet committed.
- The README count ("46 queryable records, 28 stack") disagrees with the paper's "20
  entries", so the two must be reconciled.
- There is no `DATASHEET.md` for the registry entries. Add a short datasheet covering
  how entries were produced (hand-written vs generated from logs), the maintenance plan,
  and the licence.
- HF model cards for A, B and E1 were written and pushed on 2026-10-16. Copies are in
  `huggingface/cards/`.

## 14. Crowdsourcing and human subjects: NA

## 15. IRB approvals: NA

## 16. Declaration of LLM usage: Partial

The previous declaration listed GitHub Copilot and ChatGPT Pro only. That understates
the workflow. The repository contains `AGENTS.md`, `GEMINI.md`, `CLAUDE.md` and
`autoresearch/` loops, which show that LLM agents were used substantively for code,
experiment orchestration, analysis and drafting. The paper must say so, and name the
tools and the parts of the work they touched. LLMs are also the objects of study (the
Qwen/Llama policies), which is a separate and already-declared use.

---

## Double-blind readiness

- `paper.tex` lists names and emails, and ~870 tracked files contain the GitHub handle.
  Submit only from `dist/anon_export/` after `tools/anonymize_export.py` reports zero
  residual matches.
- `paper_P6_registry.tex` has no anonymized variant, because
  `sections/_shared_author.tex` carries the author block (both names, institution, emails).
  The only anonymized sources are April-compendium files:
  `platform_local/blind_review/main_anon.tex` is missing 4 of its 7 `\input` files, and
  `platform_hybrid/paper/archive/absorbed/U01_main_compendium/main_anon.tex` is missing 12 of 89.
  Neither compiles.
- HF and W&B links in the paper resolve to the author's namespace. For review, point them
  to the anonymized export or use anonymous mirrors.

## Must-fix before submission

1. Reframe C exhibits (a)–(c) as in item 1; run the matched designed experiment.
2. Commit the `stackdiff` fixes and tests; regenerate registry entries from run logs.
3. Replace v1 with v2 for A everywhere; remove ρ=0.27 and the controller claim for B.
4. Fix two citations: `hochlehnert2025sober` is misattributed (it is about evaluation
   sensitivity, not reward hacking), and the `cobbe2021gsm8k` "35%" figure is wrong.
5. Rewrite the LLM-use declaration; add the registry datasheet; anonymize via the export
   tool.
