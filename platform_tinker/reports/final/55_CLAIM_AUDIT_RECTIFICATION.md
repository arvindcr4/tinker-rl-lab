# 55-Claim Audit Rectification Log

Source: external reviewer claim audit with 55 items across core framing,
diagnostics/statistics, capability/reward, algorithm comparison, scaling/
architecture/hyperparameters, tool-use/browser, figures/ledgers/
reproducibility, policy-drift/length-bias/instability, and ethics/cost/
licensing.

Scope: `paper/main.tex`, `paper/main_anon.tex`,
`reports/final/capstone_final_report.tex`. Many items are already scoped in
earlier passes (the abstract, intro claim-evidence-verdict table, the
Scope-boundary paragraph, the Post-selection warning, Table~10 arithmetic
fix, the BH-autocorrelation caveat, the Artifact Reconciliation paragraph,
and the Limitations section). This log records what THIS pass added.

## What this pass added

| # | Claim-audit item | Pass-3 action |
|---|---|---|
| 1 | Run-count drift (79 vs 32 vs 83). | 55-point audit paragraph in Scope boundary explicitly says the bundle contains >70 recorded run artifacts and that earlier counts reflect different inclusion rules; headline analyses use only ledger-approved completed/partial rows. |
| 3 | 61.6-89.6% prompt-token loss range. | 55-point audit paragraph says the number is a two-sequence P1-A diagnostic on Qwen3.5-4B, not a corpus-wide estimate. |
| 12 | "First systematic cross-library ZVF/GU" priority claim. | 55-point audit paragraph scopes this to "reproducible cross-library case study" and cites zero-variance prompt diagnostics in recent literature. |
| 28 | "Confirms scaling law" from short traces. | 55-point audit paragraph states exponential-saturation fits on 20-30 step traces are hypothesis-generating, not validated scaling laws. (Body text already carries matching hedges.) |
| 32 | "G=8 optimal". | 55-point audit paragraph states the single-seed group-size sweep is inconclusive and we do not claim G=8 is globally optimal. (Body Limitations section already carries a matching item.) |
| 38 | "Tool-use capability". | 55-point audit paragraph: tool-call experiments establish JSON/schema compliance under a schema-only reward, not executed tool-use capability; strict no-warmup Tinker tool-use scored 0%. |
| 39 | BrowserGym as learning evidence. | 55-point audit paragraph states the smoke test yielded reward 0.0 and no trainable datums; it is an integration check, not learning evidence. |
| 43,45 | Reproducibility / artifact-readiness overclaim. | 55-point audit paragraph: W&B step-level logging and direct KL tracking failed; Tinker is closed-source; HuggingFace checkpoint inventory is partial; we provide Docker + ledgers + local CSV traces + some model cards and do not claim end-to-end reproducibility for every run. |
| 47,48 | "Policy drift" causal claim. | 55-point audit paragraph: SI/PTD/rolling-variance are reward-instability proxies; Nemotron-120B is a reward-collapse case consistent with drift or parser mismatch, not proof of catastrophic drift. Body prose at the old "catastrophically drift" line is softened to "sometimes exhibit reward collapse consistent with drift" with an explicit note that direct KL/policy-distance telemetry is not available. |
| 49,50 | GRPO more length-biased than PPO. | 55-point audit paragraph: the rewarded-shorter pattern is specific to the GSM8K parser used here; we do not have a matched PPO-vs-GRPO algorithmic comparison sufficient to rank by length-bias susceptibility. |
| 53 | "All bases are Apache/MIT-equivalent". | Dual-use paragraph in both main.tex and main_anon.tex now names per-family licenses: Qwen Apache-2.0; Llama-3.x Meta Llama Community License; DeepSeek-V3.1 and Nemotron under their respective community/commercial terms. Explicitly says we do not claim all bases are Apache/MIT-equivalent. |
| 52 | "No direct dual-use risk". | Dual-use paragraph adds the qualification that RL post-training is dual-use in the general sense that any optimization advance can lower cost for misuse; framed as qualitative residual risk rather than zero risk. |
| 27 | GRPO-is-secretly-DPO. | 55-point audit paragraph: external theoretical connection informs interpretation of mixed groups; we do not empirically establish DPO equivalence for our runner. |

## Items already covered by earlier passes (no new edit this pass)

- Items 2, 4, 5, 6, 14, 15, 16, 17, 18, 19, 23 (core framing + capability +
  reward caveats) are already in the abstract, intro, Scope boundary, and
  Claims-We-Do-Not-Make sections from earlier audit passes.
- Item 9 (first-5-step ZVF rule, 2 positive failures) is scoped in the
  abstract as "precision 1.0, recall 1.0" with the caveat that only two
  collapsed runs exist; ZVF Lagged Regression section already carries the
  calibrated-predictor caveat.
- Item 10 (continuous ZVF ranking weak) is the ZVF Lagged Regression
  finding; it is already in the Limitations "ZVF Correlation Is Partially
  Tautological" paragraph.
- Item 11 (15-telemetry-run strong association) is caveated in the same
  section; we added no new body changes.
- Item 20 (Table 10 91.6 vs 92.08 arithmetic) was fixed in commit 1ec9a59.
- Item 24 (Qwen PPO/GRPO artifact-sensitive / Llama PPO row) is already
  scoped in the abstract, intro Claim 2, and the PPO-vs-GRPO subsection.
- Item 25 (classical PPO baselines near zero = stack mismatch) is scoped
  in the Cross-Library caption and the classic-RL sanity-baseline annotation
  from commit 82d348c.
- Item 42 (14 launched / 7 complete / 5 partial / 2 no usable) is in the
  Infrastructure Failures limitations paragraph; we did not re-scope it.
- Items 44, 46 (HuggingFace inventory; ledger as source of truth) are noted
  in the 55-point audit paragraph only, with a pointer back to the ledger.

## Not fixable in this pass

- Page-limit compliance (item 1-adjacent, hostile-review #1): no new
  content edit shrinks the manuscript to 9 main-text pages.
- Figure 5 hardware caption (item 41): requires inspection of the specific
  figure's data provenance before rewriting the caption; deferred.
- One clean replicated experiment to make 82.0 to 83.3 seeded evidence
  (item 14-adjacent): requires compute.

## Files touched

- `paper/main.tex` - dual-use license rewrite, catastrophic-drift softening,
  55-point audit paragraph inserted at top of Claims We Explicitly Do Not
  Make.
- `paper/main_anon.tex` - matching dual-use rewrite, matching drift
  softening, duplicate section heading removed (residue of an earlier
  morph miss), 55-point audit paragraph inserted.
- `reports/final/55_CLAIM_AUDIT_RECTIFICATION.md` - this log.

No new experiments, figures, or tables. Editorial + arithmetic only.
