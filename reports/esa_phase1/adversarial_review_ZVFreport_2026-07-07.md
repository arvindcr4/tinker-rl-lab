# Adversarial Review — ZVF Phase-1 Project Report (2026-07-07)

Target: `Phase1_Project_Report_ZVF.tex/.pdf` (sample-format PES Phase-1 report, solo/ZVF).

## Panel (4 of 5 independent frontier models)
| CLI | Model | Verdict | Status |
|---|---|---|---|
| `kimi -p` | Kimi K2.7 | major-revision | ✅ |
| `minimax -p` | MiniMax-M3 | major-revision | ✅ |
| `agy -p` | (agy) | major-revision | ✅ |
| `codex exec` | GPT | major-revision | ✅ |
| `zai -p` | GLM-5.2 | — | ❌ z.ai API 529 (overloaded, retried twice) |

Raw outputs: `ZVFreport_review_{kimi,minimax,agy,codex}.md`.

## Convergent findings → fixes applied
| # | Finding (models) | Fix applied |
|---|---|---|
| 1 | "no gradient" ignores KL term (agy, codex) | §1.1/§1.2/§4.3/§5.1 reworded: zero **advantage/policy-gradient** signal; KL penalty still acts |
| 2 | ZVF unit "steps" vs "prompt groups" (kimi, minimax, codex) | Abstract/§1.1 corrected to "prompt groups" |
| 3 | Invalid t-test on n=3 seeds (kimi, minimax, agy) | §6.4: replaced `t≈0.6` with descriptive stats + sign-flip note |
| 4 | Missing standalone "Contribution of the candidate" (all 4) | §1.6 renamed + expanded; built-vs-leveraged delineation + impl-completion matrix (Table 1.2) |
| 5 | "~80% code" not evidenced (minimax, codex) | Implementation-completion matrix; Code-availability statement (App B) |
| 6 | Cross-framework claim unsubstantiated (kimi, codex) | §4.1 "Scope" paragraph: only Tinker/Colab results; four-way run deferred to Phase-2 |
| 7 | P8 AUROC overfitting / small N (kimi, minimax, agy) | §6.6 overfitting caveat (N=160, CV-only, proof-of-concept) |
| 8 | fig internal inconsistency: P1 "positive", P8 "0.63" (minimax) | New `fig7_zvf.tex`: P1="null 1.0→0.11", P8="positive AUROC 0.84" |
| 9 | Heading "Bibliography" not "References" (agy) | `\renewcommand{\bibname}{References}` |
| 10 | Underpowered n=8–20 (all 4) | Owned as #1 threat; Phase-2 plan adds full-GSM8K, ≥5 seeds, CIs |
| 11 | Missing plagiarism declaration (kimi, minimax, codex) | Originality/plagiarism statement (App B), target ≤15% |
| 12 | Missing dataset/hyperparameter tables (codex) | §6.1 Datasets & Experimental Configuration (Table 6.2) |
| 13 | Missing timeline/Gantt (agy) | §1.7 Project Plan and Timeline (Table 1.3) |
| 14 | Future Work too thin (minimax) | §7.2 expanded (power, cross-framework, P8 hardening) |
| 15 | Page count short (all 4, from text-only estimate) | Now **50 pp** (47 excl. title/cert/decl; ~40 arabic body) |

## Noted but NOT changed
- **"Qwen3.5-4B" model name** — 4 models flagged as suspicious. Kept as-is because the repo/experiments use this tag consistently; **flag for the candidate to confirm the exact model ID** before final submission.
- **"Missing figures / truncated Fig 4.1"** — false positive: reviewers saw text-only extraction; the 8 TikZ figures render correctly in the PDF.
- Population vs sample std in Eq (4.1) — added clarifying note; the zero-variance condition is invariant to the choice.

## Residual (legitimate, deferred to Phase-2 by design)
- No four-way cross-framework divergence table yet (harness built; runs are Phase-2).
- Held-out n remains small (budget); framed as noise-limited, powered eval is the #1 Phase-2 upgrade.
