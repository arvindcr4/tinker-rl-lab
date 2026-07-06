# P5–P8 Improvement Brief (read this EVERY iteration)

Mission: raise papers **P5, P6, P7, P8** to the same evidence and rigor standard
as P1–P4 (which went through 150+ improvement iterations; P5–P8 have had far
fewer). Every improvement must be measured on real repo data this same
iteration — no proposals without prototypes.

## The four papers (all build at 0 LaTeX errors today — keep it that way)

| paper | file | thesis (one line) |
| --- | --- | --- |
| P5 | `paper/paper_P5_minreport.tex` | "Report the Stack, Not the Label" — RL-for-LLM results are stack-conditioned; MIN-REPORT is the reporting standard |
| P6 | `paper/paper_P6_registry.tex` | GRPO-Registry — machine-readable catalog of group-relative RL stacks and their variant deltas (`registry/schema.json`, `registry/entries/*.json`, `registry/query.py`) |
| P7 | `paper/paper_P7_zvf_controller.tex` | From Diagnostic to Controller — signal-starvation theory of GRPO + adaptive group-size intervention driven by ZVF |
| P8 | `paper/paper_P8_fraud.tex` | LLM vs XGBoost in credit-card fraud — sensor and scribe, not scorer (`fraud_data.csv`, `test_data.csv`) |

## Fresh data (merged 2026-07-04 — mostly UNANALYZED for P5–P8; mine it first)

- `experiments/results/n2_reward_tensor_resume/{grpo,aero,gift,areal}_s0_tensors.jsonl`
  — full per-(prompt × G) reward tensors, 40 steps × 4 GRPO-family methods,
  same stack. Exact ZVF per step. **P7 gold** (controller calibration on real
  tensors) and **P6 gold** (same-stack variant deltas measured, not claimed).
- `experiments/results/mega_20260704/{cells.tsv,manifests/,group_tensors/,cells_done.jsonl}`
  — one MIN-REPORT manifest per completed cell of (model × task × G × temperature
  × seed), growing live. **P5 gold**: worked examples + schema stress-test of the
  MIN-REPORT standard at scale.
- `experiments/results/n10_seed_expansion/` — 8-seed GRPO vs Dr.GRPO panel
  (growing live) with per-step ZVF. Seed-robustness material for P7 claims.
- `experiments/results/berkeley/` + `docs/berkeley_improvements/` — 20 finished
  analyses with reusable machinery: Miller error-bars recipe
  (`scripts/berkeley/adding_error_bars_to_evals.py`), eval-protocol MVSP
  (`eval_protocol_hardening.py`), Dualformer auto-G rule (row 01), AlphaProof
  tree-baseline (row 19), CDH (row 12). Reuse; do not reinvent.
- `experiments/results/zvf_iter*.tsv` — the full P2 evidence base (risk index,
  per-step features); P7 builds directly on it.

## Improvement target classes (pick ONE per iteration, name it in the ledger row)

- **T1 statistical rigor** — bootstrap CIs / TOST on every P5–P8 headline number
  (start by listing the headlines; reuse the Miller recipe). The single most
  reviewer-visible gap.
- **T2 fresh-data evidence** — run a new analysis on the N2 tensors / mega
  manifests / N10 seeds that directly strengthens (or honestly scopes) a P5–P8
  claim.
- **T3 cross-paper coupling** — e.g. P7 controller evaluated counterfactually on
  N2 tensors and connected to the Dualformer auto-G and AlphaProof γ*=0 results;
  P6 registry entries validated against what the N2 four-method run actually
  logged; P5 MIN-REPORT fields audited against what mega manifests actually
  contain (schema coverage %, missing-field table).
- **T4 related work** — verified-citation related-work hardening (arXiv MCP /
  firecrawl; NEVER cite unverified).
- **T5 presentation** — figures/tables for measured results only; captions state
  the finding.

## Deliverable conventions (mirror the Berkeley run)

- Ledger: `P5P8_IMPROVEMENTS.md` (worktree root) — same columns as
  `BERKELEY_IMPROVEMENTS.md`; statuses proposed → prototyped → validated →
  rejected. Re-rank by impact × evidence × paper-facing readiness each synthesis
  iteration. Record rejects with reasons.
- Per item: `docs/p5p8_improvements/<NN>_<slug>.md` (proposal + verified
  citations + measured result), `scripts/p5p8/<slug>.py` (≤300 lines, stdlib or
  the worktree venv), outputs to `experiments/results/p5p8/*.tsv|json`.
- Paper-facing text goes in ONLY when validated; rebuild the affected
  `paper_P{5..8}_*.tex` (pdflatex → bibtex → pdflatex ×2 into `paper/build/`)
  and keep **0 errors / 0 undefined citations** — that is the current state; do
  not regress it. New BibTeX entries go directly in `paper/references.bib`
  (verify metadata first).
- ≥1 line per iteration appended to `AUTORESEARCH_FINDINGS.jsonl` with
  `"pillar":"P5"|"P6"|"P7"|"P8"`.

## Hard rules (guardrail-enforced — do not fight them)

Write only inside this worktree; no `git push`, no `gh`, no secrets, no external
uploads, no `rm -rf`. Local commits encouraged (one per iteration). Files ≤300
lines. Zero interaction — never ask, always execute. Verify every citation
before using it.
