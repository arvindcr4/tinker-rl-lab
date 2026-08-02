# Portfolio roster disposition (zero-GPU freeze)

Date: 2026-08-02
Authority: `drafts/PORTFOLIO_DECISION.md` (`w-decide`) + 12-paper verification wave.
GPU matrix / confirmatory runner: **out of scope** for this freeze (human §4.3 gate).

## Canonical spine (ship after editorial freeze)

| ID | Disposition | Publication unit | State |
|---|---|---|---|
| **P11** | **SPINE** | Single-stack preregistered audit protocol + MDE-bounded cost of dynamic sampling | `ready_after_edits` — §3.1 applied 2026-08-02; PDF sha `faf0096798474ba7…` |

## Demoted / reassigned (not independent archival submissions)

| ID | Prior active role | Disposition | Rationale (one line) | Destination |
|---|---|---|---|---|
| P1 | Scaling laws paper | **DEMOTE → workshop/thesis note** | Negative identifiability recomputes; MoE/dense headline invalid (Nemotron is MoE) | §3.3 failed-identifiability note |
| P2 | ZVF main paper | **DEMOTE → rebuild short note** | `variance_mitigation.tsv` simulation consumed as measured; pooled correlations invalid | §3.2 sampling-model falsification note |
| P3 | Group size paper | **RETIRE as standalone** | Headline SNR/grid fabricated (`FALLBACK_ROWS`); do not merge fabrications into P2 | Thesis negative chapter; allocation plateau lives in P12 if kept |
| P4 | Length bias paper | **RETIRE / invert** | Cap non-identifiability; pseudo-replication; null carries no information | Optional ≤6pp measurement-validity note only |
| P5 | MIN-REPORT-RL | **MERGE resource** | 17× not a measurement; forced η²; keep field-coverage audit | P5+P6 reporting resource |
| P6 | GRPO registry | **MERGE resource** | Integrity: post-hoc prediction-sign flip (iter-194); disclose or delete audit | P5+P6 reporting resource |
| P7 | ZVF controller | **PARK** | No cost-matched control; PCD absent from audited controller; U-shape selective | Future experiment only; keep 0/1867 + ZVF/PCD separation for optional absorb |
| P8 | Workshop artifact | **DO NOT ship as P9 docs** | Mislabelled task; interrupted 100% row; sign-flip correlation | Regenerate docs from `run_manifest.tex` only |
| P9 | DNB benchmark | **REBUILD or park** | `make reproduce-main` missing; frontier table disagrees with sources | Artifact note after single-ledger rebuild |
| P10 | ZVF theory | **STRIP → appendix** | Placeholder figures; non-recomputing empirics | Theorem core T1–T3 → into §3.2 note only |
| P12 | Signal starvation | **PARK** | No PPO/SAO outcomes; 92.3% is by-construction base rate | Prospective methods note only |

## Deduplication lock (one home per artifact)

| Artifact | Sole home | Delete everywhere else |
|---|---|---|
| 505-task allocation / plateau | P12 (if revived) or thesis | P3 abstract orphan; P2 alternate definition |
| 17× backend exhibit | P5+P6 resource (as provenance failure, not causal) | standalone P5/P6 headlines |
| Matched-budget G=2×160 vs G=16×20 | P11 optional absorb or retired | P8, P10 E-R2b |
| `1.11e-16` identity | demote to two-line lemma in §3.2 note | P2/P12 “verified on 505 tasks” marketing |
| `FALLBACK_ROWS` grid | **nowhere** (delete) | P3, P8 grade B, P9 |

## Active submission queue (zero GPU)

1. **P11** — TMLR methods track after NeurIPS 36320 resolves or after explicit dual-submission clearance (overlap check: clean).
2. **P1 workshop note** — after §3.3 text fixes (in progress).
3. **P2 short note** — only after §3.2 deletion rebuild (not a cut-paste of the 46pp PDF).
4. Everything else: **not in the submission queue**.

## Explicit non-actions

- No confirmatory GPU launch.
- No A004 bind without human authorization.
- No TMLR upload of the flagship while 36320 is live.
- No claim that any estimator/framework ranks above another from this portfolio.

## Short units produced under zero GPU

- P2 note: `platform_hybrid/paper/paper_P2_zvf_falsification_note.pdf` sha `737b2d8cf29fa6718eaf80b1eccdf10c37666d2150bc6052e67cc6e8e1142127`
- Freeze ledger: `drafts/ZERO_GPU_FREEZE.md`

- P1 note: `platform_hybrid/paper/paper_P1_identifiability_note.pdf`
- P11 now includes absorbed matched-budget E-R2b panel (§Absorbed bounded panels)
