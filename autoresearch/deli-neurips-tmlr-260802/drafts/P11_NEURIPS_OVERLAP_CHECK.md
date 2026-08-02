# P11 ↔ NeurIPS 36320 overlap check

Date: 2026-08-02
Sources:
- Spine: `zvf-program/audit/paper_P11_reproducibility_audit.tex` (post-§3.1 reframe)
- Live NeurIPS submission 36320 flagship: `zvf-program/flagship/paper/main.tex`
  (OpenReview forum `CXbcYe69BQ`; title *Same Terminal Signal, Different Action*)

## Method

Compared titles, abstracts (5-gram Jaccard), citation key sets, and presence of
load-bearing claim phrases. No OpenReview HTML fetch required for this offline
text comparison; re-run if either PDF is revised.

## Result: **NO MATERIAL OVERLAP — dual-submission risk is low**

| Check | Result |
|---|---|
| Abstract 5-gram Jaccard | **0.0** (0 shared 5-grams) |
| Body-head 5-gram Jaccard | **0.0012** (generic RL tokens only) |
| Shared bibliography keys | only `yu2025dapo` (public DAPO paper) |
| P11 contribution | single-stack preregistered audit protocol; exact paired-t power; BH multiplicity; 40 arm–seed fail-closed execution; all four arms `INCONCLUSIVE`; secondary DAPO cost (ZVF 0.693→0.000 at 3.61× rollouts) |
| Flagship contribution | decision-theoretic result that the same all-failure history can make stopping vs prepaid retry optimal in different states; S1 CPU conformance fixtures; fail-closed objective identity |

### Phrase presence

| Term | P11 | Flagship 36320 |
|---|:---:|:---:|
| ZVF / zero-variance | yes | yes (different role) |
| DAPO | yes | yes (as related GRPO-family method) |
| stackdiff / single-stack audit | yes | no |
| INCONCLUSIVE / MDE / BH | yes | no |
| same terminal signal / minimax regret | no | yes |
| survival framing (primary) | removed in §3.1 reframe | no |

Shared vocabulary (GRPO, GSM8K, DAPO, ZVF) is **background field language**, not a
shared result table or theorem. P11 does **not** restate the flagship minimax
result, S1 fixtures, or the “same terminal signal, different action” claim.

## Residual coupling (non-blocking)

1. Older P11 drafts cited unpublished companions (`minreportrl2026`,
   `grpoRegistry2026`, `zvfaudit2026`). The reframe grounds the abstract in
   Henderson et al. (2018) and OpenRLHF; companion cites remain only as
   internal-stack context in the body. They are **not** the NeurIPS 36320 PDF.
2. If a future P11 revision re-imports flagship theorems or S1 numbers, re-run
   this check.
3. Venue: P11 targets **TMLR methods/reproducibility**. Flagship is NeurIPS
   36320. Parallel submission of *this* P11 draft with 36320 is **not** a
   dual-submission of the same paper under NeurIPS/TMLR dual-submission rules,
   based on the comparison above. Confirm against the current OpenReview PDF
   before any actual TMLR upload.

## Decision

**P11 may proceed as an independent manuscript** once §3.1 edits are frozen.
Do **not** route the flagship manuscript to TMLR while 36320 is live
(see `PUBLICATION_READINESS.md`).

Spine PDF SHA-256 at check time: `faf0096798474ba7277e6df639009c831c5b1b75e57e704b31452f4e13e0bb2f`
