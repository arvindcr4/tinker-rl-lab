# TMLR package — P11 spine (zero GPU)

Date: 2026-08-02

This package is the **submission-facing binder** for the portfolio spine.
It does **not** replace the NeurIPS 36320 flagship TMLR folder under
`zvf-program/flagship/paper/tmlr_submission/` (that track stays blocked while
36320 is live).

## What to submit (when human greenlights)

| Priority | Unit | Local PDF (gitignored `*.pdf`) | Source |
|---:|---|---|---|
| 1 | Spine | `../paper_P11_reproducibility_audit.pdf` | `../paper_P11_reproducibility_audit.tex` |
| 2 | Workshop | `../../../platform_hybrid/paper/paper_P2_zvf_falsification_note.pdf` | sibling `.tex` |
| 3 | Workshop | `../../../platform_hybrid/paper/paper_P1_identifiability_note.pdf` | sibling `.tex` |

## Integrity

See `MANIFEST.sha256` for byte hashes of tex/pdf/prereg/audit artifacts at package time.

## Anonymity checklist (before OpenReview)

1. Rebuild PDF from clean tree; compare SHA-256 to `MANIFEST.sha256`.
2. `pdftotext` scan for author name, email, institution, github user.
3. Read every claim against `ANONYMOUS_CLAIM_LEDGER.md`.
4. Confirm all four E1 arms remain `INCONCLUSIVE`.
5. Confirm DAPO cost paragraph is framed as cost, not capability.
6. Do not upload while unsure about dual-submission policy vs 36320 — re-read
   `drafts/P11_NEURIPS_OVERLAP_CHECK.md` and venue rules.

## Explicitly out of package

- Confirmatory sampler matrix / A004 / GPU runs
- Demoted 40–80pp P1–P12 long drafts as archival submissions
- Flagship "Same Terminal Signal" TMLR bundle (separate track)

## Portfolio pointers

- Decision: `autoresearch/deli-neurips-tmlr-260802/drafts/PORTFOLIO_DECISION.md`
- Disposition: `.../PORTFOLIO_ROSTER_DISPOSITION.md`
- Freeze: `.../ZERO_GPU_FREEZE.md`
