# Zero-GPU work freeze

Date: 2026-08-02 (packaging pass)

**Status: EXHAUSTED.** Content + packaging for non-GPU portfolio work is complete.
Remaining actions need a human (commit, portal upload) or a GPU (experiments).

## Shippable units

| Unit | Path | Pages | SHA-256 |
|---|---|---:|---|
| spine/tex | `zvf-program/audit/paper_P11_reproducibility_audit.tex` | — | `6e7c5f9eeb887bfe9a87c8d65901255c0a868cebbe60d6c9a3030bd3ddc183fe` |
| spine/pdf | `zvf-program/audit/paper_P11_reproducibility_audit.pdf` | 12 | `948bb2deede95aa47d4f9b8370bee6f694368dea958c1d1e5839e23072ae766d` |
| spine/prereg | `zvf-program/audit/preregistration.json` | — | `91048df7922276051cd8a07ffa939e4e6d1bbf8b8ef3c95705c0518beca9236f` |
| spine/audit_json | `zvf-program/audit/results/audit.json` | — | `385b17f43ec5b8d92b95554243b5e3cf04b1fc32e1362adf8fee472d087c8c67` |
| spine/audit_tex | `zvf-program/audit/results/audit_results.tex` | — | `a0f6d66915688b7d453899de04d8fb61e693639136991469ff6694004e9016ab` |
| spine/reanalysis | `zvf-program/audit/STATISTICAL_REANALYSIS.md` | — | `4596bd3db934c59a47da64623d3d16195394f112b33843b450125be3098d666e` |
| workshop_p2/tex | `platform_hybrid/paper/paper_P2_zvf_falsification_note.tex` | — | `8b877d356da194c7514ae8fa91f6b8fa68ab288427607ad5f30783459f37e774` |
| workshop_p2/pdf | `platform_hybrid/paper/paper_P2_zvf_falsification_note.pdf` | 3 | `737b2d8cf29fa6718eaf80b1eccdf10c37666d2150bc6052e67cc6e8e1142127` |
| workshop_p1/tex | `platform_hybrid/paper/paper_P1_identifiability_note.tex` | — | `078cca45fde1704d85e65060878c9fc7c3b5277a884ab393590cebe5b97a0fbe` |
| workshop_p1/pdf | `platform_hybrid/paper/paper_P1_identifiability_note.pdf` | 2 | `55ff5370f58217104f54a5684e65b9eff34ea759a7df3bad2612b51cb15f926c` |

## Package binder

`zvf-program/audit/tmlr_package_p11/` — README, MANIFEST.sha256, ANONYMOUS_CLAIM_LEDGER.md

## Checklist (all done without GPU)

- [x] P11 §3.1 reframe + P8 matched-budget absorb
- [x] P11↔36320 overlap check
- [x] Roster demotion
- [x] Claim ledger extension
- [x] P1 claim fixes + 2pp short note
- [x] P2 3pp falsification note
- [x] TMLR dual-track checklist
- [x] Package binder + byte manifest

## Not done

| Action | Who |
|---|---|
| `git commit` of freeze artifacts | human |
| OpenReview upload | human |
| Confirmatory GPU matrix / A004 / DAPO n=12 | human + GPU |
