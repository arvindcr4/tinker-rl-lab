# TMLR submission checklist

Updated: 2026-08-02

Two independent tracks exist. **Do not confuse them.**

---

## Track A — Portfolio spine (P11 GRPO-Stack-Audit) — zero-GPU ready

**Status:** manuscript + short companions built; portal upload is a human step.

| Item | Path |
|---|---|
| Package binder | `zvf-program/audit/tmlr_package_p11/README.md` |
| Paper tex | `zvf-program/audit/paper_P11_reproducibility_audit.tex` |
| Paper pdf (local) | `zvf-program/audit/paper_P11_reproducibility_audit.pdf` |
| Claim ledger | `zvf-program/audit/tmlr_package_p11/ANONYMOUS_CLAIM_LEDGER.md` |
| Byte manifest | `zvf-program/audit/tmlr_package_p11/MANIFEST.sha256` |
| Overlap vs NeurIPS 36320 | `drafts/P11_NEURIPS_OVERLAP_CHECK.md` (clean) |
| Workshop P2 | `platform_hybrid/paper/paper_P2_zvf_falsification_note.tex` |
| Workshop P1 | `platform_hybrid/paper/paper_P1_identifiability_note.tex` |

### Before Track A upload

- [ ] Human freezes PDF hash against `MANIFEST.sha256`
- [ ] Identity scan on PDF + source
- [ ] Claims match `ANONYMOUS_CLAIM_LEDGER.md`
- [ ] Confirm venue dual-submission comfort vs live 36320 (overlap is content-clean; policy is human)
- [ ] Enter conflicts / subject areas in OpenReview manually

Contribution types: methodology, reproducibility, negative/bounded results.

One-sentence summary:

> A single-stack preregistered audit of DAPO/GSPO/Dr.GRPO/AERO against shared GRPO finds all four comparisons INCONCLUSIVE under exact paired-t power, and reports DAPO's zero-ZVF cost (3.61× rollouts) without a capability claim.

---

## Track B — Flagship "Same Terminal Signal" — blocked while NeurIPS 36320 live

**Status:** blocked until NeurIPS submission 36320 is closed or withdrawn.

| Item | Path |
|---|---|
| Paper | `zvf-program/flagship/paper/tmlr_submission/main.pdf` |
| Supplement | `zvf-program/flagship/paper/tmlr_submission/anonymous_supplement.zip` |
| Decision record | `drafts/PUBLICATION_READINESS.md` |

### Before Track B upload

- Confirm in OpenReview that the NeurIPS paper is no longer under review.
- Recheck overlap table against the final NeurIPS record.
- Rebuild both files and confirm recorded SHA-256 digests.
- Extract the supplement and run `python3 verify_anonymous_claims.py`.
- Keep `69/100 < 95/100` and all E1 outcomes `INCONCLUSIVE` if discussed.

---

## Venue rules

- [TMLR author guide](https://jmlr.org/tmlr/author-guide.html)
- [TMLR editorial policies](https://jmlr.org/tmlr/editorial-policies.html)
- [TMLR acceptance criteria](https://jmlr.org/tmlr/acceptance-criteria.html)
- [Official TMLR style files](https://github.com/JmlrOrg/tmlr-style-file)

## Stop conditions (either track)

Stop upload if: identity leak; DISAPPEARS sold as live verdict; inconclusive sold as improvement; manifest/hash mismatch; dual-submission policy violated.
