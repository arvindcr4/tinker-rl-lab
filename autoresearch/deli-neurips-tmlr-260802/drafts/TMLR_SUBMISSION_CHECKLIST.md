# TMLR submission checklist

Status: blocked until NeurIPS submission 36320 is closed or withdrawn.

## Files

- Paper: `zvf-program/flagship/paper/tmlr_submission/main.pdf`
- Supplement: `zvf-program/flagship/paper/tmlr_submission/anonymous_supplement.zip`
- Paper source: `zvf-program/flagship/paper/tmlr_submission/main.tex`
- Internal decision record: `autoresearch/deli-neurips-tmlr-260802/drafts/PUBLICATION_READINESS.md`
- Portfolio overlap record: `autoresearch/deli-neurips-tmlr-260802/audits/18_PAPER_PORTFOLIO_REVIEW.md`

## Submission description

Contribution types: methodology, reproducibility, and a registered feasibility
postmortem.

Suggested keywords: group-relative reinforcement learning; semantic
conformance; reproducibility; gradient auditing; verifiable rewards; negative
results.

One-sentence summary:

> The paper proves that terminal reward homogeneity is insufficient to choose a
> retry action, gives an executable objective-to-gradient conformance protocol,
> and shows how that protocol preserves a failed registered mechanism gate
> instead of turning high cosine into an equivalence claim.

## Before upload

- Confirm in OpenReview that the NeurIPS paper is no longer under review.
- Recheck the overlap table against the final NeurIPS record.
- Recheck the flagship against all 18 portfolio files; do not upload a second
  derivative as a separate archival paper.
- Rebuild both files and confirm their recorded SHA-256 digests are unchanged.
- Extract the supplement in a new directory and run
  `python3 verify_anonymous_claims.py`.
- Run the identity scan on the PDF, source, file names, and supplement contents.
- Read the title, abstract, contribution list, result table, limitations, and
  conclusion against `ANONYMOUS_CLAIM_LEDGER.md`.
- Keep `69/100 < 95/100` visible in the abstract and results.
- Keep all four E1 outcomes as `INCONCLUSIVE` if E1 is discussed anywhere.
- Do not link the anonymous submission to a named repository or preprint.
- Add author names only after the review system asks for the non-anonymous final
  version.
- Enter conflicts, author order, and subject areas manually in OpenReview.

## Venue rules checked

- [TMLR author guide](https://jmlr.org/tmlr/author-guide.html)
- [TMLR editorial policies](https://jmlr.org/tmlr/editorial-policies.html)
- [TMLR acceptance criteria](https://jmlr.org/tmlr/acceptance-criteria.html)
- [Official TMLR style files](https://github.com/JmlrOrg/tmlr-style-file)

## Stop conditions

Stop the upload if any of these is true:

- the NeurIPS review is still active;
- the paper or supplement reveals an author identity;
- the 69/100 result is described as equivalence;
- an inconclusive held-out comparison is described as improvement;
- the supplement verifier or manifest fails; or
- the uploaded PDF differs from the locally reviewed digest.
