# Verify the public research release

From the repository root, run:

```sh
make public-check PYTHON=python3
```

Python 3.11 or later and Make are sufficient. No package installation, model,
GPU, account access, network request or TeX rebuild is involved. Without Make:

```sh
python3 -B tools/check_public_release.py
python3 -B tools/check_public_results.py
python3 -B -m unittest discover -s tests -p 'test_public*.py' -v
```

Both checkers return JSON and exit nonzero on failure. Their default input root
is this checkout, independent of the current working directory. Use `--root`
to inspect another local checkout. They never rewrite a manifest, ledger or PDF.

## What the checks establish

1. The reviewed 3 October snapshot's file inventory, byte counts and SHA-256
   digests match `reports/public_revision_2026-10-03/PUBLICATION_MANIFEST.json`.
   Copied scientific definitions match their exact-source provenance digests.
   Local links from the public entry-point READMEs resolve.
2. Included scientific records are internally consistent across the summary,
   JSON and CSV ledgers, document projections and selected source bindings.
   The frozen primary endpoint and the separately classified conservative
   witnesses remain distinct.
3. Synthetic parser/statistics contracts and corrupted-fixture regressions pass.
   Mutation tests operate only on temporary copies, never on published files.

The original release manifest remains an unchanged historical input. These new
verification tools and CI wiring are subsequent Git-tracked additions, outside
that snapshot's allowlist. The check fails if covered release files change;
do not refresh hashes merely to make it pass. An intentional document or data
revision requires a separately reviewed publication update.

The dedicated `public-research` CI job runs the same command on Python 3.11,
3.12 and 3.13. The normal non-LaTeX test suite also discovers these regressions.
A local pass does not imply that hosted CI has run or passed.

Run `make review-check PYTHON=python3` separately before claiming the current PDF
has a matching completed review receipt. This binds the explicitly selected PDF
to the receipt's filename, size and SHA-256 and checks its recorded review status.
It does not independently review page layout, science, privacy or reviewer identity.
The frozen October 3 quality receipt describes an older PDF and currently fails
this gate. Preserve that historical record; a fresh review must issue a truthful
receipt for the actual final PDF. Do not change hashes merely to pass the gate.

For a native PDF rebuild into new output files, use the separate
[safe public-document rebuild wrapper](BUILD_PUBLIC_RESEARCH.md). It preserves
the reviewed snapshot and requires a compatible existing TeX toolchain/cache.

## Limits

The public records preserve 11/64 frozen events and a separate 3/64 conservative
numerical-witness classification. The cohort is the first 64 eligible questions
in a deterministic screen of 4,096 questions, with 87 eligible; it is not a random
benchmark sample. Review labels remain assistant-assisted judgments.

These checks do not authenticate the manifest's author or freeze time, recover
withheld raw completions, replay inference, independently validate review labels,
certify privacy or institutional approval, or establish training or capability
gains. A file digest is a consistency commitment, not evidence of execution.

## October 4 candidate

Run `python tools/check_public_revision.py --revision reports/public_revision_2026-10-04 --pdf thesis/Thesis_Report_ArvindCR_C1_Coverage_Derived_2026-10-04_PUBLIC.pdf` to check the separately versioned candidate. This validates exact inventory and recorded PDF review binding; it does not independently perform the review or establish scientific validity. See [public review packaging](PUBLIC_REVIEW_PACKAGE.md). The historical October 3 checks remain separate.
