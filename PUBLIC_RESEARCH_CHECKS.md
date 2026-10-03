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

## Limits

The public records preserve 11/64 frozen events and a separate 3/64 conservative
numerical-witness classification. The cohort is the first 64 eligible questions
in a deterministic screen of 4,096 questions, with 87 eligible; it is not a random
benchmark sample. Review labels remain assistant-assisted judgments.

These checks do not authenticate the manifest's author or freeze time, recover
withheld raw completions, replay inference, independently validate review labels,
certify privacy or institutional approval, or establish training or capability
gains. A file digest is a consistency commitment, not evidence of execution.
