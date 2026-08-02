# Offline review bundle

`review_bundle.zip` is the self-contained artifact for auditing the claims made
by the manuscript. It is deterministic: payload order, timestamps, permissions,
compression settings, and the internal manifest are fixed.

## Verify

```bash
unzip review_bundle.zip -d review_bundle
cd review_bundle
shasum -a 256 -c MANIFEST.sha256
python3 verify_claims.py --repo-root repository
```

The outer archive digest is recorded in `REVIEW_BUNDLE.sha256`.

## Included

- paper source, PDF, bibliography, claim audit, verifier, and review disposition;
- complete S1 reference/adapters/tests, amendment, freeze, and TRL/verl receipts;
- root `pyproject.toml` and `uv.lock` for the S1 lock/wheel check;
- the frozen pilot preregistration;
- the r4-2 manifest, supervisor state, execution notes, acceptance records,
  result records, recovery receipts, A100-quota evidence, and filtered-failure log;
- frozen source-provenance archives and corpus-binding records.

The accepted r4-2 receipts bind `pilot/objective.py` to SHA-256
`980a56a1651299a5adbe7a0927c13b12d42d9d7e1a36205500a24d5eeba9b61b`.
Later experiments changed the live checkout. The exact executed source was
recovered from the prior content-addressed review bundle, frozen as
`pilot/provenance/r4-2-objective.py`, and is placed at the original path inside
this archive. The bundle builder fails if that snapshot's hash changes.

## Deliberate boundary

The archive supports offline verification of cryptographic integrity and the
internal invariants reported in the paper. It includes all 600 stored gradient
diagnostics and all six evaluation/compute ledgers. The verifier also evaluates
the frozen joint mechanism predicate and records that the completed
intended-full balanced cell passes 69/100 steps, below the required 95/100.

It does not include private model checkpoints, raw generated corpus payloads,
per-example held-out predictions, or credentials. Therefore the verifier does
not regenerate gradients, predictions, corpora, or training. The paper uses
“receipt-verified” where this distinction matters and makes no causal training
claim.

## Clean-extraction S1 gate

The S1 README deliberately splits the incompatible external stack environments;
do not use test discovery across both adapters. The documented clean-extraction
commands were run on 2026-07-27: 35/35 common+TRL tests passed in isolated
Python 3.12 with TRL 1.2.0/Transformers 5.5.4, and 10/10 verl tests passed in a
fresh Python 3.11 environment with verl 0.3.0.post1/Torch 2.4.0/Transformers
4.45.2.
