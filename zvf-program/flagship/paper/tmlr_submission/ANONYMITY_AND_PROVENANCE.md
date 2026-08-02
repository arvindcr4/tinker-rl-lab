# Anonymity and provenance

The review supplement is a numerical projection of a larger, hash-locked audit
bundle. It removes author names, email addresses, machine-local paths, and remote
account or project identifiers. Those fields are not used by any claim in the
paper.

Each projected receipt records the SHA-256 digest of its unredacted source. The
supplement also records the digest of the full review bundle and of the exact
objective source that produced the accepted r4-2 records. These digests are
anchors, not a claim that the omitted private remote objects can be fetched by a
reviewer. The full bundle can be released after double-blind review ends.

The anonymous package still lets a reviewer:

- inspect all 600 stored gradient relations;
- recompute the 62--65 joint-zero counts and the 69/100 registered-gate result;
- check all six held-out and token-ledger endpoints used in the manuscript;
- inspect the two complete S1 receipt projections and their unchanged source;
- verify every included byte against `MANIFEST.sha256`; and
- run `verify_anonymous_claims.py` without network access.

It does not contain private checkpoints, raw generated corpora, per-example
held-out predictions, credentials, or unredacted remote URLs. It therefore does
not rerun training or regenerate the stored gradients and evaluations.

