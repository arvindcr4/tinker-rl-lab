# Tinker RL Lab Submission Bundle

## Contents

- `paper_anon.pdf`: compact anonymous diagnostic-audit paper
- `code.tar.gz`: anonymized source code, document sources, and verification scripts
- `checksums.sha256`: checksums for bundle members
- `MANIFEST.md`: human-readable manifest

## Verification Files (inside code.tar.gz)

After extracting `code.tar.gz`, the following documentation files are available:

- `REVIEWER_VERIFICATION.md`: Claim → Evidence → Command mapping
- `EVAL_PROTOCOL.md`: Dataset splits, reward parsers, claim status
- `FIGURE_PROVENANCE.md`: Figure generation scripts and inputs
- `SOURCE_PRECEDENCE.md`: Discrepant values explained
- `VERSION.json`: Bundle version and metadata

## Quick Verification

```bash
# Verify checksums
sha256sum -c checksums.sha256

# Extract and verify claims
tar -xzf code.tar.gz
cd tinker-rl-lab-anon
python3 scripts/verify_claims_offline.py
```

See `REVIEWER_README.md` for the full verification guide.
