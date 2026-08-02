# TMLR submission package

This directory is generated from the canonical manuscript one level above. It
uses the official TMLR style files at commit
`7bf90efe3a0debbba703c05c43f3ff7e4d4a2992`.

The paper and supplement are anonymous. The supplement contains the complete
600-record numerical projection used for the r4-2 tables, the two S1 receipt
projections, the unchanged S1 source and tests, the executed objective snapshot,
and a verifier. Remote account names, author metadata, and machine-local paths
are omitted. Original artifact hashes remain as provenance anchors.

Build in this order:

```bash
python3 build_submission.py --paper-only
python3 /path/to/compile_latex.py main.tex
python3 build_submission.py
```

Verify the resulting supplement:

```bash
unzip anonymous_supplement.zip -d anonymous_supplement
cd anonymous_supplement
shasum -a 256 -c MANIFEST.sha256
python3 verify_anonymous_claims.py
```

Do not submit this package while an overlapping archival paper is under review.

