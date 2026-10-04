# October 4 public review package

`tools/build_public_review_package.py` creates a new ZIP from the reviewed October 4 public revision. It leaves the historical identified university package described in `SUBMISSION.md` untouched. Run it only after the final PDF, source-availability inventory, quality receipt and publication manifest have been prepared and checked.

From the repository root, using Python 3.11 or later:

```sh
mkdir -p dist
python3 tools/build_public_review_package.py \
  --output dist/TinkerRL_Public_Review_2026-10-04.zip
python3 tools/build_public_review_package.py \
  --verify dist/TinkerRL_Public_Review_2026-10-04.zip
```

An existing output path is an error. Choose a new filename when building a later candidate. Both commands print the archive SHA-256 and their result as JSON. The verifier reads the archive into bounded temporary storage and removes that storage after checking; it does not install or execute the archived sources.

The publication manifest selects the contents, and the checker requires an exact inventory of `reports/public_revision_2026-10-04/`. Required entries include the final PDF, editable Markdown and LaTeX, builders, C1 ledgers, P11 bindings, the number audit and source-availability metadata. The package adds `PUBLIC_REVIEW_PACKAGE_MANIFEST.json`, which hashes every included file and binds the publication manifest. Fixed ZIP timestamps, file order and stored compression make identical inputs produce identical archive bytes.

Only the October 4 public revision tree can be included. Private exports, operational directories and historical identified outputs are outside the allowed tree. Unlisted files inside the revision prevent a successful build. Public research files elsewhere in the repository remain external references unless reviewed copies are deliberately added to the revision and its manifests. A file-hash check cannot decide whether text is suitable for public release; publication review must precede packaging.

The build checks exact source inventory, receipt-to-PDF binding, recorded review completion and all manifest hashes. It does not perform a new visual review, authenticate reviewers, certify privacy, replay experiments, supply withheld inputs or approve scientific conclusions. PDF compilation requires the separately documented build tools and resources. The ZIP has no claim to institutional certification or signatures, and neither command uploads or submits anything.
