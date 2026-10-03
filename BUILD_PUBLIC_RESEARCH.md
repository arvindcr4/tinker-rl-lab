# Safely rebuild the frozen PUBLIC documents

Run the new wrapper instead of running the frozen document builders in place:

```sh
python3 -B tools/build_public_documents.py \
  --document all \
  --output-dir /absolute/existing/parent/new-public-rebuild
```

Replace the example with a **new directory outside this checkout**. Its parent
must already exist. Choose `--document thesis` or `--document addendum` to build
only one. `--root /path/to/checkout` selects another frozen input checkout.
The per-document timeout defaults to 1,800 seconds; `--timeout` accepts a positive
number up to 3,600 seconds. This is a local rebuild, not an upload or publication
to a hosting service.

## Requirements

- Python 3.11+, Linux or macOS, and approved `pandoc` and `tectonic` executables on
  `PATH`
- A populated Tectonic resource cache compatible with the existing document
  sources; normal `XDG_CACHE_HOME` and `XDG_CONFIG_HOME` settings are honored
- An output filesystem supporting ordinary hard links within a directory

The wrapper adds `--only-cached --untrusted` to the frozen builders' Tectonic
calls. It does not install tools. This flag requests cached TeX resources but is
**not an OS network sandbox**: Tectonic can still attempt bundle-index metadata
retrieval when a bundle has not been cached. Use an already-populated compatible
cache and external network isolation when a strict no-network guarantee is
required. An incomplete/incompatible cache fails with the native diagnostic;
prepare it separately before retrying. Existing figure PDFs are verified inputs;
the wrapper does not recompile figures.

The native-tool PATH contains only private, quoted launchers for the two resolved
executables, so a failed launcher cannot fall through to an unrestricted binary.
Each launcher restores the normalized original PATH only after it starts and
immediately before invoking its selected absolute native executable. This lets
approved native scripts find their interpreters/helpers without exposing a
compiler fallback to the document builder.

## Safety boundary

1. Verify the complete frozen publication inventory, byte counts and SHA-256
   digests against `reports/public_revision_2026-10-03/PUBLICATION_MANIFEST.json`.
   Abort before invoking any native tool if an input differs or is missing.
2. Copy only manifest-listed inputs from the selected PUBLIC document directory
   into a fresh temporary directory outside the checkout. Preserve figure PDFs;
   exclude old final PDFs, scratch build trees, generated master/addendum TeX,
   hidden/cache paths and unrelated file types. Recheck each copied input digest.
3. Execute the unchanged builder in that stage, with a process-group timeout.
   Require successful exit and a newly produced regular PDF with a plausible
   header and EOF trailer. Reverify the frozen source inventory before publishing.
4. Reserve the chosen output directory exclusively. Publish complete PDFs and
   build logs through no-replace hard links, then write `BUILD_RECEIPT.json` last
   as the completion marker. Any existing file, directory or dangling symlink at
   the destination causes failure; the wrapper never replaces or removes it.

The original manifest, document sources, reviewed PDFs and existing build
products are never rewritten. Temporary staging is cleaned up. A failure during
publication can leave a **new, incomplete output directory**; it is retained for
inspection and must not be mistaken for a completed build. Only a directory
with the completion receipt is successful. Retry into a different new directory.
No user-owned directory is cleaned up automatically.

The receipt records the input manifest digest, selected input paths, requested
Tectonic flags, lack of network isolation, and output PDF hashes and sizes. Logs
may contain temporary staging paths. The wrapper assumes trusted local tools and
a stable local filesystem; it is not a security sandbox against malicious
builders or concurrent hostile filesystem changes.

## What a successful rebuild establishes

A success means the requested native builders ran from verified PUBLIC input
bytes in a clean stage and produced plausible PDFs. It does **not** imply
byte-identical output, matching page counts, a parser/visual-layout review,
scientific validation, privacy certification, source authentication or replay of
withheld executions. Tool versions and TeX caches can change the output.

Review the new PDFs independently before sharing them or proposing a separately
reviewed revision. Do not replace the reviewed snapshot or regenerate its
manifest to make altered files pass validation.

## Regression tests

```sh
python3 -B -m unittest discover -s tests -p 'test_public_document_build.py' -v
```

These tests use fabricated native tools in temporary fixtures. They cover stale,
missing, empty, invalid, truncated and failed outputs; process timeout; missing
tools; frozen-source preservation; corrupt/unlisted inputs; stage-copy changes;
figure retention; spaced/relative executable paths; failed shim execution;
symlinks; existing destinations; and multi-document/partial publication.
Passing these tests is **not** evidence that a native TeX rebuild has run.

## Observed native validation (3 October 2026)

With Pandoc 3.1.11.1, Tectonic 0.17.0 and the existing populated toolchain cache:

- The addendum rebuilt successfully to a fresh external directory: 9 pages,
  47,845 bytes. Its complete `pdftotext -layout` output matched the frozen PDF
  exactly (29,325 characters). The PDF binary digest differs, so this is not a
  byte-identical reproduction claim. Rendered pages 1, 3, 7 and 9 showed readable
  tables, math and margins.
  The cached `xstring.tex` emitted UTF-8 warnings; no corresponding text change
  was found. This was a sampled visual check, not a new full document review.
- The thesis attempt failed before publication because that cache supplied
  `pgfplots` compatibility 1.17 while the frozen preamble requires 1.18. No thesis
  PDF was published, and no source/cache override was added to force a pass.
- A default-bundle attempt was stopped after uncached bundle-index retrieval
  returned HTTP 403 despite `--only-cached`. This is why the network-isolation
  limitation above is explicit. No new tools or resources were installed.

The 108-file frozen snapshot still passed its original digest/inventory checks
after these attempts. The reviewed PDFs remain the canonical release artifacts.
