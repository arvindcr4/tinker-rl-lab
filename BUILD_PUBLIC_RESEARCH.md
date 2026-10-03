# Safely rebuild the frozen PUBLIC documents

Run the new wrapper instead of running the frozen document builders in place:

```sh
python3 -B tools/build_public_documents.py \
  --document all \
  --output-dir /absolute/existing/parent/new-public-rebuild
```

Replace the example with a **new directory outside this checkout**. Its parent
must already exist. Supply a canonical output path without `..` components,
so symlink aliases cannot change the intended destination. Choose `--document thesis` or `--document addendum` to build
only one. `--root /path/to/checkout` selects another frozen input checkout.
The per-document timeout defaults to 1,800 seconds; `--timeout` accepts a positive
number up to 3,600 seconds. This is a local rebuild, not an upload or publication
to a hosting service.

## Requirements

- Python 3.11+, Linux or macOS, and approved `pandoc` and `tectonic` executables on
  `PATH`
- A populated Tectonic resource cache compatible with the existing document
  sources, or an explicitly prepared local ZIP bundle (recipe below); normal
  `XDG_CACHE_HOME` and `XDG_CONFIG_HOME` settings are honored
- An output filesystem supporting ordinary hard links within a directory

The wrapper adds `--only-cached --untrusted` to the frozen builders' Tectonic
calls. Optional `--tectonic-bundle /absolute/path/public-tex-bundle.zip` selects
a validated local ZIP copied into the isolated stage; URLs (including `file://`)
are rejected and there is no remote fallback. Omitting the option leaves the
existing cache-selection behavior unchanged. It does not install tools. This flag requests cached TeX resources but is
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
Tectonic flags, lack of network isolation, and output PDF hashes and sizes. With
`--tectonic-bundle`, it additionally records the supplied ZIP's SHA-256 and byte
count, canonical resource-inventory identity, resource count and staged filename.
The wrapper verifies ZIP inventory, resource digests and CRCs, then rechecks both
the supplied and staged ZIP before publishing; a missing or changed bundle fails
closed. The bundle is a toolchain input, never an addition to or replacement for
the frozen scientific manifest, and is not copied into the PDF output directory. Logs
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

## Prepare a compatible local bundle without downloads

`tools/prepare_public_tex_bundle.py` uses only Python's standard library and
explicit existing resource paths. It does not search for resources, download,
install, edit the system/cache, or alter document sources. It reads a **flat**
Tectonic bundle-data cache, then overlays selected `.tex`, `.sty`, `.def` and
`.cfg` files recursively from each named resource root. Basenames must be unique
across all overlay roots, even if duplicate bytes are identical. Cache-to-overlay
replacements are intentional and individually recorded; every other ambiguity,
unsafe selected basename, symlink (including directory/root aliases), nonregular
file, missing root, or absent license-evidence label causes failure. Roots must
contain at least one selected runtime file. Use canonical real input paths,
including on macOS where `/tmp` or `/var` may be aliases; resolve those paths
before supplying them. Symlink aliases and `..` path components are deliberately
rejected.

Choose existing local paths and **new external output directories** below;
output paths must also omit `..` components.
The compatible native test used a cached TeX Live 2021 resource set, installed
PGF 3.1.10 and PGFPlots 1.18.1. These paths are examples to fill in, not resources
provided by this repository. Ensure `PATH` selects approved native Pandoc and
Tectonic executables, without an old launcher adding a conflicting bundle URL.

```sh
CACHE="/absolute/existing/tectonic/bundles/data/CACHED_BUNDLE_ID"
TEX="/absolute/existing/texmf-dist/tex"
WORK="/absolute/existing/parent"

python3 -B tools/prepare_public_tex_bundle.py \
  --cache-dir "$CACHE" \
  --cache-origin "Previously populated TeX Live 2021 cache; record its original identity here" \
  --resource-root "pgf=$TEX/generic/pgf" \
  --resource-root "pgf=$TEX/latex/pgf" \
  --resource-root "pgf=$TEX/plain/pgf" \
  --resource-root "pgfplots=$TEX/generic/pgfplots" \
  --resource-root "pgfplots=$TEX/latex/pgfplots" \
  --resource-root "pgfplots=$TEX/plain/pgfplots" \
  --license-evidence "cache=Mixed package licenses; review TeX Live LICENSE.TL and original package notices" \
  --license-evidence "pgf=Installed PGF headers offer LPPL and/or GPL; review and preserve applicable notices" \
  --license-evidence "pgfplots=Installed PGFPlots headers identify GPL-3.0-or-later; review applicable notices" \
  --output-dir "$WORK/new-local-tex-bundle"

XDG_CACHE_HOME="$WORK/new-private-runtime-cache" \
XDG_CONFIG_HOME="$WORK/new-private-runtime-config" \
python3 -B tools/build_public_documents.py \
  --document all \
  --tectonic-bundle "$WORK/new-local-tex-bundle/public-tex-bundle.zip" \
  --output-dir "$WORK/new-public-rebuild"
```

The preparation output is `public-tex-bundle.zip` plus `PROVENANCE.json`, written
last as the completion marker. The JSON records per-file absolute source path,
size, SHA-256, source label and caller-supplied license evidence; previous and
replacement records for every overlay; original cache provenance; and the final
ZIP and resource-inventory digests. Absolute paths stay in this local output;
review them before any separately authorized sharing. `--cache-origin` is inert
provenance text, including when it contains a URL. It is never fetched.

The ZIP uses sorted flat names, fixed 1980 timestamps, fixed permissions and
uncompressed stored entries, so the same selected resource bytes produce the
same ZIP bytes without depending on a zlib version. Provenance is also stable
for the same explicit paths, labels and notes. Bounds are 10,000 resource files,
64 MiB per file and 512 MiB total, with room reserved for inventory metadata.
Preparation rejects existing destinations and outputs within the checkout or
resource roots. It leaves any new, incomplete output for inspection after a
publication failure, never replacing existing files. It assumes trusted local
resources and a stable filesystem, not hostile concurrent changes. It rechecks
selected/replaced file contents before publishing; it does not re-enumerate roots
to detect concurrently added resources.

These mixed-license toolchain resources are **not automatically committed,
shared or published**. The supplied license notes are evidence to review, not a
license audit, ownership claim or permission to redistribute. Preserve all
applicable package notices and satisfy their terms before any separate sharing.
The scientific publication manifest continues to cover exactly the original
frozen public artifacts, independently of this toolchain provenance.

This is a document-specific resource subset, not a complete TeX distribution.
The tested historical cache lacks the default Computer Modern `cmr10.pfb` font:
a generic article probe failed, while the thesis's `newtxtext,newtxmath` font
setup and both complete PUBLIC documents succeeded. The recipe cannot repair an
arbitrary incomplete cache. It preserves `--only-cached --untrusted`; it neither
uses ignored `-Z search-path` overrides nor establishes an OS network sandbox.

## Regression tests

```sh
python3 -B -m unittest discover -s tests -p 'test_public_document_build.py' -v
python3 -B -m unittest discover -s tests -p 'test_public_tex_bundle.py' -v
```

These tests use fabricated native tools in temporary fixtures. They cover stale,
missing, empty, invalid, truncated and failed outputs; process timeout; missing
tools; frozen-source preservation; corrupt/unlisted inputs; stage-copy changes;
figure retention; spaced/relative executable paths; failed shim execution;
symlinks; existing destinations; and multi-document/partial publication. Bundle
tests additionally cover deterministic bytes, explicit overlays and provenance,
duplicate/unsafe ZIP members, CRC/content integrity, changed or missing bundles,
size bounds, local-only paths, staged bundle selection, quoted paths and preserved
safety flags. They require no actual TeX installation.
Passing these tests is **not** evidence that a native TeX rebuild has run.

## Observed native validation (3 October 2026)

The comparison below is bound to the original 108-file publication snapshot at
Git commit `17312e2e7c407635a6c99bb8a565f57336a66758`, whose unchanged manifest has
SHA-256 `aae48f58ce7056b9e5870f2d258e12431651a4201090e42fd66399041e2977b9`.
It does not validate later manuscript/PDF changes made by concurrent sessions.
The integrity gate intentionally rejects a checkout that differs from that
manifest; a later publication requires a separately reviewed versioned manifest,
not silently refreshed historical hashes.

Pandoc 3.1.11.1 and Tectonic 0.17.0 initially rebuilt the addendum but failed the
thesis with the existing cache: it supplied PGFPlots compatibility 1.17 while the
frozen preamble requires 1.18. Tectonic's `-Z search-path` override is ignored in
`--untrusted` mode. An uncached default-bundle metadata request also returned
HTTP 403 despite `--only-cached`; that route was stopped, not retried to force a
pass. No tools or resources were installed or downloaded.

An explicit local compatible bundle resolved the thesis failure while preserving
both safety flags. It combined 578 already-cached TeX Live 2021 resources with
353 existing PGF/PGFPlots runtime files, recording 105 replacements and a final
826-resource inventory. The successful native rebuild produced:

- Thesis: 295 pages, 1,516,500 bytes; complete `pdftotext -layout` output matched
  the frozen PDF exactly (801,095 characters)
- Addendum: 9 pages, 47,845 bytes; complete extracted text matched exactly
  (29,325 characters)
- All 304 pages rendered pixel-identically at 72 dpi with PyMuPDF 1.26.6
- Representative dense tables, equations, code and appendix pages were also
  visually inspected; no new clipping or missing-glyph defect was found

The resource-inventory identity was
`92ba8092f23efa05c15b7e835351829a050eba39225c7555e59e7e5bde861aab`.
The initial feasibility ZIP used deflate and had SHA-256
`a23b927f88a168e971e6bbe1d46b97335096fbd1d070b6f961c71cedac9e2c2a`;
the helper deliberately uses stored entries for compression-independent
reproducibility. Its 25,959,204-byte ZIP had SHA-256
`b57c09efc29509303b4773956cc5f251d1008b870ff5e68265f6155a286a5762`
and the same resource-inventory identity. A second complete native rebuild using
this helper and the integrated `--tectonic-bundle` option reproduced the page
counts, byte counts, complete-text equality and all-page 72-dpi pixel equality
above. ZIP digest equality is distinct from resource-inventory equality.

PDF binary hashes differ, including creation-time metadata: this is **not** a
byte-identical PDF reproduction claim. Existing `xstring.tex` UTF-8 warnings and
small overfull-box warnings remain. Complete-text and 72-dpi pixel equality found
no regression against the reviewed snapshot under this toolchain; they are not
a new scientific/privacy review, a general-purpose TeX compatibility guarantee,
or proof that different toolchains or higher-resolution renders are identical.
No separate OS-level network isolation was established.

The 108-file frozen snapshot still passed its original digest/inventory checks
after these attempts. The reviewed PDFs remain the canonical release artifacts.
