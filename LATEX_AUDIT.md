# Legacy LaTeX audit

This gate checks the three legacy paper sources under
`platform_tinker/reports/final/`: the named paper, anonymous paper, and
supplementary appendix. It does not rebuild the public thesis/C1 revision or
change institutional pages or signatures.

Run from the repository root with the development dependencies installed:

```bash
python -m pytest tests/test_audit_runner.py -m latex -v --tb=short
```

The audit builds into a temporary output directory, checks for a nonempty PDF,
and rejects references/citations still unresolved in the final TeX log. The
pdfLaTeX path runs LaTeX, BibTeX, LaTeX, LaTeX separately for each document.
Tectonic runs its own required passes and retains its final log only in the
scratch directory. Existing source-side PDFs and auxiliary files are not
replaced. Each compiler invocation has a 120-second timeout; bounded compiler
output is retained in structured audit findings even after scratch cleanup.

## Select the toolchain

`TINKERRL_LATEX_ENGINE` accepts:

- `auto` (default): use `pdflatex` when both `pdflatex` and `bibtex` are on
  `PATH`; otherwise use `tectonic`
- `pdflatex`: require the installed pdfLaTeX/BibTeX toolchain
- `tectonic`: require Tectonic, even when pdfLaTeX is also installed

```bash
TINKERRL_LATEX_ENGINE=pdflatex python -m pytest tests/test_audit_runner.py -m latex -v
TINKERRL_LATEX_ENGINE=tectonic python -m pytest tests/test_audit_runner.py -m latex -v
```

`auto` checks executable availability, not whether a TeX distribution has been
fully configured. A present binary with missing format files, fonts, packages,
or configuration is an environment failure. The audit deliberately does not
silently switch engines after a compilation failure, since that could hide a
real source defect. Invalid engine settings and missing/unlaunchable tools
produce explicit findings rather than being reported as successful builds.

## Offline builds and writable caches

Tectonic always runs with `--only-cached`. Its bundle and every needed resource
must already be available in its configured cache. A cache warmed for a
different document is not necessarily complete for these papers. Cache warming
or installing missing packages is a separate setup step, outside the audit.
No network access is needed by a correctly prepared audit build.

Preserve your working TeX configuration when invoking the audit. On a machine
with a read-only home directory, point `XDG_CACHE_HOME` and `XDG_CONFIG_HOME` at
appropriate writable Tectonic directories containing the intended configuration
and warmed cache. For TeX Live, `TEXMFVAR`, `TEXMFCONFIG`, and `TEXFORMATS` can
select writable generated-file locations and prepared formats. Do not point a
compiler at an empty directory and assume its resources have been installed.
The audit inherits these environment variables and does not repair system-wide
TeX configuration.

## Reproduce the offline TeX Live validation

On 3 October 2026, the validation image had TeX Live source packages and binaries
under `/usr/share/texlive/texmf-dist` and `/usr/bin`, but lacked its generated
format/configuration/database files. The default pdfLaTeX invocation failed with
`I can't find the format file 'pdflatex.fmt'`. The available Tectonic cache also
lacked `times.sty` for the legacy papers.

The following uses only those already-installed TeX Live resources. It writes
all setup files under a new temporary directory, provides the installed English
hyphenation configuration, generates a local format, and assembles a local font
map. It makes no system installation or repository-source changes. These paths
are specific to that Linux TeX Live layout; they are not a replacement for a
normal TeX installation on other systems.

```bash
set -euo pipefail
TEX_WORK=$(mktemp -d /tmp/tinker-texlive-local-XXXXXX)
mkdir -p "$TEX_WORK/texmf/tex/generic/config" \
         "$TEX_WORK/texmf/fonts/map/pdftex" "$TEX_WORK/formats"
cp /usr/share/texlive/texmf-dist/tex/generic/config/language.us \
   "$TEX_WORK/texmf/tex/generic/config/language.dat"
cat /usr/share/texlive/texmf-dist/fonts/map/dvips/tetex/pdftex35.map \
    /usr/share/texlive/texmf-dist/fonts/map/dvips/amsfonts/*.map \
    > "$TEX_WORK/texmf/fonts/map/pdftex/pdftex.map"
export TEXMF="{$TEX_WORK/texmf,/usr/share/texlive/texmf-dist,/usr/share/texmf}"
export TEXMFVAR="$TEX_WORK/var" TEXMFCONFIG="$TEX_WORK/config"
export TEXFORMATS="$TEX_WORK/formats//:"
export TINKERRL_LATEX_ENGINE=pdflatex
pdflatex -ini -etex -interaction=nonstopmode -halt-on-error \
  -jobname=pdflatex -output-directory="$TEX_WORK/formats" pdflatex.ini
python -m pytest tests/test_audit_runner.py -v --tb=short
```

Use the Python interpreter from your development environment. The validation
image used `/workspace/shared/research-tools/venv/bin/python`. Keep the exported
environment in the same shell for the test command; subsequent shells must
re-export it or prepare their own scratch configuration.
