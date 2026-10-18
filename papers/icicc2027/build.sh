#!/usr/bin/env bash
# Build only the manuscript; no network, experiments, or external state changes.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
# Standard TeX Live/MiKTeX installations use latexmk without a local repair.
# This optional fallback supports read-only Debian images with missing ls-R maps.
if ! kpsewhich article.cls >/dev/null || ! kpsewhich lmodern.sty >/dev/null || ! kpsewhich pdftex.map >/dev/null; then
  dist=/usr/share/texlive/texmf-dist
  extra=/usr/share/texmf
  test -f "$dist/tex/latex/base/article.cls"
  test -f "$extra/tex/latex/lm/lmodern.sty"
  cache="$PWD/build/texmf"
  mkdir -p "$cache"/{formats,maps,var,config,fonts}
  export TEXMFCNF="$dist/web2c"
  export TEXMF="{$dist,$extra}"
  export TEXMFVAR="$cache/var" TEXMFCONFIG="$cache/config" VARTEXFONTS="$cache/fonts"
  export TEXINPUTS=".:$dist/tex//:$extra/tex//:"
  export TFMFONTS="$dist/fonts/tfm//:$extra/fonts/tfm//:"
  export T1FONTS="$dist/fonts/type1//:$extra/fonts/type1//:"
  export ENCFONTS="$dist/fonts/enc//:$extra/fonts/enc//:"
  export BSTINPUTS=".:$dist/bibtex/bst//:"
  export TEXFORMATS="$cache/formats:"
  export TEXFONTMAPS="$cache/maps:"
  cat "$dist/fonts/map/dvips/amsfonts/cm.map" \
      "$dist/fonts/map/dvips/amsfonts/latxfont.map" \
      "$dist/fonts/map/dvips/amsfonts/symbols.map" \
      "$extra/fonts/map/dvips/lm/lm.map" > "$cache/maps/pdftex.map"
  if [[ ! -s "$cache/formats/pdflatex.fmt" ]]; then
    if ! ( cd "$cache/formats"; pdftex -ini -etex -jobname=pdflatex -progname=pdflatex \
        -interaction=nonstopmode -halt-on-error pdflatex.ini > format-build.log 2>&1 ); then
      echo "pdfLaTeX format build failed; see $cache/formats/format-build.log" >&2
      tail -n 40 "$cache/formats/format-build.log" >&2
      exit 1
    fi
  fi
fi
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build paper_camera.tex
