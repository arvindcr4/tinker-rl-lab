#!/usr/bin/env bash
# Source-only report build; never invokes training/evaluation or the network.
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v tectonic >/dev/null; then
  export THESIS_TEX_ENGINE=latexmk
  dist=/usr/share/texlive/texmf-dist
  extra=/usr/share/texmf
  if ! kpsewhich article.cls >/dev/null || ! kpsewhich pdftex.map >/dev/null; then
    cache="$PWD/build/texmf"
    mkdir -p "$cache"/{formats,maps,var,config,fonts}
    export TEXMFCNF="$dist/web2c" TEXMF="{$dist,$extra}"
    export TEXMFVAR="$cache/var" TEXMFCONFIG="$cache/config" VARTEXFONTS="$cache/fonts"
    export TEXINPUTS=".:$dist/tex//:$extra/tex//:"
    export TFMFONTS="$dist/fonts/tfm//:$extra/fonts/tfm//:"
    export T1FONTS="$dist/fonts/type1//:$extra/fonts/type1//:"
    export ENCFONTS="$dist/fonts/enc//:$extra/fonts/enc//:"
    export BSTINPUTS=".:$dist/bibtex/bst//:"
    export TEXFORMATS="$cache/formats:" TEXFONTMAPS="$cache/maps:"
    cat "$dist/fonts/map/dvips/amsfonts/cm.map" \
        "$dist/fonts/map/dvips/amsfonts/latxfont.map" \
        "$dist/fonts/map/dvips/amsfonts/symbols.map" \
        "$dist/fonts/map/dvips/rsfs/rsfs.map" \
        "$dist/fonts/map/dvips/tetex/ps2pk35.map" \
        "$extra/fonts/map/dvips/lm/lm.map" > "$cache/maps/pdftex.map"
    if [[ ! -s "$cache/formats/pdflatex.fmt" ]]; then
      (cd "$cache/formats"; pdftex -ini -etex -jobname=pdflatex -progname=pdflatex \
        -interaction=nonstopmode -halt-on-error pdflatex.ini > format-build.log)
    fi
  fi
  if ! kpsewhich IEEEtran.bst >/dev/null; then
    export THESIS_BIB_STYLE=ieeetr
  fi
fi
python3 compile_figures.py -j 4
python3 build_thesis.py
