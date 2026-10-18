#!/usr/bin/env bash
# Source-only report build; never invokes training/evaluation or the network.
set -euo pipefail
cd "$(dirname "$0")"
engine="${THESIS_TEX_ENGINE:-}"
if [[ -z "$engine" ]]; then
  if command -v tectonic >/dev/null; then engine=tectonic; else engine=latexmk; fi
fi
case "$engine" in
  tectonic) required=(python3 pandoc tectonic) ;;
  latexmk) required=(python3 pandoc latexmk pdflatex bibtex kpsewhich) ;;
  *) echo "THESIS_TEX_ENGINE must be tectonic or latexmk" >&2; exit 1 ;;
esac
for tool in "${required[@]}"; do
  command -v "$tool" >/dev/null || { echo "missing required tool: $tool" >&2; exit 1; }
done
export THESIS_TEX_ENGINE="$engine"
if [[ "$engine" == latexmk ]]; then
  system_map="$(kpsewhich pdftex.map 2>/dev/null || true)"
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
    if [[ -n "$system_map" && -s "$system_map" ]]; then
      if [[ "$system_map" != "$cache/maps/pdftex.map" ]]; then
        cp "$system_map" "$cache/maps/pdftex.map"
      fi
    else
      # ps2pk35 includes the Times/mathptmx mappings. Do not select newtx
      # against this emergency map; its additional mappings are not assumed.
      export THESIS_PORTABLE_FONTS=1
      cat "$dist/fonts/map/dvips/amsfonts/cm.map" \
        "$dist/fonts/map/dvips/amsfonts/latxfont.map" \
        "$dist/fonts/map/dvips/amsfonts/symbols.map" \
        "$dist/fonts/map/dvips/rsfs/rsfs.map" \
        "$dist/fonts/map/dvips/tetex/ps2pk35.map" \
        "$extra/fonts/map/dvips/lm/lm.map" > "$cache/maps/pdftex.map"
    fi
    if [[ ! -s "$cache/formats/pdflatex.fmt" ]]; then
      (cd "$cache/formats"; pdftex -ini -etex -jobname=pdflatex -progname=pdflatex \
        -interaction=nonstopmode -halt-on-error pdflatex.ini > format-build.log)
    fi
  fi
fi
python3 compile_figures.py -j 4
python3 build_thesis.py
