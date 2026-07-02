# paper/tikz/ — INDEX

**Purpose:** Hand-drawn TikZ diagrams as standalone `.tex` sources compiled to `.pdf`, embedded by `../main.tex` via `\includegraphics{tikz/*.pdf}`. Regenerate a PDF by pdflatex-compiling its `.tex`.

**Key files:**
- `taxonomy.tex` → `taxonomy.pdf` — RL post-training method taxonomy (main.tex ~line 118)
- `reward_flow.tex` → `reward_flow.pdf` — reward computation/flow diagram (~line 217)
- `pipeline.tex` → `pipeline.pdf` — end-to-end training/eval pipeline (~line 297)
- `architecture.tex` → `architecture.pdf` — system architecture (not currently embedded; see FIGURE_AUDIT)

**Find it fast:**
- to change a diagram → edit its `.tex`, recompile with pdflatex, PDF is what main.tex loads
- to see where each is used → `../FIGURES.tex` registry
