# ZVF Program -- Pillar 2 (Theory)

The formal proof paper `zvf_theory.tex` (and its compiled `zvf_theory.pdf`).

**Status: CONDITIONAL THEORY NOTE.** T1--T3 now have complete proofs under
their explicit assumptions. `THEORY_NOTES.md` records the proof audit and the
remaining research extensions. The results must still be cited with their
scope: i.i.d. prompt groups, binary deterministic rewards, non-degenerate
difficulty support, and the declared T3 proxy objective.

## Compiling

```
cd /Users/arvind/Developer/tinker-rl-lab/zvf-program/theory
pdflatex zvf_theory.tex
pdflatex zvf_theory.tex   # second pass for cross-references
```

The current PDF and page count are recorded by the corpus inventory. Numerical
claims in the empirical-validation section point to checked-in artifacts; the
theorems themselves are algebraic.

## What's here

| File | Role |
|---|---|
| `zvf_theory.tex` | The paper itself. Compiles with `pdflatex x2`. |
| `zvf_theory.pdf` | Compiled artifact (8 pages). |
| `zvf_theory.log` | pdflatex log; not committed clean, kept for the last compile's diagnostics. |
| `zvf_theory.aux` | pdflatex aux file; regenerated each compile. |
| `zvf_theory.out` | pdflatex hyperref output. |
| `THEORY_NOTES.md` | Proof-audit ledger, assumptions, and research extensions. |

## What ties this to the rest of the program

- Pillar 1 (`sweep/`) is the empirical side: it scales the ZVF/GU audit
  to 300-500 runs so the asymptotic claims in this paper can be
  checked against real measurements. The 0.008 worked example is
  the same number Pillar 1 audits under matched configs.
- Pillar 3 (`zvf-triage/`) is the operationalization: the formal
  ZVF definition used here (`\mathrm{ZVF}_t = ...` in
  `zvf_theory.tex` §Formal Definition) is exactly the diagnostic the
  callback computes at every step. The eps=1e-6 / ddof=1 / K=1
  convention matches addendum 01 of the parent paper.
- Pillar 4 (`position/`) is the policy side: it argues that RL
  post-training reporting needs to include the "stack" levers
  (sampler, backend, precision, etc.) that this paper's theorems
  treat as fixed.

## Cross-pillar consistency

The "0.008 worked example" is the same number in `THEORY_NOTES.md`
and in the Pillar-1 audit's diagnostic output. The `eps=1e-6` ZVF
threshold, the `ddof=1` unbiased variance, and the `K=8` default
group size are the same constants in this paper, the parent paper
addenda 01 / 05, the Pillar-1 audit, and the Pillar-3 callback.
If you find a place where they disagree, that's a bug.
