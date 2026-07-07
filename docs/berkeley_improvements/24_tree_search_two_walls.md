# 24 — Tree Search for LM Agents → the two walls of ZVF (coverage vs saturation)

**Source lecture:** SP25 L6 — Ruslan Salakhutdinov, *Tree Search for Language
Model Agents* (Koh, McAleer, Fried, Salakhutdinov 2024, **arXiv:2407.01476**).
Citation verified via arXiv abstract (title / all four authors / 2024 / cs.AI;
reported gains +39.7% VisualWebArena, +28.0% WebArena over baseline).

**Target:** A5 (inference-time reasoning) × Pillar 2 (ZVF / contrastive signal).

**Status:** validated (5/5 hypotheses DECISIVE; one sub-conjecture falsified and
recorded honestly).

---

## The idea

Koh et al. replace **parallel best-of-N sampling** with a **best-first,
value-guided sequential tree search**, and beat best-of-N at matched compute —
most on *hard* tasks. Their entire gain comes from breaking one specific
barrier: the **coverage wall** — finding *at least one* success on a prompt
where i.i.d. sampling keeps failing.

GRPO's zero-variance fraction (ZVF, our Pillar-2 diagnostic) is the fraction of
groups that produce **no gradient** because every advantage is zero. It splits
**exactly** into two walls that behave oppositely:

```
ZVF(g) = P[all wrong]  +  P[all correct]
         \___________/    \____________/
         COVERAGE wall     SATURATION wall
         (1-p)^g           p^g
```

- **Coverage wall** = all-wrong groups. Tree search (Koh et al.) *can* rescue
  these — condition later expansions on earlier failures until one path
  succeeds.
- **Saturation wall** = all-correct groups. **No search over the same policy can
  fix these**: you cannot make an all-correct group contrastive. Tree search
  finds *a* solution; GRPO needs a *contrastive pair*. This is the clean,
  paper-facing distinction between inference-time search and an RL training
  signal.

Both walls are computed **exactly** from real per-prompt data (600 Qwen3-8B
GSM8K groups, native G=8, 3 seeds × 200 prompts): all-wrong = C(8−k,g)/C(8,g),
all-correct = C(k,g)/C(8,g); g≤8 exact hypergeometric, g>8 i.i.d. extrapolation
(row-18 convention).

## Results (`experiments/results/berkeley/tree_search_*`)

| g | all-wrong (coverage) | all-correct (saturation) | ZVF | coverage share | saturation share |
|---|---|---|---|---|---|
| 2 | 0.130 | 0.518 | 0.649 | 0.201 | **0.799** |
| 4 | 0.054 | 0.310 | 0.364 | 0.148 | **0.852** |
| 8 | 0.032 | 0.127 | 0.158 | 0.200 | **0.800** |
| 32 | 0.032 | 0.130 | 0.162 | 0.197 | **0.803** |

- **H1 — two walls decompose (DECISIVE).** ZVF = all-wrong + all-correct
  exactly, by construction. Crossover g\* = 2: even at the smallest group the
  saturation wall already exceeds the coverage wall.
- **H2 — saturation dominates at *every* g (DECISIVE; sub-conjecture
  FALSIFIED).** The saturation share is 0.80–0.85 across all g∈{2…32}; the
  coverage share (the part tree search could rescue) never exceeds **0.20**. My
  original conjecture that the coverage share decreases monotonically is
  **false** — it is small and U-shaped. The corrected, sharper claim: on this
  policy/task **≥80% of GRPO's dead signal is saturation, which tree search
  provably cannot touch.**
- **H3 — hardness targeting (DECISIVE).** The coverage wall lives almost
  entirely on hard prompts: the bottom-p tertile (p≤1/3, 48 prompts) holds
  **89.8%** of all-wrong mass and **0%** of all-correct mass. The easy tertile
  (p>2/3, 362 prompts) holds **96.0%** of all-correct mass and **0%** of
  all-wrong mass. Near-perfect orthogonality — and exactly Koh et al.'s "search
  helps hard tasks."
- **H4 — sequential compute advantage (DECISIVE).** An oracle best-first search
  covers a solvable prompt in E[1/p] = **1.66** expansions; parallel sampling
  needs g\*=**3** draws to reach 0.90 mean coverage → parallel costs **1.81×**
  the oracle-sequential compute for the same coverage. **3.2%** of prompts have
  p=0 and are uncoverable by *any* method over this policy.
- **H5 — saturation wall is search-invariant (DECISIVE, falsification guard).**
  At g=32 the saturation share is **0.80** and still dominant; tree search
  cannot rescue GRPO's large-g signal collapse because that collapse is
  saturation, not coverage.

## Recommendation — **GO** (as a scope-limiting result, not a new method)

The paper-facing takeaway is a **negative/orthogonality result** that sharpens
our ZVF story: inference-time tree search (Koh et al.) and RL training-signal
collapse are *largely orthogonal problems on this task*. Tree search addresses
at most ~20% of dead GRPO signal (the hard, all-wrong tail); the other ~80% is
all-correct saturation that only a **harder curriculum / higher-p-diversity
prompt mix** — not search — can fix. Concretely for TinkerRL-Bench:

1. Report ZVF **split into its two walls** in the Pillar-2 diagnostic; a single
   ZVF number conflates a search-fixable and a search-unfixable failure.
2. Use the coverage/saturation split as a **routing signal**: hard (coverage)
   prompts → benefit from search/more draws; easy (saturation) prompts → drop or
   replace (they burn compute for zero gradient regardless of G).

This composes with the iso-yield grouping (rows 13/iter46) and the Lean-STaR
yield-by-difficulty result (row 21): all three say the same thing from
different lectures — **contrastive yield, not raw coverage, is what GRPO needs.**

**Reproduce:** `python3 scripts/berkeley/tree_search_two_walls.py`
Outputs: `tree_search_two_walls.tsv`, `tree_search_hardness_targeting.tsv`,
`tree_search_compute_ratio.tsv`, `tree_search_summary.json`.
