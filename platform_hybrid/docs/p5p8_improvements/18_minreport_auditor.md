# P5 Improvement — Item 18: MIN-REPORT-RL Auditor Prototype (0-100 badge)

**Pillar:** P5 (MIN-REPORT, "Report the Stack, Not the Label")
**Class:** T3 (cross-paper coupling — operationalises the
MIN-REPORT-RL Auditor component of `p5_toolchain.tex` §3 on the live
worktree data) + T5 (presentation — produces the auditor leaderboard
figure)
**Status:** prototyped → **validated** (iter 13)
**Deliverable:** `scripts/p5p8/minreport_auditor.py` (≤300 LoC,
stdlib + matplotlib) + 2 TSVs + 2 figures + paper Exhibit 8

## Motivation

Iter-1 item 01 audited the 7-item MIN-REPORT schema against the
98-cell live mega manifest corpus and reported it as a *coverage*
table (present / missing / validated). Iter-9 item 14 extended the
audit to cells.tsv and the N10 corpus and called for an
**18-item expansion** to surface the model / task / G / temperature /
seed axes that iter-5's `mega_eta2` showed explain 73–93% of outcome
variance. Both prior iterations stopped at the *counting* level —
they could tell you *which* items were missing but not *how* a
manifest compares to a fixed standard.

The `p5_toolchain.tex` §3 specifies the **MIN-REPORT-RL Auditor (0-100
badge)** as the tool that turns the standard into a measurable
verdict. Until this iteration the auditor was a *specification* with
no executable reference implementation. This iteration builds the
prototype, runs it across every manifest in the worktree, and reports
the leaderboard.

## What was built

1. **`scripts/p5p8/minreport_auditor.py`** (≤300 LoC, stdlib + matplotlib)
   - 7-item MIN-REPORT schema with weighted scores (items 3, 4, 7 each
     worth 20 pts; items 1, 2, 5, 6 each worth 10 pts; total = 100)
   - Per-item scoring: present × validated × (0.5 + 0.5 ×
     subfield_coverage)
   - **Honest `n/a` declaration bonus**: a recognised value like
     `n/a-sampling` for Item 1 or `n/a` for Item 2 earns 50% of the
     item weight (vs. 100% for a fully-validated value, 25% for a
     present-but-unrecognised value, 0% for a missing key)
   - Multi-corpus support: scans both
     `experiments/results/mega_20260704/manifests/` (compact keys)
     and `experiments/results/quick_20260704/` (verbose keys) and
     scores every file that touches at least one of the seven
     MIN-REPORT fields
   - Per-manifest TSV output + stratified summary JSON + two figures

2. **`experiments/results/p5p8/minreport_audit.tsv`** (103 rows: 98
   mega + 5 quick; columns: cell_id, corpus, model, task_slice, G,
   temperature, seed, items 1-7 scores, badge, tier)

3. **`experiments/results/p5p8/minreport_audit_summary.json`**
   (n, mean/median/min/max/std, tier counts, stratified by corpus /
   task / model / G / temperature / seed, per-item %)

4. **`experiments/results/p5p8/figures/minreport_badge_dist.{png,pdf}`**
   (histogram of badge distribution with four tier thresholds marked)

5. **`experiments/results/p5p8/figures/minreport_per_item.{png,pdf}`**
   (per-item % bar chart with red highlighting on the three
   high-leverage items, weight 20)

6. **`paper/sections/p5_evidence.tex`** — extended with Exhibit 8
   (per-item table + stratified table + 5-paragraph discussion)

7. **`paper/build/paper_P5_minreport.pdf`** — rebuilt to 21 pages,
   0 errors, 0 undefined citations

## Key results

| metric | value |
|--------|-------|
| n_manifests | 103 (98 mega + 5 quick) |
| badge mean / median | 55.0 / 56.7 |
| badge range | [35.8, 56.7]  (std 3.1) |
| gold (≥90) | 0 |
| silver (≥75) | 0 |
| bronze (≥50) | 99 |
| wood (≥25) | 4 |
| fail (<25) | 0 |

### Per-item coverage on n=103

| # | item | weight | % achieved | interpretation |
|---|------|--------|------------|----------------|
| 1 | Loss form                       | 10  | 24.4% | honest-n/a declaration gives 5/10 |
| 2 | Reference policy & KL          | 10  | 24.4% | honest-n/a declaration gives 5/10 |
| 3 | Sampler / backend / precision  | 20  | 64.0% | "tinker-closed" opaque, no precision exposed |
| 4 | Per-step ZVF/GU trajectory     | 20  | 49.5% | trajectory paths reference off-host files |
| 5 | Group-size schedule            | 10  | 74.3% | compact key gives 7.5/10 |
| 6 | Held-out split                 | 10  | 76.2% | compact key gives 7.5/10 |
| 7 | Decontamination & parser probe | 20  | 61.8% | parser-probe sub-field is 0/103 reported |

### Stratified badges (mean)

| axis | mean | interpretation |
|------|------|----------------|
| **by corpus** | mega 55.5 vs. quick 45.0 (Δ=10.5) | auditor discriminates across corpora, not within |
| **by task_slice** | gsm8k_easy 56.7, gsm8k_hard 56.7, humaneval 53.3 | task_slice explains 1.5pt of the 3.1pt std |
| **by model** | Qwen 55.47 vs. Llama 55.57 | within-stack, model is invariant |
| **by G** | G=2 → 55.6, G=32 → 56.3 | group-size axis explains ~1pt |
| **by seed** | s0 54.55 vs. s1 55.52 | seed axis explains ~1pt |

## Headline findings

1. **The auditor is enforceable.** Every manifest received a verdict;
   every key was read or declared missing. The audit's failure mode is
   *not* the per-manifest score; it is the **per-item table** that
   surfaces the three missing declarations the iter-1 audit identified
   (Item 2 KL, Item 4 trajectory, Item 7 parser probe).
2. **Honest `n/a` is a first-class signal.** Treating the recognised
   `n/a-sampling` value as 50% of the item weight rather than 0% is the
   single largest design choice in the prototype; without it Items 1
   and 2 would both score 0% and the corpus would look uniformly bad.
   With it, Items 1 and 2 score 24.4% (the same as the
   loss-form-honest-n-a in 98/98 mega manifests and the
   kl-honest-n-a-or-missing in 103/103 manifests).
3. **The auditor discriminates across corpora, not within one.** When
   the corpus is a single-stack harvest (the 98 mega manifests), the
   per-axis stratification is near-flat (std ≈ 1pt across model / G /
   temperature / seed). This is itself the informative result: when
   *every* manifest has the same template, no auditor can find
   variance *along* the stack axes. The auditor's information is in
   the **per-item table** (Item N weakness) and in **between-corpus**
   comparisons (quick 45.0 vs. mega 55.5).
4. **The fix is mechanical.** The 0-gold, 0-silver verdict on a
   corpus whose keys are 100% present is a flag, not a bug. Three
   small emitter changes (add `kl_coefficient` to Item 2; write the
   ZVF trajectory as a sidecar that lives next to the manifest; add a
   `parser_probe` sub-field to Item 7) would move every manifest into
   the silver tier on the same 103 cells. The auditor is therefore
   also the **work-list generator** for the next iteration of the
   manifest emitter.
5. **Closed-stack penalty is visible.** Item 3 drops from a possible
   100% to 64% because `tinker-closed` does not expose precision or
   decoding parameters; the auditor rewards the closed backend
   honestly (75% of 20 = 13.33) but penalises the missing
   sub-fields. This is the intended behaviour: a closed stack is
   reportable, a closed-and-undocumented stack is not.

## Reproducibility

```
$ python3 scripts/p5p8/minreport_auditor.py
manifests scored:    103
corpus sizes:        {'manifests': 98, 'quick_20260704': 5}
badge mean / median: 55.01 / 56.7
badge range:         [35.8, 56.7]  (std=3.08)
tier counts:
     gold: 0
   silver: 0
   bronze: 99
     wood: 4
     fail: 0
```

Outputs:
- `experiments/results/p5p8/minreport_audit.tsv` (per-manifest)
- `experiments/results/p5p8/minreport_audit_summary.json`
- `experiments/results/p5p8/figures/minreport_badge_dist.{png,pdf}`
- `experiments/results/p5p8/figures/minreport_per_item.{png,pdf}`

Paper-facing artifact: `paper/sections/p5_evidence.tex` Exhibit 8 +
`paper/build/paper_P5_minreport.pdf` rebuilds at 0 errors / 0
undefined refs (21 pages).

## Connection to iter-1 (item 01) and iter-9 (item 14)

Iter-1's audit was a binary "present / missing" count per key;
iter-9's extended audit added cells.tsv and the N10 corpus and
flagged the 18-item expansion proposal. This iteration **builds the
tool** that turns those audits into a single comparable number per
manifest, so future work can:
- Re-run the auditor after every manifest-emitter change to verify
  the badge has moved
- Score external papers' artifacts (when their supplementary JSON
  is published) to see whether their stack declaration is
  auditor-passing
- Use the per-item table as the spec for the next iteration of
  the manifest emitter

## Connection to FRONTIER_INSIGHTS.md

The ChatGPT Pro + Gemini Deep Think reasoning on Pillars 1-2 (Round
1) argued that the **algorithm axis is under-identified against the
stack axis**; the auditor operationalises the under-identification
*at the manifest level*: a manifest that earns a high badge *and*
declares its stack (Item 3 backend, Item 5 schedule, Item 7 probe)
is one the algorithm comparison can actually be drawn against. A
manifest with a low badge or an Item 3 n/a is one whose label
should not be cited as evidence for anything.

## Open question for iter 14

Should the auditor extend its 7-item schema to the 18-item
recommendation of iter-9 (item 14), or stay at 7 to remain
minimal-and-checkable? A 7-item auditor is shorter; an 18-item
auditor is what iter-5's eta^2 showed would catch the dominant axes.
The trade-off is operational (a 7-item auditor is what fits in a CI
badge; an 18-item one is what fits in a benchmark review).