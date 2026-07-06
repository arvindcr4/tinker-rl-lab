# P7 (ZVF Theory + Adaptive-G Controller) Claim–Evidence Lint

**Contract:** `research_prompts/writing/claim-evidence-linter.md` — label each major claim in the
abstract + results + conclusion of `paper/paper_P7_zvf_controller.tex` as **supported / weakly
supported / unsupported**, citing the exact artifact in `experiments/results/`, `registry/entries/`,
or the companion-paper record that backs (or refutes) it. Every number below was re-checked against
the on-disk TSV/JSON/registry entries on 2026-07-04. Fixes were applied directly to the tex; the
"Fix" column records what changed.

**Sections linted:** `p7_abstract.tex`, `p7_intro.tex`, `p7_results_intro.tex`, `p7_theory.tex`,
`p7_ushape.tex`, `p7_e1_validation.tex`, `p7_controller.tex`, `p7_design_rules.tex`,
`p7_synthesis.tex`, `p7_limitations.tex`, `p7_conclusion.tex`, `p7_appendix_derivations.tex`.

**Headline tally: 41 claims — 33 supported, 6 weakly supported (all provenance-bounded W&B
internal records, now flagged as such in Limitations), 2 unsupported-as-written (both corrected:
the "+1.44 rollouts per halving" math error and the "60 runs" sweep count).**

Support-level key:
- **S** supported — number reproduces from the cited/available artifact.
- **W** weakly supported — direction and value consistent, but the artifact is an internal
  program record (W&B project aggregate) rather than a released repo file, or the scope needed
  tightening.
- **U** unsupported as written — number/statement contradicted by the artifact or by arithmetic;
  corrected in the tex.

---

## 1) Claim table

### Abstract (`sections/p7_abstract.tex`)

| ID | Claim | Label | Evidence / check | Fix applied |
|----|-------|-------|------------------|-------------|
| A1 | ZVF indicator expectation h_G(p)=p^G+(1-p)^G; S=p(1-p)(1-h_G) model; T1–T3 | **S** | Closed-form derivations in `p7_appendix_derivations.tex`; T1/T2 elementary binomial facts; matches P2's frontier-synthesis Eq. (zvf-inelasticity). | none |
| A2 | 368-run audit "across seven model families", U-shape ZVF ≈0.95 at extremes, ≈0.25 at reward 0.4–0.5 | **W** | Internal program record (W&B `zvf-audit`); table values identical to companion P5 `tab:p5-ushape`. But: table has seven *models* (Llama-3.2-1B/3B are one family), extremes are 0.95–0.97, and the 0.25 minimum sits at reward **0.35** (Qwen3-32B); reward 0.50 gives 0.29. | "seven model families"→"seven models"; "≈0.95 / ≈0.25 at reward 0.4–0.5" → "0.95–0.97 / 0.25–0.29 at mid-range rewards of 0.35–0.50" |
| A3 | Model×G grid monotone: Qwen3-32B 0.33→0.18→0.08 (G=4/8/16) | **W** | Internal program record; identical to P5 `tab:p5-gsize`. One non-monotone cell (Qwen3-4B-Instruct 0.89→0.84→0.91) is flagged in the text as the boundary case theory predicts. | abstract now says "monotone G effect **at interior accuracy**" |
| A4 | E1: corr(‖∇‖, p(1-p)) = +0.71, 0.5B model, synthetic arithmetic, directional only | **W** | Internal record (W&B `zvf-colab-experiments`/`E1_grad_signal`); same number quoted as Stratum-C in companion P5 (`p5_stack.tex`, `p5_limitations.tex`). Directional scoping already exemplary (`p7_e1_validation.tex` "Honest scope"). | provenance flag added to Limitations |
| A5 | PCD = (G-1)/G·E[p(1-p)]; micro-jitter collapses batch ZVF 0.158→0.000 with PCD unchanged | **S** | `experiments/results/pcd_vs_zvf_summary.tsv`: zvf 0.1583→0.0000, pcd 0.153802→0.153802 (600 groups); identity derived in appendix; regenerable via `scripts/pcd_vs_zvf.py`. | none |
| A6 | E3: adaptive-G matches best fixed-recipe held-out gain (+0.575) at 186 rollouts; DAPO dynamic sampling ZVF 0.00 at +45% rollout cost | **S** | `registry/entries/colab-open_{grpo,drgrpo,dapo,grpo-adaptiveg}_e3.json`: deltas +0.500/+0.575/+0.550/+0.575; ZVF 0.25/0.27/0.00/0.23; rollouts 120/120/174/186. 174/120 = +45% ✓; 186/120 = +55% ✓. | none |
| A7 | "All interventional results are single-task, small-n, and scoped accordingly" | **S** | Matches registry notes ("Toy scale: directional evidence only") and Limitations. | none |

### Intro (`sections/p7_intro.tex`)

| ID | Claim | Label | Evidence / check | Fix applied |
|----|-------|-------|------------------|-------------|
| I1 | P2 "deliberately stopped at description … declined to promote to predictive/causal" | **S** | `p2_abstract.tex` lines 10–11, `p2_conclusion.tex` lines 4–6 — verbatim consistent. | none |
| I2 | Jitter of order 1e-4 zeroes ZVF "without changing the learning signal at all" | **U→S** | Overclaim: jitter does perturb rewards infinitesimally; artifact shows PCD shift < 1e-6. | reworded to "while shifting the measured within-group contrast by less than 1e-6" |
| I3 | U-shape "across seven model families" | **U→S** | Seven models, not seven families (see A2). | "seven models" |
| I4 | Contribution (i): each theorem "paired with an empirical check at population scale" | **U→S** | T3's check (E1) is toy-scale, not population-scale — contradicted two lines later by contribution (iii). | dropped "at population scale" |
| I5 | Related work: DAPO/GRESO/RL-ZVP/Dr.GRPO/GSPO positioning | **S** | All bib keys present in `references.bib`; characterizations match the delta records in `registry/entries/delta_*.json`. | none |

### Results opener (`sections/p7_results_intro.tex`)

| ID | Claim | Label | Evidence / check | Fix applied |
|----|-------|-------|------------------|-------------|
| RI1 | 12-cell head-to-head (Qwen3.5-4B, GSM8K, 3 seeds): last-10 0.744/0.742/0.723/0.710, mean ZVF 0.578/0.500/0.567/0.511 for dapo/drgrpo/grpo/gspo | **S** | `registry/entries/tinker_{dapo,drgrpo,grpo,gspo}_qwen3.5-4b_gsm8k.json` outcomes — all eight numbers reproduce exactly; seeds [42,123,456]; 4 recipes × 3 seeds = 12 ✓. | none |
| RI2 | Same "DAPO" label: ZVF 0.58 closed (clip surrogate, no dynamic sampling) vs 0.00 open | **S** | `tinker_dapo_...json` (mean_zvf 0.578, dynamic_sampling "absent", clip "surrogate") vs `colab-open_dapo_e3.json` (mean_zvf 0.0, dynamic_sampling "implemented"). | none |
| RI3 | Backend swap moved final reward 5.0%→84.4% (17×) | **W** | Internal program record; identical claim in companions P5 (`p5_abstract/stack/evidence`) and P6 (`p6_abstract/intro`); 84.4/5.0 = 16.9 ≈ 17× ✓. Attributed in draft to "Pillar-2 findings" but it is a P5/P6 exhibit. | attribution reworded to "companion papers (Pillar-2 … Pillar-5/6)" |

### Theory + U-shape + PCD (`p7_theory.tex`, `p7_ushape.tex`)

| ID | Claim | Label | Evidence / check | Fix applied |
|----|-------|-------|------------------|-------------|
| T1a | Binned GSM8K tensors: indicator 1.000 at p̂∈{0,1}, 0.000 at all interior bins (G=8, 3 seeds, only runs with full per-group tensors) | **S** | `pcd_vs_zvf_shape.tsv` (9 bins, exact match); `tinker_gsm8k_zvf_summary.json` (G=8, seeds [42,123,456], 600 groups). Endpoint values are definitional, now said so. | added "by construction at the endpoints, and empirically in the interior collapse" |
| T2a | Repo sweep: mean ZVF 0.838/0.764/0.691/0.631 for G=2/4/8/16; h_G(p̄) 0.731/0.554/0.326/0.114; p̄ 0.840/0.863/0.869/0.873; held-out 0.982/0.988/0.990/0.978 (SE 0.004/0.002/0.003/0.006) | **S** | `groupsize_zvf_sweep.tsv` — all 20 table cells reproduce (0.7635→0.764 etc.). Measured>predicted gap correctly explained as Jensen gap. | none |
| T2b | Model×G grid rows (Llama-3.2-1B 0.98/0.97/—, …, Qwen3-32B 0.33/0.18/0.08) | **W** | Internal record; identical to P5 `tab:p5-gsize` cell-for-cell; boundary-case reading (decay rate -ln max(p,1-p)) is correct. | provenance flag in Limitations |
| U1 | U-shape table (7 rows, reward 0.01→0.95, ZVF 0.97→0.96) | **W** | Internal record; identical to P5 `tab:p5-ushape` cell-for-cell. | text range corrected (see A2) |
| U2 | Jitter perturbs contrast "at most O(1e-8) in variance" | **U→S** | Per-group sample cross terms can reach O(1e-5); only the *expected* shift is Var(ε)=δ²/12 ≤ 8.4e-10. Empirical batch shift < 1e-6 ✓ (`pcd_vs_zvf_summary.tsv`). | main text now states expected shift ≤ 1e-9; appendix restated with zero-mean cross-term argument |
| U3 | Dense sub-reward batch ">15% dead" read as healthy | **S** | zvf_batch_before_jitter = 0.1583 ✓. | none |
| U4 | ρ=0.95 (mean reward) vs 0.56 (ZVF) on 80-run summary corpus; "ρ≈0.27 on the wider Pillar-2 corpus" | **U→S** | `pcd_vs_zvf_summary.tsv`: spearman_meanreward_outcome 0.9527, spearman_zvf_outcome 0.5638, n_runs_crossrun 80 ✓. But P2's ρ=0.27 is on its **23-cell** cross-experiment matrix (`zvf_cross_experiment_diagnostic.tex`, `zvf_failure_correlation.tsv` n=23) — not a "wider" corpus. | reworded to "the companion Pillar-2 paper reports ρ = 0.27 on its 23-cell cross-experiment matrix" |

### E1 + controller + E3 (`p7_e1_validation.tex`, `p7_controller.tex`)

| ID | Claim | Label | Evidence / check | Fix applied |
|----|-------|-------|------------------|-------------|
| E1a | corr(‖∇‖, p̂(1-p̂)) = +0.71, single configuration, no CI, sign-only transfer claim | **W** | See A4. The "Honest scope" paragraph already makes every reservation the linter would demand. | none |
| C1 | Adaptive schedule G: 4→6→8 on live ZVF spikes; changes only sampler budget | **S** | `colab-open_grpo-adaptiveg_e3.json` adaptation_rule: "zvf-triage callback escalates G 4->6->8 on ZVF spikes (live)". | none |
| C2 | E3 four-arm table (all 12 cells) | **S** | Registry entries reproduce every cell (see A6). | none |
| C3 | "Competitive with, not superior to, the best fixed recipe" | **S** | +0.575 tie with Dr.GRPO; claim correctly graded as the weakest reading. | none |
| C4 | Closed-stack "DAPO" arm mean ZVF 0.58 | **S** | `tinker_dapo_...json`: 0.578. | none |

### Design rules (`p7_design_rules.tex`)

| ID | Claim | Label | Evidence / check | Fix applied |
|----|-------|-------|------------------|-------------|
| D1 | Rule 1 boundary arithmetic (escalation buys O(ε)/rollout at mastery) | **U→S** | Draft said "h_G(p)≈1-Gp(1-p) improvements of order Gε" — garbled; appendix gives h_G≈1-Gε, marginal rollout recovers O(ε). | rewritten to "improvements of only O(ε) per added rollout, since h_G(p) ≈ 1-Gε near the boundary" |
| D2 | Rule 2 "+1.44 rollouts per halving at p=1/2" | **U (math error)→S** | h_G = 2^{1-G} at p=1/2, so **each +1 rollout halves h_G**; 1/ln2 ≈ 1.44 rollouts is the *e-folding* cost, not the halving cost. Same error in appendix T2. | both places corrected: "+1 rollout halves; e-folding costs 1/ln2 ≈ 1.44" |
| D3 | Rule 2 static point: G=16 costs 4× G=4 for ZVF 0.76→0.63, no held-out gain | **S** | `groupsize_zvf_sweep.tsv`: 0.7635→0.6312; held-out 0.988 vs 0.978 (overlapping SEs). | none |
| D4 | Rule 3: lag-1 autocorrelation ≈ 0.94 (Pillar 2) | **S** | P2 `zvf_dynamics.tex` line 172 ("≈0.94 across the board"); P2 lint A5 verified method-pooled 0.939 from `zvf_dynamics_phase.tsv`. Correctly attributed to Pillar 2. | none |
| D5 | Rule 4: backend-swap + label-flip cross-stack caution; 7-item minimum stack | **S/W** | Same artifacts as RI2/RI3; 7-item list matches P5's items 1–7. | none |
| D6 | Rule 5: ZVF=0 always purchasable at +45% | **S** | E3 dapo arm; scoped to "on our task" with D2/D4 flagged as open. | none |

### Synthesis (`p7_synthesis.tex`)

| ID | Claim | Label | Evidence / check | Fix applied |
|----|-------|-------|------------------|-------------|
| SY1 | Cross-examination proposed the binomial form, aliasing, jitter test, PCD | **S** | `frontier_synthesis_zvf.tex`: inelasticity theorem (p^G+(1-p)^G), aliasing paragraph, micro-jitter falsification, Gemini's PCD with identical (G-1)/G expectation. | none |
| SY2 | ρ ≥ 0.45 bar set for replacement diagnostics; "we have not cleared that bar" | **S** | `frontier_synthesis_zvf.tex` line ~94: "stated target of ρ≳0.45 — a falsifiable bar, not a claimed result". P7 correctly reports it unmet and defers to D1. | none |
| SY3 | Sign-resolution half: ρ=0.95 vs 0.56 on 80-run corpus | **S** | `pcd_vs_zvf_summary.tsv` (see U4). | none |

### Limitations + conclusion (`p7_limitations.tex`, `p7_conclusion.tex`)

| ID | Claim | Label | Evidence / check | Fix applied |
|----|-------|-------|------------------|-------------|
| L1 | Four load-bearing limitations (single task, small n, LoRA/closed arm, toy gradients) | **S** | Matches registry notes and E1/E3 scoping; "deltas separated by 0.025–0.075" tightened (two top arms are separated by 0.000). | "separated by at most 0.075" |
| L2 | (added) provenance boundary for W&B internal records | **S** | New paragraph mirrors P5 `p5_limitations.tex` first limitation; points to appendix provenance note. | added |
| L3 | D1–D5 proposed experiments extend `experiments/FRONTIER_EXPERIMENT_BACKLOG.md` | **S** | File exists. | none |
| CO1 | Conclusion restates T1/T2/T3, U-shape/368 runs, PCD repair, E3 tie, +45% pricing | **S** | All restatements match the corrected section-level numbers; "directionally confirmed … at +0.71" keeps the scope marker. | none |
| AP1 | Appendix provenance: sweep = "60 runs" | **U→S** | `groupsize_zvf_sweep.json`: **12 runs** (4 G × 3 seeds). | corrected to "12 runs: 4 values of G × 3 seeds" |

---

## 2) Missing evidence list

1. **E1 (+0.71)** — no repo artifact; lives only in W&B `zvf-colab-experiments/E1_grad_signal`.
   Paper now flags this in Limitations (provenance boundary) in addition to the existing
   "directional only" scoping. Decisive version specified as D3.
2. **368-run audit + model×G grid** — W&B `zvf-audit` aggregates only; no per-run TSV in
   `experiments/results/`. Values are cell-identical to companion P5's tables, so the two papers
   cannot drift apart, but neither is independently re-derivable from the released repo. Flagged.
3. **E3 seeds** — registry entries record `"seeds": null` for all four arms; the paper's "seeds
   shared across arms" (table caption) is the protocol statement, not a recorded artifact. Left
   as-is (consistent with registry notes) but D2 requires n ≥ 5 recorded seeds.
4. **PCD as outcome predictor** — deliberately absent; the paper's D1 pre-registers the test
   instead of proxying it. No fix needed (this is the correct behavior).

## 3) Consistency with companion papers

- **P2 (descriptive ZVF):** No contradiction. P7's framing of P2 ("declined to promote to
  predictive/causal") matches P2's abstract/conclusion verbatim. Numbers P7 borrows from P2
  (lag-1 ≈ 0.94; ρ = 0.27 — now correctly scoped to the 23-cell matrix; ρ ≥ 0.45 bar; PCD
  proposal; jitter test provenance) all check against P2's sections. P7 is the interventional
  sequel P2's synthesis called for; the division of labor is stated in both directions.
- **P1 (no scaling law):** No contradiction. P7 makes no cross-scale claims; the model×G and
  U-shape results are within-model group-size/accuracy-regime effects, and E1 explicitly
  disclaims coefficient transfer across scales ("only the sign and ordering").
- **P5/P6:** Backend-swap (5.0%→84.4%) and DAPO label-flip exhibits quoted with identical
  numbers; attribution now names Pillar-5/6 alongside Pillar-2; 7-item minimum stack list
  matches P5's items.

## 4) Build

`pdflatex → bibtex → pdflatex ×2` on `paper/paper_P7_zvf_controller.tex`: **0 errors, 0 undefined
citations, 0 undefined references** (the two dangling `sec:frontier`/`sec:stat_rigor` refs in
`_shared_methods.tex`, inherited by every standalone pillar paper, were replaced with textual
pointers). 17 pages.
