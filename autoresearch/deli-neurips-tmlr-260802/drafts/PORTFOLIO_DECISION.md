# Portfolio decision

Source tag: `w-decide` | Date: 2026-08-02 | Inputs: 12 independent paper verifications
(P1/P6 wave 1, P2–P5/P7–P12 wave 2), `audits/18_PAPER_PORTFOLIO_REVIEW.md`,
`drafts/PUBLICATION_READINESS.md`, `state/confirmatory_execution_plan.md`, `state/task_spec.md`.

## Bottom line up front

**Recommended spine: P11, reframed and retitled.** It is the only manuscript in the roster
whose every headline statistic recomputes from checked-in raw data, whose held-out prompt
sequence is provably identical across all 40 arm-seed units, and which already obeys evidence
rules 1–5 and 8. Its blocking defects are four text fixes and one figure node. Zero GPU hours.

**Ready verdict: `ready_after_edits`** — for a methods + reproducibility paper (TMLR-shaped),
not for a NeurIPS empirical claim. The task_spec stop condition **triggers**: no empirical
capability claim in this portfolio survives verification. The narrower paper that does survive
is named in §5.

**The GPU option is a real option and it is not the recommended first spend.** The
preregistered sampler matrix is 92 rows / ~640–680 GPU-h / ~$700–950 / ~15 days at K=2 *after*
a confirmatory runner that does not yet exist is written and hash-bound. A ~$40, ~38-A100-hour
alternative upgrades the recommended spine from INCONCLUSIVE to a preregistered positive
verdict. If any GPU money is spent, spend that first. See §4.

---

## 1. Paper-by-paper decision table

Evidence health scale — **SOUND**: headline claims recompute, defects are editorial;
**MIXED**: a real recomputable core sits inside unsupported framing; **BROKEN**: the majority
of headline claims are contradicted, unreproducible, or arithmetically forced.

| Paper | Roster verdict | Verifier adjudication | Evidence health | Cheapest path |
|---|---|---|---|---|
| **P1** scaling limits | Cut to workshop note on failed identifiability, or thesis chapter | `minor_issues`. Core negatives recompute exactly: flat cross-scale slope, λ-at-bound on 4/5 fits, constant model wins AICc on all 5, Nemotron zero-reward frac 0.55. Defects are peripheral. | **MIXED-HIGH** (best empirical health after P11) | Zero GPU. Fix 5-of-20 off-by-one; delete the no-Nemotron sensitivity (recomputes +0.278, p≈0.049 — reverses the stated conclusion) and the MoE/dense split (Nemotron `A12B` suffix may make it MoE; offline-unverifiable); drop "pre-registered"; reconcile 671B vs 685B; fix `rafailov2024rlhfscaling`. → 6–8pp negative identifiability note. |
| **P2** ZVF | KEEP_AFTER_CUT → short ZVF stratification note; blocker = "descriptive not predictive/causal" | `major_issues`, **too lenient, wrong blocker**. The stated blocker is P2's own declared non-claim. Real blocker: `variance_mitigation.tsv` is a **simulation** (negative rewards, negative accuracies, 2710 continuous ZVF values) consumed as measured evidence in 4 of 5 sections — 45 of 80 summary rows, 9 of 17 correlation cells. Collapse correlation's positive class is two dead-verifier runs; stated n=23 is really 17; the abstract's `1.11e-16` identity is a tautology with no body support; AERO misattributed to Le et al. | **BROKEN** (1 of 7 claims recomputes clean) | Zero GPU, but a rebuild not a cut. Delete everything sourced from `variance_mitigation.tsv`; delete all pooled cross-experiment correlations; demote the identity to a two-line lemma; rebuild around the one real asset — the 600-row Qwen3-8B/GSM8K per-group tensor + 480-step G-sweep. → 6–8pp sampling-model falsification note. |
| **P3** group size | MERGE into P2 or thesis; blocker = G=32 reconstructed, G=4 proxy-only | `major_issues`, **too lenient, wrong blocker**. The stated blocker is stated as a non-claim in abstract, intro, every caption, and conclusion. Real: the headline SNR statistic is degenerate (`advantage_variance` is two-valued; SNR ≡ √(q/(1−q)), a re-expression of 1−ZVF, bounded so it cannot track √G); all 20 grid cells are `FALLBACK_ROWS` hardcoded constants whose own script says the ±0.03 band "is NOT a measured bootstrap CI"; every TOST p, the T_crit, the 1024M extrapolation and the 56.2% saving derive from that constant. | **BROKEN** | Zero GPU. **Do not merge into P2** — merging imports the fabricated grid. Delete the reconstructed chain and the SNR mechanism; the 2–3pp residue (12 runs, 0.5B, synthetic arithmetic, n=3, no separable G effect) goes to the thesis negative-results chapter. The 505-task allocation result already lives in P12 — strengthen it there as a **plateau** (G=3–6 within 4% of each other), not a selection. |
| **P4** length bias | PARK → bounded negative note; blocker = 200-tok cap + short horizon | `major_issues`, **too lenient, wrong blocker**. The cap problem is *understated* (the "uncapped" 1024-tok panel starts at 93–100% of cap, per-run maxima exactly 1024.0 — no length-headroom regime exists anywhere in the repo), but it is not binding. Binding: the strongest positive claim is pseudo-replicated 3× (same 200 held-out items pooled into 600 McNemar outcomes; de-replicated Dr.GRPO p=0.076); the GRPO-vs-Dr.GRPO CI [−1.96, +4.29]pp contains the paper's own +2–5pp decision threshold; the one DECISIVE result is measured on 4.74-token completions. | **BROKEN** | Zero GPU. **Retire the bounded_negative_note** — the null carries no information. Invert: ship a ≤6pp measurement-validity note on cap-induced non-identifiability of length bias. Report McNemar per seed; strike "independent" from the audit description; fix the Qwen3-8B/Qwen2.5-1.5B abstract attribution (standing stop rule, still unfixed); delete the 10 contentless TikZ diagrams. |
| **P5** MIN-REPORT-RL | MERGE into P5+P6 resource; blocker = confounded 17× + no external adoption | `major_issues`, **understated**. The 17× low arm is not confounded, it is **not a measurement**: `clipped_ratio=1`, `mean_terminated_length=0`, 25/30 steps exactly 0.0 — no completion ever terminated. Also not same-label (`loss_type=dapo` vs `method=grpo`). Missed by roster: Exhibit 5's η²(G)=1.0000 is arithmetically forced (`per_group_n='1,1,1,1'`, so SS_within≡0) and the 22×/133× ratios inherit it; Exhibit 9 certifies Exhibit 3's Tinker number with a bootstrap computed on a different corpus, model and task. | **BROKEN** (1 of 8) | Zero GPU. Retract Exhibit 1 as an empirical exhibit; recast as a provenance-failure vignette (stronger evidence for "report the stack" than the number ever was). Delete η²(G) and the 22×/133× ratios. Delete Exhibit 9's certification sentence. What survives is Exhibit 6 (n=98 field-coverage audit, 0/98 carry a concrete `kl-*` value) + Exhibit 7 (25 η² values reproduce to 4dp) → ~12pp reporting resource with P6. |
| **P6** GRPO registry | MERGE with P5; release, recruit external entries, measure extraction error | Wave-1 findings. 17× exhibit and badge/stackdiff machinery **recompute exactly**. But: **iter-194 amendment records flipped the AERO/AREAL predicted signs from ≥0 to ≤0 *after* the negative deltas were observed**, turning 2 Contradicts into `n_contradicts=0, supports_rate=1.0`; the amendment appears nowhere in the P6 tex. Registry counts in abstract (3 deltas) contradict the overview (11) and the shipped artifact (28 stack + 18 delta + 2 amendment = 48). | **MIXED-LOW** — integrity flag | Zero GPU. **Disclose the iter-194 amendment prominently and restore the 2 Contradicts, or delete the iter190 prediction audit.** This is the most serious integrity finding in the portfolio and the roster missed it entirely. Then reconcile registry counts to the shipped artifact and merge with P5. |
| **P7** ZVF controller | PARK → future controller experiment; blocker = no prospective matched-cost advantage | `major_issues`, **wrong unit**. PARK is right and the stated blocker is real (E3 has no cost-matched fixed-G arm: adaptive spends 186 rollouts vs baselines pinned at 120), but it is not binding, because the retrospective layer fails audit on its own: E3 is described as "GSM8K-style" and is two-digit addition; the audited controller contains **zero occurrences of PCD**, the paper's contribution (ii); the U-shape table publishes 7 of 9 models and drops the sole non-monotone point; T2 grid pools a 10× lr range while claiming "only G varies". | **MIXED-LOW** (3 of 8) | Zero GPU. Retire `future_controller_experiment` as the near-term unit. Keep the two things that recompute to the digit: 0/1867 structurally degenerate groups restorable by escalating to G=16, and the ZVF/PCD separation (micro-jitter collapses ZVF 0.1583→0.0000 with PCD invariant at 0.153802). **Note:** the sibling 92.3% figure is disqualified — see the P12 row. Delete the 12 trailing TikZ figures after `\bibliography` (all 13 multiply-defined-label warnings). |
| **P8** workshop artifact | SUPPORT_DOCUMENTATION → P9 artifact documentation; blocker = 7-gram overlap 0.427/0.306 | `major_issues`, **wrong unit**. Overlap is a merge problem curable by rewriting; the binding problem is provenance and is curable only by deleting rows. The only multi-seed open-stack headline row is labelled GSM8K and is synthetic two-digit addition (`trl_grpo_math.py:99-101`), contradicted by the paper's own manifest (`task=math/split=gen`); the 100%/100% Qwen3-235B row is `partial=true, steps_completed=15/20, checkpoint='training_interrupted'`, Tier C by the paper's own exclusion rule; the r=−0.769 headline **flips sign to +0.350 (p=0.0021) at run level** on the same released traces; the repo's verifier for it returns `True` unconditionally. | **BROKEN** (2 of 9) | Zero GPU. **Do not ship as P9's documentation** — that installs a mislabelled task and an interrupted run into the artifact release. Regenerate P9's documentation from `run_manifest.tex` instead (the manifest is the honest object; the main text systematically upgrades what the manifest downgrades). Migrate the two claims that recompute — 5-seed held-out negative control (t=1.32323, p=0.25629) and the 2-seed matched-budget G=2×160 vs G=16×20 panel — into the spine or the P2 note. |
| **P9** DNB benchmark | ARTIFACT_CANDIDATE → benchmark artifact; blocker = public anonymous clean-machine release unverified | `major_issues`, **too lenient**. The release-channel blocker is real but downstream. Binding: the artifact's central affordance does not exist (`make reproduce-main` is absent from the Makefile; the source comment concedes it is a TODO); the ledger does not reconcile with itself (all 8 frontier rows disagree with their named source JSON, reported at 30 steps where the CSV says 26, one cell twice at 84.4% and 85.6%, "Last-10" computed over 3–5 points); Tier A is defined four incompatible ways; the compute card is off by 60–180× and on the wrong hardware (claimed 96 H100-h vs 0.5423 GPU-h of L4 elapsed time). | **BROKEN** (1 of 9) | Zero GPU, ~1 focused day, but cuts the paper to a third of its claims. Rebuild every table from **one** named ledger; drop rows whose trace is shorter than the window the statistic names; exclude `status=failed` runs from the ZVF pool and check the regenerated pool in; state the ZVF identity correctly as E[ZVF]=E_p[p^G+(1−p)^G] with the Jensen caveat; delete the 79-run/7-library/MATH-500 framing. → 8–10pp artifact + instrumentation note. |
| **P10** ZVF theory | MERGE → theory appendix; blocker = conditional results written too broadly | `major_issues`, **too lenient, wrong blocker**. The stated blocker **does not survive checking** — T3 is correctly scoped in abstract, theorem, controller section and limitations, and T1/T2/T3 all prove out. Binding instead: E-P4's "Dr.GRPO" arm removes only `/std` (`live_zvf_probe.py:109`) while Liu et al. define it as removing `1/|o_i|` **and** std and show the length term drives length — so the reported null is manufactured, in the one paragraph making a claim about someone else's work; all 10 figures are placeholder TikZ flowcharts with captions describing plots that do not exist, two duplicated; the non-claims paragraph ("contains no experiments of its own") is false. | **MIXED** — theory sound, empirics broken | Zero GPU, same-day. Keep only T1, T2, T3 + E-T1 + E-T2 (both fully recompute). **Delete E-P4 outright** — not fixable by rewording. Move E-T3a/E-R2b/E-B to P7. Delete all 10 figures. Fix the E-R2b ZVF range (0–0.50, not 0–0.25) and the curriculum coverage range (0.07–0.43). Result is a correct 5–6pp theory note that is not a paper — it is the methods layer of the P2 note. |
| **P11** repro audit | KEEP_BOUNDED → reproducibility audit; blocker = one model/task/stack, short horizon, low power | `minor_issues`, **agree on disposition, blocker not binding**. The stated blocker is a generalization caveat the paper already states in its own Limitations. Binding instead: (a) `published_delta` is **null for all five arms**, so "survival" — the title, the abstract framing, named contribution 2 — was never computed and RETAINS is structurally unreachable; (b) `fig:r08_fig4` still renders "Disappears Verdict" for DAPO in the built PDF, the exact label the repo's own `STATISTICAL_REANALYSIS.md` forbids in any manuscript; (c) the pilot table says two seeds where the repo's own ledger says n=1, `INSUFFICIENT_N`. | **SOUND** with editorial defects (6 of 9 recompute, 0 fabricated) | Zero GPU, ~1 working day. Reframe from "protocol + four-way null" to "single-stack audit protocol + MDE-bounded cost of dynamic sampling". Full list in §3.1. |
| **P12** signal starvation | PARK → prospective methods note; blocker = no PPO/SAO outcomes or controller test | `major_issues`, **agree on disposition and unit, blocker not binding**. A proposal paper is allowed to have no outcomes, and P12 fences PPO/SAO explicitly. Binding instead: both empirical groundings are informationless. The `1.11e-16` identity is float reordering (`analyze_breakthroughs.py:78-83`). The 92.3% escalation asymmetry is the **base rate of a rule that fires iff k∈{0,8} by construction** — verified by crosstab: fires=1 on exactly the 1,723 k=8 and 144 k=0 rows, fires=0 on all 693 rows with k∈1..7. Plus undisclosed cohort censoring (p_x restricted to (0.05,0.95) removes ~73% saturated prompts) and Figure 2 contradicting Proposition 2. | **MIXED** (numbers correct, headlines carry no information) | Zero GPU. Delete both headline numerals; state the firing rule in the text; disclose the censoring and the denominator. The G=4 allocation result **survived a harder double bootstrap than the paper's own** — reframe it as the plateau result and make it the paper's actual contribution. → workshop-short methods + proposal note. |

---

## 2. Where the verifiers disagreed with the roster, and which disagreement matters

### The pattern

Across ten fresh adjudications: **zero** verifiers called the roster too harsh. Eight called it
too lenient or aimed at the wrong unit. **Ten of ten** said the roster named the wrong binding
blocker.

The roster grades manuscripts on **scope and generalization caveats** — one model, one task, no
causal claim, no external adoption, short horizon. Several of the audited papers *already state
those caveats as explicit non-claims*, which is exactly what evidence rule 8 asks for. Grading a
paper on its own declared non-claim lets it off the hook.

The verifiers found the binding failures are almost all **rule 2 and rule 6** violations:

- simulated data consumed as measured evidence (P2: 45 of 80 summary rows);
- hardcoded constants presented as measurements with derived inferential statistics (P3: 20
  cells + a hardcoded 0.03 driving every TOST p-value; P8/P9 inherit the same grid graded "B");
- broken, partial or failed runs inside headline aggregates (P5: 100%-clipped zero-termination
  arm; P8/P9: `partial=true, steps_completed=15/20` reported as 100%/100%; P9: "Last-10"
  computed over 3-point traces);
- statistics that are arithmetically forced and therefore measure nothing (P5: η²=1 at n=1 per
  group; P3: SNR ≡ √(q/(1−q)); P12: 92.3% is a by-construction base rate; P2/P12: `1.11e-16` is
  float reordering).

None of these is a compute problem. All are correction problems. That is the single most
consequential finding in this synthesis: **the portfolio's gap is not evidence quantity, it is
evidence bookkeeping.**

### The seven disagreements that change a decision

1. **P6's post-hoc sign flip (roster: not mentioned at all).** Iter-194 amendment records
   changed the AERO and AREAL predicted signs from ≥0 to ≤0 *after* the negative deltas were
   observed, converting 2 Contradicts and an 85.7% support rate into `n_contradicts=0,
   supports_rate=1.0`; the amendment appears nowhere in the P6 source. **Why it matters:** this
   is a prediction-registry paper whose entire value proposition is that predictions are
   registered before observation. If it ships as written it is a falsified claim in a paper
   about not making falsified claims. This needs a human ruling before P5/P6 merge work starts.

2. **P2's simulated corpus (roster: "descriptive not causal").** The roster's prescribed fix —
   cut to a stratification note — carries `variance_mitigation.tsv` forward. **Why it matters:**
   the roster's remedy does not remove the disqualifying material.

3. **P3 → merge into P2 (roster) vs. do not merge (verifier).** Merging imports 20 hardcoded
   constants and a degenerate SNR into the one ZVF paper the roster is trying to save. **Why it
   matters:** a contamination decision, and the roster's instruction points the wrong way.

4. **P8 → P9's artifact documentation (roster) vs. regenerate documentation from the manifest
   (verifier).** Shipping P8 as documentation installs a synthetic-arithmetic row labelled GSM8K
   and an interrupted run labelled 100%/100% into a public artifact release. **Why it matters:**
   this is a shipping decision with an external blast radius.

5. **P5's 17× "confounded" (roster) vs. "not a measurement" (verifier) vs. "recomputes exactly"
   (P6 verifier).** All three are literally true — the arithmetic reproduces from
   `master_results.json`, *and* the low arm never terminated a single completion. **Why it
   matters:** P5 and P6 both currently publish this exhibit, both would defend it as
   "recomputes", and only a joint ruling removes it from both.

6. **P11's binding blocker (roster: generalization) vs. unanswerable estimand (verifier).**
   `published_delta` null for all five arms means the paper's *title* names a quantity it never
   computed. **Why it matters:** it changes the title and framing of the recommended spine.

7. **P10's "universal written too broadly" (roster) — does not survive checking.** T3 is
   correctly scoped; the real defects are a mis-implemented Dr.GRPO ablation contradicted by its
   own citation and 10 placeholder figures. **Why it matters:** the roster prescribes a wording
   fix where a deletion is required, and the roster's own self-review recorded **zero** flags for
   this paper.

### One disagreement that does not matter

The roster and the verifiers converge on direction in 12 of 12 cases: nothing is submittable
unchanged. The disputes are entirely about *which* repair to make and *what unit survives* — not
about whether repair is needed.

---

## 3. Publication units reachable with zero new GPU runs, ranked

Ranked by (evidence health × remaining work × venue credibility). Units below rank 4 exist but
are not worth the calendar time unless a specific external need appears.

### 3.1 — Rank 1: **Single-stack preregistered audit protocol + MDE-bounded cost of dynamic sampling** (from P11)

*This is the recommended spine.*

The positive claim is already in the checked-in data and is not an equivalence claim: DAPO drives
mean ZVF from 0.693 to exactly 0.000 at **3.61× the rollouts** (1734 vs 480) and **1.44× wall
clock** (4.47h vs 3.11h), and its seed-paired held-out gain is **bounded above by +0.0068 at 95%**
(CI [−0.00450, +0.00675]). Cost and capability stay separate (rule 3); the bound is stated as a
bound, not as equivalence (rule 4).

Remaining work (~1 working day):
1. Delete the "Disappears Verdict" node from `fig:r08_fig4` — the exact label the repo's own
   `STATISTICAL_REANALYSIS.md` forbids in any manuscript. **Blocking.**
2. Retitle and reframe away from "survival". Either register metric-compatible published deltas
   or drop the survival framing; `published_delta` is null for all five arms, so survival
   fractions were never computed and RETAINS was structurally unreachable. **Blocking.**
3. Restate the pilot as n=1 per arm per the repo's own ledger (`p7_iter123_headline_cis.tsv`,
   class `C3_single_seed`), or cut Table 2 entirely. **Blocking.**
4. Fix Table 2's "Held-out Δ" header — the GRPO baseline row is 0.500, so it is a within-arm
   pre/post change, not a delta versus GRPO.
5. Add an explicit non-claims paragraph: no published delta was registered; DISAPPEARS was
   unreachable a priori at n=8 (smallest achievable MDE80 across arms is 0.0101 against a 0.01
   margin); replay perturbs a held-out score by up to 0.004, which exceeds the 0.001 DAPO point
   difference; DAPO consumed 3.61× the rollouts, so the stack lock controls implementation but
   not sample budget.
6. Fix the `qwen2025gspo` bib entry (fabricated author "Zhou, Jianwei", absent from
   arXiv:2507.18071) and the `tong2025drgrpo` non-author.
7. Re-ground the abstract's opening premise in public prior art (Henderson et al. 2017; SLM Lab;
   OpenRLHF) rather than in three unpublished companion manuscripts that this same triage has
   blocked or merged.
8. Set `preregistration.json` status from `preregistered-not-run` to completed, and disclose that
   the A100 amendment's `locked_at` is date-only on the same calendar day the first confirmatory
   unit completed — priority cannot be established from the artifacts alone. State it; do not
   hide it.
9. Optionally absorb P8's two recomputing claims (5-seed held-out negative control t=1.32323,
   p=0.25629; 2-seed matched-budget G=2×160 vs G=16×20 panel) — ~1 page, and both are exactly the
   kind of bounded result this paper is built to carry.

**Venue class: methods + reproducibility paper.** TMLR is the natural home (it explicitly
accepts negative and bounded methodological results and does not require a positive delta). A
workshop version can stay narrower. It does **not** collide with the flagship's NeurIPS overlap —
but run the overlap check, since P11 currently cites the flagship as a companion.

### 3.2 — Rank 2: **ZVF sampling-model falsification note** (rebuilt from P2, with P10's theory core and P7's structural-inertness result)

The i.i.d. Bernoulli ZVF model is systematically wrong **and its sign is model-dependent**:
delta_div = ZVF_iid − ZVF_obs is **+0.1224 [+0.1115, +0.1338]** on real GSM8K reasoning and
**−0.0668 [−0.0792, −0.0560]** on synthetic arithmetic. Both CIs exclude zero; the signs are
opposite. Consequence: iso-G rollout sizing derived from the binomial model is miscalibrated in
*both* directions (G=13→G=5 on real tails, G=8→G=9 on the synthetic frontier), which is a
falsifiable, practitioner-relevant correction to the sizing rules AERO/GRESO/DAPO-style prompt
filtering rests on.

Sources, all fully offline-recomputable: `tinker_gsm8k_zvf_s{42,123,456}.json` (3 seeds × 200
GSM8K problems, Qwen3-8B, G=8, full per-group reward vectors — ZVF recomputes to 0.1300 / 0.1900
/ 0.1550, pooled 0.15833) and `groupsize_zvf_sweep.json` (480 logged steps, Qwen2.5-0.5B,
G ∈ {2,4,8,16}, 3 seeds).

Remaining work (1–2 days, mostly deletion):
1. Delete — do not caveat — everything sourced from `experiments/results/variance_mitigation.tsv`
   (ZVF-by-library, AERO-vs-GRPO, the 9-method dynamics rows, lead-time, iter130 risk index, the
   AUROC 0.929 eval protocol, the self-debug section). One caption already calls it a simulation
   projection; the data has negative rewards and negative accuracies.
2. Delete all pooled cross-experiment correlations and say why: "we do not report a pooled
   ZVF-outcome correlation because the only cells that would drive it are runs where the verifier
   returned zero on every rollout" is a stronger sentence than ρ=0.27.
3. Demote `pass@G − p^G = 1 − ZVF` to a two-line lemma with an algebraic proof. Delete "verified
   to 1.11e-16 on 505 tasks" everywhere it appears.
4. Import P10's T1, T2, T3 with proofs and E-T1/E-T2 as the methods layer. Delete P10's E-P4.
5. Import the within-GSM8K stratification (the absorbed R02 result) as the "do not pool ZVF
   across tasks" warning — independently corroborated by P9's verifier, who found the pooled r
   flips from −0.769 to +0.40 within GSM8K and collapses to −0.03 under task fixed effects, and
   by P8's verifier, who found +0.3501 (p=0.0021) at run level on the released traces.
6. Optionally import P7's structural-inertness result (0/1867 degenerate groups restorable by
   escalating to G=16) and the ZVF/PCD micro-jitter separation. Do **not** import P7's 92.3%
   figure — P12's verifier proved it is a by-construction base rate.
7. Three mandatory fixes: cite `zhang2026aero` (arXiv:2602.14338) for AERO and `le2025rlzvp`
   (arXiv:2509.21880) for RL-ZVP, dropping the "also reported as" merge; fix the seed-42/seed-123
   accuracy swap in `zvf_summary.tsv`; re-run every bootstrap with a fixed seed so the published
   CIs match the artifacts to the last digit.

**Venue class: workshop short / short note (6–8pp).** Not a NeurIPS or TMLR main-track empirical
paper. Stated as a measurement/negative-result note it is honest and checkable.

### 3.3 — Rank 3: **Failed-identifiability note on RL post-training scale** (from P1)

The useful result is negative and it recomputes exactly: model size, stack, recipe and budget move
together in every available anchor, so cross-scale differences cannot be assigned to scale — flat
slope, λ pinned at the bound on 4 of 5 fits, and a constant model winning AICc on all 5.

Remaining work (~half a day):
1. Fix "4 of 20 steps > 0.25" → 5 of 20 (body text and `fig:scaling-elevated` caption).
2. **Delete the no-Nemotron sensitivity numbers.** The paper says the gap shrinks to ~+0.20 with
   p>0.05; recompute gives +0.278 with one-sided permutation p≈0.049 — which reverses the stated
   direction — and no checked-in TSV backs the sensitivity numbers at all (rule 6).
3. **Delete the MoE-vs-dense stratification** (perm p=0.023). Nemotron-120B is classified dense but
   `NVIDIA-Nemotron-3-Super-120B-A12B` follows MoE active-params naming; the classification is not
   verifiable offline and HF fetch is prohibited. Removing it removes the only positive claim and
   leaves a clean negative — which is the paper.
4. Strike "pre-registered" from the abstract: the three-phase falsification predictions are
   internal autoresearch iteration-ledger entries inside the same section files, not an immutable
   external artifact (rule 1).
5. Reconcile DeepSeek-V3.1 at 671B (paper) vs 685B (every TSV), and reconcile "roster max ~671B"
   against a 12-anchor fitted pool that includes Kimi-K2 at 1T — the source of the stated 2.4-OOM
   span.
6. Fix `rafailov2024rlhfscaling` (4 of 8 authors listed, Hejna mis-ordered).
7. Lift the stale `claim_to_run_table.md` P1-C2 caveat: the 0.55 zero-reward fraction *is*
   re-verifiable from the checked-in per-step trace (0.55 / peak 0.875 / last-10 0.1625, exact).
   The separate W&B quarantine stands.

**Venue class: workshop note, or thesis chapter.** 45 pages must become 6–8. Direct recent work
occupies the broad scaling claim (arXiv:2509.25300, 2507.18014, 2607.13389), so the contribution
must be framed strictly as an identifiability failure in *these* anchors, not as a scaling law.

### 3.4 — Rank 4: **Small auditable artifact + instrumentation note** (from P9, documented from the manifest, not from P8)

Zero GPU but the most bookkeeping labor of any unit, and the honest scope after repair is modest:
~75 runs, one closed backend, three of seven claimed libraries with no data, three of four
classic-RL arms that did not learn.

Remaining work (~1 focused day, cuts the paper to a third of its claims):
1. Delete every reproducibility claim that is not currently true — the three `make` targets, the
   ~96 GPU-hour estimate, the H100 attribution, `REPRODUCIBILITY_HASHES.json` — **or** actually
   add the targets (they are thin wrappers over already-released Modal scripts and the held-out
   harness, so this is cheap). Do not ship the current text.
2. Rebuild the evidence ledger from **one** named source. Report the true step counts (26, not
   30). Drop every row whose `reward_trace` is shorter than the window the statistic names. Fix
   the duplicated Qwen3-8B-Base row (84.4% / 85.6%) and the Kimi-K2 double count.
3. Regenerate the ZVF pools from checked-in traces excluding every `status=failed` run, and check
   the regenerated file in. If r=−0.769 does not survive exclusion of the failed tool-use cluster,
   say so — that is a better result than the one currently printed.
4. Retitle to what is bound: the analytic ZVF result stated correctly as
   E[ZVF] = E_p[p^G + (1−p)^G] with the Jensen caveat (the plug-in form is off 14× on a
   two-stratum counterexample); the task-confound demonstration; the 5-seed held-out control.
5. Fix `jordan2024benchmarking` ("Jordan, Emma" → Scott M. Jordan), `le2025rlzvp` (ICLR 2026
   acceptance, not preprint), `shao2024deepseekmath` (two dropped mid-list authors, no "and
   others").
6. Regenerate P9's documentation from `run_manifest.tex`. **Do not use P8's main text.**

**Venue class: dataset-benchmark artifact (D&B track) or workshop artifact.** The 4open.science
anonymous clean-machine reproduction remains a gate, but it is downstream of the ledger rebuild —
publishing the repo today would expose the missing `reproduce-main` target, not fix it.

### 3.5 — Rank 5: **Reporting + registry resource** (P5 + P6)

Achievable and cheap on the P5 side (Exhibit 6 + Exhibit 7 + manifest emitter / stackdiff /
registry machinery, ~12pp), but **gated on a human ruling on P6's iter-194 sign flip** (§2.1).
Also gated on the roster's own condition: external entries, extraction-agreement measurement, and
a user decision study — none of which exist and none of which are GPU work, but all of which are
calendar work with external dependencies.

**Venue class: resource / position track.** Not reachable this cycle without external
participants.

### 3.6 — Rank 6: fragments that should not become papers

- **P4** cap-induced non-identifiability note (≤6pp) — real but thin; better as a limitations
  subsection of §3.1 or §3.2 than as a standalone.
- **P3** residue (2–3pp negative, 0.5B synthetic arithmetic, n=3) — thesis negative-results
  chapter.
- **P12** — workshop-short methods + proposal note, once both hollow headlines are deleted and
  the G=4 plateau is made the actual contribution.
- **P7** controller framing, **P8** as a paper, **P10** as a standalone theory note — retire.

### Deduplication warning

The reachable unit count is smaller than it looks, because the same evidence appears in multiple
"papers" under different names:

- the 505-task allocation result: P3's abstract (orphaned, zero body support), P12 §5 (its real
  home), and P2's body (where "505" is a *different* quantity — the p̂∈(0.05,0.95) subset);
- the 17× exhibit: P5 and P6;
- the matched-budget G=2/G=16 panel: P8 and P10 (E-R2b);
- the `1.11e-16` identity: P2 and P12;
- the hardcoded `FALLBACK_ROWS` grid: P3, P8 (graded "B"), P9.

Any merge plan must assign each artifact to exactly one unit and delete it everywhere else.

---

## 4. The GPU option, priced — a decision for the human, not for me

### 4.1 Option A: the preregistered sampler matrix (the current spine candidate)

**Scope:** 92 task-arm-seed rows (23 paired seeds × 4 cells) under amendment A003. The frozen
results contract will not permit the main table to be generated from anything less; the tasking's
64-run / 16-seed framing is superseded and planning to it would guarantee INCONCLUSIVE.

**Price:** ~640–680 GPU-h; ~360 wall-clock hours (~15 days) at K=2 concurrent Colab A100
sessions; **$700–950**. Contingency +4 runs (n=24) ≈ +$35. If the eval-batch lever is frozen
larger than the preflighted batch of 2, this drops to ~380–420 GPU-h, ~9 days, ~$450–520 — but
that choice defines the evaluator and must be hash-bound before row 1.

**What is not in that price, and what makes the calendar estimate optimistic:**

| Blocker | State |
|---|---|
| Preflight gate | `blocked`. 3 of 4 mixed-update seams missing (gsm8k/contrast, math500/contrast, math500/grpo_g8). Zero confirmatory launches permitted until it regenerates green. |
| Confirmatory runner | **Does not exist.** All four launchers are preflight-only (`max_steps=1`, `heldout_n=8` hard-coded) and do not emit the required telemetry (`policy_ratio_q05_q50_q95`, `clip_fraction_by_advantage_sign`, both KL fields, `parser_disagreement`, `two_sample_false_homogeneity`). Writing it *and hash-binding it via a prospective amendment before row 1* is unpriced engineering. |
| Eval-batch freeze | Undecided; must be inside that amendment. |
| Multi-provider parallelism | Non-conformant. The stack fingerprint binds provider identity, Colab CLI version and accelerator; `verify_preflight_matrix.py` requires exactly one fingerprint across receipts. A split matrix fails hash validation. Colab-only unless an A004-style amendment lands first. |
| HF Jobs | HTTP 402, credits exhausted. |
| Kaggle | Allocates P100 despite A100 request → no bf16 → `remote_preflight` fails closed. ~30 GPU-h/week quota against a ~640 GPU-h need. Excluded outright. |
| GCP | Hash-bound 90-minute Spot ceiling against 5.6–8.8 h single-shot runs, no exact-resume in the bound stack. Can only ever produce `failed_infrastructure` rows. |

**The scientific risk on top of the engineering risk:** the GSM8K contrast arm is *structurally
unlikely* to observe the mixed-update seam at all — Qwen3-8B non-thinking is accurate enough on
GSM8K that G=2 groups come out all-correct (observed 3/3 seeds). That is a design problem, not an
infrastructure problem, and it threatens two of the four cells.

**The preregistered downside:** boundary-crossing interval, failed power receipt, or any
incomplete cell yields **INCONCLUSIVE** — not equivalence, not failure. There is a real,
preregistered, non-trivial probability of spending $700–950 and 3–6 weeks of calendar to produce
no publishable positive claim. Honest calendar including runner engineering, seam recovery, and
Colab allocation variance: **3–6 weeks, not 15 days.**

### 4.2 Option B: finish P11's own preregistered verdict (the cheap GPU spend)

At the observed DAPO paired sd of 0.00875, **n=9 yields MDE80 = 0.00934 ≤ the 0.01 margin**. That
is **2 additional A100 runs** (one GRPO seed + one DAPO seed, ~7.6 A100-hours, roughly $10) to
convert DAPO from INCONCLUSIVE to a genuine preregistered DISAPPEARS. Because the sd is estimated
on 7 df, a safe design targets n=12 for DAPO: ~10 extra runs, **~38 A100-hours, roughly $45**.
Full four-arm coverage needs n = 9/11/13/15 (DAPO/GSPO/AERO/Dr.GRPO) plus matched GRPO seeds —
about 23 extra runs, ~104 A100-hours, roughly $125.

This spend upgrades the **recommended spine** rather than starting a new paper, and it runs on
infrastructure that already produced 40 completed units.

### 4.3 The decision, stated as a decision

**I am not taking this call.** The three coherent positions:

1. **Zero GPU.** Ship §3.1 (P11 reframed) as the canonical manuscript to TMLR; ship §3.2 as a
   workshop short. Cost: ~3 days of editing. Ceiling: a good methods/repro paper and a good
   negative-result note. No empirical capability claim, ever, from this portfolio.
2. **Minimal GPU (recommended if any GPU spend happens).** Option B at n=12 for DAPO — ~38
   A100-h, ~$45, ~2 days of wall clock on proven infrastructure. Converts the spine's headline
   from "all four INCONCLUSIVE" to one preregistered positive verdict plus three honest
   inconclusives. Then ship as in position 1.
3. **Full GPU.** Option A. Requires first authorizing unpriced runner engineering, a prospective
   protocol amendment, and acceptance of a preregistered chance of INCONCLUSIVE. Only worth it if
   the goal is specifically a NeurIPS main-track empirical claim and 3–6 weeks plus ~$950 is
   acceptable against that.

Positions 2 and 3 are not exclusive, but 3 should not start before the gate is green, the runner
exists and is hash-bound, and the GSM8K-contrast seam problem has a prospective amendment.

---

## 5. Ready / not-ready, against task_spec success criteria

| Success criterion | State | Gap |
|---|---|---|
| One canonical manuscript and one canonical PDF | **Not met** | 12 active roots, 486 pages. §3.1 designates the spine; the other 11 must be demoted, merged or retired in writing. |
| Claim ledger with source paths, hashes, status, allowed inference | **Partially met** | `claim_to_run_table.md` exists and carries 18 rows across P1–P8 — but **zero rows for P10 and zero for P12**, and P1-C2 carries a stale caveat the verifier lifted. Every binding for those papers had to be done by ad-hoc recomputation. |
| No unresolved contradiction in any headline result | **Not met** | Live in the built PDFs today: P11's "Disappears Verdict" figure; P11's pilot at 2 seeds vs the ledger's n=1; P9's frontier table disagreeing with all three of its named sources; P2's abstract aggregating undisclosed versions of scattered disclosures; P3's flatness claim contradicting its own arithmetic; P10's non-claims paragraph contradicting its own body. |
| No causal or capability claim supported only by tests, receipts, or toy fixtures | **Not met as-is; met after the §3 edits** | P7's E3 (two-digit addition, n=20 held-out, 10 steps, n=2 seeds) currently carries an intervention claim; P3's mechanism rests on toy arithmetic at 0.98 accuracy; P12's two headlines are internal-consistency artifacts. |
| Reproducible build, no undefined references or clipped content | **Met mechanically, failed substantively** | All 12 roots build clean (latexmk exit 0, no undefined refs or citations). But P7 has 13 multiply-defined labels from 12 duplicated content-free TikZ figures, P8 has 16 from 8 figures pasted three times, and P10 ships 10 placeholder figures whose captions describe plots that do not exist. |
| Artifact verification commands pass from the documented environment | **Not met** | P9's advertised `make reproduce-main` does not exist in the Makefile; P12's README instructs compiling a `main.tex` that is not git-tracked; P9's declared `experiments/analysis/v5/` provenance directory does not exist. |
| Independent reviewer can summarize method, result and boundary accurately | **Not met for 10 of 12** | Only P11 and P1 have abstracts whose claims match their bodies. |
| Venue policy, anonymity, overlap, dual-submission checks documented | **Met** | Documented in `PUBLICATION_READINESS.md`. Flagship TMLR route stays blocked while NeurIPS 36320 is live. Re-run the overlap check for P11 specifically — it currently cites the flagship as a companion. |

### Verdict: `ready_after_edits`

Not `ready` — three headline contradictions are live in the recommended spine's own PDF. Not
`not_ready_needs_runs` — nothing on the shortest path requires a GPU. Not `no_go` — the evidence
supports real, bounded, checkable claims.

### Stop-condition assessment

**The task_spec stop condition triggers.** The checked-in evidence cannot support a NeurIPS or
TMLR **empirical capability** claim: not "training improved GSM8K", not "ZVF predicts failure",
not a sampler cost saving, not a controller advantage, not a framework or estimator ranking. Every
candidate for such a claim failed on rule 2 (partial/simulated cells in aggregates), rule 4
(non-significance sold as equivalence), rule 5 (heterogeneous pooling), or rule 6 (numbers with no
resolvable source).

The narrower paper that **does** survive, per the stop condition's own instruction:

> **A single-stack preregistered audit protocol for RLVR variant claims, with an
> MDE-bounded cost result for dynamic sampling.** (§3.1, from P11.)
>
> Methods + reproducibility class. TMLR. Its contribution is the protocol, the exact
> paired-t power with registered four-test BH correction, the reconciled 40 arm-seed units,
> the provably identical held-out prompt sequence, and one directional cost-anchored result:
> DAPO eliminates zero-variance groups entirely (ZVF 0.693 → 0.000) at 3.61× the rollouts
> and 1.44× wall clock, with its held-out gain bounded above by +0.0068 at 95%. Its
> non-claims are stated at the same prominence: no published delta was registered, so
> survival fractions were never computed; DISAPPEARS was unreachable a priori at n=8; and
> replay perturbs a held-out score by up to 0.004, which exceeds the DAPO point difference.

The second, independent negative-results paper is §3.2 (the ZVF sampling-model falsification
note). The third, if calendar allows, is §3.3 (failed identifiability of RL post-training scale).

No no-go is warranted. But the honest framing of this portfolio is: **it is a strong
measurement-and-audit program that has been repeatedly written up as an empirical-results
program.** The shortest real path is to stop doing that.

---

## 6. Recommended sequencing

1. **Day 0 (human, ~30 min):** rule on P6's iter-194 sign flip (§2.1) and on whether the 17×
   exhibit is retracted from both P5 and P6 (§2.5). Both are integrity calls, not editing calls.
2. **Days 1–2:** execute §3.1's nine items. Rebuild P11's PDF. Re-run the overlap check against
   the live NeurIPS submission.
3. **Day 3:** freeze the spine PDF hash; regenerate the claim ledger with P10 and P12 rows and
   the lifted P1-C2 caveat; record the demotion of the other 11 roots in writing.
4. **Days 4–6 (parallel, optional):** execute §3.2. This is mostly deletion and can run
   independently.
5. **Decision gate:** the human picks position 1, 2 or 3 from §4.3. Nothing in steps 1–4 depends
   on that choice, and nothing in steps 1–4 should wait for it.
