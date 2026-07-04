# P4 Ablation Gap Report — Length Bias & Held-Out Generalization

Contract: `research_prompts/revision/ablation-gap-finder.md` (pre-submission ablation reviewer).
Paper: `paper/paper_P4_length_bias.tex` + `paper/sections/p4_*.tex`, `length_bias*.tex`,
`frontier_synthesis_length_bias.tex`. All claims below were checked against
`experiments/results/` and `experiments/FRONTIER_EXPERIMENT_BACKLOG.md`.

---

## Placeholder fill (per contract)

**Main claim (one sentence).** In the short-horizon GSM8K-CoT regime, GRPO and Dr.GRPO
produce statistically similar held-out gains and *neither inflates length* (mean completion
length drifts ~193 → ~188 tokens); the verbosity-trap signature is absent, and this null is
the *expected* outcome "when length is already controlled" (p4_abstract.tex, p4_results_intro.tex).

**Method components (design choices that carry the claim).**
- GRPO vs Dr.GRPO (per-response 1/L normalization removed), paired seeds.
- Two task cells: Qwen2.5-0.5B arithmetic (40 steps, n=5/algo, `MAX_NEW=10`) and
  Qwen2.5-1.5B-Instruct GSM8K-CoT (30 steps, n=3/algo, **`MAX_NEW=200`** —
  `experiments/modal/modal_drgrpo_gsm8k_cot.py:45`).
- Per-step aggregate logging only (`mean_reward`, `mean_comp_len`; no per-completion lengths).
- Held-out pre/post eval (n=200) generated **under the same 200-token cap** (same `gen_batch`,
  lines 103–120 of the modal script).
- Diagnostic stack: Spearman triples, trap-onset windows, decile E[R|L], dL/dR mechanism
  regressions, ZVF coupling/mediation, drift/forecast, iters 28–136 time-series analyses.

**Current ablation coverage (verified in `experiments/results/`).**
- Algorithm ablation: GRPO vs Dr.GRPO, both cells, paired bootstrap CIs
  (`drgrpo_vs_grpo.json`, `drgrpo_gsm8k_cot_full.json`, `length_bias*.tsv`).
- Task/scale ablation: easy 0.5B vs hard 1.5B cell.
- Horizon (partial): a 100-step arithmetic crossval (`arithmetic_metrics.jsonl`,
  `length_bias_crosval.tsv`) — which the paper itself discounts because the model sits at the
  5-token truncation cap ("the trap is mechanically impossible in that regime", length_bias.tex).
- ~30 re-analysis iterations (iter 24–136) — all computed on the *same* per-step traces.
- Backlog check: A1 (PCD/LARQ, Pillar 2) is the only executed item; A3 (eval-time truncation
  sweep), A4 (CLMP mediation), C1 (length-confounded regime, caps 512–1024), C3
  (memorization ladder) are proposed, **not run**. No result file anywhere contains a
  GRPO-vs-Dr.GRPO cell trained or evaluated with a completion cap above 200 tokens.

---

## 1) Missing ablation (the most dangerous one)

**A generation-cap (censoring) ablation: rerun the GSM8K-CoT GRPO vs Dr.GRPO comparison with
an uncensored completion cap (MAX_NEW = 512), everything else identical.**

The paper's entire "hard task" cell — the basis of the headline "neither inflates length"
and of essentially every Pillar-4 mechanism analysis — was trained *and* evaluated under a
**200-token hard cap that the pre-RL policy already saturates**. Verified from
`drgrpo_gsm8k_cot_full.json` step logs:

| run | step-0 mean len | max mean len | cap |
|---|---|---|---|
| GRPO s42 | 195.9 | 198.3 | 200 |
| GRPO s123 | 189.8 | 196.3 | 200 |
| GRPO s456 | 196.1 | 197.6 | 200 |
| Dr.GRPO s42 | 195.6 | 199.0 | 200 |
| Dr.GRPO s123 | 189.2 | 197.8 | 200 |
| Dr.GRPO s456 | 196.1 | 198.7 | 200 |

A *mean* of ~196 under a hard max of 200 implies the bulk of individual completions are
truncated at the cap (e.g., if non-capped completions averaged 150 tokens, the capped
fraction at step 0 is ≥ ~92%). Three consequences:

1. **"Neither inflates length" is unfalsifiable by construction.** Length inflation above
   200 tokens is mechanically impossible; the observed 193 → 188 "compression under
   reinforcement" is the only direction length *can* move. This is precisely the criticism
   the paper levels at its own 100-step arithmetic crossval ("the model is at the truncation
   cap... the trap is mechanically impossible") — but never applies to the GSM8K cell, and
   the 200-token cap is **disclosed nowhere in the P4 sections** (grep: no mention of the cap,
   `MAX_NEW`, or censoring outside the arithmetic paragraphs).
2. **The negative ρ(len, reward) core is plausibly a censoring artifact.** Capped completions
   are cut off before the final boxed answer → reward 0 → steps with more truncation have
   both higher mean length and lower reward. The decile E[R|L] monotonicity, the dL/dR
   regressions, the mediation and drift analyses (iters 16–136) all inherit this artifact.
3. **The held-out equivalence is cap-confounded too.** Pre/post eval (0.20 → 0.26 GRPO,
   0.21 → 0.26 Dr.GRPO) is generated under the same 200-token cap, so "statistically similar
   held-out gains" is measured in a regime where the policies' length behavior is clipped.
4. **The framing becomes circular.** The abstract's defense — the null is expected "when
   length is already controlled" — reads as a property of the task; in fact length is
   controlled *by the authors' sampler configuration*.

## 2) Why reviewers will ask for it (reviewer framing)

Likely NeurIPS reviewer wording:

> "The headline claim that neither GRPO nor Dr.GRPO inflates length is measured under a
> 200-token generation cap that the pre-RL policy already saturates (step-0 mean length
> 189–196 of 200; released code, `modal_drgrpo_gsm8k_cot.py`, `MAX_NEW=200`). Length
> inflation is therefore impossible by construction, and the negative length–reward coupling
> that drives Sections 4.x–4.y is exactly what right-censoring at the cap predicts
> mechanically. The authors themselves dismiss their 100-step arithmetic run on identical
> grounds. Until the comparison is repeated with an uncensored cap (512–1024 tokens, as the
> authors' own frontier synthesis specifies), none of the Pillar-4 conclusions — no
> inflation, no verbosity trap, GRPO ≈ Dr.GRPO held-out — are informative. Reject/major revision."

This is highest-risk because it attacks **validity, not scope**. The other known weaknesses
(30–40-step horizon, n=3 seeds, 1.5B scale vs the Qwen3-8B headline) are disclosed and
scoped repeatedly, and reviewers tolerate disclosed limits. The cap censoring is
*undisclosed*, discoverable in one line of released code, contradicted by the paper's own
reasoning about the arithmetic cap, and it converts the paper's negative-control framing
into circularity. Every one of the ~30 analysis iterations reuses the same censored traces,
so no amount of existing re-analysis defends against it.

Runner-up gaps, and why they rank below:
- **200+-step horizon run** (the paper's own stated falsifier): disclosed ~6 times and
  explicitly scoped; also dominated by this gap — extending horizon at cap 200 still cannot
  show inflation, so the cap ablation must come first (ideally combined: cap 512 × 100 steps).
- **Backlog A3 eval-time truncation sweep** ("one truncation sweep away", not executed):
  embarrassing to leave unrun, but it evaluates checkpoints *trained* under the 200 cap, so
  it cannot rescue the training-dynamics claims either way.
- **Dr.GRPO at Qwen3-8B scale**: a claims-consistency issue (the abstract mixes model
  families), but expensive and less likely to be the stated reject reason.

## 3) Minimal way to run it (cheapest credible version)

**Design.** One-line config change in the existing script
(`experiments/modal/modal_drgrpo_gsm8k_cot.py`): `MAX_NEW: 200 → 512`. Same model
(Qwen2.5-1.5B-Instruct), same 30 steps, same seeds {42, 123, 456}, both algorithms
→ 6 new runs; the existing cap-200 runs are the other arm of the ablation. Add two log
fields the paper already flags as "one log-field away": per-completion lengths (gives
within-step std(len) and the herding channel) and per-step fraction-at-cap.

**Compute cap (per contract's resource-limit failure mode).** The existing 30-step run took
803.9 s (seed 42, `drgrpo_gsm8k_cot_full.json`). Generation dominates and scales ~linearly
with tokens: ~2.5× → ~35 min/run → **~3.5–4 GPU-hours total for all 6 runs** on the same
Modal single-GPU setup. Optional strengthening at ~3× that cost: extend to 100 steps,
simultaneously closing most of the horizon objection (~12 GPU-hours).

**Deliverables (reuse existing pipelines).** Re-emit the exact headline tables on the
cap-512 arm: fraction-at-cap per step, ρ(step,len), ρ(step,reward), ρ(len,reward), trap
flag, decile E[R|L], paired bootstrap Dr.GRPO−GRPO deltas, held-out pre/post + McNemar
(evaluated at cap 512). Scripts `scripts/length_bias*.py` run unchanged on the new traces.

## What result would change the paper's conclusion

- **Conclusion flips** if, with headroom to 512, GRPO shows ρ(step,len) > 0 with flat or
  falling reward (or simply a length drift that Dr.GRPO attenuates by the frontier rule's
  ≥30%): then the verbosity trap *does* engage in this regime and was previously masked by
  the cap. The abstract's three headline statements — no inflation, no trap signature,
  "null expected because length is already controlled" — all invert, and the paper becomes
  (weak) evidence *for* Dr.GRPO rather than a scoped null. The recommendation "we do not
  recommend Dr.GRPO as a necessary length-bias mitigation on these task scales" would have
  to be withdrawn.
- **Conclusion survives and is strengthened** if mean length stays ~185–195 tokens despite
  2.5× headroom and the diagnostic stack reproduces (negative trends, no trap flag,
  held-out equivalence): "length is already controlled" becomes a demonstrated property of
  the task/model rather than of the sampler config, the censoring objection is defused with
  data, and the negative-control framing becomes reviewer-proof. Either outcome, the
  ablation converts the paper's most dangerous undisclosed confound into a result.

**Handoff (per contract workflow):** if this arrives as actual reviewer feedback, paste the
reviewer wording block from item 2 into `{{reviewer_comments}}` of the Rebuttal Strategy
Builder, together with the 6-run cap-512 result table as the evidence artifact.

---

*Evidence trail:* `experiments/modal/modal_drgrpo_gsm8k_cot.py` (MAX_NEW=200, shared
train/eval generator); `experiments/results/drgrpo_gsm8k_cot_full.json` (step-0 mean lengths
189–196, per-run maxima 196–199, elapsed 804 s); `paper/sections/length_bias.tex` (arithmetic
cap paragraphs, "mechanically impossible"; no GSM8K cap disclosure);
`experiments/FRONTIER_EXPERIMENT_BACKLOG.md` (A3/A4/C1 unexecuted; no cap ablation listed).
