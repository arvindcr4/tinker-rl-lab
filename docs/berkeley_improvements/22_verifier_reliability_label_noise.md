# 22 — Verifier-Reliability Audit: label-noise robustness of the verifiable-reward claims

**Source lecture:** Berkeley F25 *Agentic AI* L4 — **Jiantao Jiao (NVIDIA), "Post-Training
Verifiable Agents"** (SWE-bench Verified + BrowseComp).
**Target:** A3 (post-training / verifiable-reward science) + A1 (statistical rigor).
**Status:** validated · **Pillar:** B-F25 · **Ledger row:** 22
**Evidence:** 5/5 pre-registered hypotheses DECISIVE.

## Verified citations (2026-07-04, WebFetch on arxiv.org/abs)

- **SWE-bench** — Jimenez, Yang, Wettig, Yao, Pei, Press, Narasimhan, *"SWE-bench: Can
  Language Models Resolve Real-World GitHub Issues?"* **arXiv:2310.06770**, **ICLR 2024**.
  (The **Verified** subset is OpenAI's 2024-08 human-audited 500-task clean split, created
  precisely because the original verifier was noisy — ~38% of tasks under-specified or with
  broken/insufficient tests.)
- **BrowseComp** — Wei, Sun, Papay, McKinney, Han, Fulford, Chung, Passos, Fedus, Glaese,
  *"BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents,"* **arXiv:2504.12516**,
  **2025**. Design principle: **easy-to-verify, hard-to-solve** (short verifiable answers) —
  the low-false-positive end of the verifier-reliability spectrum.

## The idea

Every claim in the four TinkerRL-Bench papers rests on a **verifiable reward** — a binary
"is this rollout correct?" signal. Jiao's lecture (and the very existence of SWE-bench
**Verified**) makes the point that **a verifiable reward is only as trustworthy as its
verifier**: a verifier with false positives (declares a wrong rollout correct) or false
negatives (declares a correct rollout wrong) systematically biases every downstream number.
Rows 09/15/17/19 used the verifiable-reward *framing* but **never tested robustness to a
noisy verifier**. This row does.

Model the verifier as an **asymmetric binary label-noise channel** with false-positive rate
`alpha` and false-negative rate `beta`:

```
p_obs = alpha + (1 - alpha - beta) * p_true
```

and propagate it through the headline claims on **real** in-repo data.

## Data (all real, in-repo)

| source | role |
| --- | --- |
| `experiments/results/samestack_ppo_grpo.json` | flagship GRPO≈PPO, 5 seeds × 2 algos, heldout + per-step |
| `experiments/results/group_size_effect.tsv` | heldout accuracy per group size G ∈ {2,4,8,16} |
| `experiments/results/berkeley/verifiable_zvf_percell.tsv` | real per-cell `(n_correct, G)` group counts for ZVF |

## Hypotheses & results — 5/5 DECISIVE

**H1 — Flagship equivalence is point-estimate-invariant under verifier noise.** Monte-Carlo
corrupting each seed's heldout through the channel over a 20-cell (α,β) grid, the paired
GRPO−PPO delta **stays at −0.002 in every cell** (max |mean delta| < 0.002). Because the same
verifier acts on both algorithms, verifier noise is a **common-mode error that cancels in the
paired difference** — it can never manufacture a false GRPO≠PPO gap. **However**, the
false-negative channel β inflates per-seed sampling variance, **widening the TOST equivalence
bound 3.7×** (0.029 → 0.109 at β=0.20). *A noisy verifier costs the equivalence claim its
tightness (power), not its direction* — this is exactly why SWE-bench **Verified** drives β
down. DECISIVE.
→ `verifier_reliability_h1_flagship_invariance.tsv`

**H2 — Measured effects attenuate by `(1−α−β)` → our numbers are conservative lower bounds.**
The observed group-size effect equals `(1−α−β)` times the true effect (empirical = analytic to
1e-9). Verifier noise **only shrinks measured effects toward zero, never inflates them** — so
every reported verifiable-reward effect in the benchmark is a conservative *lower* bound on the
truth. (This is an identity by construction; its value is the corollary, not a discovery.)
DECISIVE.
→ `verifier_reliability_h2_attenuation.tsv`

**H3 — The closed-form ZVF-under-noise model is validated.** For a group with `k` true-correct
of `G`, observed all-pass w.p. `(1−β)^k·α^(G−k)` and all-fail w.p. `β^k·(1−α)^(G−k)`. The
analytic `ZVF_obs` matches Monte-Carlo corruption of the **real** per-cell group counts to
**max |dev| = 0.0016** across five (α,β) settings. The corruption model is trustworthy.
DECISIVE.
→ `verifier_reliability_h3_zvf_model_validation.tsv`

**H4 — α-vs-β dominance is regime-dependent (crossover at p=0.5) → SWE-bench-Verified does not
fix the bigger threat for sparse RL rewards.** `bias(p) = α(1−p) − βp`, so `|∂bias/∂α| = 1−p`
and `|∂bias/∂β| = p`: **α (false positives) dominates for p < 0.5, β (false negatives)
dominates for p > 0.5**. The flagship heldout (p = 0.991) is **β-dominated** — SWE-bench-Verified's
false-negative cleaning is the right fix there. But a **sparse RL step reward** (measured
p = 0.113 on real per-cell data) is **α-dominated**: Verified removes false *negatives*, yet the
larger bias for sparse verifiable RL is the false *positive* (a wrong rollout the verifier waves
through). **The two real regimes straddle the 0.5 crossover** — so the standard "Verified fixes
it" intuition is regime-specific and *incomplete for the sparse-reward setting that RL lives in*.
DECISIVE.
→ `verifier_reliability_h4_alpha_beta_dominance.tsv`

**H5 — Collapse-signal masking: a modest FP rate deflates ZVF sharply.** A fully-collapsed
regime (all groups 0-correct, ZVF_true = 1) reads `ZVF_obs = (1−α)^G + α^G`. At **α = 0.05,
G = 8** this is **0.663 — a 34pp deflation**. The Pillar-2 ZVF collapse detector therefore
**under-reports collapse severity** whenever the verifier admits false positives: a false
positive turns an all-fail group into a mixed group, erasing the zero-variance signal. Reported
ZVF is a *biased-low* estimate of true reward sparsity. DECISIVE.
→ `verifier_reliability_h5_zvf_collapse_masking.tsv`

## Cross-pillar bridges

- **Row 20/21 (Sida Wang error bars):** H1's "bias vs variance" split mirrors row 20's
  regime-dependent DEFF — verifier noise, like seed clustering, does not move the *point
  estimate* of the flagship null but *widens the honest interval*. Two independent noise
  sources (seed clustering, verifier reliability) both attack the equivalence claim's *power*,
  never its *direction*.
- **Row 11 (eval-protocol hardening, Pillar-2 ZVF):** H5 adds a verifier-noise floor to the ZVF
  risk metric — the magnitude channel that row 11 decomposed is deflated by α, so the ZVF risk
  ranking is a lower bound.
- **BrowseComp vs SWE-bench:** BrowseComp's "easy-to-verify" design (low α) is the safe end of
  H4's spectrum; the original SWE-bench (noisy tests, high β and non-trivial α) is the unsafe
  end that motivated Verified.

## Go / no-go

**GO — validated.** Paper-facing (one-sentence robustness note per pillar, plus a short
appendix): *"All headline verifiable-reward numbers are reported under a perfect verifier;
under an asymmetric verifier-noise channel (FP α, FN β) the flagship GRPO≈PPO point estimate is
invariant — verifier noise is common-mode and cancels in the paired difference — while measured
effects attenuate by (1−α−β) (conservative lower bounds) and the TOST equivalence bound widens
up to 3.7× with the FN rate. For the sparse-reward regime the dominant bias is the false
positive (p < 0.5), which SWE-bench-Verified-style false-negative cleaning does not address; and
a 5% verifier FP rate deflates a fully-collapsed ZVF by 34pp (G=8), so the ZVF collapse detector
is a conservative lower bound on collapse severity."*

## Reproduce

```
python3 scripts/berkeley/verifier_reliability_audit.py
# -> experiments/results/berkeley/verifier_reliability_{h1..h5}*.tsv
#    + verifier_reliability_summary.json   (5/5 DECISIVE)
```
