# 22 — Chain-of-Verification (factored vs joint) ≡ RLOO leave-one-out vs naive GRPO baseline

**Source lecture:** SP25 L2 — Jason Weston (Meta) — *Learning to Reason*.
**Untried technique:** **Chain-of-Verification (CoVe)** — the third technique in
Weston's lecture, flagged in the brief but never mined (row 02 took DPO + Iterative
RPO; CoVe was left on the table).
**Target:** A3 (post-training science) + **Pillar 3** (group-size / GRPO baseline geometry).
**Status:** prototyped → **4/6 paper-facing, 6/6 DECISIVE**.

## Verified citations (both checked 2026-07-04 via arXiv Atom API)
- **CoVe** — Dhuliawala, Komeili, Xu, Raileanu, Li, Celikyilmaz (Jason Weston senior
  author). *Chain-of-Verification Reduces Hallucination in Large Language Models.*
  **arXiv:2309.11495**, published 2023-09-20, EACL 2024 Findings.
- **RLOO** — Ahmadian, Cremer, Gallé, Fadaee, Kreutzer, Pietquin. *Back to Basics:
  Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs.*
  **arXiv:2402.14740**, published 2024-02-22, ACL 2024.

## The mapping
CoVe's central design choice is **factored** verification (the verifier answers
verification questions *independently*, without attending to the draft) beating
**joint** verification (the verifier conditions on the draft and so repeats its
errors). Cast onto GRPO's advantage baseline for a group of `G` rollouts with `k`
correct (binary reward `Rᵢ∈{0,1}`):

| CoVe | GRPO baseline | formula |
| --- | --- | --- |
| **joint** (self-referential) | naive group mean | `bᵢ = k/G` — includes rollout *i* in its own baseline |
| **factored** (independent) | RLOO leave-one-out | `b₋ᵢ = (k−Rᵢ)/(G−1)` — excludes rollout *i* |

This is exactly the RLOO estimator (Ahmadian et al.): the leave-one-out baseline
decorrelates the baseline from the sample it scores, the same way CoVe's factored
verifier decorrelates the check from the draft.

## Exact per-rollout algebra (the load-bearing identity)
```
A_naive_i = Rᵢ − k/G
A_LOO_i   = Rᵢ − (k−Rᵢ)/(G−1) = (G/(G−1)) · A_naive_i          [EXACT]
self-confirmation bias:  correct  β⁺ = (G−k)/(G(G−1))
                         wrong    β⁻ = −k/(G(G−1))
```
The LOO ("factored") advantage is a **pure scalar rescale** `G/(G−1)` of the naive
("joint") advantage — *for a self-consistent verifier* (verifier == reward parser).
Direction is identical; only the effective step size changes, and the change vanishes
as `G→∞`. **CoVe's genuine value appears only when the verifier is *independent* of the
reward parser** (H6) — precisely the paper's factored-beats-joint mechanism.

## Prototype & measured result
`scripts/berkeley/cove_factored_baseline.py` on **600 real GSM8K rollout groups**
(Qwen3-8B, 3 seeds × 200 prompts, native G=8, **exact hypergeometric subsampling** to
G∈{2,4,8} — no Monte-Carlo). Outputs in `experiments/results/berkeley/cove_*`.

| H | claim | result | verdict |
| --- | --- | --- | --- |
| **H1** | self-confirmation bias identity `β⁺=(G−k)/(G(G−1))` holds exactly | max abs err **8.3e-17** | **DECISIVE** |
| **H2** | per-rollout bias decays as `1/G` (CoVe matters most at small G) | log-log slope **−1.0000**; bias 0.1266 (G2) → 0.0317 (G8) | **DECISIVE** |
| **H3** | LOO leaves ZVF **exactly invariant** (no tail recovery — unlike STaR row-21) | `zvf_naive ≡ zvf_loo` to 1e-12 at every G | **DECISIVE** |
| **H4** | lone-outlier catch: single wrong rollout in a k=G−1 group gets LOO adv **exactly −1** (naive: −(G−1)/G); amplification `G/(G−1)` | verified at G=2,4,8 | **DECISIVE** |
| **H5** | factored ≡ pure step-size rescale `G/(G−1)` of joint (verifier==reward) | `rescale_emp≡G/(G−1)`, max err **2.2e-15** | **DECISIVE** |
| **H6** | genuine CoVe value = verifier **decorrelation**: residual-beyond-rescale is 0 at verifier-error e=0, grows monotonically with e | resid 0.000 (e=0) → 0.028 (e=.05) → 0.111 (e=.20) | **DECISIVE** |

**6/6 DECISIVE.**

## Interpretation (three paper-facing consequences)
1. **The joint-vs-factored baseline is under-identified up to a scalar** on a
   self-consistent verifier (H5) — a Pillar-3-native echo of Pillar-1's "estimator
   doesn't matter, stack does" (frontier synthesis): swapping GRPO's group-mean
   baseline for RLOO changes only the effective step size by `G/(G−1)`, not the update
   direction. At the G=8 our stack uses, that is a **14.3%** step inflation; at the
   large G reported in RLVR papers it is negligible — a clean explanation for why
   "RLOO vs GRPO" ablations rarely move the needle at scale.
2. **CoVe-baseline is ZVF-invariant and recovers *none* of the zero-advantage mass**
   (H3, `cove_star_contrast.tsv`) — the exact complement of STaR/rejection-sampling
   (row 21), which recovered the all-correct tail `ZVF_hi`. Baseline choice cannot
   manufacture within-group contrast; only a different *filter* (STaR) or a different
   *verifier* (H6) can. This sharpens the ZVF pillar's causal story.
3. **CoVe's real mechanism is verifier decorrelation, not baseline arithmetic** (H6).
   The `G/(G−1)` rescale is free; the paper's factored-beats-joint gain requires the
   verifier to make *independent* errors from the policy — operationalised here as
   reward-parser disagreement rate `e`, where the residual-beyond-rescale grows
   linearly (0.055 per 0.10 of e). This is the concrete lever a paper section would
   claim: **an independent verifier buys signal a leave-one-out baseline cannot.**

## Cross-pillar bridges
- **Row 21 (STaR/Lean-STaR)**: complementary — STaR recovers `ZVF_hi`; CoVe-baseline
  is ZVF-invariant. Together they bracket what changes-of-estimator can and cannot do.
- **Row 12 / 16 (CDH)**: the `G/(G−1)` step-size scalar is the RLOO analogue of the
  critic-degeneracy result — a nominal algorithmic change that reduces to a scalar
  once the stack is fixed.
- **Row 17 (Self-Debug / Huang no-self-correct)**: H6's "independent verifier only"
  finding is the RL-baseline restatement of Huang et al.'s "intrinsic self-correction
  fails" — a *joint* (self-referential) check adds nothing beyond a rescale.

## Go / no-go
**GO — one-sentence P3 stabilizer + one appendix identity.** Add to the Pillar-3
same-stack narrative: *"Replacing GRPO's group-mean baseline with the RLOO
leave-one-out baseline is, for binary rewards, an exact `G/(G−1)` rescale of every
advantage (App. X); it leaves ZVF invariant and its effect vanishes at large G,
so RLOO-vs-GRPO is under-identified up to a per-group step size."* No new experiment
required — the identity is analytic and reproduced to machine precision on real data.

## Reproduce
```
python3 scripts/berkeley/cove_factored_baseline.py
# -> experiments/results/berkeley/cove_{baseline_identity,verifier_noise,
#     outlier_catch,star_contrast}.tsv + cove_summary.json
```
