From: arvindcr4 <[your-academic-email]>
To: [Editor-in-Chief / Track Chair]
Subject: Submission: Reward Contrast, Not Algorithm Labels — A Claim-to-Evidence Audit of Group-Relative RL (v0.9-pre-submission)

Dear [Editor / Track Chair],

Please find attached our manuscript, "Reward Contrast, Not Algorithm Labels:
A Claim-to-Evidence Audit of Group-Relative Reinforcement Learning for LLMs,"
for consideration at [Venue Name]. The paper is an engineering diagnostic,
not a benchmark or leaderboard result. We study when critic-free group-relative
RL has a usable learning signal, and how training-set reward fails to
transfer to held-out capability.

=== Why this fits [Venue Name] ===

The paper addresses [one of the venue's stated themes — e.g., "rigorous
empirical methodology for LLM post-training" / "audit-style contributions
that scope claims to evidence" / "production-scale RL infrastructure"].
Our diagnostic framework — Zero-Variance Fraction (ZVF) and Gradient
Utilization (GU) — gives practitioners a pre-screen triage metric for
collapsed runs, paired with explicit held-out capability controls on a
GSM8K 200-example slice (82.0% → 83.3%, p = 0.26; not significant).

=== What changed in v0.9-pre-submission ===

This revision adds:

1. **LoRA-vs-full-FT controlled experiment (E2, new).** The Tinker platform
   used throughout the audit corpus is LoRA-only, which left open whether
   the LoRA restriction is itself a confound. We address this with an
   out-of-platform controlled experiment on Qwen/Qwen3-4B-Instruct-2507
   (3 seeds, matched task/data/seed-reset/held-out N = 50). LoRA mean
   heldout_delta = +0.160 (std 0.020) vs. full-FT +0.100 (std 0.020);
   gap = +0.060. The full-FT arm shows lower mean ZVF (0.758 vs. 0.954),
   consistent with theory T2: in saturation regimes the parameter-efficient
   path is also the more stable one. This defends the LoRA-only Tinker
   constraint as not handicapping the audit corpus.

2. **Audit suite repair.** Four audit scripts had false positives that
   suppressed legitimate scope-language edits. These are fixed and the
   full 13-audit suite now passes with zero issues.

3. **Upstream integration.** The repo at github.com/arvindcr4/tinker-rl-lab
   is now synchronized with the canonical upstream at pes-llm-research/
   tinker-rl-lab; the canonical NeurIPS integrator and the Qwen3.5-4B
   Tinker swap are integrated.

=== Scope and limits (recap for reviewer clarity) ===

- All numbers are training-set reward unless explicitly flagged.
- HumanEval is a 50-problem subset, not the full canonical harness.
- The held-out GSM8K 200-example lift is small and not significant;
  we report it as a scope bound, not as a capability claim.
- Algorithm-label comparisons (PPO/GRPO/DPO) are backend-, sampler-,
  reward-, LoRA-, and checkpoint-confounded; we treat the labels as
  insufficiently specified experimental treatments.

=== Reproducibility ===

- Code, training scripts, evaluation scripts, prompt templates, run logs:
  https://github.com/arvindcr4/tinker-rl-lab (tag v0.9-pre-submission)
- W&B project: tinker-rl-lab-world-class (368 runs)
- ZVF-Program out-of-platform follow-ups: zvf-colab-experiments (E1-E7)
- 13/13 paper/submission audits pass on this tag.

=== Suggested reviewers ===

- [Reviewer 1: GRPO methodology / low-budget RL for LLMs]
- [Reviewer 2: ZVF / group-relative diagnostics / saturation analysis]
- [Reviewer 3: NeurIPS-style empirical methodology / claim scoping]

=== Conflicts of interest ===

[None / list as appropriate.]

We thank the [Venue] reviewers and area chair for their consideration.

Sincerely,

[Your Name]
[Affiliation]
[ORCID]
[Contact]