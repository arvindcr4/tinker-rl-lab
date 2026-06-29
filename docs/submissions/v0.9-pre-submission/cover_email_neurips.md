# Cover Email — NeurIPS / ICML variant

## Recipient
- To: [Area Chair / Senior Area Chair] <[ac-XXX@neurips.cc]>
- CC: [Program Chairs if required]
- Subject: NeurIPS 2026 Submission: Reward Contrast, Not Algorithm Labels — A Claim-to-Evidence Audit of Group-Relative RL (paper ID placeholder)

## Body

Dear Area Chair,

We submit our manuscript, "Reward Contrast, Not Algorithm Labels: A
Claim-to-Evidence Audit of Group-Relative Reinforcement Learning for
LLMs," for consideration at NeurIPS 2026 [Main / Datasets & Benchmarks
/ Workshop on ... — fill as appropriate]. The paper is an engineering
diagnostic, not a benchmark or leaderboard result.

### Contribution in scope of [Track]

The contribution is methodological: a triage diagnostic (ZVF/GU)
paired with explicit held-out capability controls. We frame the
positive result as scoped (learned schema-valid tool-call emission
under custom evaluation; 0% → 92% JSON validity on one custom
pipeline) and the negative result as scoped (held-out GSM8K
82.0% → 83.3%, p = 0.26; not significant; HumanEval 0% reward at 50
steps on a 50-problem subset).

### Specific responses to NeurIPS-style concerns

- **Reproducibility.** Code, run logs, W&B traces, and a 13-audit
  hygiene suite are released at the v0.9-pre-submission tag.
- **Statistical rigor.** 5-seed GSM8K run with bootstrap 95% CI;
  paired per-prompt p-values where applicable.
- **Broader impact.** Discussed in §[X]; the methodology is
  claim-scoping, not capability-claim.
- **Limitations.** A dedicated limitations section calls out: 30-step
  Tinker horizon, custom parsers (not canonical harness), 50-problem
  HumanEval subset, single-seed Tinker runs, no PPO/REINFORCE
  baselines.

### New in this version (vs. v2.0-world-class / v3.0-neurips-submission)

1. **E2 LoRA-vs-full-FT controlled experiment.** Out-of-platform on
   Qwen/Qwen3-4B-Instruct-2507 (3 seeds, held-out N = 50). LoRA mean
   heldout_delta = +0.160 (std 0.020) vs. full-FT +0.100 (std 0.020);
   LoRA-full gap +0.060. Defends the LoRA-only Tinker constraint;
   consistent with ZVF theory T2 (parameter-efficient path = more
   stable in saturation regimes).
2. **Audit-suite repair.** Four false-positive audit failures
   (abstract_scope, submission_claim, blind_review_package,
   export_guard) fixed; capstone abstract aligned to paper scope.
3. **Upstream sync.** Repo synchronized with pes-llm-research/tinker-
   rl-lab; NeurIPS-2026 integrator integrated.

### Conflicts of interest

[None / list as appropriate.]

### Suggested reviewers

- [Reviewer 1: GRPO / RL-for-LLMs methodology]
- [Reviewer 2: ZVF / group-relative diagnostics]
- [Reviewer 3: NeurIPS empirical methodology / scope language]

We appreciate your time and look forward to the reviewer feedback.

Sincerely,

[Your Name]
[Affiliation]
[ORCID]
[Contact]

P.S. Per NeurIPS policy, the camera-ready version (if accepted) will
include the corresponding anonymised repo URL. For now, the author-
identified v0.9-pre-submission tag is at github.com/arvindcr4/tinker-
rl-lab/releases/tag/v0.9-pre-submission.