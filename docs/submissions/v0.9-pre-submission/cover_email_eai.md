# Cover Email — EAI Endorsed Transactions variant

## Recipient
- To: EAI Editorial Office <editorial@eai.org>
- CC: [Editor-in-Chief of EAI Endorsed Transactions]
- Subject: New Submission: Reward Contrast, Not Algorithm Labels — A Claim-to-Evidence Audit of Group-Relative RL for LLMs

## Body

Dear EAI Editorial Team,

Please find attached our manuscript for consideration at EAI Endorsed
Transactions on [Track Name — e.g., "AI and Machine Learning" /
"Knowledge Discovery"]. The paper is an engineering diagnostic of
critic-free group-relative reinforcement learning for LLMs.

### Suitability for EAI

EAI Endorsed Transactions emphasise [practitioner-grade empirical work
with rigorous scope-bounding / open-access dissemination of audit-style
contributions]. Our submission fits via three axes:

1. **Diagnostic, not benchmark.** We do not claim a SOTA result. We
   audit when group-relative RL has usable learning signal and when it
   does not, with explicit held-out capability controls.
2. **Open-data + open-code.** Code, run logs, W&B traces, and the
   13-audit hygiene suite are all released; the paper tags a v0.9-pre-
   submission state with a reproducible verification path.
3. **Honest scope language.** All headline numbers carry scope caveats:
   training-set reward vs. held-out, custom parsers vs. canonical
   harness, 50-problem subsets vs. full benchmarks.

### Novelty highlights (in scope order)

- Zero-Variance Fraction (ZVF) and Gradient Utilization (GU) as
  pre-screen triage diagnostics for cold-start collapse, validated on
  a 22-run set with precision 1.0 / recall 1.0 at a first-five-step
  rule.
- A clean held-out GSM8K 200-example control (82.0% → 83.3%, p = 0.26;
  not significant) that scopes training-reward gains to dynamics
  evidence, not capability.
- Out-of-platform LoRA-vs-full-FT experiment (E2, Qwen3-4B-Instruct-
  2507, 3 seeds) defending the LoRA-only constraint of the Tinker
  platform and tying to ZVF theory T2.
- Audit suite (13 paper/submission hygiene checks) that future
  contributors can extend.

### What changed since the prior internal R1/R2 review loop

[Reference commit b7b7bd97 "EAI Endorsed Transactions submission draft
+ ARIS R1/R2 review loop" if applicable.]

- The new E2 result (above) addresses the previously outstanding
  reviewer concern about the LoRA-only confound.
- Four audit-script false positives were repaired so the suite
  reflects the actual scope language of the paper.
- The repo is synchronized with the upstream canonical at
  pes-llm-research/tinker-rl-lab.

### Reproducibility

- Tag: v0.9-pre-submission
- Code: https://github.com/arvindcr4/tinker-rl-lab
- W&B: https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab-world-class
- Audits: 13/13 passing on this tag (python3 platform_local/run_all_audits.py)

### Author confirmation

- All authors have approved this submission.
- The manuscript is not under review at any other venue.
- Conflicts of interest: [None / list as appropriate.]

We appreciate the EAI editorial team's time and look forward to the
review.

Sincerely,

[Your Name]
[Affiliation]
[ORCID]
[Contact]