# Title

Agentic GRPO Experiments for Tool-Using Language Models

# Keywords

GRPO, reinforcement learning for language models, tool use, BrowserGym, WebArena,
MiniWoB, Tinker, Atropos, GSM8K, reward shaping, curriculum learning, group
relative policy optimization, zero-variance frontier, agentic evaluation

# TL;DR

Generate high-leverage, low-risk experiments that improve or stress-test a GRPO
research project on mathematical reasoning and browser/tool-use agents. Propose
experiments that can be expressed as bounded configs or small isolated patches,
then validated through the repository's existing result ledger.

# Abstract

This project studies reinforcement learning for language models, focusing on
GRPO-style optimization, reasoning benchmarks, and agentic tool/browser control.
The current evidence suggests a capacity threshold around the 3B to 4B scale on
GSM8K-style math reasoning, plus early BrowserGym/MiniWoB smoke evidence that
browser and tool use should be represented more strongly. The goal is to use AI
Scientist-v2 to propose concrete experiments that either strengthen, falsify, or
sharpen these claims.

The most valuable proposals are not new manuscripts. They are executable
research directions that can be safely mapped into this repository's existing
experiment surfaces: `research_loop/train.py` YAML configs, BrowserGym/Atropos
config proposals, and small isolated analysis scripts. Strong proposals should
name the hypothesis, the minimal experiment, expected failure mode, required
metrics, and how the result would change the paper.

# Current Empirical Context

- The project has GRPO results across Tinker, TRL, Modal, Atropos, and related
  evaluation harnesses.
- The main unresolved scientific question is whether the observed 3B failure is
  a hard capacity wall or a recoverable recipe failure.
- Existing research-loop knobs include model, seed, LoRA rank, steps, learning
  rate, group size, batch size, temperature, max tokens, advantage
  normalization, reward shape, curriculum, and evaluation subset size.
- Browser and tool-use coverage is currently lighter than the core math RL
  evidence. BrowserGym/MiniWoB smoke testing exists, while WebArena-style
  benchmark work needs stronger baselines and safer infrastructure.
- The repository maintains a result ledger, NIA-indexed artifacts, and a
  submission builder with secret scanning.

# Preferred Experiment Families

1. Capacity-wall rescue:
   Test whether reward shaping, curriculum, group size, batch diversity,
   temperature, advantage normalization, or LoRA rank can rescue small-model
   GSM8K GRPO runs.

2. Reward-signal engineering:
   Break all-zero group advantage saturation with partial-credit rewards,
   format rewards, verifier ensembles, or curriculum-dependent reward shaping.

3. Browser/tool-use representation:
   Propose MiniWoB, WorkArena-L1, or WebArena-verified ablations for observation
   format, action schema, invalid-action penalties, and stepwise ReAct rollouts.

4. Robustness and audit experiments:
   Stress-test length bias, prompt masking, train/test leakage, zero-variance
   frontier metrics, and cross-backend reproducibility.

5. NIA-driven paper improvements:
   Use indexed logs and manifests to identify missing baselines, unsupported
   claims, weak result framing, or experiments that would most improve the
   paper/report.

# Constraints

- Do not request access to API keys, local secret files, browser cookies, SSH
  keys, W&B credentials, Hugging Face credentials, or Tinker credentials.
- Prefer ideas that can be represented as safe YAML configs or small isolated
  analysis scripts.
- Do not propose live cloud runs without a budget cap and a smoke-test stage.
- Do not modify paper claims until the result is reproduced and reconciled with
  the canonical result ledger.
- For BrowserGym/WebArena work, start with MiniWoB or WorkArena-L1 smoke tasks
  before any full WebArena run.

# Evaluation Criteria

Rank ideas by:

- Scientific value for the current paper/report.
- Ability to falsify or sharpen a central claim.
- Cost and runtime.
- Compatibility with the existing research-loop config schema.
- Robustness to seed variance and benchmark artifacts.
- Safety under the experiment contract.

# Expected Idea Format

Each idea should include:

- Title.
- Hypothesis.
- Minimal experiment.
- Suggested config changes or benchmark target.
- Primary metric and secondary diagnostics.
- Expected positive result.
- Expected negative result.
- How this would change the paper/report.
- Safety or infrastructure risks.
