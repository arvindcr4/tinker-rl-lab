---
title: 'Value-model track status optional unmeasured prompt baseline'
id: value-track-status-2026-10-03
tags:
- rl-verification
created: '2026-10-03T16:30:00Z'
status: review
tier: institutional
type: note
content_type: review
summary: 'Optional CPU prompt-value baseline is wired into the GRPO loop and off by default. It is unmeasured and not a thesis result. GAE stays a trajectory helper, not the token-loop advantage.'
---

# Value-model track status (2026-10-03)

Context: [[final_report_rl-verification]].

The trainer can subtract a prompt-level value from the reward before the
batch-wide advantage normalization. `PromptValueCritic` is a CPU
hash-embedding plus a small MLP. It predicts E[reward | prompt]. It does
not read a backbone value head, because the Tinker closure only returns
sampled-token logprobs. `critic_enabled` defaults to false. No measured
run exists. Do not put this baseline on a defense slide or in the thesis.

What the loop does when the flag is on:
- Advantages are R − V(x), then normalized across the batch. Per-group
  centering would cancel the baseline, so the critic forces the batch path.
- The logged explained variance is the pre-update baseline, including
  truncated rewards, because V is subtracted from those rewards too.
- Optional pretraining samples batches with the policy frozen, and only
  on a fresh run. A later resume replays those draws before the step draws.
- Checkpoints store the module and its Adam state beside the JSON receipt.
  A mid-run resume with no critic file raises. A raw state dict from an
  older file still loads, without optimizer moments.

Still not a value track:
- `compute_gae_advantages` and `value_loss_mse` remain trajectory helpers.
  The token loop does not use GAE.
- There is no length-adaptive λ, no learned value head on the policy, and
  no live-run evidence that this baseline changes held-out accuracy.
