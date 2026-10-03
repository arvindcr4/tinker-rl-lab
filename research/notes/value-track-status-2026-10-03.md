---
title: 'Value-model track status critic foundation only'
id: value-track-status-2026-10-03
tags:
- rl-verification
created: '2026-10-03T16:30:00Z'
status: review
tier: institutional
type: note
content_type: review
summary: 'Value-track status: GAE + half-MSE value loss implemented and tested as pure helpers (critic.py); value head, pretraining, and loop integration remain future work.'
---

# Value-model track status (2026-10-03)

Context: [[final_report_rl-verification]].

Shipped (pure, tested, no loop wiring — there is no critic to wire to):
- `compute_gae_advantages` — GAE, Schulman et al. 2016 Eq. 11–12, with
  bootstrap-length and dones validation (fail-closed).
- `value_loss_mse` — half-MSE value loss, PPO convention (Schulman et
  al. 2017 §7), shape-checked.

Explicitly remaining for a VAPO-style value track: a value head on the
policy (or separate critic), value pretraining, length-adaptive GAE
tuning, and training-loop integration with live runs to validate. The
helpers above are the pinned-math foundation so that work does not start
from scratch. Do not claim a working value track until a trained critic
exists.
