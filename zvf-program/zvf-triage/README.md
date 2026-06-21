# zvf-triage

**Zero-Variance Fraction (ZVF) triage for GRPO-style RL fine-tuning.**

Pillar 3 of the ZVF Program — a drop-in callback that watches a GRPO training
run for the failure mode where *group-relative advantage collapses to zero
gradient signal*, classifies the regime, adapts the rollout group size, drops
dead prompts, and auto-stops doomed runs before they waste compute.

It is a **triage diagnostic**, not a calibrated performance predictor: it tells
you *which regime* a run is in (cold-start collapse, saturation, or healthy
exploitable contrast), so you can intervene early.

---

## Why ZVF?

In GRPO, the advantage of a rollout is computed *relative to the other rollouts
of the same prompt-group*. If all `K` rollouts of a group earn the **same**
reward (all right, or all wrong), the group's reward variance is zero, the
group-relative advantage is zero, and that group contributes **no gradient** to
the update. ZVF is the fraction of the batch that is in exactly this state.

For a GRPO batch `B_t` of prompt-groups, each with `K` rollout rewards
`r_{g,1..K}`:

```
ZVF_t = (1 / |B_t|) * Σ_g  1[ Var_hat(r_{g,1..K}) ≤ ε ]      (unbiased variance, ε = 1e-6)
GU_t  = 1 − ZVF_t                                            (Group Uniformity / "Gradient Utilization")
```

`GU` is the fraction of groups still carrying usable within-group contrast.

ZVF is **not** a tautological restatement of mean reward: two batches with the
same mean reward `r̄ = 0.5` can have wildly different ZVF depending on *how*
success mass is distributed across prompts (a bimodal `p_g ∈ {0,1}` population
gives `ZVF = 1`; a uniform-hard `p_g = 0.5, K = 8` population gives
`ZVF ≈ 0.008`). See the ZVF Program paper appendix *"Formal Definition of the
Zero-Variance Fraction, Non-Tautology, and Scope"* for the full argument and the
boundary conditions (dense rewards, `K = 1`, SFT-saturated baselines).

The math in this package is reused verbatim from the ZVF Program experiments and
paper; `zvf_triage/core.py` documents the source file for each definition.

---

## Install

```bash
pip install zvf-triage                # core (numpy only)
pip install "zvf-triage[trl]"         # + torch / transformers / trl adapter
pip install "zvf-triage[wandb]"       # + Weights & Biases panel
pip install "zvf-triage[tensorboard]" # + TensorBoard panel
pip install "zvf-triage[dev]"         # + pytest
```

Python ≥ 3.9. The core package imports with **only numpy**; torch, trl,
transformers, wandb and tensorboard are all imported lazily, only when their
adapter/panel is actually used.

---

## Quickstart

### Intended TRL API (from the deck)

```python
from zvf_triage import ZVFCallback

trainer = GRPOTrainer(model, dataset, callbacks=[ZVFCallback(
    window=5,
    zvf_max=0.85,
    on_collapse="warm_start",
    adaptive_G=True,
    wandb_panel=True,
).as_trl_callback()])
```

`as_trl_callback()` returns a `transformers.TrainerCallback` that reads per-step
rollout rewards + group ids out of the trainer log dict and sets
`control.should_training_stop` on auto-stop or an abort collapse.

### Framework-agnostic loop (works anywhere — Tinker, verl, custom)

```python
from zvf_triage import ZVFCallback

cb = ZVFCallback(window=5, zvf_max=0.85, on_collapse="warm_start", adaptive_G=True)

for step in training_loop:
    rewards, group_ids = run_rollouts(...)   # flat reward list + matching group ids
    decision = cb.on_step(rewards, group_ids)

    print(decision.regime, decision.zvf, decision.group_size)
    G = decision.group_size                  # adaptive rollout count for next step
    drop(decision.dropped_prompts)           # prompts dead for K steps

    if cb.should_stop:                       # auto-stop or abort collapse
        break
```

### Pure functions

```python
from zvf_triage import zvf, group_uniformity, peak_to_tail_drift, rolling_zvf

zvf([1, 1, 1, 0], [0, 0, 1, 1])              # -> 0.5  (g0 uniform, g1 mixed)
group_uniformity([1, 1, 1, 0], [0, 0, 1, 1]) # -> 0.5
peak_to_tail_drift([1.0, 0.8, 0.2, 0.0], tail=2)  # -> 0.9
rolling_zvf([0, 1, 1, 0], window=2)          # -> [0, 0.5, 1.0, 0.5]
```

---

## The triage state machine

Each step the controller computes `zvf` and batch mean reward `r̄` and emits a
structured `StepDecision`:

| Condition | Regime | Signal | Action |
|---|---|---|---|
| `zvf > θ_hi` and `r̄ < ε_lo` | `cold_start_collapse` | `on_collapse` | nothing solved, no contrast → warm-start easier prompts |
| `zvf > θ_hi` and `r̄ > 1 − ε_hi` | `saturation` | `lift_difficulty` | everything solved → lift task difficulty |
| `zvf > θ_hi` (mid reward) | `cold_start_collapse` | `on_collapse` | degenerate, warn operator |
| `zvf ≤ θ_hi` | `exploitable_contrast` | `noop` | healthy, spend rollout budget |

- `θ_hi = zvf_max` (default `0.85`), `ε_lo = eps_lo` (`0.05`), `ε_hi = eps_hi` (`0.05`).
- `on_collapse ∈ {"warm_start", "abort", "noop"}` selects what a cold-start
  collapse does: fire your `warm_start_fn` hook, stop training, or nothing.

### Adaptive group size

When ZVF is high, the controller *increases* the rollout group size so more
rollouts per prompt raise the odds of producing within-group reward contrast:

```
G_t = clip(G0 * f(zvf),  Gmin,  Gmax)
```

Default `f(zvf) = 1 + zvf` (1× at ZVF 0, 2× at ZVF 1). Override with any
`adaptive_fn: [0,1] → float`. Enabled with `adaptive_G=True`.

### Per-prompt drop & auto-stop

- **Drop a prompt** once its group ZVF == 1 (zero-variance) for `drop_k`
  consecutive steps — it is contributing no gradient. Dropped ids are exposed on
  `decision.dropped_prompts` and `callback.dropped_prompts`.
- **Auto-stop the run** once the *global* batch ZVF == 1 for `stop_k`
  consecutive steps — the entire batch has collapsed. Sets `callback.should_stop`
  and `decision.auto_stop`, and emits the `"abort"` signal.

---

## W&B / TensorBoard panel

`ZVFPanel` logs `zvf`, `gu`, `rolling_zvf`, `mean_reward`, `group_size`,
`dropped_prompt_count`, `auto_stop`, and a categorical `regime_code` — with no
hard dependency on either backend:

```python
from zvf_triage import ZVFCallback, ZVFPanel

cb = ZVFCallback(wandb_panel=True)                       # attaches to wandb.run if live
cb = ZVFCallback(panel=ZVFPanel(backend="tensorboard", tb_logdir="runs/zvf"))
cb = ZVFCallback(panel=ZVFPanel(backend="memory"))       # offline; history in panel.history
```

`backend="auto"` tries wandb, then tensorboard, then an in-memory buffer —
training never crashes for lack of a logger.

---

## Scope (when ZVF is *not* informative)

From the paper's boundary conditions — ZVF degenerates and you should switch to
a per-step reward-variance diagnostic instead:

- **Dense / continuous rewards** → variance is almost surely nonzero, `ZVF → 0`.
- **`K = 1`** → group variance undefined, `ZVF` trivially 1.
- **SFT-saturated baselines** (`p_g → 1` everywhere) → `ZVF → 1`, no discriminative power.
- **Non-outcome-reward RL** (DPO / regression) → no rollout groups, ZVF ill-defined.

Intended scope: outcome-reward GRPO on verifiable tasks with `K ≥ 4` and
non-saturated reward support.

---

## Roadmap to v0.2

- **Upstream adapters** as first-class callbacks for **TRL**, **verl**,
  **OpenRLHF**, and **NeMo-RL** (PRs), each reading native rollout-reward
  structures instead of the generic log-dict shim.
- **Tinker integration**: native hook in the ZVF Program's `live_zvf_probe`
  training loop.
- **Calibrated thresholds**: ship per-task default `θ_hi / ε_lo / ε_hi` derived
  from the predictive-validation sweep instead of fixed constants.
- **Continuous-reward fallback**: per-step reward-variance diagnostic for dense
  / process-reward regimes where ZVF degenerates.
- **Richer warm-start actions**: difficulty-binned prompt resampling driven by
  per-prompt ZVF history (the causal-ZVF "Goldilocks zone" binning).

---

## Example

A runnable, numpy-only end-to-end demo lives in
[`examples/quickstart.py`](examples/quickstart.py): it drives synthetic GRPO
rollouts from healthy contrast into collapse and prints the regime transitions,
adaptive-`G` changes, prompt drops, and the auto-stop firing. See
[`examples/README.md`](examples/README.md).

```bash
python examples/quickstart.py
```

## Native adapters

The working reference adapter is the TRL / HuggingFace callback returned by
`ZVFCallback.as_trl_callback()`. Honest stubs (raising `NotImplementedError`
with the intended extraction documented) for **verl**, **OpenRLHF**, and
**NeMo-RL** ship under `zvf_triage.integrations` and land in v0.2.

## License

Apache-2.0 — see [`LICENSE`](LICENSE). Chosen to match the Apache-2.0 licensing of
TRL / verl, which the deck targets for upstreaming. See also
[`CHANGELOG.md`](CHANGELOG.md).
