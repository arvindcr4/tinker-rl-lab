# zvf-triage examples

Runnable, dependency-light demos. Everything here uses **numpy only** — no GPU,
no network, no torch / trl / wandb.

## `quickstart.py`

An end-to-end run of the ZVF triage callback on **synthetic GRPO rollouts**.

```bash
python examples/quickstart.py
python examples/quickstart.py --seed 7 --steps 24 --groups 8 --K 16
```

### What it simulates

Each prompt-group `g` has a latent solve probability `p_g(t)`. At `K` rollouts
per group, a group is **zero-variance** (contributes no GRPO gradient) iff all
`K` rollouts agree — probability `p_g^K + (1 - p_g)^K`. The script drives a
three-phase schedule:

1. **Healthy** — `p_g ≈ 0.5`: groups are mixed, lots of within-group contrast,
   `ZVF ≈ 0`, regime `exploitable_contrast`.
2. **Drift** — every `p_g` decays toward 0: groups start collapsing to all-wrong,
   `ZVF` climbs, adaptive `G` grows to fish for contrast, dead prompts get
   dropped.
3. **Terminal collapse** — `p_g = 0`: every group is uniformly wrong, global
   `ZVF == 1`, regime flips to `cold_start_collapse`, and after `stop_k`
   consecutive fully-collapsed steps the run **auto-stops** (signal `abort`).

### What it prints

- A per-step table: `zvf`, rolling `zvf`, mean reward `r_bar`, the adaptive
  group size `G`, the classified `regime`, the emitted `signal`, and event flags
  (`REGIME-CHANGE`, `G a->b`, `DROP [...]`, `AUTO-STOP`).
- A summary block: the regime transitions, the adaptive-`G` change log, the
  cumulative dropped-prompt set, and the step at which auto-stop fired.
- The in-memory `ZVFPanel` history count and the last logged metrics, showing the
  panel captured every step with no W&B / TensorBoard backend installed.

The script ends with assertions that the run started healthy (low ZVF) and
finished in a fired auto-stop, so it doubles as a smoke test of the full
controller → callback → panel pipeline.

### Maps to the public API

| Demo step | API used |
|---|---|
| build the watcher | `ZVFCallback(window, zvf_max, on_collapse, adaptive_G, G0/Gmin/Gmax, drop_k, stop_k, panel)` |
| feed a batch | `cb.on_step(rewards, group_ids) -> StepDecision` |
| read the verdict | `decision.regime`, `.signal`, `.zvf`, `.rolling_zvf`, `.group_size`, `.dropped_prompts`, `.auto_stop` |
| early-stop the loop | `cb.should_stop` |
| offline logging | `ZVFPanel(backend="memory").history` |

For the intended TRL integration (`as_trl_callback()`) and the framework-agnostic
loop, see the top-level [`README.md`](../README.md).
