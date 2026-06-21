# Colab-only experiments E4–E7 — one per ZVF-Program pillar

**Rule:** every experiment must require a capability that closed, LoRA-only,
fixed-stack **Tinker** structurally cannot provide. Same honest pilot scope as
E1–E3 (Qwen2.5-0.5B-Instruct on synthetic arithmetic, T4): these validate
*measurement design and direction*, not publishable effect sizes.

The 4 pillars (from `zvf-program/*/README.md`):
- **P1 `sweep/`** — empirical: scale the ZVF/GU audit; check asymptotic claims vs real measurement.
- **P2 `theory/`** — formal ZVF theory: signal `S = p(1-p)·(1-h_G(p))`, closed form `ZVF(p,K)=p^K+(1-p)^K`, worked example p=0.5,K=8 → 2·0.5^8 ≈ 0.0078.
- **P3 `zvf-triage/`** — operationalization: `ZVFController.step(rewards, group_ids) → StepDecision` (adaptive G, drop, auto-stop).
- **P4 `position/`** (MIN-REPORT-RL) — policy: RL reporting must include the *stack* levers (sampler, backend, precision) theory holds fixed.

Existing: E1→P2 (grad-norm↔p(1-p)), E2→P4 (LoRA vs full-FT), E3→P3+P4 (open loss audit + adaptive-G).
**Gap: P1 has no Colab experiment.** New batch covers all four with *new* Tinker-impossible angles.

| Exp | Pillar | Question | Tinker-impossible lever |
|-----|--------|----------|--------------------------|
| **E4** | P1 | Does empirical ZVF follow the closed form `ZVF(p,K)=p^K+(1-p)^K` across a K-sweep, and does it shift with numerical precision? | fp32 vs bf16 audit (Tinker pins precision) + full reward-matrix logging at controlled large K |
| **E5** | P2 | What fraction of the GRPO gradient is *wasted* on zero-variance groups, as a function of ZVF? (theory: ZVF groups contribute exactly 0 gradient) | open backward pass / per-rollout gradient access |
| **E6** | P3 | Does the *real* `ZVFController` driving live G-escalation + prompt-drop beat a fixed-G baseline at matched compute? | reading internal per-step ZVF and mutating group size / curriculum mid-run |
| **E7** | P4 | Holding task/data/seed/compute/algorithm fixed, how much does flipping ONE unreported stack lever move ZVF and held-out Δ? | precision / attention-backend / sampler are fixed by Tinker |

## E4 — P1: ZVF–K scaling law, two precisions (`e4_scaling_law.py`)
- Grid: difficulty ∈ {trivial, easy, medium, hard} × K ∈ {2,4,8,16}. Per cell: ~16 prompts, sample K rollouts each, reward = exact-match.
- Estimate per-prompt p from a larger reference sample (e.g. 24 gens) at the cell's difficulty; compute empirical ZVF (frac of groups with 0 reward variance); compare to closed form `p^K+(1-p)^K` using the measured p.
- Report per-K R²(empirical, predicted), and where empirical ZVF crosses 0.008 vs the p=0.5,K=8 prediction.
- Repeat whole grid at bf16 and fp32; report ΔZVF(precision) per cell.
- **Output:** `E4_RESULT {by_cell, r2_by_K, precision_delta_zvf}`. Run `--gpu T4 --timeout 1800`.

## E5 — P2: gradient-waste vs ZVF (`e5_grad_waste.py`)
- Across the difficulty grid, per group compute advantage-weighted seq-logprob loss and `.backward()`.
- Measure `g_live` = ‖gradient from groups with reward variance > 0‖ and `g_all` = ‖gradient if dead groups *also* contributed‖ (they contribute 0 by construction → g_live == g_all numerically; the *point* is the realized vs potential rollout budget). Operational metric: **gradient efficiency = (#live groups)/(#groups)** and **effective-signal magnitude per rollout** vs ZVF and vs p(1-p).
- Confirm: effective signal is inverted-U in difficulty and → 0 as ZVF → 1 at both ends; corr(signal, p(1-p)) > corr(signal, GU).
- **Output:** `E5_RESULT {by_difficulty:{p,zvf,signal_per_rollout,live_frac}, corr_signal_p1mp, corr_signal_gu}`. Run `--gpu T4 --timeout 1200`.

## E6 — P3: live ZVFController vs fixed-G baseline (`e6_live_triage.py`)
- `pip install -e zvf-triage` (or copy the package) so the experiment imports the **real** `ZVFController`.
- Two arms at matched total-rollout budget, seeds {0,1}:
  - `baseline`: fixed G=G0, no drop.
  - `triage`: `ZVFController(adaptive_G=True, G0=4, Gmax=12, drop_k=3, stop_k=4)`; each step feed (rewards, group_ids) → use `decision.group_size` for next step, exclude `decision.dropped_prompts`, honor `decision.auto_stop`.
- Metrics: held-out Δ, total rollouts, mean ZVF, steps-to-first-ZVF<0.2, dropped-prompt count.
- **Output:** `E6_RESULT {by_arm:{heldout_delta, mean_zvf, rollouts, zvf_suppress_step}}`. Run `--gpu T4 --timeout 1500`.

## E7 — P4: single-lever stack sensitivity (`e7_stack_levers.py`)
- Fixed: task, data, seed, compute, GRPO algorithm, steps. Reference config = bf16 / sdpa / temp 1.0 / top_p 0.95.
- One-at-a-time flips: precision→fp32; attention→eager; temp→0.7; top_p→1.0. Each arm = short GRPO run (10–12 steps) + held-out eval.
- Report ΔZVF and Δheldout vs reference per lever → quantifies how much an *unreported* lever moves the headline (the MIN-REPORT thesis).
- **Output:** `E7_RESULT {reference, by_lever:{delta_zvf, delta_heldout}}`. Run `--gpu T4 --timeout 1500`.

## Persistence
Generalize `persist_results.py` to parse `E4_RESULT`..`E7_RESULT` from each run log,
write `results/e{4..7}_*.json`, log to W&B project `zvf-colab-experiments`, refresh `results/README.md`.

## Codex review (gpt-5.5, read-only) — incorporated 2026-06-21
Verdict: GO as-is none; fix E4/E6/E7; E5 cut *as written*, kept only redesigned around real gradients/Fisher.
Revisions applied below:
- **E4**: power fix — **generate-once, subsample ≥128 groups/K around p≈0.5 with bootstrap CIs** (16 groups can't resolve 0.008). Precision (fp32 vs bf16) is the *only* clean Tinker-blocked lever → demoted to a K=8 side-check, not the main grid.
- **E5**: dropped tautological `live_frac` (= 1−ZVF). Redesigned to measure **per-rollout gradient norm, within-group advantage-weighted gradient SNR / cosine alignment, and a Fisher-trace proxy `E[‖∇logp‖²]`** on the last decoder layer — quantities that require the open backward pass.
- **E6**: **fixed prompt pool with stable integer IDs** (so drop logic works), **match total rollouts** (primary; optimizer steps secondary), **3 arms**: fixed-G / adaptive-G / adaptive-G+drop, all driven by the controller logic mirrored from `zvf_triage.controller`.
- **E7**: **paired + replicated** — same prompts/seeds per arm, ≥2 seeds, report mean±std; headline on **ZVF / ERF / reward trajectory** (held-out Δ over 10 toy steps is too noisy). Trim to ref + {fp32, eager, temp0.7}.
