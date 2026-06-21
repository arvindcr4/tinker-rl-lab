# Changelog

All notable changes to **zvf-triage** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned (v0.2)
- Implement the native adapter bodies for **verl**, **OpenRLHF**, and **NeMo-RL**
  (currently honest `NotImplementedError` scaffolds in
  `zvf_triage.integrations`).
- Calibrated per-task default thresholds (`zvf_max` / `eps_lo` / `eps_hi`) from
  the predictive-validation sweep.
- Continuous-/process-reward fallback diagnostic for regimes where ZVF
  degenerates.
- Richer difficulty-binned warm-start resampling driven by per-prompt ZVF
  history.

## [0.1.0] - 2026-06-14

Initial public release.

### Added
- **Core ZVF math** (`zvf_triage.core`, numpy-only): `zvf`, `group_uniformity`
  (`gu`), `per_prompt_zvf`, `rolling_zvf`, `rolling_variance`,
  `peak_to_tail_drift`, `stability_index`, and the `DEFAULT_VAR_EPS` tolerance.
  Definitions reused verbatim from the ZVF Program paper/experiments with
  per-function provenance in the module docstring.
- **Triage controller** (`zvf_triage.controller.ZVFController`): the four-regime
  state machine (`healthy` / `exploitable_contrast` / `cold_start_collapse` /
  `saturation`), adaptive rollout group-size scheduling
  (`G_t = clip(G0 * f(zvf), Gmin, Gmax)`), per-prompt drop after `drop_k`
  consecutive zero-variance steps, and global auto-stop after `stop_k`
  consecutive fully-collapsed steps. Returns a structured `StepDecision`.
- **Callback** (`zvf_triage.callback.ZVFCallback`): framework-agnostic
  `on_step(rewards, group_ids)` entry point plus `as_trl_callback()`, a lazily
  built `transformers.TrainerCallback` (the working reference adapter) that sets
  `control.should_training_stop` on auto-stop / abort collapse. `warm_start_fn`,
  `abort_fn`, and `on_decision` user hooks.
- **Panel** (`zvf_triage.panel.ZVFPanel`): backend-agnostic metric sink with
  `auto` / `wandb` / `tensorboard` / `memory` / `none` backends, all imported
  lazily so training never crashes for lack of a logger.
- **Native-adapter scaffolds** (`zvf_triage.integrations`): `BaseZVFAdapter` plus
  honest `NotImplementedError` stubs `VERLZVFAdapter`, `OpenRLHFZVFAdapter`, and
  `NeMoRLZVFAdapter` documenting the intended native rollout-reward extraction
  for each framework.
- **Typing**: `py.typed` marker; public API carries type hints.
- **Example**: `examples/quickstart.py` — a runnable, numpy-only end-to-end demo
  on synthetic GRPO rollouts that drift from healthy contrast into collapse,
  showing regime transitions, adaptive-`G` changes, prompt drops, and auto-stop.
- Packaging: PyPI metadata, classifiers, keywords, and project URLs;
  Apache-2.0 license; GitHub Actions CI across Python 3.9–3.12.

[Unreleased]: https://example.invalid/TODO-set-repo-url/compare/v0.1.0...HEAD
[0.1.0]: https://example.invalid/TODO-set-repo-url/releases/tag/v0.1.0
