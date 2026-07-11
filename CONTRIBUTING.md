# Contributing to TinkerRL

This repository combines reusable Python modules, experiment launchers, research
artifacts, and paper outputs. Changes must preserve both software correctness and
scientific provenance.

## Set up

Use the locked environment:

```bash
make bootstrap
```

This installs the development dependencies from `uv.lock` and enables the local
pre-commit checks. Run the same quality gate used by CI before opening a pull
request:

```bash
make check
```

## Change discipline

- Keep reusable logic in the supported modules under `platform_local/`,
  `platform_tinker/tinkerrl/`, or `utils/`; keep one-off exploration under an
  experiment or archive directory.
- Add or update tests for behavior changes. Test through a module's public
  interface rather than duplicating its implementation in the test.
- Make paid or long-running experiment entry points resumable. Persist progress
  atomically, validate configuration before resume, and never mix partial runs.
- Use deterministic seeds where possible and record model, dataset, sampler,
  checkpoint, and evaluation configuration in result artifacts.
- Never fabricate, hand-edit, or silently replace measured results. Preserve
  invalidated artifacts with an explicit status and reason.
- Do not commit credentials, `.env` files, W&B caches, model checkpoints, or
  generated build outputs.
- Keep commits scoped. Do not sweep unrelated dirty-worktree changes into a fix.

The canonical research-engineering interfaces are:

- `platform_tinker.tinkerrl.grpo.run_grpo` for GRPO experiment execution;
- `utils.audit_utils.AuditResult` for submission and scientific audits;
- `platform_hybrid.paper.figure_module.FigureModule` for paper figures.

Historical script paths may remain as compatibility adapters, but must not grow
independent training loops, audit exit policy, or rendering implementations.

## Pull requests

Describe the behavior change, verification performed, provenance impact, and any
known limitations. CI must pass on every supported Python version. Architectural
changes should update the relevant `CONTEXT.md`; durable trade-offs should be
recorded as an ADR under `docs/adr/`.

## Generated training scripts

Generate TRL launchers through the supported package interface:

```bash
python -m platform_local.unified --framework trl --algorithm grpo \
  --train-data train.json --generate-script train_grpo.py
```

Generated scripts detect the latest compatible Trainer checkpoint and resume it
automatically.
