# AntiVibe Full-Repo Deep Dive — Tinker RL Lab

**Generated**: 2026-08-02 · AntiVibe full-repo pass (mid level, full mode)

> Start here. This is the index; each subsystem has its own file in `deep-dive/`.

## What this repo is

A consolidated research vault for **reinforcement learning with large language models**. One codebase reproduces a single frozen RL experiment — Qwen3-8B · GSM8K · GRPO · 30 steps · group size 8 · LoRA r=16 — across a **2-D matrix**:

- **5 frameworks** (the "what trains"): `trl`, `tinker`, `verl`, `openrlhf`, `skyrl`
- **6 compute backends** (the "where it runs"): `local`, `modal`, `colab`, `vast`, `gcp`, `hfspaces`

The scientific goal is cross-framework **equivalence** under one protocol, with reviewer-facing preregistration, hash-anchored receipts, and CI gates that substitute for GPU access. It spans two academic phases (Group 6 capstone → solo continuation) that share infrastructure but separate deliverables (see `PROJECT_HISTORY.md`).

## Architecture at a glance

```
                         ┌─────────────────────────────┐
                         │  CanonicalSpec (frozen)      │   one experiment
                         │  Qwen3-8B · GSM8K · GRPO     │   every cell reproduces
                         └──────────────┬───────────────┘
                                        │
                          platform_local/unified  ◄── entry: `tinkerrl` CLI
                          UnifiedLauncher.run()
                                        │
            ┌───────────────────────────┴───────────────────────────┐
            │ backend dispatch (where)                                │ framework dispatch (what)
            ▼                                                         ▼
  local / colab  → dispatch_framework() ──►  _run_trl / _run_tinker / _run_verl / _run_openrlhf / _run_skyrl
  modal / vast / gcp → shell out to per-backend driver, which on the box re-enters `--backend local`
  hfspaces → fetch-only (no training)
                                        │
                                        ▼
                    results / receipts (HF · W&B · GCS) → aggregate → paper/decks
```

Two ideas make the whole thing tractable: (1) **plan before run** — `Backend.plan()` resolves a cell into a `LaunchPlan` without compute, so the 30-cell matrix is testable in CI; (2) **remote backends re-enter the local dispatch** — Modal/vast/GCP provision a box, then run the *same* `dispatch_framework()` code local uses, so "vast runs verl" is literally the same code path as "local runs verl."

## Subsystem deep-dives

Read top-to-bottom, or jump to the subsystem you care about:

| # | File | Subsystem | The one-line hook |
|---|------|-----------|-------------------|
| 1 | [01-core-launcher-and-matrix](01-core-launcher-and-matrix-2026-08-02.md) | `platform_local/` + `tests/` | The 2-axis dispatch, `LaunchPlan` dry-run seam, and the test that finally caught the matrix lying. |
| 2 | [02-framework-integrations](02-framework-integrations-2026-08-02.md) | tinker, verl, openrlhf, skyrl, trl | Three integration shapes (in-process / subprocess+CLI / recipe), GRPO-without-a-critic, the `PYTHONPATH` shadow trap. |
| 3 | [03-drivers-and-compute-backends](03-drivers-and-compute-backends-2026-08-02.md) | experiments + `platform_{modal,gcp,vast,colab,hf_spaces}` | Six interchangeable backends behind one `Backend` ABC; the vendored `platform_modal` blob (only 2 of 537 files are ours). |
| 4 | [04-zvf-program-preflight](04-zvf-program-preflight-2026-08-02.md) | `zvf-program/` | Frozen protocol → secret-free Spot VM → three-channel receipts → three-tier verification. The most rigorous code in the repo. |
| 5 | [05-tooling-and-outputs](05-tooling-and-outputs-2026-08-02.md) | `outputs/`, `utils/`, `tools/`, root scripts | Code-as-deck, stats rigor (IQM/rliable), audit-as-structured-data, and the throwaway root scripts. |

Also see [`framework-backend-dispatch-2026-08-02.md`](framework-backend-dispatch-2026-08-02.md) — a focused dive on the recent change that made every cell genuinely run its own framework (the work that motivated the `test_each_cell_threads_its_framework` gate).

## Cross-cutting themes

- **Frozen protocol over flexibility.** `CanonicalSpec`, `preregistration.json`, hash-anchored amendments, pinned package versions — comparability beats knob count.
- **Plan/test without GPUs.** `LaunchPlan` + `--dry-run` + invariant tests let CI cover a 30-cell GPU matrix. The tests prove *plumbing*, not gradient correctness — a stated, accepted limit.
- **Single dispatch, many substrates.** Whether it's a serverless H100, a Spot VM, or a rented vast.ai box, the on-box command converges on `python -m platform_local.unified --framework <fw> --backend local`.
- **Receipts as evidence.** Every run writes to multiple independent channels (HF + W&B + GCS); agreement across them is the trust signal.
- **Code-as-artifact.** Decks and audits derive numbers from the checkout at build time — nothing hand-typed that can drift from evidence.

## Per-file deep dives

These subsystem dives are the 10,000-foot view. For the **file-by-file** drill-down, see **[`per-file/INDEX.md`](per-file/INDEX.md)** — one AntiVibe deep dive per real source file (Python, shell, YAML/TOML, key JSON), mirrored under `per-file/<module>/` and regenerable with `python tools/apply_antivibe.py`. The interactive `/antivibe` skill now ships in the repo at `.claude/skills/antivibe/`.

## What's *not* code (deliberately skipped)

`autoresearch/`, `research/` (hyperresearch vault), `capstone-literature-survey/`, `thesis/`, `docs/`, and the `.tex`/`.md` paper + audit prose are documentation and research notes, not source — listed in the README structure map but outside AntiVibe's scope. The `platform_modal/scripts/{berkeley,p5p8}` trees (~322 files) and `platform_modal/openrlhf/` are **vendored** experiment scripts kept for reproducibility, not invoked by the unified launcher.

---
*Generated by AntiVibe · full-repo pass · `/antivibe` on any file or directory for a deeper drill-down.*
