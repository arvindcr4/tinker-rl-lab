# ZVF Program — Pillar-1 / M1 Sweep Harness

A **runnable orchestration harness** for two things:

1. **Scale the ZVF/GU audit** from ~95 runs to **300–500 runs**, logging
   Zero-Variance Fraction (ZVF) and Gradient Utilization (GU) **per step**.
2. **Matched-compute comparison** of canonical **GRPO vs DAPO vs GSPO** at
   equal total rollouts/tokens.

> **No results are fabricated. This harness only orchestrates and aggregates
> real runs.** Every metric printed by `aggregate_sweep.py` is read from a file
> a real training run wrote. Missing files are reported `MISSING`, never guessed.
> `matched_compute.py` never declares a winner — that is computed post-hoc from
> real logs.

---

## What's here

| File | Role |
|------|------|
| `sweep_config.yaml` | Declarative grid: model families, RL frameworks, seeds, group sizes G, difficulty buckets, held-out suite. Every field commented. |
| `run_sweep.py` | Orchestrator. Expands the config into a manifest and **shells out** to your existing runner per cell. Tracks status, resumable. **Dry-run is the default.** |
| `matched_compute.py` | Defines the GRPO/DAPO/GSPO arms at matched compute and emits their launch commands. Has a `# RESULTS:` placeholder filled only after real runs. |
| `aggregate_sweep.py` | Reads completed run JSONs, joins per-step ZVF/GU with held-out scores → `aggregate_table.csv`. Reports `MISSING` for absent files. |
| `run_manifest.json` | The enumerated grid (written by `run_sweep.py`, including dry-run). |
| `run_status.csv` | Per-cell status ledger (written when you `--execute`). |
| `aggregate_table.csv` | Final results table (written by `aggregate_sweep.py`). |

### Definitions (match the existing in-repo runners)
- **ZVF** = fraction of prompts in a step where all `G` completions got the
  *same* reward → GRPO advantage collapses to 0 → no usable gradient.
- **GU** = `1.0 - ZVF`.
Both are logged per step by the existing runners
(`experiments/tinker-runs/campaign_v2.py`, `live_zvf_probe.py`,
`tinker_parallel_runner.py`) into each run's `step_log`.

---

## How dry-run works (default, launches nothing)

```bash
cd zvf-program/sweep
python3 run_sweep.py            # dry-run is the DEFAULT
python3 run_sweep.py --dry-run  # explicit, identical
```

It prints the manifest summary (cell count per block, resume status), writes
`run_manifest.json`, and prints the **literal commands it would run** — but
executes nothing. Current grid expands to **403 cells** (385 audit + 18
matched-compute), inside the 300–500 target band.

Other dry-run options:
```bash
python3 run_sweep.py --filter audit     # only audit cells
python3 run_sweep.py --filter matched   # only matched-compute cells
python3 run_sweep.py --limit 5          # cap to 5 cells (smoke test)
python3 run_sweep.py --no-resume        # don't skip already-completed cells
```

---

## What YOU must provide before launching (this harness does NOT create it)

The harness **shells out** to a per-cell runner script. It does **not**
implement training. You provide one thin shim that wraps your existing Tinker
GRPO loop and accepts the flags the harness passes:

**`experiments/tinker-runs/cell_runner.py`** (you own this) must accept:
```
--tag <str> --model <hf_id> --task <gsm8k|...> --loss <grpo|dapo|gspo|importance_sampling>
--seed <int> --group-size <int> --lr <float> --steps <int> --rank <int>
--difficulty <easy|medium|hard> --out <path/to/result.json>
```
and write `--out` as a JSON containing at minimum:
```json
{ "status": "completed",
  "step_log": [ {"step": 1, "reward": 0.3, "loss": 0.1, "zvf": 0.5, "gu": 0.5}, ... ],
  "reward_trace": [...], "peak_reward": 0.6, "last10_avg": 0.4,
  "first5_avg": 0.3, "zero_reward_pct": 10.0 }
```
This is the **exact schema the existing runners already emit** — your shim can
be ~30 lines that import `campaign_v2.run_experiment` /
`tinker_parallel_runner.run_single`, add the `--loss` / `--difficulty` branch
(map difficulty → a dataset subset; map loss → grpo/dapo/gspo surrogate), and
dump the result. The harness deliberately leaves training logic to you so it
can never fabricate a metric.

To use **Modal** instead of Tinker: set `runner.kind: modal` in
`sweep_config.yaml`; `run_sweep.py` then builds commands against
`experiments/modal_runner.py` (which already has its own `--dry-run`).

### (Optional) held-out scores
After training, your eval step should write `<tag>.heldout.json` next to each
result with `{"GSM-Plus": <float>, "AIME-2025": <float>, "BFCL-v3": <float>}`
(datasets named in `sweep_config.yaml` → `held_out_suite`). If absent,
`aggregate_sweep.py` reports those columns `MISSING`.

---

## How to actually launch on your compute

### Credentials (same as the rest of the repo; see `.env.example`)
```bash
export TINKER_API_KEY="tml-..."     # required for Tinker runs
export WANDB_API_KEY="..."          # optional, for W&B logging
export HF_TOKEN="hf_..."            # optional, for HF checkpoint upload
# Modal path only:
export MODAL_TOKEN_ID="ak-..."  MODAL_TOKEN_SECRET="as-..."
```

### Launch the audit + matched cells
```bash
cd zvf-program/sweep

# 1. confirm the plan (nothing runs)
python3 run_sweep.py --dry-run

# 2. launch for real (resumable; re-run anytime to pick up where it stopped)
python3 run_sweep.py --execute --max-parallel 6

# 3. (or) launch ONLY the matched-compute study with pinned equal budgets
python3 matched_compute.py            # preview arms + commands
python3 matched_compute.py --execute  # launch GRPO/DAPO/GSPO arms

# 4. aggregate real results -> aggregate_table.csv
python3 aggregate_sweep.py
python3 aggregate_sweep.py --matched  # just the GRPO/DAPO/GSPO comparison
```

`run_sweep.py --execute` is **resumable**: cells whose result JSON already
exists (or whose tag is `completed` in `experiments/master_results.csv`) are
skipped. Kill it and re-run freely.

---

## Cost & wall-clock — **ESTIMATE ONLY (not a measurement)**

> These are order-of-magnitude planning numbers, **not** observed results.
> Your actual cost depends on Tinker/Modal pricing, model sizes, queue time,
> and failure/retry rates. Verify against a small `--limit 5` batch first.

Per-cell sizing follows the existing runners (≈30 steps, group size G,
batch 2 prompts, 512 max tokens):

| Tier | Models | Approx per-cell wall-clock* | Notes |
|------|--------|------------------------------|-------|
| small (8B) | qwen3-8b, llama-8b-inst | ~5–15 min | bulk of the audit |
| mid (30–32B) | qwen3-32b, MoEs, gpt-oss-20b, nemotron-30b | ~15–40 min | |
| frontier (120B–235B) | qwen3-235b, deepseek-v3.1, nemotron-120b, llama-70b, gpt-oss-120b | ~30–90 min | thinned grid, fewer cells |

\* *ESTIMATE — heavily dependent on Tinker backend throughput and parallelism.*

**Ballpark for the full 403-cell grid (ESTIMATE):**
- With `--max-parallel 6`, wall-clock on the order of **~1–3 days** of
  orchestrated runtime.
- Compute spend is **whatever Tinker/Modal bill** for ~403 short LoRA-GRPO runs
  weighted toward small models. **Run `--limit 5` first, read the actual
  per-run cost from your Tinker/Modal dashboard, then multiply.** Do not trust
  any single number here as a quote.

To shrink cost: trim `model_families` tiers, drop `group_sizes` entries, reduce
`seeds`, or set `sweep_design.audit.full_grid_tiers: [small]` in the config.

---

## Provenance guarantee

- `run_sweep.py` derives a cell's status **only** from the subprocess exit code
  **and** the presence/parse-ability of the runner's own result JSON.
- `aggregate_sweep.py` reads ZVF/GU **only** from `step_log` on disk; absent →
  `MISSING`.
- `matched_compute.py`'s `# RESULTS:` block stays `<pending>` until you fill it
  from real aggregated logs.

**Nothing in this directory invents a run, a metric, or a "win."**
