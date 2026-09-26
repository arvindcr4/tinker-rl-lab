# E1–E14 small-scale base-model arm — 2026-09-26

Goal: every lane E1–E14 gets a fresh, honestly-scoped number from one consistent actor at small scale.

## Why a new arm
The Tinker training run for the trained seed809 LoRA is purged, so the adapter cannot be sampled on Tinker.
(Correction 2026-09-26: the adapter itself survives on HF branch `checkpoint-seed809-stepfinal-9f777c4018b6`; see
`ADAPTER_AVAILABILITY_CHECK_2026-09-26.json`.) This arm uses the base model. Its numbers are a **new arm**, never pooled
with or presented as the original lane scores (project evidence rule).

## Actor (fixed for all lanes)
- `Qwen/Qwen3.6-35B-A3B` base weights, no adapter, sampled on Tinker (`tinker==0.30.4`).
- Non-thinking chat template (`enable_thinking=False`) unless the lane's native protocol needs reasoning; record which.
- Default `temperature=0`, per-lane `max_tokens` recorded.
- Known-good Python: `/Users/arvind/.local/share/uv/tools/tinker/bin/python` (tinker 0.30.4, transformers, jinja2).
  Load key with `set -a; source /Users/arvind/Developer/agentic_repos/tinker-rl-lab/.env; set +a`. Never print or copy key values.
- Encode prompts with `tok.apply_chat_template(..., tokenize=False)` then `tok.encode(text, add_special_tokens=False)`;
  passing the BatchEncoding directly fails validation.
- An OpenAI-compatible shim is fine if a harness needs one; keep it local.

## Compute routes
- Tinker: model sampling. Cap per lane group: 6M total tokens (prefill + sample).
- Colab (`colab` CLI, 110 units shared): Linux x86 harness hosts. Prefer `colab run` (auto-releases VM) and CPU/T4.
  Cap 15 units per lane group. Name sessions `e<lane>-…`; `colab stop` anything you start.
- Local Docker (aarch64, 12 GB) or Modal (profile arvindcr4, existing infra under `outputs/modal_e1_e14/`,
  `zvf-program/flagship/modal_*`) for container grading. Modal cap $10 per lane group.

## Scale
Small: target 10–50 scored items per lane (fewer only if each item is expensive, minimum 3). Pick items by a
fixed seed (`seed=20260926`) from the canonical ordering; record the item IDs.

## Benchmark choice per lane
Use the original benchmark if any public items can be run. Otherwise use a public substitute from
`outputs/E2_E14_ALTERNATIVE_BENCHMARK_MAP_2026-09-20.json` or `outputs/WHY_ALTERNATIVE_BENCHMARKS_2026-09-20.md`,
and state the gap in one sentence.

## Required output per lane: `outputs/e1_e14_small_scale_2026-09-26/E<n>/result.json`
```json
{
  "lane": "E<n>", "original_benchmark": "...", "benchmark_run": "...", "scope": "original-public-subset | substitute",
  "substitute_gap": "... or null", "actor": "Qwen/Qwen3.6-35B-A3B base (no adapter), Tinker",
  "thinking": false, "temperature": 0, "max_tokens": 0,
  "n_selected": 0, "n_scored": 0, "n_errors": 0, "item_ids": [],
  "metric": "...", "value": 0.0, "numerator": 0, "denominator": 0, "wilson95": [0.0, 0.0],
  "grader": "native | re-implemented | llm-judge(model)", "compute_route": "...",
  "cost": {"tinker_tokens": 0, "colab_units": 0.0, "modal_usd": 0.0},
  "started_utc": "...", "finished_utc": "...", "commands": ["..."], "raw_dir": "E<n>/raw/",
  "caveats": ["..."]
}
```
Errors and timeouts count in the denominator as failures. Also write `E<n>/README.md` (≤25 lines: what ran, how to rerun).

## Hard rules
- No fabricated, estimated, or copied numbers. `value` must be recomputable from files in `raw/`.
- Don't modify or delete existing result files elsewhere in `outputs/`. Write only under this directory
  (scratch code may go in `E<n>/code/`).
- No git commits, pushes, emails, issues, forum posts, or publishing. No new paid accounts or quota requests.
- Stay within caps; if a cap would be exceeded, reduce n, never exceed.
- Stop every Colab session / Modal app / container you start.

## Paired vLLM arm (trained vs base, same engine) — added 2026-09-26
The seed809 adapter's effect is about the same size as Tinker-vs-vLLM engine noise, so the adapter is compared **only within vLLM**.
The Tinker base-model results above remain a separate arm and are never differenced against vLLM results.

- Endpoints (both already deployed; do NOT deploy, redeploy or stop them — they scale to zero after 300 s idle, and the lead stops them at the end):
  - trained: `https://arvindcr4--pavlov-trained-actor-e1e14-serve.modal.run`, model `pavlov-public-portfolio-bf16`
  - base:    `https://arvindcr4--pavlov-trained-actor-e1e14-basectl-serve.modal.run`, model `qwen36-base-bf16`
  - Auth: `set -a; source outputs/e1_e14_small_scale_2026-09-26/trained_actor/.env.local; set +a` gives `TRAINED_ACTOR_API_KEY`
    (same key for both). Never print it. The client snippet and cold-start notes are in `TRAINED_ACTOR_ENDPOINT.md`.
- For each lane, rerun the **identical item IDs, prompts, harness, grader and decoding settings** as the lane's Tinker-base run, twice:
  once against each endpoint. Change nothing but the endpoint and model id. If the harness used Tinker's raw sampler, use
  `/v1/completions` with the same chat-templated prompt string. Otherwise use chat with `chat_template_kwargs.enable_thinking=false`.
- Concurrency: ≤ 4 in-flight requests per endpoint per lane group (one H200 each, max 16 sequences, shared by all lanes).
  The first request may take 3–5 min (cold start); wait on `/v1/models`.
- Output: `E<n>/vllm_trained/result.json` and `E<n>/vllm_base/result.json` (same schema; `actor` names the endpoint and
  model id; `compute_route` = "Modal vLLM H200"), plus `E<n>/paired.json` with per-item outcomes for both arms and:
  - n_items, trained value, base value, difference;
  - for binary items: discordant counts b/c and an exact McNemar p;
  - for continuous items: paired mean difference with a bootstrap 95% CI (10k resamples, seed 20260926).
- LLM judges (E10, E12, E14) must use the **same judge** in both arms as in the lane's Tinker run.
- Cost: record wall time and tokens per arm. The Modal GPU bill is shared and the lead reconciles it. Total cap across all
  lanes is $70; if a lane's paired run would take more than ~45 min of wall time per arm, reduce n and document it.
- Claims: report the per-lane paired difference with its CI/p, and nothing stronger. No pooling across lanes.
