# Research Loop — 3B GRPO Rescue

Bitter-lesson-style parallel-agent autoresearch, modeled on Ryan Li's
Paradigm Optimization Arena winner. Target the single most valuable
open question in this repo: **can the 3B model's GSM8K failure be
rescued through hyperparameter / curriculum changes, or is it a hard
capacity wall?**

If yes → the paper's "capacity threshold at 3B–4B" claim needs to be
softened and a new recipe added to Section 7. If no → the capacity
claim is strengthened with 1000+ experiments of evidence.

## Phases

| Phase | Target | Compute | Runs | Cost estimate |
|-------|--------|---------|------|---------------|
| 1 — Proxy loop       | Qwen-0.6B / Qwen-1.7B, 100 steps, GSM8K 500-example subset | Tinker API (cloud) | ~500 | see budget |
| 2 — Scale verify     | Qwen-3B + Llama-3B, full 50-step GSM8K, 5 seeds | vast.ai H100 | ~150 | see budget |
| 3 — From-scratch reset | One agent designs a new recipe ignoring all prior configs | Tinker API | ~50 | see budget |
| **Total**            |                                                          |                    | **~700** |               |

The 1000-run number is the *total upper bound including aborted / failed runs*;
expect ~700 productive runs. Ryan Li ran 1,039 unique strategies + 2,000 evals.

## Budget (HARD cap)

- Tinker API: $200 (Phase 1 + 3). Tripwire at $150 — coordinator blocks new waves.
- vast.ai:    $100 (Phase 2). Tripwire at $75.
- Wall-clock: 2 weeks max.
- Agent concurrency: max 8 active at once (Phase 1), max 3 (Phase 2).

Set `RESEARCH_LOOP_BUDGET_USD` in env to enforce. `coordinator.py status`
shows running total.

## Metric

Primary: `last10_avg_accuracy` — mean reward over last 10 training steps
(already computed by `grpo_gsm8k_base.py`).

Secondary (tracked for robustness, not optimized directly):
- `peak_accuracy`
- `first5_avg_accuracy` (detect fast-starts)
- `zero_reward_pct`
- `zero_loss_pct`

Phase 2 re-evaluates the top 10 by Phase 1 `last10_avg_accuracy` using
`reports/final/evaluate_gsm8k_test.py` for real held-out accuracy. Phase 1
is the cheap proxy; Phase 2 is the scientific claim.

## Files

```
research_loop/
├── README.md                   this file
├── train.py                    parameterized GRPO trainer with 9 knobs
├── run_one.py                  single-experiment runner, appends to results.jsonl
├── coordinator.py              wave orchestration, budget enforcement, briefings
├── best_recipe.yaml            current best config + score (updated after each wave)
├── learnings.md                confirmed-optimal + dead-end knobs (read before new hypotheses)
├── hypotheses_seed.md          20 initial hypotheses to seed Wave 1
├── agent_brief_template.md     per-agent prompt template
├── results.jsonl               append-only log of all (variant_id, config, metrics) tuples
├── queue.jsonl                 proposed-but-not-yet-run configs
├── wave_briefs/wave_NNN/       per-wave directory, one markdown per agent
└── variant_configs/wave_NNN/   generated variant YAMLs from agents
```

## Knobs (train.py)

1. `model` — Qwen/Qwen3-0.6B, Qwen/Qwen3-1.7B, Qwen/Qwen3-3B (Phase 2), Qwen/Qwen3-4B, etc.
2. `seed` — RNG seed
3. `rank` — LoRA rank (4, 8, 16, 32, 64)
4. `steps` — training steps (50, 100, 200)
5. `lr` — learning rate
6. `group_size` — GRPO group size (4, 8, 16, 32, 64, 128)
7. `batch` — prompts per step
8. `temperature` — sampling temperature (0.3 – 1.5)
9. `adv_norm` — advantage normalization: `std` | `none` | `rank`
10. `reward_shape` — `binary` | `graded` | `partial`
11. `curriculum` — `random` | `easy_first` | `hard_first`
12. `max_tokens` — response token limit

KL-to-SFT and entropy bonus are deferred to Phase 3 from-scratch agents
(they require tracking reference logprobs, which the current trainer doesn't).

## How to run

**Prereqs:**
```bash
export TINKER_API_KEY="tml-..."          # cloud training
export WANDB_API_KEY="..."                 # logging (optional)
export RESEARCH_LOOP_BUDGET_USD=200        # hard cap
```

**One run (smoke test):**
```bash
cd ~/tinker-rl-lab/research_loop
python run_one.py --config variant_configs/smoke_test.yaml
tail -1 results.jsonl | jq
```

**Start a wave:**
```bash
python coordinator.py wave new --size 8 --phase 1
# Reads queue.jsonl (or hypotheses_seed.md for wave 1), writes 8 agent briefs to
# wave_briefs/wave_NNN/agent_NNN.md. Then spawn Claude Code agents pointing at
# each brief. Each agent writes variant_configs/wave_NNN/vNNN.yaml, runs run_one.py,
# reports METRIC lines.
```

**Ingest results:**
```bash
python coordinator.py wave ingest wave_001
# Parses all result JSONs, updates best_recipe.yaml, appends to results.jsonl,
# appends dead-ends to learnings.md, re-scores queue.jsonl.
```

**Status:**
```bash
python coordinator.py status
# Shows: current best, total runs, waves completed, budget spent, learnings count.
```

## Anti-overfitting

Per Ryan Li's writeup — when a knob looks promising on seed=0, it needs
to hold across 4+ seeds before we accept it. The coordinator auto-promotes
a variant to "confirmed winner" only after 4-seed validation.

Single-seed exploration happens in Phase 1 proxy loop. Multi-seed validation
happens in Phase 2 scale verification. Do not optimize for a single seed.

## From-scratch reset (Phase 3 trigger)

When `last10_avg_accuracy` stops improving for 3 consecutive waves (`coordinator.py`
flags this), spawn a single agent with zero context except: (a) the challenge
(rescue 3B GSM8K), (b) current best score, (c) forbidden list of knobs already
exhausted. Ryan Li's biggest jump came from such a reset.

## AI Scientist-v2 idea source

AI Scientist-v2 can be used as a bounded upstream idea generator for this loop.
Do not give it direct access to live Tinker/W&B/HF credentials or unrestricted
repo writes. Prepare a redacted context pack, run ideation, import ideas into the
queue, then launch normal coordinator waves:

```bash
python3 scripts/prepare_ai_scientist_v2.py --output-dir .ai_scientist_v2_runs/latest
bash .ai_scientist_v2_runs/latest/run_ideation.sh
python3 scripts/import_ai_scientist_v2_ideas.py --ideas .ai_scientist_v2_runs/latest/tinker_rl_lab_topic.json --dry-run
python3 scripts/import_ai_scientist_v2_ideas.py --ideas .ai_scientist_v2_runs/latest/tinker_rl_lab_topic.json
python3 research_loop/coordinator.py wave new --size 4 --phase 1
```

See `ai-scientist-v2/experiment_contract.md` for the execution boundary.
