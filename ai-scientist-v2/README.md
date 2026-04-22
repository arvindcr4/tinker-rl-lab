# AI Scientist-v2 Integration

This directory wires Sakana AI Scientist-v2 into the Tinker RL Lab workflow as
an idea generator and experiment triage system. It is intentionally constrained:
AI Scientist-v2 should propose experiments, while this repo's existing runners
decide what is allowed to execute.

Upstream project: https://github.com/sakanaai/ai-scientist-v2

## Recommended Workflow

1. Build a redacted context pack from the current repo:

```bash
python3 scripts/prepare_ai_scientist_v2.py \
  --output-dir .ai_scientist_v2_runs/latest
```

2. Run AI Scientist-v2 ideation against the generated topic file:

```bash
export AI_SCIENTIST_V2_DIR="$HOME/AI-Scientist-v2"
cd "$AI_SCIENTIST_V2_DIR"
python ai_scientist/perform_ideation_temp_free.py \
  --workshop-file "/Users/arvind/tinker-rl-lab/.ai_scientist_v2_runs/latest/tinker_rl_lab_topic.md" \
  --model "${AI_SCIENTIST_MODEL:-gpt-4o-2024-11-20}" \
  --max-num-generations 12 \
  --num-reflections 3
```

For the model requested in this run:

```bash
python ai_scientist/perform_ideation_temp_free.py \
  --workshop-file "/Users/arvind/tinker-rl-lab/.ai_scientist_v2_runs/latest/tinker_rl_lab_topic.md" \
  --model gemini-3.1-pro-preview \
  --max-num-generations 1 \
  --num-reflections 2
```

Set `S2_API_KEY` if you want the upstream literature-search tool to work
reliably. Without it, Semantic Scholar can rate-limit and the upstream script may
stall or fail before writing usable ideas.

3. Import the generated ideas into the safe research-loop queue:

```bash
cd /Users/arvind/tinker-rl-lab
python3 scripts/import_ai_scientist_v2_ideas.py \
  --ideas .ai_scientist_v2_runs/latest/tinker_rl_lab_topic.json \
  --dry-run

python3 scripts/import_ai_scientist_v2_ideas.py \
  --ideas .ai_scientist_v2_runs/latest/tinker_rl_lab_topic.json
```

4. Launch a bounded wave through the existing coordinator:

```bash
python3 research_loop/coordinator.py wave new --size 4 --phase 1
```

Each wave brief asks an agent to write a YAML config for
`research_loop/train.py`, not to edit arbitrary source files.

## Full AI Scientist-v2 BFTS Mode

Run full `launch_scientist_bfts.py` only in a sandboxed clone with no live
Tinker, W&B, Hugging Face, SSH, or cloud credentials mounted. Upstream explicitly
warns that the system runs LLM-written code. For this repo, full BFTS should be
used to generate candidate patches and analysis artifacts, then those artifacts
should be reviewed and ported into the safe queue manually.

Use `bfts_config.override.yaml` as a conservative reference configuration. It is
not a drop-in upstream config replacement; it documents the intended limits.

## What AI Scientist-v2 Should Explore

- GRPO rescue experiments for the 3B/4B capacity-threshold claim.
- Reward shaping to break all-zero group advantage saturation.
- Group-size, batch-diversity, temperature, LoRA-rank, and curriculum ablations.
- BrowserGym/MiniWoB/WebArena tool-use experiments as config proposals.
- Robustness checks for length bias, prompt masking, and zero-variance frontier.

## Guardrails

Read `experiment_contract.md` before running any generated idea. The core rule is
simple: AI Scientist-v2 can propose, summarize, and draft, but live training
runs must go through local validators, budget caps, and the existing result
ledger.
