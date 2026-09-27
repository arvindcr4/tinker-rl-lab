#!/bin/bash
# E5 Tau3 (tau2-bench banking_knowledge, a2c02472) native run: thin runner replacing the fail-closed
# zvf-program/e5_successor27 controller (its launch path needs a lead-bound runtime adapter + reservation
# paperwork; superseded by outputs/finish_pending_2026-09-27/AUTHORIZATION.json).
# Native CLI + native grader; conditions = outputs/public_portfolio_2026-09-05/tau3_setup/execution_plan.proposed.json
# except: actor api_base (dedicated Modal 65k-context copy of the campaign actor) and OpenAI roles via relay.
# usage: run_tau3.sh <save_name> [--task-ids ...]
set -euo pipefail
REPO=/Users/arvind/Developer/agentic_repos/tinker-rl-lab
S=/private/tmp/claude-501/-Users-arvind-Developer-agentic-repos-tinker-rl-lab/9b594818-8c2e-47c9-a641-947970ba4bba/scratchpad/e5
set -a; source $REPO/outputs/e1_e14_small_scale_2026-09-26/trained_actor/.env.local; set +a
ACTOR=https://arvindcr4--e5-trained-actor-65k-0927-serve.modal.run/v1
RELAY=https://arvindcr4--e5-openai-relay-0927-relay.modal.run/v1
export OPENAI_API_KEY="$(cat $S/relay_token)" OPENAI_BASE_URL=$RELAY OPENAI_API_BASE=$RELAY
unset OPENROUTER_API_KEY
export PATH=$S/npm/node_modules/.bin:$S/tau2/.venv/bin:$PATH LITELLM_LOCAL_MODEL_COST_MAP=True PYTHONDONTWRITEBYTECODE=1
NAME=$1; shift
AGENT_ARGS=$(python3 -c "import json,os;print(json.dumps({'temperature':0.0,'api_base':'$ACTOR','api_key':os.environ['TRAINED_ACTOR_API_KEY'],'max_tokens':4096,'timeout':1800}))")
cd $S/run
exec tau2 run --domain banking_knowledge --agent llm_agent --user user_simulator \
  --agent-llm openai/pavlov-public-portfolio-bf16 --agent-llm-args "$AGENT_ARGS" \
  --user-llm gpt-4.1-2025-04-14 --user-llm-args '{"temperature":0.0}' \
  --retrieval-config alltools --task-split-name base --num-trials 1 --max-steps 200 --max-errors 10 \
  --max-concurrency 3 --seed 300 --save-to "$NAME" --verbose-logs --llm-log-mode all "$@"
