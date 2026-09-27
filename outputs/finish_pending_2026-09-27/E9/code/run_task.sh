#!/bin/bash
# Run ONE native ML-Dev-Bench task (calipers hydra entrypoint, OpenHands CodeActAgent) with the
# trained actor behind the loopback relay. Runs as e9builder (UID 1001, docker group).
# usage: run_task.sh <task_id> [wall_timeout_s]
set -u
T=$1; WALL=${2:-4500}
R=/opt/e9/runs/$T; mkdir -p $R
cd /opt/e9/controller
set -a; . /opt/e9/secrets/secrets.env; set +a
export HOME=/home/e9builder POETRY_VIRTUALENVS_IN_PROJECT=true
export PATH=/opt/e9/controller/.venv/bin:/usr/local/bin:/usr/bin:/bin:/opt/e9/tooling/bin
# task credentials forwarded into the OpenHands sandbox (native SANDBOX_ENV_ mechanism)
export SANDBOX_ENV_WANDB_API_KEY=$WANDB_API_KEY SANDBOX_ENV_HF_TOKEN=$HF_TOKEN
unset TRAINED_ACTOR_API_KEY TRAINED_ACTOR_BASE_URL
date -u +%FT%TZ > $R/start
timeout -k 60 $WALL /opt/e9/controller/.venv/bin/python -m calipers.scripts.run_hydra_evaluation \
  task=$T agent=openhands \
  agent.model_name=openai/pavlov-public-portfolio-bf16 \
  +agent.max_iterations=50 \
  +agent.model_config.base_url=http://127.0.0.1:18019/v1 \
  +agent.model_config.api_key=local-relay \
  +agent.model_config.temperature=0.0 \
  +agent.model_config.top_p=1.0 \
  +agent.model_config.max_input_tokens=28000 \
  +agent.model_config.max_output_tokens=4096 \
  +agent.model_config.num_retries=4 \
  +agent.model_config.timeout=1200 \
  +agent.model_config.native_tool_calling=true \
  output_dir=$R/results clone_workspace_to=$R/workspace \
  +env_file=$R/no-env-file \
  hydra.run.dir=$R/hydra > $R/eval.log 2>&1
echo $? > $R/rc
date -u +%FT%TZ > $R/end
# OpenHands agent never closes its runtime: remove this task's sandbox container(s) (bind-mount match)
ids=$(docker ps -aq --filter volume=$R/workspace); [ -n "$ids" ] && docker rm -f $ids > $R/containers_removed 2>&1
exit 0
