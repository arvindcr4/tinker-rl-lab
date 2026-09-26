#!/bin/bash
# usage: run_tau2.sh <arm: base|vllm_trained|vllm_base> <agent_port> <domain> <task ids...>
# Agent LLM -> local shim on <agent_port> (Tinker base, or vLLM endpoint via shim --backend vllm, same chat-templated prompt ids).
# User simulator -> shim on ${USER_PORT:-18743}: Tinker base (18743) for the Tinker arm; vLLM base endpoint shim for both vLLM arms (Tinker token cap exhausted).
S=${SCRATCH:-/private/tmp/claude-501/-Users-arvind-Developer-agentic-repos-tinker-rl-lab/f6a17e89-20dd-4b9b-b647-18831abd53b1/scratchpad}
E=/Users/arvind/Developer/agentic_repos/tinker-rl-lab/outputs/e1_e14_small_scale_2026-09-26/E5
K=$(command cat $S/shimkey)
arm=$1; port=$2; dom=$3; shift 3
unset OPENAI_API_KEY OPENAI_BASE_URL   # never reach real OpenAI
export GEMINI_API_KEY="$(security find-generic-password -s GEMINI_API_KEY -w)"   # NL-assertion judge
cd $S/tau2
.venv/bin/tau2 run --domain $dom --task-split-name test --task-ids "$@" --num-trials 1 \
  --agent llm_agent --agent-llm openai/agent \
  --agent-llm-args "{\"api_base\":\"http://127.0.0.1:$port/v1\",\"api_key\":\"$K\",\"temperature\":0,\"max_tokens\":4096}" \
  --user user_simulator --user-llm openai/user \
  --user-llm-args "{\"api_base\":\"http://127.0.0.1:${USER_PORT:-18743}/v1\",\"api_key\":\"$K\",\"temperature\":0,\"max_tokens\":4096}" \
  --max-concurrency 4 --save-to e5_${arm}_${dom} --log-level INFO
mkdir -p $E/$([ $arm = base ] && echo raw || echo $arm/raw)
cp -r data/simulations/e5_${arm}_${dom} $E/$([ $arm = base ] && echo raw || echo $arm/raw)/
