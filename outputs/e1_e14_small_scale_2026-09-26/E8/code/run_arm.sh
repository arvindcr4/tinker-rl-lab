#!/bin/bash
# Paired vLLM arm driver for E11 -> E14 actor -> E8 -> E10, sequential so <=4 in-flight requests hit the endpoint.
# Usage: ARM=trained|base ACTOR_PORT=18766 JUDGE_PORT=18765 run_arm.sh   (Tinker judge shim must already be listening on JUDGE_PORT)
set -u
B=/Users/arvind/Developer/agentic_repos/tinker-rl-lab/outputs/e1_e14_small_scale_2026-09-26
S=/private/tmp/claude-501/-Users-arvind-Developer-agentic-repos-tinker-rl-lab/f6a17e89-20dd-4b9b-b647-18831abd53b1/scratchpad
set -a; source /Users/arvind/Developer/agentic_repos/tinker-rl-lab/.env; source $B/trained_actor/.env.local; set +a
export ARM
ts() { date -u +%FT%TZ; }
lane() { mkdir -p $B/$1/vllm_$ARM/raw; [ -f $B/$1/vllm_$ARM/raw/started_utc.txt ] || echo "$(ts)" > $B/$1/vllm_$ARM/raw/started_utc.txt; echo "resume $(ts)" >> $B/$1/vllm_$ARM/raw/segments.txt; }
done_() { echo "$(ts)" > $B/$1/vllm_$ARM/raw/finished_utc.txt; echo "end $(ts)" >> $B/$1/vllm_$ARM/raw/segments.txt; }

lane E11; (cd $B/E11/code && $S/venv/bin/python run_e11.py >> ../vllm_$ARM/run.log 2>&1); done_ E11
lane E14; (cd $B/E14/code && $S/venv/bin/python run_actor.py >> ../vllm_$ARM/run.log 2>&1); done_ E14
lane E8;  (cd $B/E8/code && $S/venv/bin/python run_e8.py >> ../vllm_$ARM/run.log 2>&1); done_ E8
lane E10
(cd $B/E10/code && $S/venv/bin/python shim.py $ACTOR_PORT $B/E10/vllm_$ARM/raw/shim_log.jsonl > $B/E10/vllm_$ARM/raw/shim_stderr.log 2>&1) &
SHIM=$!
until nc -z 127.0.0.1 $ACTOR_PORT 2>/dev/null; do sleep 1; done
(cd $B/E10/code && OPENAI_BASE_URL=http://127.0.0.1:$ACTOR_PORT/v1 JUDGE_BASE_URL=http://127.0.0.1:$JUDGE_PORT/v1 \
  OPENAI_API_KEY=local-shim INSPECT_DISPLAY=plain $S/ve10/bin/python run_e10.py >> ../vllm_$ARM/run.log 2>&1)
kill $SHIM; pkill -f "shim.py $ACTOR_PORT"; done_ E10
echo "arm $ARM done $(ts)"
