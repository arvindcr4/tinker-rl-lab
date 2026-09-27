#!/bin/bash
# Run all 34 released ML-Dev-Bench task configs, one run each, 4 at a time (actor concurrency cap).
cd /opt/e9/controller/ml_dev_bench/conf/task
ls *.yaml | sed 's/\.yaml$//' > /opt/e9/runs/task_list.txt
date -u +%FT%TZ > /opt/e9/runs/batch_start
xargs -P 4 -I{} /opt/e9/runs/run_task.sh {} 4500 < /opt/e9/runs/task_list.txt
date -u +%FT%TZ > /opt/e9/runs/batch_end
