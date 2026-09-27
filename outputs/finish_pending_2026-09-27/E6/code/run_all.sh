#!/bin/bash
# E6: run the five splits with the native evaluator. w1-w4 in parallel (disjoint sites; actor concurrency 4),
# then w5 (cross-site gitlab/reddit/shopping) after they finish. Resumable (driver skips done task_ids).
cd /opt/e6/webarena && . /opt/e6/venv/bin/activate && . /opt/e6/env
for w in w1_shopping w2_shopping_admin w3_gitlab w4_reddit_map; do
  python /opt/e6/run/e6_driver.py --task_ids @/opt/e6/run/splits/$w.txt --result_dir /opt/e6/results/$w > /opt/e6/results_$w.log 2>&1 &
done
wait
python /opt/e6/run/e6_driver.py --task_ids @/opt/e6/run/splits/w5_cross_reddit.txt --result_dir /opt/e6/results/w5_cross_reddit > /opt/e6/results_w5_cross_reddit.log 2>&1
echo ALL_SPLITS_DONE > /opt/e6/results/ALL_DONE
