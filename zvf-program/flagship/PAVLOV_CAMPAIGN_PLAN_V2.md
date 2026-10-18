# Prospective campaign plan v2

Plan hash: 7bf92ea9e6cfe36caceeb85adb26f09570d1e5d398efc3c0580e7aaea7a670d8

This amendment does not rerun or reinterpret historical experiments.

- smoke: 1 entrant(s), 10 total steps, 1 survivor(s); train xlam; evaluate sealed_selection_only
- round_1: 3 entrant(s), 50 total steps, 2 survivor(s); train xlam; evaluate sealed_selection_only
- round_2: 2 entrant(s), 100 total steps, 1 survivor(s); train xlam; evaluate sealed_selection_only
- winner_extension: 1 entrant(s), 200 total steps, 1 survivor(s); train xlam; evaluate sealed_selection_only
- portfolio_extension: 1 entrant(s), frozen_in_launch_manifest total steps, 1 survivor(s); train all_12_suites; evaluate none
- final: 1 entrant(s), 0 total steps, 1 survivor(s); train none; evaluate paired_base_final_all_14_suites_once

Ranking: perfect_call_count_desc, strict_mean_reward_desc, conservative_cost_asc, learning_rate_asc, arm_id_lexical

Estimand: fixed_adapter_paired_task_clusters_conditional_on_recorded_training_seed
