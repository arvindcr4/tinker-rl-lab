# Redactions

tau2-bench serialises its LLM client arguments into `results.json`, including an `api_key` field. Every such value (12 occurrences in 6 files) was replaced with `REDACTED` on 2026-09-26 before commit. Rewards, trajectories and all scored fields are unchanged.
