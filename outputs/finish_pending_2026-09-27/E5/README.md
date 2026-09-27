# E5: Tau3 (tau2-bench banking_knowledge) replacement scope, 2026-09-27

- **Result: 8/97 = 8.25% pass^1 (Wilson 95% CI 4.2–15.4%).** Infrastructure errors count as failures over all 97 tasks. The native pass^1 over the 72 tasks that did not error is 11.1%. Not pooled with original-contract numbers.
- 25 tasks ended in infrastructure errors that were the model's own failures: 14 hit the 65,536-token context limit and 11 returned empty assistant messages.
- Actor: `pavlov-public-portfolio-bf16` on a dedicated 65k-context Modal H200, temperature 0, non-thinking. The user simulator was gpt-4.1 via a spend-capped relay.
- Spend: $18.42 against the $80 cap. Both Modal apps are stopped.
- Recovery history: the main run stopped at 92/97. The 5 unstarted tasks were run separately and merged in. An accidental auto-resume dropped the 24 error records from `results.json`, so they were reconstructed in `infra_errors_reconstructed.json`. See the caveats in `result.json`.
- Code: `code/run_tau3.sh` (runner), `code/score_and_redact.py` (native metric, lane rule, redaction), `code/e5_actor_65k.py`, `code/openai_relay.py`.
