# E9 — ML-Dev-Bench replacement (2026-09-27)

**Result: 10/34 = 29.4% task success (Wilson 95% CI 16.8–46.2%).** Replacement scope. Not pooled with original-contract numbers.

- Ran all 34 released ml-dev-bench task configs once each with the native calipers hydra entrypoint and the per-task validators. The agent was OpenHands CodeActAgent (max 50 iterations). The actor was `pavlov-public-portfolio-bf16` through a loopback relay (`code/relay.py`), at temperature 0 with thinking disabled.
- Every task finished with rc=0 and a graded result. There were no errors or timeouts. Had any occurred, they would have counted as failures.
- Passed: channel_vit_implementation, channel_vit_implementation_easy, hello_world, mcts_implementation, nan_loss_debug, parse_logs, pretrained_model_load_from_torchvision, shape_mismatch_output, small_dataset_overfit, wandb_logging.
- Deviations:
  - Ran on GCP n2-standard-16, not AWS, because the AWS quota was denied.
  - Applied an OpenHands context-window hang fix (`code/patch_openhands_ctxwindow.py`, a backport of the upstream one-liner).
  - Set max_input_tokens to 28000 because the actor's context is 32k.
  - Full list in `result.json`.
- Excluded: smoke runs (`raw/_smoke`) and the pre-patch aborted partial batch (`raw/_aborted_batch1`).
- Spend: about $1.67 for the VM and disk (cap $60). Actor time is billed to the shared endpoint budget.
- The VM and its disk were deleted and absence was verified at 2026-09-27T02:50Z.
