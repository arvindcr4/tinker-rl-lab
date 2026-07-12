# Static defense walkthrough

Open `index.html` and present the frames in order. These are frames captured from the verified MP4 recordings, so they remain available without network access.

1. `01_hf_tool_call_result.png` — hosted Qwen returns a structured `get_stock_price(AAPL)` tool call.
2. `02_hf_math_691.png` — hosted Qwen Math solves the compact arithmetic prompt; result `691`.
3. `03_hf_math_14.png` — hosted DeepSeek distill solves the multi-step word problem; result `14`.
4. `04_wandb_g2_signal_collapse.png` — matched-budget G=2 reaches reward 1.0 while ZVF reaches 1.0.
5. `05_wandb_g16_contrast.png` — matched-budget G=16 retains estimator contrast; final ZVF 0.25.
6. `06_invalid_run_preserved.png` — invalid P4 arm is retained for audit but has no valid history.
7. `07_corrected_drgrpo_run.png` — corrected Dr.GRPO run has complete history and provenance.

Defense-safe wording: the exact completion-length contraction range is approximately 3.8–12.2%, so say “roughly 4–12%.”
