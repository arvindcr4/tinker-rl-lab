# Qwen3-8B GSM8K held-out re-evaluation: truncation vs thinking mode (P8 Q1, P6 §5.2 Q2)

Date 2026-10-18. Every number below comes from fresh greedy generations on Colab (vLLM 0.32.0, bf16, A100-SXM4-40GB). Raw generations are in `raw/gen_<model>__<cond>.jsonl.gz` (gzip), one row per item with the text, token count, finish reason and both scores. Per-condition summaries are in `raw/summary.jsonl.gz`.

## Short answers

**Q1 (P8 Qwen3-8B, training reward 34.4% vs held-out base ~82%).** The gap comes from **truncation caused by thinking mode**. The reward parser does not cause it.
- The training script (`world_class_experiments.py`, April version `c68110a25`) builds a raw ChatML prompt that ends at `<|im_start|>assistant\n`. It has no `<think></think>` stub, so Qwen3 thinks by default. Sampling uses `max_tokens=512`.
- Qwen3-8B's thinking traces on GSM8K average about 1,800 tokens. At 512 tokens about 99% of responses are cut off mid-thought.
- At the training budget, base held-out accuracy is **27.0%**, which is close to the training reward. With thinking disabled and a 4,096-token budget it is **97.5%**.
- The P8 trained adapter (HF `arvindcr4/tinker-rl-scale_gsm8k_qwen3-8b-qwen3-8b`, `final/`) scores 31.0% / 96.5% / 93.5% under the same three conditions. That is indistinguishable from base: paired McNemar p = 0.32 / 0.5 / 1.0.
- The W&B runs (`tinker-rl-lab-world-class/vv6uu72m` and `wbue9ji9`) are backfilled summaries with 0 history rows. They log no response length or truncation. The 512-token cap is established from the script source, and the 99–100% truncation from this re-run.

**Q2 (P6 §5.2 matched TRL-vs-Tinker: all trained runs below the untrained base of 0.345, with ~98% truncation).** The "below base" result is **a truncation artefact**.
- Under the original protocol (512 tokens, thinking on), 88.5% of base outputs never close `</think>` and 88.5% contain no `\boxed{}`. About 97% of the original "accuracy" is scored on truncated text: whichever number happens to come last in an unfinished chain of thought.
- In that regime the base vs adapter differences are noise. They swing between −0.015 and +0.04 here, with 34–50 discordant items out of 200.
- With a budget that does not truncate, the three Tinker adapters score **96.5 / 97.5 / 97.5%** against a base of **96.0%** (thinking off, 0% truncation). With thinking on at 4,096 tokens they score **94.5 / 95.0 / 93.5%** against 92.5%. None of them is below base. The gains are +0.5 to +2.5 pp and none is significant (McNemar p ≥ 0.06).
- The three TRL adapters could not be re-evaluated. They existed only at `/content/matched/trl_s*_adapter` on the original Colab VM, and no HF or Tinker copy exists. Their original texts show the same pattern: 98–98.5% truncation and 89% unclosed `</think>`.

## Protocols (read from the repo)

| | Q1 (P8 `scale_gsm8k_qwen3-8b`) | Q2 (P6 §5.2 matched run) |
|---|---|---|
| Training script | `platform_hybrid/experiments/tinker-runs/scripts/world_class_experiments.py` (lr 3e-5, r32, G=8, 2 prompts/step, 30 steps, seed 42) | `platform_colab/publication_fixes/matched_common.py` + `matched_tinker.py` / `matched_trl_colab.py` (lr 1e-5, r32, G=8, 2 prompts/step, 30 steps, seeds 42/123/456) |
| Prompt | system "math assistant … \boxed{}" + raw question, raw ChatML, no think stub | same, plus the suffix " Provide a numerical answer without units, written inside \boxed{}." |
| Decoding | temp 0.8, top_p 0.95, **max_tokens 512** | train: temp 0.8 / 0.95 / **512**. Eval (`matched_eval_colab.py`): **greedy, 512**, eos [151645, 151643] |
| Reward | first `\boxed{}` match, else **last number in the text** (`reward_math`) | identical (`reward_fn`, campaign_v2) |
| Held-out set | none defined for this run, so this re-run uses GSM8K test first 200 | GSM8K test first 200 (`eval_heldout.json`) |

Thinking mode is on in both, because Qwen3's default is to think when the prompt carries no `<think>\n\n</think>` stub.

Side note on the P6 text: `p6s_worked_examples.tex` says "16 prompts/step, G=8", but `matched_common.py` sets `PROMPTS_PER_STEP = 2`, giving 16 rollouts per step. Worth correcting.

## Conditions run

- `<prompt>_think_512` reproduces training and the original eval: raw prompt, thinking on, greedy, 512 tokens.
- `<prompt>_nothink_4096` is the raw prompt plus `<think>\n\n</think>\n\n` (what the Qwen3 template emits for `enable_thinking=False`), greedy, 4,096 tokens.
- `<prompt>_think_4096` is a diagnostic: thinking on with a non-truncating budget (≤9% truncation).
- `wc` is the Q1 prompt and `matched` is the Q2 prompt.

The metrics are defined as follows:
- **acc (training reward):** the verbatim campaign reward, i.e. boxed match with a last-number fallback.
- **strict boxed:** the boxed-only score.
- **truncation:** vLLM `finish_reason == "length"`.
- **parse fail:** no `\boxed{}` in the output.

Adapters were merged into the base with PEFT `merge_and_unload` before generation:
- Tinker `unembed_tokens` was renamed to `lm_head`.
- The Q1 adapter trains lm_head as well. The matched adapters train only the 7 projections.
- All adapters loaded with 0 all-zero lora_B tensors (`raw/merge_*.log`).

## Results (n = 200 each, GSM8K test items 0–199, greedy)

| model | condition | n | accuracy (training reward) [95% Wilson CI] | strict-boxed acc | truncation rate | parse failures (no `\boxed`) | `</think>` unclosed | mean tokens | runtime (gen s) |
|---|---|---|---|---|---|---|---|---|---|
| base | wc_think_512 | 200 | 0.270 [0.213, 0.335] | 0.040 | 1.000 | 0.960 | 0.960 | 512 | 15 |
| base | wc_nothink_4096 | 200 | **0.975** [0.943, 0.989] | 0.915 | 0.000 | 0.000 | - | 281 | 12 |
| base | wc_think_4096 | 200 | 0.940 [0.898, 0.965] | 0.930 | 0.085 | 0.065 | 0.065 | 1864 | 151 |
| q1_scale8b (P8 adapter) | wc_think_512 | 200 | 0.310 [0.250, 0.377] | 0.055 | 0.995 | 0.945 | 0.930 | 512 | 15 |
| q1_scale8b (P8 adapter) | wc_nothink_4096 | 200 | **0.965** [0.930, 0.983] | 0.900 | 0.000 | 0.000 | - | 277 | 13 |
| q1_scale8b (P8 adapter) | wc_think_4096 | 200 | 0.935 [0.892, 0.962] | 0.930 | 0.085 | 0.065 | 0.065 | 1787 | 139 |
| base | matched_think_512 | 200 | 0.305 [0.245, 0.372] | 0.090 | 0.990 | 0.910 | 0.900 | 512 | 14 |
| base | matched_nothink_4096 | 200 | **0.960** [0.923, 0.980] | 0.960 | 0.000 | 0.000 | - | 261 | 18 |
| base | matched_think_4096 | 200 | 0.925 [0.880, 0.954] | 0.920 | 0.085 | 0.065 | 0.070 | 1729 | 133 |
| tinker_s42 | matched_think_512 | 200 | 0.295 [0.236, 0.362] | 0.125 | 0.985 | 0.875 | 0.880 | 512 | 15 |
| tinker_s42 | matched_nothink_4096 | 200 | **0.965** [0.930, 0.983] | 0.965 | 0.000 | 0.000 | - | 256 | 12 |
| tinker_s42 | matched_think_4096 | 200 | 0.945 [0.904, 0.969] | 0.935 | 0.070 | 0.045 | 0.055 | 1685 | 131 |
| tinker_s123 | matched_think_512 | 200 | 0.320 [0.259, 0.388] | 0.105 | 0.995 | 0.895 | 0.885 | 512 | 15 |
| tinker_s123 | matched_nothink_4096 | 200 | **0.975** [0.943, 0.989] | 0.975 | 0.000 | 0.000 | - | 257 | 14 |
| tinker_s123 | matched_think_4096 | 200 | 0.950 [0.910, 0.973] | 0.940 | 0.055 | 0.055 | 0.055 | 1695 | 129 |
| tinker_s456 | matched_think_512 | 200 | 0.290 [0.232, 0.356] | 0.110 | 0.980 | 0.890 | 0.885 | 511 | 15 |
| tinker_s456 | matched_nothink_4096 | 200 | **0.975** [0.943, 0.989] | 0.975 | 0.000 | 0.000 | - | 258 | 12 |
| tinker_s456 | matched_think_4096 | 200 | 0.935 [0.892, 0.962] | 0.925 | 0.090 | 0.065 | 0.070 | 1723 | 132 |

The model-load time for each vLLM engine was about 145 s per model, on top of the runtimes above. Merging took 80–98 s per adapter on CPU.

Accuracy splits into a clean pattern by whether the output was cut off:
- **Untruncated outputs** score 98–100% under the thinking conditions and 96–97.5% under nothink_4096.
- **Truncated outputs** score 18–43%. That is the last-number fallback guessing from inside an unfinished chain of thought.

### Adapter minus base, paired on the same 200 items

The middle column gives the discordant counts: (base wrong / adapter right, base right / adapter wrong).

| adapter | think_512 Δ (discordant, McNemar p) | nothink_4096 Δ | think_4096 Δ |
|---|---|---|---|
| q1_scale8b | +0.040 (29/21, p=0.32) | −0.010 (0/2, p=0.5) | −0.005 (2/3, p=1.0) |
| tinker_s42 | −0.010 (16/18, p=0.86) | +0.005 (3/2, p=1.0) | +0.020 (5/1, p=0.22) |
| tinker_s123 | +0.015 (21/18, p=0.75) | +0.015 (3/0, p=0.25) | +0.025 (5/0, p=0.06) |
| tinker_s456 | −0.015 (20/23, p=0.76) | +0.015 (3/0, p=0.25) | +0.010 (4/2, p=0.69) |

### Original P6 eval re-scored from its stored texts (`eval_heldout.json`, transformers, greedy 512)

`orig_p6_eval_diagnostics.md` has the full table. In short:
- Base was 0.345 with 98.5% truncation, 88.5% of outputs had no `\boxed`, and 88.5% never closed `</think>`.
- On the truncated subset, base scores 0.335. Nearly all of its "accuracy" therefore comes from truncated text.
- All six trained runs show the same signature: 97.5–99.5% truncation and 86–90% unclosed `</think>`.

In this re-run, base under the same protocol scores 0.305 with vLLM instead of 0.345 with transformers. The difference is within the CI and reflects how fragile last-number scoring of truncated text is to small numerical differences in decoding.

## Caveats

- n = 200 per condition, single greedy decode, on the first 200 test items only. P8's "~82% base" figure comes from a different, unspecified protocol. Here base reaches 96–97.5% without thinking at 4,096 tokens.
- The P8 W&B runs have no history. The Tinker run `3154dee5…` has no live checkpoints, but the HF `final/` adapter matches its URI (`training_run.json`).
- The matched Tinker adapters were fetched read-only from Tinker sampler checkpoints `m{42,123,456}_final`; archive SHA-256 values are in `tinker_adapters_meta.json`. The TRL adapters are not recoverable.

## Compute

- **L4** (session `gsmeval`, about 14:58–15:33 UTC): about 0.58 h. Setup only. It was abandoned because a 14k-token KV cache gave 3× concurrency at 4.6k context, and an early vLLM `ninja` JIT failure ended the base run before it produced results.
- **A100-SXM4-40GB** (15:33–16:07 UTC): about 0.57 h, which covers all results above.
- **Total about 1.15 GPU-hours.** Both sessions were stopped; `colab sessions` reports "No active sessions found on server."

## Files

- `scripts/gsm8k_eval.py` is the eval (merge + vLLM generation + scoring).
- `scripts/run_all.sh` and `scripts/bootstrap.sh` are the remote drivers.
- `scripts/fetch_tinker_adapters.py` is the read-only Tinker checkpoint fetch.
- `scripts/upload_adapter.sh` and `scripts/rsh.py` are Colab helpers.
- `scripts/table.py` renders the results table.
- `raw/` holds gzipped generations and summaries (logs are gitignored). The downloaded Tinker matched adapters (about 1 GB) are not checked in; their archive SHA-256 values are in `tinker_adapters_meta.json`.
