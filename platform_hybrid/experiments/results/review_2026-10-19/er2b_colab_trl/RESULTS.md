# E-R2b on Colab with TRL: four arms trained, held-out GSM8K pass@k

Run on 2026-10-19 on one Colab A100-SXM4-40GB (session `er2b`). Wall time was about 3h50m from the session start at 03:09 UTC to the stop at 07:00 UTC. That covers setup, the smoke test, about 2h45m of training (2126 + 2141 + 2732 + 2734 s of in-arm wall time plus model loads) and 42m of eval. The run used **20.22 compute units** (balance went from 62.61 to 42.39), under the 40-CU cap. The session is stopped and `colab sessions` lists none.

## Summary

- **G=16 arms reproduce the July Tinker training traces.** Last-10 reward was 0.284 / 0.354 here against 0.322 / 0.389 in July. Final-20 ZVF was 0.213 / 0.119 against 0.194 / 0.113.
- **G=2 arms match July only up to about step 60.** In both July G=2 runs, completion length dropped from 512 to about 230–280 tokens between steps 60 and 100. Train reward rose to about 0.9–0.97 at the same time, and ZVF rose to about 0.98. Neither TRL G=2 run did this within 160 steps. Length stayed at about 510, last-10 reward was 0.46 / 0.43, and final-20 ZVF was 0.66 / 0.69 (block table below). So the July G=2 headline numbers (last10 ≈ 0.9, ZVF ≈ 0.98) did **not** replicate on this stack.
- **Held-out, on one backend (Colab vLLM), both arms beat base at pass@1.** G=2 pooled pass@1 was 0.483 against base 0.306 (Δ +0.177, CI [0.155, 0.199]). G=16 pooled was 0.356 (Δ +0.050, CI [0.036, 0.064]). G=2 minus G=16 pooled, paired by problem: Δpass@1 +0.127 [0.111, 0.143], p = 1e-4; Δpass@8 +0.048 [0.034, 0.063], p = 1e-4; Δpass@32 +0.025 [0.0075, 0.0425], p = 0.012. These pass@k gaps also include the 8x difference in optimizer steps (160 vs 20) that the matched-budget design implies.

## Protocol

### Training

Script: `scripts/er2b_trl_colab.py`, adapted from `platform_colab/publication_fixes/matched_trl_colab.py`, which was not modified. The driver is `scripts/run_all.py`.

| item | July Tinker (`live_zvf_probe_dd6c231.py`, args from the W&B metadata) | this run (TRL) |
|---|---|---|
| model | Qwen/Qwen3-8B | same, bf16 weights |
| adapter | LoRA rank 4, Tinker defaults | LoRA r=4, alpha=32, dropout 0, on q,k,v,o,gate,up,down. PEFT keeps the adapter in fp32 (the safetensors file is 44 MB). |
| optimizer | Tinker Adam: lr 3e-5, betas (0.9, 0.95), eps 1e-8 | torch `adamw_torch_fused`: lr 3e-5 constant, betas (0.9, 0.95), eps 1e-8, weight decay 0, no warmup, no grad clipping (`max_grad_norm=0`) |
| prompts | `load_gsm8k_examples(256, seed)`, then `rng.sample(examples, 8)` per step with `rng = Random(seed)` | the same functions, copied verbatim, give the same 8 prompts per step. `stream_order_ok = true` in every train JSON. |
| prompt format | raw ChatML with SYSTEM_PROMPT and QUESTION_SUFFIX, thinking on, no think stub | same strings, passed as non-conversational prompts so TRL applies no chat template |
| sampling | T 1.0, top_p 0.95, 512 tokens, unseeded | T 1.0, top_p 0.95, top_k off, 512 new tokens, eos ids [151645, 151643]. HF `generate` in chunks of 160 sequences, torch RNG seeded by the trainer seed. |
| budget | G=2 x 160 steps, G=16 x 20 steps, 8 prompts per step, 2,560 rollouts per arm | identical |
| sampler refresh | `save_every=5`: sampler weights are refreshed after every 5 optimizer steps | `steps_per_generation = 5 x grad_accum`: each 5-step block is sampled from the policy at the start of the block. TRL's shuffle of the generation batch is disabled, so each optimizer step holds exactly that July step's 8 groups. |
| reward | `reward_fn`: any boxed match, else the last number | `reward_fn` copied verbatim |
| advantage | (r − mean_g) / (population std_g + 1e-8) | TRL's advantage is overwritten with the same formula on the same rewards. The script asserts that sign and zero pattern match TRL's own advantage. |
| loss | mean over sequences of −A · Σ logprobs, with no ratio, KL or clip | `loss_type="luspo"`, token level. That is (−coef·A·mask).sum over tokens, then mean over sequences. `old_per_token_logps` is dropped, so coef = exp(logp − sg(logp)) = 1 and the gradient is −A·∇Σlog p. Micro-batches are 8 completions with grad-accum 2 (G=2) or 16 (G=16), all the same size, so the mean over sequences is exact. beta = 0. |
| logging | per step: reward, ZVF, mean completion length | the same, plus truncation fraction and the TRL log, in `train/er2b_trl_<arm>.json` (`step_log`) |

### Held-out eval

Script: `scripts/er2b_vllm_passk.py`.

- **Problems:** GSM8K test, `random.Random(42).shuffle`, first 200.
- **Prompt and grader:** `build_prompt`, `GSM8K_SUFFIX`, `gsm8k_reward` (parser v2: last boxed value, else last number) and `pass_at_k`, copied verbatim from `zvf-program/experiments-next/modal_passk.py`.
- **Sampling:** 32 samples per problem, T 1.0, top_p 1.0, max_tokens 512.
- **Backend:** vLLM 0.32.0 on the same A100, base bf16. Adapters were loaded as vLLM LoRA (`max_lora_rank=8`), with CUDA graphs on and prefix caching on.
- **Seeds:** each sample is its own request with its own seed, `42*1_000_003 + i*32 + j`, as in `tinker_passk.py`. The 32 samples were checked for collapse: `min_unique_texts_per_problem = 32` for all five models, so no two samples of any problem were identical.
- **Recorded per model:** truncation (`finish_reason == "length"`) and mean tokens.
- **Statistics:** `scripts/analyze.py`, a copy of `er2b_heldout_eval/scripts/analyze.py` with only stems and labels changed. Paired-by-problem percentile bootstrap (10,000 resamples) and two-sided sign-flip test (10,000 flips), both with seed 0.

### Smoke test (`smoke/`)

- **Training:** G=16 s123, 2 steps with block = 2. Step 1, which included sampling 256 completions, took 187 s. Step 2, training only, took 32 s.
- **Eval:** base and the smoke adapter on 10 problems x 32 samples, about 30 s per model.
- **Projection:** about 2.8 h of training plus about 0.8 h of eval, roughly 20–22 CU. The run came in at 20.22 CU.

## Environment

Python 3.13.15, torch 2.13.0+cu130, transformers 5.18.0, trl 1.16.0, peft 0.21.1, datasets 4.8.5, accelerate 1.15.0, vllm 0.32.0. GPU: NVIDIA A100-SXM4-40GB, driver 580.82.07. The preinstalled torchao was uninstalled. TRL warns that it supports vLLM only up to 0.31; this did not matter, because TRL training does not use vLLM here and vLLM was only called directly for eval.

## Differences from the July Tinker runs that could not be removed

1. **Unembedding not trained.** `create_lora_training_client(base_model, rank)` uses Tinker's defaults. Per the probe note in `matched_common.py`, those defaults also train the unembedding (lm_head) LoRA. TRL 1.16 / PEFT rejects lm_head LoRA here, so only the 7 projections are trained.
2. **LoRA alpha is assumed.** Alpha = 32 comes from the Tinker adapter config probed at r = 32. The value Tinker uses at r = 4 was not probed.
3. **Spurious target token in the July loss.** The July datum is `target_ids = full_ids[1:] + [0]`, and `loss_fn` sums logprobs over every target position.
   - The prompt-token terms cancel exactly within a group, because the advantages sum to zero and the prompt logprob is the same for every member.
   - The last term, −A_i · log p(token 0 | x, y_i), does not cancel. It is one spurious token per sequence and is absent here.
4. **Optimizer internals.** Tinker's server-side Adam state precision and update numerics are not visible. This run used torch fused AdamW (wd 0) on fp32 LoRA parameters with a bf16 base, and a bf16 HF forward with gradient checkpointing.
5. **Sampler.** Generation is HF `generate` with an unmerged PEFT LoRA, not Tinker's sampler. Random streams differ (July sampling was unseeded). Tinker's default stop handling was not specified in the July call, while TRL stops on eos ids [151645, 151643]. Logprobs are full-softmax at T = 1, as on Tinker.
6. **Loss normalisation.** The July per-sequence logprob sum also includes the prompt and dummy terms in item 3. Otherwise the `luspo` reduction (sum over completion tokens, mean over sequences) is the July reduction, with no constant rescaling.
7. **Eval backend.** Colab vLLM differs from the Tinker sampler used for the October Tinker-rerun pass@k (`er2b_heldout_eval/raw`). Those numbers are listed only as a cross-backend reference. Base agrees across backends at pass@1 (0.306 here, 0.303 on Tinker, 0.306 on Modal vLLM per commit 0cc717ea2).

## Training: per arm vs July

Last-10 train reward and final-20 ZVF, from `analysis.md`:

| arm | steps (July / TRL) | July last10 | TRL last10 | July ZVF final20 | TRL ZVF final20 | TRL truncation, last 20 steps |
|---|---:|---:|---:|---:|---:|---:|
| G=2 s123 | 160 / 160 | 0.9000 | 0.4625 | 0.9750 | 0.6625 | 0.972 |
| G=2 s456 | 160 / 160 | 0.9625 | 0.4250 | 0.9813 | 0.6875 | 0.947 |
| G=16 s123 | 20 / 20 | 0.3219 | 0.2836 | 0.1938 | 0.2125 | 0.998 |
| G=16 s456 | 20 / 20 | 0.3891 | 0.3539 | 0.1125 | 0.1188 | 0.990 |

G=2 trajectories in 20-step blocks. Each cell is July / TRL, as mean reward, mean ZVF and mean completion length:

| steps | s123 reward | s123 ZVF | s123 length | s456 reward | s456 ZVF | s456 length |
|---|---|---|---|---|---|---|
| 1–20 | 0.278 / 0.284 | 0.719 / 0.731 | 512 / 512 | 0.291 / 0.281 | 0.644 / 0.650 | 512 / 512 |
| 41–60 | 0.300 / 0.334 | 0.700 / 0.656 | 512 / 511 | 0.431 / 0.322 | 0.613 / 0.681 | 508 / 511 |
| 61–80 | 0.350 / 0.334 | 0.650 / 0.631 | 509 / 511 | 0.781 / 0.362 | 0.850 / 0.650 | 321 / 511 |
| 81–100 | 0.912 / 0.300 | 0.925 / 0.725 | 274 / 512 | 0.963 / 0.366 | 0.975 / 0.644 | 230 / 511 |
| 141–160 | 0.925 / 0.456 | 0.975 / 0.662 | 280 / 511 | 0.959 / 0.425 | 0.981 / 0.688 | 222 / 508 |

The October Tinker rerun of G=2 s123 died at step 55 on billing. Its first 55 steps match both July and TRL: rewards 0.259 / 0.275 / 0.258 and ZVF 0.656 / 0.662 / 0.617 over steps 1–20, 21–40 and 41–55. So the gap opens only at the July length collapse, which this TRL stack never reached. Items 1–5 above list the stack differences that could explain it; none was isolated in this run.

## Held-out GSM8K (200 problems x 32 samples, Colab vLLM, one backend)

Each row compares against base, paired by problem:

| model | pass@1 | pass@8 | pass@32 | truncation | mean tokens | Δpass@1 [95% CI], p | Δpass@32 [95% CI], p |
|---|---:|---:|---:|---:|---:|---|---|
| base | 0.3058 | 0.7809 | 0.8950 | 0.9930 | 511.62 | — | — |
| G=2 s123 | 0.4880 | 0.8593 | 0.9350 | 0.9547 | 508.78 | +0.1822 [0.1584, 0.2061], p=1e-4 | +0.0400 [0.0100, 0.0750], p=0.041 |
| G=2 s456 | 0.4772 | 0.8511 | 0.9300 | 0.9506 | 508.88 | +0.1714 [0.1486, 0.1945], p=1e-4 | +0.0350 [0.0050, 0.0700], p=0.067 |
| G=16 s123 | 0.3478 | 0.7993 | 0.8950 | 0.9852 | 511.12 | +0.0420 [0.0277, 0.0563], p=1e-4 | 0.0000 [−0.0250, 0.0300], p=1.0 |
| G=16 s456 | 0.3634 | 0.8149 | 0.9200 | 0.9828 | 510.96 | +0.0577 [0.0408, 0.0744], p=1e-4 | +0.0250 [−0.0100, 0.0600], p=0.27 |
| G=2 pooled | 0.4826 | 0.8552 | 0.9325 | 0.9527 | 508.83 | +0.1768 [0.1545, 0.1991], p=1e-4 | +0.0375 [0.0100, 0.0700], p=0.024 |
| G=16 pooled | 0.3556 | 0.8071 | 0.9075 | 0.9840 | 511.04 | +0.0498 [0.0363, 0.0637], p=1e-4 | +0.0125 [−0.0125, 0.0400], p=0.48 |

A p of 1e-4 is the floor of the 10,000-flip test, (0 + 1) / 10001. The full-precision table and JSON are in `analysis.md` and `analysis.json`.

Cross-backend reference, not comparable row by row because the backends differ: the October Tinker-trained G=16 adapters scored on Tinker (`er2b_heldout_eval/raw`) give pass@1 0.394 / 0.441 and pass@32 0.94 / 0.95, against Tinker base 0.303 / 0.920.

## Adapters (not in the repo)

The adapter weights are stored locally at `/private/tmp/claude-501/-Users-arvind-Developer-agentic-repos-tinker-rl-lab/72e169ca-1f53-48fd-8bf4-ac450a9c4e66/scratchpad/er2b/adapters/er2b_trl_<arm>/`. That directory holds `adapter_model.safetensors`, `adapter_config.json`, `README.md` and `DONE`. It is session scratch space: move it somewhere durable to keep it. The Colab copy is gone now that the runtime is stopped.

| arm | adapter_model.safetensors sha256 | adapter_config.json sha256 |
|---|---|---|
| G=2 s123 | 988e61d72b2d85f57d7ef6eb62145cc3e3aa01792c03698eb50d323e516e5ec7 | 9e3793f3230b05e058c7f702e867dcc391ef8cf4bf0450bea299a090d96b238d |
| G=2 s456 | d55d6c0dbc8b7f91ee421f3482996289a6d00531f91c2ad03f881b7519df6fc8 | 79d679bd9be352f61ef355afc460320b84d2b7084ae14a3892146adec42d7baa |
| G=16 s123 | a39c731de1ade6fd95e9caea772840c20aa2ebc741156411dc67f77f7f48a34a | a73a11d74b02185851ecc6d035ec2148588c7c2f79eb437eedb3009ab5fef45e |
| G=16 s456 | e59808c330293ab2b8417d0f5cd69c0cdebae366cdeab57a47c68347bad93d5b | 85702623ca55dd6f3c0afc40224eb11eabea78c085276ad6f2e21f93cf7c2e81 |

## Files

- `scripts/er2b_trl_colab.py`: trains one arm.
- `scripts/run_all.py`: Colab driver. It skips trained arms and finished evals on restart.
- `scripts/er2b_vllm_passk.py`: held-out eval.
- `scripts/analyze.py`: statistics.
- `train/er2b_trl_{g2,g16}_s{123,456}.json`: per-step logs with config and environment.
- `raw/{base,er2b_trl_*}.json`: per-problem correct counts, unique-sample counts, truncation and mean tokens.
- `analysis.md`, `analysis.json`: output of `analyze.py`.
- `logs/`: per-arm train logs, eval log, driver log, and the keepalive / CU log.
- `smoke/`: smoke-test train JSON, smoke evals and smoke log.
