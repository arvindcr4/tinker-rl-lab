# Trained E1–E14 actor endpoint (Modal, vLLM): runbook

**Status (2026-09-26 14:35 UTC): deployed once, smoke-tested, then STOPPED.** Nothing is running. Code lives in `trained_actor/`.

## What is served
- Actor: `Qwen/Qwen3.6-35B-A3B`@`995ad96e` + seed809 stepfinal LoRA
  (`arvindcr4/pavlov-portfolio-qwen36-seed809-stepfinal-tinker-cf0ad8c1-1f1b-5ff-9f777c4018b6`@`64444133`, branch `checkpoint-seed809-stepfinal-9f777c4018b6`).
- Route: this is the one the original campaign used. The LoRA was **pre-merged into BF16 weights** on 2026-08-17 (`streaming_lora_delta_merge_v1`: 862/862 adapter tensors consumed, 331 targets, scaling 1.0). The merged weights are cached in Modal Volume `pavlov-e1-qwen36-hf-cache` at `/e1-qwen36-seed809-merged-5fb6490f740f` (pointer file `/e1-qwen36-seed809-merged-pointer.json`, per-shard sha256 listed there). The volume is mounted **read-only**. Nothing was re-merged.
- Server: vLLM 0.28.0+cu129 image and flags copied from `zvf-program/flagship/modal_public_portfolio_runtime.py` and `zvf-program/e1_wave10/recovered/public_colab_runtime_fast.py::server_command`. Flags: bf16, seed 809, decode-only CUDA graphs, `--reasoning-parser qwen3`, no prefix caching, max-model-len 32768. Two additions here: `--enable-auto-tool-choice --tool-call-parser qwen3_coder`, and `--max-num-seqs 16`.
- GPU: 1× **H200** (141 GB; the 72 GB of weights leave about 52 GB for KV cache), 4 CPU, 64 GiB RAM. `max_containers=1`, scale to zero after 300 s idle.
- **Model id: `pavlov-public-portfolio-bf16`**
- Endpoints: `/v1/chat/completions`, `/v1/completions` (raw), `/v1/models`. Tool calling and images (≤4 per prompt) are also available.

## Start (one command)
```bash
cd /Users/arvind/Developer/agentic_repos/tinker-rl-lab/outputs/e1_e14_small_scale_2026-09-26/trained_actor && modal deploy modal_trained_actor.py
```
The first request after a deploy, or after scale-to-zero, cold-starts the GPU. Plan for **≈3–5 min**. Warm it up and wait with:
```bash
set -a; source trained_actor/.env.local; set +a
curl -sf -m 900 -H "Authorization: Bearer $TRAINED_ACTOR_API_KEY" "$TRAINED_ACTOR_BASE_URL/v1/models"
```

## Stop
```bash
modal app stop -y pavlov-trained-actor-e1e14
modal app list | grep pavlov-trained-actor   # must show "stopped"
```
Idle containers scale to zero on their own after 300 s, but **the last lane to finish should stop the app explicitly.** Once stopped, run the deploy command again to restart. The URL stays the same.

## Base URL and auth
- `TRAINED_ACTOR_BASE_URL` = `https://arvindcr4--pavlov-trained-actor-e1e14-serve.modal.run` (append `/v1` for OpenAI SDKs).
- `TRAINED_ACTOR_API_KEY` is the bearer token that vLLM `--api-key` checks. Its value is stored in:
  - `trained_actor/.env.local` (gitignored by `.env.*`, chmod 600);
  - Modal Secret `pavlov-trained-actor-api`.

  Never print it or commit it.
- Load both vars with `set -a; source outputs/e1_e14_small_scale_2026-09-26/trained_actor/.env.local; set +a`.

## Client snippet
```python
import os
from openai import OpenAI
c = OpenAI(base_url=os.environ["TRAINED_ACTOR_BASE_URL"] + "/v1", api_key=os.environ["TRAINED_ACTOR_API_KEY"], timeout=900)
r = c.chat.completions.create(model="pavlov-public-portfolio-bf16", temperature=0, max_tokens=256,
        messages=[{"role": "user", "content": "What is 17 * 23?"}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})   # match PROTOCOL.md non-thinking default
print(r.choices[0].message.content)
# Raw completions: c.completions.create(model=..., prompt=<chat-templated string>, temperature=0, max_tokens=..., logprobs=1)
```
- Leave thinking on (the Qwen template default) only if the lane's protocol needs it. The reasoning text then comes back in `message.reasoning_content`.
- The endpoint is shared by 6 lanes on one GPU. vLLM runs at most 16 sequences at once and queues the rest.
- **Keep per-lane concurrency ≤ 4.** Use `stream=True` for generations longer than about 2 min, in case the Modal HTTP request limit applies (not tested).

## Smoke-test results (`trained_actor/smoke_test.py`, raw data in `trained_actor/smoke/`)
Setup: 5 fixed prompts, identical chat-templated prompt strings, `enable_thinking=False`, temperature 0, max 64 tokens. Three arms:
- Tinker base (`tinker 0.30.4`);
- this endpoint with the trained weights;
- a same-engine control: the same Modal/vLLM app serving the **base** snapshot from the same volume (app `…-basectl`, since stopped).

The logprob columns score the Tinker-base greedy tokens under each model (teacher-forced).

| Prompt | Tinker base = vLLM base text | vLLM base = **trained** text | mean / max \|Δlogprob\| trained vs vLLM base |
|---|---|---|---|
| 17×23 | yes | yes (`391`) | 0.000 / 0.000 |
| Fibonacci fn | yes | yes | 0.006 / 0.189 |
| shell: count lines | yes (`wc -l < /etc/passwd`) | **no** (`wc -l /etc/passwd`) | 0.025 / 0.193 |
| one-sentence summary | yes | **no** (diverges after 60 chars) | 0.020 / 0.136 |
| refuse unknown transfer | no (engine noise, diverges after 163 chars) | yes | 0.025 / 0.220 |

- **The adapter is active.** On the same engine, trained and base outputs differ in text on 2 of 5 prompts, and per-token logprobs shift by up to 0.22 nats.
- On both of those prompts, Tinker base and vLLM base agree with each other. The divergence is therefore caused by the weights, not by the serving stack.
- The effect is small, which fits a light RL LoRA. Engine noise between Tinker and vLLM (mean \|Δ\| of about 0.017 nats) is the same order as the adapter effect. **Compare results only within one engine.**
- Chat completions returned the same text as raw completions on all 5 prompts.

## Performance and cost
| Item | Value |
|---|---|
| GPU | Modal H200 ×1 (4 CPU, 64 GiB) |
| $/hr while a container is up | **$5.24** (H200 $4.54 + CPU 4×$0.0473 + RAM 64×$0.008, from `modal billing rates` on 2026-09-26) |
| Cold start (request → first `/v1/models` response) | 282 s trained (first boot after deploy); 196 s base control |
| Tokens/s, 1 stream (256-token greedy) | 97.6 |
| Tokens/s, 16 concurrent × 256 tokens | 404 aggregate |
| Idle tail per session | up to 300 s ≈ $0.44 unless stopped explicitly |
| Spend for this prep | **$0.80** from `modal billing report`, 14:00 UTC bucket: trained $0.49 + base control $0.32 (the report may still lag slightly) |

The trained app was up about 5.6 min and the base control about 3.7 min. The image build hit the existing layer cache for vLLM and ran on CPU only.
