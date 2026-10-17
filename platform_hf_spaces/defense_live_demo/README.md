---
title: Tinker RL Defense Demo
emoji: 🧪
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: Live and offline evidence for the Tinker RL defense
---

# Tinker RL Defense Demo

Defense-ready evidence surface for the M.Tech project **Reliable RL for LLMs**.

The four tabs are deliberately independent:

1. **Hosted tool call** uses the Hugging Face Inference Router when `HF_TOKEN` is available, and otherwise shows a clearly labeled deterministic fallback.
2. **Hosted math** exercises ordinary mathematical reasoning through the same Router. This is separate from tool calling.
3. **W&B evidence** presents a frozen, bundled snapshot of the matched-budget and run-hygiene results with links to the source runs.
4. **Offline provenance** verifies the bundled evidence locally, without a network request.

## Model and artifact boundary

- Live Router model: [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), served by an available Hugging Face Inference Provider.
- Project tool-use artifact: [`arvindcr4/tool-call-lora-qwen2.5-7b`](https://huggingface.co/arvindcr4/tool-call-lora-qwen2.5-7b).
- Project math artifact: [`arvindcr4/gsm8k-qwen3-4b`](https://huggingface.co/arvindcr4/gsm8k-qwen3-4b).

The project artifacts are linked as training evidence. They currently have no hosted Inference Provider mapping, so this Space does **not** mislabel the Router model as a served project checkpoint.

## Local run

```bash
python -m pip install -r requirements.txt
python app.py
```

`HF_TOKEN` is optional. Without it, both interactive inference tabs remain usable through explicitly labeled, precomputed defense fallbacks.


## Review-10 serving and evidence gates

The default is static evidence even when HF_TOKEN is configured. Live inference
requires explicit `LIVE_INFERENCE_POLICY=bounded_public`, an absolute persistent
`LIVE_BUDGET_DB`, and `LIVE_TOTAL_CALL_LIMIT` between 1 and 100. The cap is a
persistent lifetime call count, reserved before requests, not a currency budget.
Use a shared quota service for independent replicas; separate databases do not
form a global cap. Inputs are limited to 4096 characters and requests time out
at 30 seconds. This policy was tested offline only and was not enabled.

Offline PASS now requires `EVIDENCE_MANIFEST_PATH` and an independently trusted
`EVIDENCE_MANIFEST_SHA256`. No trusted release manifest was supplied by review10,
so absent anchors return FAIL. Live result fetching requires a reviewed
`results_index.json` with exact framework/task/model-revision/seed/protocol
identities and pinned source locations. Missing indices report unavailable;
previous broad project listings are not relabeled as requested experiments.
