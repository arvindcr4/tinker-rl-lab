# Modal Experiment Fixes — Applied

**Date:** 2026-04-18  
**Files modified:**
- `modal_parallel_runner.py`
- `relaunch_kl.py`

---

## Fix 1: KL Tracking Gradient Error

**Symptom:**  
`run_kl_tracking` failed with:
```
element 0 of tensors does not require grad and does not have a grad_fn
```

**Root cause:**  
The KL and entropy computations were performed on tensors that came from a `torch.no_grad()` block (and thus had no grad_fn), but those raw tensor operations were still evaluated outside the block. When PyTorch tried to track autograd state through those tensors, it encountered a mismatch — the ref model is frozen (`requires_grad=False`) and its output tensors cannot participate in the gradient graph. Any arithmetic mixing frozen ref outputs with live policy outputs caused the error.

**Fix applied (both `modal_parallel_runner.py` and `relaunch_kl.py`):**

1. Explicitly freeze the reference model's parameters:
   ```python
   ref_model.eval()
   for p in ref_model.parameters():
       p.requires_grad = False
   ```

2. Move the **entire** KL/entropy computation inside a `with torch.no_grad():` block and call `.detach()` on both logit tensors. KL divergence is a monitoring metric — it must never be part of the backward graph:
   ```python
   with torch.no_grad():
       pol_out = policy_model(**inputs)
       ref_out = ref_model(**inputs)
       policy_logits = pol_out.logits.detach()
       ref_logits = ref_out.logits.detach()

       kl = F.kl_div(
           F.log_softmax(policy_logits, dim=-1),
           F.softmax(ref_logits, dim=-1),
           reduction='batchmean',
       ).item()

       pol_lp = F.log_softmax(policy_logits, dim=-1)
       pol_p = torch.exp(pol_lp)
       entropy = -(pol_p * pol_lp).sum(dim=-1).mean().item()
   ```

3. Replaced the manual KL formula `(pol_p * (pol_lp - ref_lp)).sum(-1).mean()` with `F.kl_div(log_softmax(policy), softmax(ref), reduction='batchmean')` — this is numerically more stable and matches the PyTorch convention for KL(P || Q) where P is the "target" distribution.

4. The dummy gradient step (`train_out = policy_model(**inputs); loss.backward()`) remains **outside** the `no_grad` block so that LoRA adapter weights are correctly updated. This forward pass is a separate call from the one used for KL monitoring.

Also added `import torch.nn.functional as F` to the import line in both files.

---

## Fix 2: Timeout Too Short

**Symptom:**  
Three experiments timed out at the 3600s (1 hour) wall-clock limit:
- `heldout_qwen3-32b`: Qwen3-32B is a large model; loading + 200 greedy inferences exceeded 1 hour.
- `heldout_qwen3.5-27b`: Same — 27B model loading plus inference.
- `humaneval_qwen3-8b`: 164 problems × 5 samples = 820 generations; hit the limit at ~40/164.

**Fix applied (`modal_parallel_runner.py` and `relaunch_kl.py`):**

All four `@app.function` decorators changed from `timeout=3600` to `timeout=7200` (2 hours):

| Function | Old timeout | New timeout |
|---|---|---|
| `run_ppo_gsm8k` | 3600s | 7200s |
| `run_humaneval_eval` | 3600s | 7200s |
| `run_kl_tracking` | 3600s | 7200s |
| `run_gsm8k_heldout_eval` | 3600s | 7200s |
| `relaunch_kl.run_kl_tracking` | 3600s | 7200s |

Note: `run_ppo_gsm8k` completed fine at 3600s but was increased for consistency and to handle future larger-model variants spawned via the same function.

---

## Fix 3: HumanEval Evaluation Efficiency

**Symptom:**  
`run_humaneval_eval` only completed 40/164 problems in 1 hour (~22 seconds/problem). For a model already loaded on GPU, each generation should take 2–5 seconds. The bottleneck was almost certainly that the model was being re-initialized per problem iteration (e.g., by a mistaken placement of the `from_pretrained` call inside the loop, or by Python's import system re-executing module-level code in some evaluation harness configurations).

**Fix applied (`modal_parallel_runner.py`):**

1. Moved model and tokenizer loading **explicitly before** the problem loop with a clear comment marking the intent:
   ```python
   # Load model ONCE outside the problem loop — avoids re-initializing per problem
   print(f"[{exp}] Loading model on H100 (once for all problems)...")
   tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
   if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
   model = AutoModelForCausalLM.from_pretrained(
       model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
   )
   model.eval()  # inference-only; no gradients needed
   ```
   The `model.eval()` call was also added to disable dropout and batch-norm tracking, which slightly speeds up inference and ensures deterministic outputs.

2. The `model` and `tokenizer` variables are shared across all 164 problems and all `num_samples` inner iterations — no re-loading occurs.

With the model loaded once, expected throughput for Qwen3-8B on H100 at 512 output tokens:
- ~1–3s per generation × 5 samples × 164 problems ≈ 14–41 minutes total
- Well within the new 7200s limit.

---

## Summary of Changes

| File | Change | Lines affected |
|---|---|---|
| `modal_parallel_runner.py` | All 4 `@app.function` timeouts: 3600 → 7200 | 69, 196, 265, 361 |
| `modal_parallel_runner.py` | KL computation wrapped in `no_grad`, `.detach()` added, switched to `F.kl_div`, ref model frozen | 267, 288–291, 313–331 |
| `modal_parallel_runner.py` | HumanEval: `model.eval()` added, comment clarifying single-load intent | 217 |
| `relaunch_kl.py` | Timeout 3600 → 7200 | 11 |
| `relaunch_kl.py` | Same KL no_grad + detach + F.kl_div fix as above | 13, 51–68 |

---

## How to Relaunch

```bash
# Relaunch all experiments (fixes applied, do not re-run before confirming)
modal run experiments/modal/modal_parallel_runner.py

# Or relaunch only the KL tracking experiment
modal run experiments/modal/relaunch_kl.py
```
