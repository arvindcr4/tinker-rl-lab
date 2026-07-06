#!/usr/bin/env python
"""
P1 white-box — SCALED per-layer adaptation profile under GRPO-style updates.

Scaled version of experiments/openings/p1_layer_profile.py. Differences:
  * Larger model:  Qwen/Qwen2.5-3B-Instruct   (was 1.5B)
  * REAL data:     openai/gsm8k (main/train), ~24 problems   (was hardcoded arithmetic)
  * More steps:    10                                        (was 5)
  * Multi-seed:    seeds {0, 1}, aggregated across seeds      (was single run)

Measures, per seed:
  * per-layer LoRA grad-norm over GRPO-style advantage-weighted updates
  * step1 -> final top-k (top-25% of layers) overlap
  * top-25% gradient-norm concentration share
then aggregates (mean +/- spread) across seeds.

Needs a GPU (per-layer gradient access; Tinker can't do this). Launch via `colab run`.
Prints `RESULT: {json}` to stdout.
"""
import subprocess
import sys


# ----------------------------------------------------------------------------
# 0. Self-install deps (fresh Colab VM). Idempotent-ish; quiet.
# ----------------------------------------------------------------------------
def _pip_install():
    pkgs = [
        "torchao>=0.16",
        "transformers",
        "peft",
        "datasets",
    ]
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-U", *pkgs]
    )


# Fresh Colab VMs ship torchao 0.10, which imports fine but fails transformers'
# runtime version check (>=0.16) at model-load. So upgrade UNCONDITIONALLY before
# importing transformers — a guard on import success is not enough.
_pip_install()

import json
import re
import statistics

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
N_PROBLEMS = 24
N_STEPS = 10
SEEDS = [0, 1]
GROUP_SIZE = 4          # rollouts per problem (GRPO group) — trimmed to fit L4 24GB
MAX_NEW_TOKENS = 192
LR = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


# ----------------------------------------------------------------------------
# 1. GSM8K helpers
# ----------------------------------------------------------------------------
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def gold_answer(answer_field: str) -> str:
    """GSM8K gold answer is after '####'."""
    tail = answer_field.split("####")[-1].strip()
    tail = tail.replace(",", "").replace("$", "")
    m = _NUM_RE.search(tail)
    return m.group(0).replace(",", "") if m else tail


def extract_pred(text: str) -> str:
    """Last number in the generated completion."""
    nums = _NUM_RE.findall(text.replace(",", ""))
    return nums[-1] if nums else ""


def reward_fn(completion: str, gold: str) -> float:
    pred = extract_pred(completion)
    if pred == "":
        return 0.0
    try:
        return 1.0 if abs(float(pred) - float(gold)) < 1e-4 else 0.0
    except ValueError:
        return 1.0 if pred == gold else 0.0


def load_problems(tokenizer):
    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.select(range(N_PROBLEMS))
    problems = []
    for ex in ds:
        msgs = [
            {
                "role": "system",
                "content": "Solve the math problem. End with 'The answer is <number>'.",
            },
            {"role": "user", "content": ex["question"]},
        ]
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        problems.append({"prompt": prompt, "gold": gold_answer(ex["answer"])})
    return problems


# ----------------------------------------------------------------------------
# 2. Per-layer LoRA grad-norm bookkeeping
# ----------------------------------------------------------------------------
_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def layer_of(param_name: str):
    m = _LAYER_RE.search(param_name)
    return int(m.group(1)) if m else None


def per_layer_grad_norms(model, n_layers):
    norms = [0.0] * n_layers
    for name, p in model.named_parameters():
        if p.grad is None or "lora" not in name.lower():
            continue
        li = layer_of(name)
        if li is None or li >= n_layers:
            continue
        norms[li] += p.grad.detach().float().norm().item() ** 2
    return [n ** 0.5 for n in norms]


# ----------------------------------------------------------------------------
# 3. One GRPO-style step: sample a group, weight logprobs by advantage, backward
# ----------------------------------------------------------------------------
def grpo_step(model, tokenizer, problem):
    model.eval()
    enc = tokenizer(problem["prompt"], return_tensors="pt").to(DEVICE)
    prompt_len = enc.input_ids.shape[1]

    with torch.no_grad():
        gen = model.generate(
            **enc,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            num_return_sequences=GROUP_SIZE,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    seqs = gen  # (G, prompt_len + gen_len)
    rewards = []
    for g in range(GROUP_SIZE):
        comp = tokenizer.decode(seqs[g, prompt_len:], skip_special_tokens=True)
        rewards.append(reward_fn(comp, problem["gold"]))

    rt = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
    adv = rt - rt.mean()
    if rt.std() > 1e-6:
        adv = adv / (rt.std() + 1e-6)
    # Degenerate group (all same reward) -> no signal this problem.
    if float(adv.abs().sum()) < 1e-6:
        return False

    # Advantage-weighted policy-gradient loss over the group.
    model.train()
    attn = (seqs != (tokenizer.pad_token_id or tokenizer.eos_token_id)).long()
    out = model(input_ids=seqs, attention_mask=attn)
    logits = out.logits[:, :-1, :]
    targets = seqs[:, 1:]
    # Memory-efficient token log-prob (the L4 OOM culprit was log_softmax(logits.float())
    # which materializes a full-vocab float32 tensor). log p(t) = logit_t - logsumexp(logits),
    # computed in the model dtype without a (G, L, V) float32 intermediate.
    tok_logp = (logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                - torch.logsumexp(logits, dim=-1))

    # Mask to completion tokens only.
    comp_mask = torch.zeros_like(targets, dtype=torch.float32)
    comp_mask[:, prompt_len - 1:] = 1.0
    comp_mask = comp_mask * attn[:, 1:].float()

    seq_logp = (tok_logp * comp_mask).sum(dim=-1) / comp_mask.sum(dim=-1).clamp(min=1)
    loss = -(adv * seq_logp).mean()
    loss.backward()
    return True


# ----------------------------------------------------------------------------
# 4. Run one seed
# ----------------------------------------------------------------------------
def run_seed(seed, problems, tokenizer):
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE
    ).to(DEVICE)
    lcfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lcfg)
    n_layers = model.config.num_hidden_layers
    opt = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )

    step_norms = []  # per-step per-layer grad norms
    for step in range(N_STEPS):
        prob = problems[step % len(problems)]
        opt.zero_grad(set_to_none=True)
        did = grpo_step(model, tokenizer, prob)
        if not did:
            step_norms.append([0.0] * n_layers)
            continue
        step_norms.append(per_layer_grad_norms(model, n_layers))
        opt.step()

    # Aggregate.
    k = max(1, round(n_layers * 0.25))
    step1 = step_norms[0]
    total = [sum(step_norms[s][l] for s in range(N_STEPS)) for l in range(n_layers)]

    top1 = set(sorted(range(n_layers), key=lambda l: step1[l], reverse=True)[:k])
    topF = set(sorted(range(n_layers), key=lambda l: total[l], reverse=True)[:k])
    overlap = len(top1 & topF) / k

    tot_sum = sum(total) or 1.0
    conc = sum(sorted(total, reverse=True)[:k]) / tot_sum

    mean_by_layer = [t / N_STEPS for t in total]

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return {
        "seed": seed,
        "n_layers": n_layers,
        "step1_predicts_final_topk_overlap": overlap,
        "concentration_top25pct_share": conc,
        "top_layers_overall": sorted(topF),
        "mean_gradnorm_by_layer": [round(x, 4) for x in mean_by_layer],
    }


# ----------------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------------
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    problems = load_problems(tokenizer)

    per_seed = [run_seed(s, problems, tokenizer) for s in SEEDS]

    overlaps = [r["step1_predicts_final_topk_overlap"] for r in per_seed]
    concs = [r["concentration_top25pct_share"] for r in per_seed]

    def _std(xs):
        return statistics.pstdev(xs) if len(xs) > 1 else 0.0

    result = {
        "model": MODEL_NAME,
        "dataset": "openai/gsm8k:main:train",
        "n_problems": N_PROBLEMS,
        "steps": N_STEPS,
        "group_size": GROUP_SIZE,
        "seeds": SEEDS,
        "n_layers": per_seed[0]["n_layers"],
        "step1_predicts_final_topk_overlap_mean": round(statistics.mean(overlaps), 4),
        "step1_predicts_final_topk_overlap_std": round(_std(overlaps), 4),
        "concentration_top25pct_share_mean": round(statistics.mean(concs), 4),
        "concentration_top25pct_share_std": round(_std(concs), 4),
        "per_seed": per_seed,
        "compute": "Colab GPU (colab run)",
    }
    # write to a file too, so a dropped colab-run/exec connection doesn't lose the result
    try:
        open("/content/p1_scaled_result.json", "w").write(json.dumps(result, indent=2))
    except Exception:
        pass
    print("RESULT: " + json.dumps(result))


if __name__ == "__main__":
    main()
