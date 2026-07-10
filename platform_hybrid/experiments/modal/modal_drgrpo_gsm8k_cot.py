"""Modal: Dr. GRPO vs GRPO on GSM8K with chain-of-thought (the long-output
regime where Dr. GRPO's length-bias fix actually applies) + a measured pre->post
held-out McNemar generalization test (pillar 4).

Two arms share everything (Qwen2.5-1.5B-Instruct, GSM8K boxed-answer correctness
reward, token-level clipped surrogate, group size, compute, eval) and differ ONLY
in the two Dr. GRPO fixes:
  GRPO    : A=(r-mean_g)/(std_g+eps) ; loss normalized per-response length
  Dr.GRPO : A= r-mean_g             ; loss normalized by a constant (max_new)

Unlike the 5-token arithmetic probe, CoT answers are long and variable-length, so
the length-bias fix has something to act on. We log the mean completion-length
TRAJECTORY (Dr. GRPO is designed to curb GRPO's response-length inflation).

Each run also evaluates the SAME held-out GSM8K slice greedily BEFORE (pre-RL,
base) and AFTER training, storing per-item correctness -> exact McNemar pre->post.

Usage:
  modal run experiments/modal/modal_drgrpo_gsm8k_cot.py
"""
import modal
import json
import os
import time

app = modal.App("tinkerrl-drgrpo-gsm8k-cot")
results_vol = modal.Volume.from_name("tinkerrl-results", create_if_missing=True)
RESULTS_DIR = "/results"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.3.0", "transformers>=4.46.0", "peft>=0.13.0",
                 "datasets>=3.0.0", "numpy>=1.26.0,<2.0.0", "accelerate>=1.0.0",
                 "safetensors>=0.4.0", "huggingface-hub>=0.26.0")
)

SEEDS = [42, 123, 456]
ALGOS = ["grpo", "dr_grpo"]
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
N_STEPS = 30
N_PROMPTS = 8
GROUP = 8
K_EPOCHS = 2
CLIP = 0.2
MAX_NEW = 200
N_EVAL = 200
EPS = 1e-6
CHUNK = 4


@app.function(image=image, gpu="A10G", timeout=7200, volumes={RESULTS_DIR: results_vol}, retries=1)
def run_arm(algo: str, seed: int) -> dict:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import random
    import re
    import numpy as np
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = get_peft_model(
        AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, trust_remote_code=True).to(device),
        LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                   target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"),
    )
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-5)

    ds = load_dataset("openai/gsm8k", "main")
    train = ds["train"].shuffle(seed=seed)
    test = ds["test"].shuffle(seed=0).select(range(N_EVAL))

    def gold_of(ans):
        return ans.split("####")[-1].strip().replace(",", "")

    def extract(t):
        m = re.findall(r"\\boxed\{([^}]+)\}", t)
        cand = m[-1] if m else (re.findall(r"-?\d[\d,]*", t) or [None])[-1]
        return cand.replace(",", "").replace("$", "").strip() if cand else None

    def build(q):
        msgs = [{"role": "user", "content": q + "\nThink step by step, then give the final answer as \\boxed{ANSWER}."}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def reward(text, gold):
        p = extract(text)
        return 1.0 if (p is not None and p == gold) else 0.0

    def gen_batch(prompts, greedy):
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        plen = enc["input_ids"].shape[1]
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=not greedy,
                               temperature=(1.0 if not greedy else None),
                               top_p=(1.0 if not greedy else None),
                               pad_token_id=tok.pad_token_id)
        return enc, plen, g[:, plen:]

    def heldout():
        model.eval()
        corr = []
        for i in range(0, len(test), 16):
            ch = list(test)[i:i + 16]
            _, _, c = gen_batch([build(x["question"]) for x in ch], greedy=True)
            for x, t in zip(ch, tok.batch_decode(c, skip_special_tokens=True)):
                corr.append(int(reward(t, gold_of(x["answer"])) == 1.0))
        model.train()
        return corr

    pre_correct = heldout()  # base / pre-RL

    def token_logps_grad(full, attn, plen):
        outs_lp, outs_m = [], []
        for cs in range(0, full.shape[0], CHUNK):
            ce = min(cs + CHUNK, full.shape[0])
            logits = model(input_ids=full[cs:ce], attention_mask=attn[cs:ce]).logits[:, :-1, :].float()
            tgt = full[cs:ce][:, 1:]
            lp = F.log_softmax(logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[:, plen - 1:]
            cm = (tgt != tok.pad_token_id).float()[:, plen - 1:]
            outs_lp.append(lp); outs_m.append(cm); del logits
        return torch.cat(outs_lp), torch.cat(outs_m)

    step_log = []
    t0 = time.time()
    train_iter = iter(train)
    for step in range(N_STEPS):
        batch = []
        for _ in range(N_PROMPTS):
            try:
                batch.append(next(train_iter))
            except StopIteration:
                train_iter = iter(train.shuffle(seed=seed + step)); batch.append(next(train_iter))
        prompts, golds = [], []
        for ex in batch:
            for _ in range(GROUP):
                prompts.append(build(ex["question"])); golds.append(gold_of(ex["answer"]))
        enc, plen, comp_ids = gen_batch(prompts, greedy=False)
        comp_txt = tok.batch_decode(comp_ids, skip_special_tokens=True)
        rewards = np.array([reward(t, g) for t, g in zip(comp_txt, golds)], dtype=np.float32)

        R = rewards.reshape(N_PROMPTS, GROUP)
        gmean = R.mean(1, keepdims=True); gstd = R.std(1, keepdims=True)
        zvf = float((gstd[:, 0] <= EPS).mean())
        A = (R - gmean) / (gstd + EPS) if algo == "grpo" else (R - gmean)
        adv = torch.tensor(A.reshape(-1), dtype=torch.float32, device=device)

        full = torch.cat([enc["input_ids"], comp_ids], dim=1)
        attn = (full != tok.pad_token_id).long()
        B = full.shape[0]
        with torch.no_grad():
            old_lp, mask = token_logps_grad(full, attn, plen)
        comp_len = float(mask.sum(1).mean().item())

        for _ep in range(K_EPOCHS):
            opt.zero_grad()
            for cs in range(0, B, CHUNK):
                ce = min(cs + CHUNK, B)
                logits = model(input_ids=full[cs:ce], attention_mask=attn[cs:ce]).logits[:, :-1, :].float()
                tgt = full[cs:ce][:, 1:]
                lp = F.log_softmax(logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[:, plen - 1:]
                m = mask[cs:ce]; old = old_lp[cs:ce]; a_i = adv[cs:ce].unsqueeze(1)
                ratio = torch.exp(lp - old)
                ptl = -torch.min(ratio * a_i, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * a_i) * m
                if algo == "grpo":
                    loss_chunk = (ptl.sum(1) / m.sum(1).clamp(min=1.0)).sum() / B
                else:
                    loss_chunk = ptl.sum() / (B * MAX_NEW)
                loss_chunk.backward(); del logits, lp, ptl
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        step_log.append({"step": step, "mean_reward": float(rewards.mean()),
                         "zvf": zvf, "mean_comp_len": comp_len})

    post_correct = heldout()  # post-GRPO

    import numpy as _np
    pre = _np.array(pre_correct); post = _np.array(post_correct)
    res = {"experiment": "drgrpo_gsm8k_cot", "algo": algo, "seed": seed, "model": MODEL,
           "heldout_pre_acc": float(pre.mean()), "heldout_post_acc": float(post.mean()),
           "n_eval": int(len(pre)),
           "improved_wrong_to_right": int(((pre == 0) & (post == 1)).sum()),
           "regressed_right_to_wrong": int(((pre == 1) & (post == 0)).sum()),
           "last10_avg": float(_np.mean([s["mean_reward"] for s in step_log[-10:]])),
           "mean_comp_len_first5": float(_np.mean([s["mean_comp_len"] for s in step_log[:5]])),
           "mean_comp_len_last5": float(_np.mean([s["mean_comp_len"] for s in step_log[-5:]])),
           "mean_zvf": float(_np.mean([s["zvf"] for s in step_log])),
           "elapsed_seconds": time.time() - t0,
           "pre_correct": pre_correct, "post_correct": post_correct, "step_log": step_log}
    os.makedirs(f"{RESULTS_DIR}/drgrpo_gsm8k", exist_ok=True)
    with open(f"{RESULTS_DIR}/drgrpo_gsm8k/{algo}_s{seed}.json", "w") as f:
        json.dump(res, f)
    results_vol.commit()
    print(f"[{algo} s{seed}] pre={pre.mean():.3f} post={post.mean():.3f} "
          f"len {res['mean_comp_len_first5']:.0f}->{res['mean_comp_len_last5']:.0f} last10={res['last10_avg']:.3f}")
    return res


@app.local_entrypoint()
def main():
    import numpy as np, math
    jobs = [(al, s) for al in ALGOS for s in SEEDS]
    print(f"Dr.GRPO vs GRPO on GSM8K-CoT: {len(jobs)} runs")
    results = [r for r in run_arm.starmap(jobs) if r]

    def mcnemar(pairs):  # list of (pre,post) arrays pooled
        b = sum(int(((p == 1) & (q == 0)).sum()) for p, q in pairs)   # right->wrong
        c = sum(int(((p == 0) & (q == 1)).sum()) for p, q in pairs)   # wrong->right
        n = b + c
        # exact two-sided binomial p (k=min(b,c), p=0.5)
        k = min(b, c)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)) if n > 0 else 1.0
        return {"right_to_wrong": b, "wrong_to_right": c, "mcnemar_p_exact": p}

    summ = {}
    for al in ALGOS:
        rs = [r for r in results if r["algo"] == al]
        pairs = [(np.array(r["pre_correct"]), np.array(r["post_correct"])) for r in rs]
        summ[al] = {
            "n_seeds": len(rs),
            "heldout_pre_mean": float(np.mean([r["heldout_pre_acc"] for r in rs])),
            "heldout_post_mean": float(np.mean([r["heldout_post_acc"] for r in rs])),
            "delta_mean": float(np.mean([r["heldout_post_acc"] - r["heldout_pre_acc"] for r in rs])),
            "comp_len_first5": float(np.mean([r["mean_comp_len_first5"] for r in rs])),
            "comp_len_last5": float(np.mean([r["mean_comp_len_last5"] for r in rs])),
            "mcnemar_pre_to_post": mcnemar(pairs),
        }
    out = {"summary": summ, "runs": [{k: v for k, v in r.items()
                                      if k not in ("pre_correct", "post_correct", "step_log")} for r in results]}
    # keep full per-item arrays separately
    full = {"summary": summ, "runs": results}
    with open("experiments/results/drgrpo_gsm8k_cot.json", "w") as f:
        json.dump(out, f, indent=2)
    with open("experiments/results/drgrpo_gsm8k_cot_full.json", "w") as f:
        json.dump(full, f)
    print("\n=== Dr.GRPO vs GRPO on GSM8K-CoT (Qwen2.5-1.5B-Instruct, measured) ===")
    for al in ALGOS:
        s = summ[al]; mc = s["mcnemar_pre_to_post"]
        print(f"  {al:8s}: heldout {s['heldout_pre_mean']:.3f} -> {s['heldout_post_mean']:.3f} "
              f"(Δ{s['delta_mean']:+.3f}) | len {s['comp_len_first5']:.0f}->{s['comp_len_last5']:.0f} | "
              f"McNemar w->r {mc['wrong_to_right']} r->w {mc['right_to_wrong']} p={mc['mcnemar_p_exact']:.3f}")
    print("Saved experiments/results/drgrpo_gsm8k_cot.json")
