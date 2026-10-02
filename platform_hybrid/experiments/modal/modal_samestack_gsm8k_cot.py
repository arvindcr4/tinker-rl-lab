"""Modal: same-stack PPO vs GRPO (and GRPO G=2 vs G=8) on the UNSATURATED
GSM8K-CoT configuration (thesis review item M1(b)).

The 0.5B arithmetic same-stack contrast (modal_samestack_ppo_grpo.py) sits at
held-out ~0.99, so its null cannot separate "no effect" from "no headroom".
This rerun uses the configuration where pre->post gains were measurable
(Qwen2.5-1.5B-Instruct, GSM8K boxed-answer reward, 200-token CoT; the setup
of modal_drgrpo_gsm8k_cot.py) and holds compute fixed across arms:

  grpo_g8 : 8 prompts x 8 completions, A=(r-mean_g)/(std_g+eps)
  grpo_g2 : 32 prompts x 2 completions, same estimator
  ppo     : 64 prompts x 1 completion, A = r - V(prompt) (value head on the
            prompt-end hidden state, 0.5*MSE value loss), batch-normalised

All arms: 64 generations/step, 30 steps, K=2 inner epochs, token-level clipped
surrogate (clip 0.2) with per-response length normalisation, LoRA r=16 on
q/k/v/o, Adam lr 1e-5, grad clip 1.0, A10G, bf16. Each run greedily evaluates
the same 200 GSM8K test items before and after training and stores per-item
correctness. Five paired seeds.

Pre-specified analysis (fixed before launch, 2026-10-02): unit = seed; primary
contrasts are post-training accuracy grpo_g8 - ppo and grpo_g8 - grpo_g2;
paired t 95% CI and TOST equivalence at a margin of +/-2.0 accuracy points.

Usage:
  modal run experiments/modal/modal_samestack_gsm8k_cot.py
"""
import json
import math
import os
import time

import modal

app = modal.App("tinkerrl-samestack-gsm8k-cot")
results_vol = modal.Volume.from_name("tinkerrl-results", create_if_missing=True)
RESULTS_DIR = "/results"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.3.0", "transformers>=4.46.0", "peft>=0.13.0",
        "datasets>=3.0.0", "numpy>=1.26.0,<2.0.0", "accelerate>=1.0.0",
        "safetensors>=0.4.0", "huggingface-hub>=0.26.0",
    )
)

SEEDS = [42, 123, 456, 789, 1024]
ARMS = ["grpo_g8", "grpo_g2", "ppo"]
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
N_STEPS = 30
N_GEN = 64
GROUP = {"grpo_g8": 8, "grpo_g2": 2, "ppo": 1}
K_EPOCHS = 2
CLIP = 0.2
LR = 1e-5
MAX_NEW = 200
N_EVAL = 200
EPS = 1e-6
CHUNK = 4
MARGIN = 0.02  # TOST equivalence margin, accuracy fraction


@app.function(image=image, gpu="A10G", timeout=3 * 3600, volumes={RESULTS_DIR: results_vol},
              retries=1, secrets=[modal.Secret.from_name("huggingface-secret")])
def run_arm(arm: str, seed: int) -> dict:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    out_path = f"{RESULTS_DIR}/samestack_gsm8k/{arm}_s{seed}.json"
    if os.path.exists(out_path):  # resume: a finished arm is not re-run
        with open(out_path) as f:
            return json.load(f)

    import random
    import re
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = "cuda"

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = get_peft_model(
        AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device),
        LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                   target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"),
    )
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    vhead = None
    if arm == "ppo":
        vhead = nn.Linear(model.config.hidden_size, 1).to(device=device, dtype=torch.float32)
        params = params + list(vhead.parameters())
    opt = torch.optim.Adam(params, lr=LR)

    ds = load_dataset("openai/gsm8k", "main")
    train = ds["train"].shuffle(seed=seed)
    test = list(ds["test"].shuffle(seed=0).select(range(N_EVAL)))

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
        kw = dict(do_sample=False) if greedy else dict(do_sample=True, temperature=1.0, top_p=1.0)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id, **kw)
        return enc, plen, g[:, plen:]

    def heldout():
        model.eval()
        corr = []
        for i in range(0, len(test), 16):
            ch = test[i:i + 16]
            _, _, c = gen_batch([build(x["question"]) for x in ch], greedy=True)
            for x, t in zip(ch, tok.batch_decode(c, skip_special_tokens=True)):
                corr.append(int(reward(t, gold_of(x["answer"])) == 1.0))
        model.train()
        return corr

    def forward(full, attn, plen, cs, ce, want_value):
        out = model(input_ids=full[cs:ce], attention_mask=attn[cs:ce], output_hidden_states=want_value)
        logits = out.logits[:, :-1, :].float()
        tgt = full[cs:ce][:, 1:]
        lp = F.log_softmax(logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[:, plen - 1:]
        v = vhead(out.hidden_states[-1][:, plen - 1, :].float()).squeeze(-1) if want_value else None
        del out, logits
        return lp, v

    pre_correct = heldout()

    G = GROUP[arm]
    n_prompts = N_GEN // G
    step_log = []
    t0 = time.time()
    train_iter = iter(train)
    for step in range(N_STEPS):
        batch = []
        for _ in range(n_prompts):
            try:
                batch.append(next(train_iter))
            except StopIteration:
                train_iter = iter(train.shuffle(seed=seed + step)); batch.append(next(train_iter))
        prompts, golds = [], []
        for ex in batch:
            for _ in range(G):
                prompts.append(build(ex["question"])); golds.append(gold_of(ex["answer"]))
        enc, plen, comp_ids = gen_batch(prompts, greedy=False)
        comp_txt = tok.batch_decode(comp_ids, skip_special_tokens=True)
        rewards = np.array([reward(t, g) for t, g in zip(comp_txt, golds)], dtype=np.float32)
        rew_t = torch.tensor(rewards, device=device)

        full = torch.cat([enc["input_ids"], comp_ids], dim=1)
        attn = (full != tok.pad_token_id).long()
        attn[:, :plen] = enc["attention_mask"]
        B = full.shape[0]
        is_ppo = arm == "ppo"
        with torch.no_grad():
            olds, masks, vals = [], [], []
            for cs in range(0, B, CHUNK):
                ce = min(cs + CHUNK, B)
                lp, v = forward(full, attn, plen, cs, ce, is_ppo)
                olds.append(lp); masks.append((full[cs:ce][:, 1:] != tok.pad_token_id).float()[:, plen - 1:])
                if is_ppo:
                    vals.append(v)
            old_lp, mask = torch.cat(olds), torch.cat(masks)
        comp_len = float(mask.sum(1).mean().item())

        if is_ppo:
            V_old = torch.cat(vals)
            adv = rew_t - V_old
            adv = (adv - adv.mean()) / (adv.std() + EPS)
            zvf = float("nan")
        else:
            R = rewards.reshape(n_prompts, G)
            gstd = R.std(1, keepdims=True)
            zvf = float((gstd[:, 0] <= EPS).mean())
            A = (R - R.mean(1, keepdims=True)) / (gstd + EPS)
            adv = torch.tensor(A.reshape(-1), dtype=torch.float32, device=device)

        vloss_sum = 0.0
        for _ep in range(K_EPOCHS):
            opt.zero_grad()
            for cs in range(0, B, CHUNK):
                ce = min(cs + CHUNK, B)
                lp, v = forward(full, attn, plen, cs, ce, is_ppo)
                m = mask[cs:ce]; a_i = adv[cs:ce].unsqueeze(1)
                ratio = torch.exp(lp - old_lp[cs:ce])
                ptl = -torch.min(ratio * a_i, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * a_i) * m
                loss = (ptl.sum(1) / m.sum(1).clamp(min=1.0)).sum() / B
                if is_ppo:
                    vl = 0.5 * ((v - rew_t[cs:ce]) ** 2).sum() / B
                    loss = loss + vl
                    vloss_sum += float(vl.item())
                loss.backward()
                del lp, ptl, loss
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            opt.step()
        step_log.append({"step": step, "mean_reward": float(rewards.mean()), "zvf": zvf,
                         "mean_comp_len": comp_len, "grad_norm": gnorm,
                         "value_loss": vloss_sum / K_EPOCHS if is_ppo else None})
        print(f"[{arm} s{seed}] step {step} r={rewards.mean():.3f} len={comp_len:.0f}", flush=True)

    post_correct = heldout()
    pre, post = np.array(pre_correct), np.array(post_correct)
    res = {"experiment": "samestack_gsm8k_cot", "arm": arm, "seed": seed, "model": MODEL,
           "group": G, "prompts_per_step": n_prompts, "n_gen": N_GEN, "n_steps": N_STEPS,
           "k_epochs": K_EPOCHS, "lr": LR, "max_new": MAX_NEW,
           "heldout_pre_acc": float(pre.mean()), "heldout_post_acc": float(post.mean()),
           "n_eval": int(len(pre)),
           "wrong_to_right": int(((pre == 0) & (post == 1)).sum()),
           "right_to_wrong": int(((pre == 1) & (post == 0)).sum()),
           "last10_avg": float(np.mean([s["mean_reward"] for s in step_log[-10:]])),
           "elapsed_seconds": time.time() - t0,
           "pre_correct": pre_correct, "post_correct": post_correct, "step_log": step_log}
    os.makedirs(f"{RESULTS_DIR}/samestack_gsm8k/adapters", exist_ok=True)
    model.save_pretrained(f"{RESULTS_DIR}/samestack_gsm8k/adapters/{arm}_s{seed}")
    with open(out_path, "w") as f:
        json.dump(res, f)
    results_vol.commit()
    print(f"[{arm} s{seed}] pre={pre.mean():.3f} post={post.mean():.3f} last10={res['last10_avg']:.3f}")
    return res


def _t_cdf(t, df):
    """Student-t CDF via the regularized incomplete beta (scipy-free)."""
    if t == 0:
        return 0.5
    x = df / (df + t * t)
    def betacf(a, b, x):
        qab, qap, qam = a + b, a + 1, a - 1
        c, d = 1.0, 1 - qab * x / qap
        d = 1 / (d if abs(d) > 1e-300 else 1e-300); h = d
        for m in range(1, 300):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1 / (1 + aa * d or 1e-300); c = 1 + aa / c or 1e-300; h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1 / (1 + aa * d or 1e-300); c = 1 + aa / c or 1e-300; de = d * c; h *= de
            if abs(de - 1) < 3e-14:
                break
        return h
    a, b = df / 2.0, 0.5
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    ib = bt * betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - bt * betacf(b, a, 1 - x) / b
    return 1 - 0.5 * ib if t > 0 else 0.5 * ib


def _t_ppf(p, df):
    lo, hi = -50.0, 50.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _t_cdf(mid, df) < p else (lo, mid)
    return (lo + hi) / 2


def paired(a: dict, b: dict) -> dict:
    seeds = sorted(set(a) & set(b))
    d = [a[s] - b[s] for s in seeds]
    n = len(d)
    md = sum(d) / n
    sd = math.sqrt(sum((x - md) ** 2 for x in d) / (n - 1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n)
    df = n - 1
    t = md / se if se > 0 else float("inf")
    p = 2 * (1 - _t_cdf(abs(t), df)) if se > 0 else 0.0
    tc = _t_ppf(0.975, df)
    p_lo = 1 - _t_cdf((md + MARGIN) / se, df) if se > 0 else 0.0   # H0: diff <= -margin
    p_hi = _t_cdf((md - MARGIN) / se, df) if se > 0 else 0.0       # H0: diff >= +margin
    return {"n_seeds": n, "seeds": seeds, "diffs": d, "mean_diff": md, "sd": sd,
            "ci95": [md - tc * se, md + tc * se], "t": t, "df": df, "p_two_sided": p,
            "tost_margin": MARGIN, "tost_p": max(p_lo, p_hi),
            "equivalent_at_margin": max(p_lo, p_hi) < 0.05}


@app.local_entrypoint()
def main():
    jobs = [(a, s) for a in ARMS for s in SEEDS]
    print(f"Same-stack GSM8K-CoT: {len(jobs)} runs ({ARMS} x {SEEDS})")
    results = [r for r in run_arm.starmap(jobs, return_exceptions=True) if isinstance(r, dict)]
    by = {a: {r["seed"]: r for r in results if r["arm"] == a} for a in ARMS}

    summ = {}
    for a in ARMS:
        rs = list(by[a].values())
        if not rs:
            continue
        post = [r["heldout_post_acc"] for r in rs]
        summ[a] = {"n_seeds": len(rs),
                   "heldout_pre_mean": sum(r["heldout_pre_acc"] for r in rs) / len(rs),
                   "heldout_post_mean": sum(post) / len(rs),
                   "delta_mean": sum(r["heldout_post_acc"] - r["heldout_pre_acc"] for r in rs) / len(rs),
                   "last10_mean": sum(r["last10_avg"] for r in rs) / len(rs)}
    post_of = {a: {s: r["heldout_post_acc"] for s, r in by[a].items()} for a in ARMS}
    contrasts = {}
    for x, y in [("grpo_g8", "ppo"), ("grpo_g8", "grpo_g2")]:
        if len(set(post_of[x]) & set(post_of[y])) >= 2:
            contrasts[f"{x}_minus_{y}"] = paired(post_of[x], post_of[y])

    out = {"config": {"model": MODEL, "n_steps": N_STEPS, "n_gen": N_GEN, "group": GROUP,
                      "k_epochs": K_EPOCHS, "lr": LR, "max_new": MAX_NEW, "n_eval": N_EVAL,
                      "seeds": SEEDS, "tost_margin": MARGIN, "unit": "seed",
                      "metric": "greedy accuracy on 200 GSM8K test items after training"},
           "summary": summ, "contrasts": contrasts,
           "runs": [{k: v for k, v in r.items() if k not in ("pre_correct", "post_correct", "step_log")}
                    for r in results]}
    with open("experiments/results/samestack_gsm8k_cot.json", "w") as f:
        json.dump(out, f, indent=2)
    with open("experiments/results/samestack_gsm8k_cot_full.json", "w") as f:
        json.dump({**out, "runs": results}, f)
    print("\n=== Same-stack GSM8K-CoT (Qwen2.5-1.5B-Instruct) ===")
    for a, s in summ.items():
        print(f"  {a:8s}: {s['heldout_pre_mean']:.3f} -> {s['heldout_post_mean']:.3f} "
              f"(Δ{s['delta_mean']:+.3f}, n={s['n_seeds']}) last10={s['last10_mean']:.3f}")
    for k, c in contrasts.items():
        print(f"  {k}: {c['mean_diff']:+.4f} CI [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] "
              f"p={c['p_two_sided']:.3f} TOST(±{MARGIN}) p={c['tost_p']:.3f}")
    print("Saved experiments/results/samestack_gsm8k_cot.json")
