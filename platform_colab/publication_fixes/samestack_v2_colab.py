"""Colab port of platform_hybrid/experiments/modal/modal_samestack_gsm8k_cot_v2.py.

Training/eval logic of run_arm() is copied verbatim from the Modal script (same
model, reward, cap 512, critic warm-up, LoRA, seeds, full 1,319-item test eval,
stored greedy texts). Only the Modal wrapper is removed: results go to
--out_dir, and an `env` block (hardware, package versions, wallclock) is added.

Usage on a GPU VM:
  python samestack_v2_colab.py --arm ppo --seed 4096 --out_dir /content/v2s
"""
import argparse
import itertools
import json
import os
import platform
import subprocess
import time

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
N_STEPS = 30
N_GEN = 64
GROUP = {"grpo_g8": 8, "grpo_g2": 2, "ppo": 1, "ppo_8x8": 8}
K_EPOCHS = 2
CLIP = 0.2
LR = 1e-5
MAX_NEW = 512
N_EVAL = None  # None = full 1,319-item test split
EVAL_BS = 64
EPS = 1e-6
CHUNK = 4
V_LR = 1e-3
V_ITERS = 50
W_WARM = 4


def env_info():
    import torch, transformers, peft, datasets, numpy, accelerate
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    return {"platform": "google_colab", "gpu": smi, "python": platform.python_version(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__,
            "peft": peft.__version__, "datasets": datasets.__version__, "numpy": numpy.__version__,
            "accelerate": accelerate.__version__, "hostname": platform.node()}


def run_arm(arm: str, seed: int, out_dir: str, smoke: bool = False) -> dict:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if arm not in GROUP:
        raise ValueError("Unknown arm")
    n_steps, n_eval, w_warm = (2, 32, 1) if smoke else (N_STEPS, N_EVAL, W_WARM)
    protocol = "samestack-colab-v2-validated-reuse"
    out_dir = os.path.join(out_dir, protocol, "smoke" if smoke else "full")
    out_path = f"{out_dir}/{arm}_s{seed}.json"
    if os.path.exists(out_path):
        import gzip
        with open(out_path) as handle:
            cached = json.load(handle)
        expected = {"protocol": protocol, "arm": arm, "seed": seed, "smoke": smoke,
                    "model": MODEL, "n_steps": n_steps, "n_eval": n_eval or 1319,
                    "group": GROUP[arm], "n_gen": N_GEN, "lr": LR, "max_new": MAX_NEW,
                    "k_epochs": K_EPOCHS}
        if any(cached.get(k) != v for k, v in expected.items()):
            raise ValueError("Existing result manifest does not match request")
        with gzip.open(f"{out_dir}/{arm}_s{seed}_texts.json.gz", "rt") as handle:
            texts = json.load(handle)
        if any(len(texts.get(k, [])) != expected["n_eval"] for k in ("pre", "post")):
            raise ValueError("Existing completion bundle is incomplete")
        return cached
    wall_start_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

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
    # A completion that stopped contains an end token (or post-stop padding); one cut at the cap has neither.
    stop_ids = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id, tok.pad_token_id}
    model = get_peft_model(
        AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device),
        LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                   target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"),
    )
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=LR)
    is_ppo = arm.startswith("ppo")
    vhead = vopt = None
    if is_ppo:
        vhead = nn.Linear(model.config.hidden_size, 1).to(device=device, dtype=torch.float32)
        nn.init.zeros_(vhead.weight); nn.init.zeros_(vhead.bias)
        vopt = torch.optim.Adam(vhead.parameters(), lr=V_LR)

    ds = load_dataset("openai/gsm8k", "main")
    train = ds["train"].shuffle(seed=seed)
    test = list(ds["test"]) if n_eval is None else list(ds["test"].shuffle(seed=0).select(range(n_eval)))

    def gold_of(ans):
        return ans.split("####")[-1].strip().replace(",", "")

    def norm_num(s):
        s = re.sub(r"\\[!,;: ]|\\\$|\$|,|\\text\{[^}]*\}|\\%|%", "", s).strip().rstrip(".")
        try:
            return f"{float(s):.6g}"
        except ValueError:
            return s

    def extract(t):
        # Boxed-only: no last-number fallback, so a truncated answer-less completion scores 0.
        m = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})+)\}", t)
        return norm_num(m[-1]) if m else None

    def build(q):
        msgs = [{"role": "user", "content": q + "\nThink step by step, then give the final answer as \\boxed{ANSWER}."}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def reward(text, gold):
        p = extract(text)
        return 1.0 if (p is not None and p == norm_num(gold)) else 0.0

    def truncated(ids):
        return [not any(int(t) in stop_ids for t in row) for row in ids]

    def gen_batch(prompts, greedy):
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        plen = enc["input_ids"].shape[1]
        kw = dict(do_sample=False) if greedy else dict(do_sample=True, temperature=1.0, top_p=1.0)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id, **kw)
        return enc, plen, g[:, plen:]

    def heldout():
        model.eval()
        corr, trunc, texts = [], [], []
        for i in range(0, len(test), EVAL_BS):
            ch = test[i:i + EVAL_BS]
            _, _, c = gen_batch([build(x["question"]) for x in ch], greedy=True)
            trunc += truncated(c)
            for x, t in zip(ch, tok.batch_decode(c, skip_special_tokens=True)):
                corr.append(int(reward(t, gold_of(x["answer"])) == 1.0))
                texts.append(t)
        model.train()
        return corr, [int(t) for t in trunc], texts

    def forward(full, attn, plen, cs, ce, want_feat):
        out = model(input_ids=full[cs:ce], attention_mask=attn[cs:ce], output_hidden_states=want_feat)
        logits = out.logits[:, :-1, :].float()
        tgt = full[cs:ce][:, 1:]
        lp = F.log_softmax(logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[:, plen - 1:]
        feat = out.hidden_states[-1][:, plen - 1, :].detach().float() if want_feat else None
        del out, logits
        return lp, feat

    train_iter = iter(train)
    reshuffles = itertools.count(1)

    def next_batch(n_prompts, G):
        nonlocal train_iter
        batch = []
        for _ in range(n_prompts):
            try:
                batch.append(next(train_iter))
            except StopIteration:
                train_iter = iter(train.shuffle(seed=seed + next(reshuffles))); batch.append(next(train_iter))
        prompts, golds = [], []
        for ex in batch:
            for _ in range(G):
                prompts.append(build(ex["question"])); golds.append(gold_of(ex["answer"]))
        return prompts, golds

    def rollout(prompts, golds):
        enc, plen, comp_ids = gen_batch(prompts, greedy=False)
        comp_txt = tok.batch_decode(comp_ids, skip_special_tokens=True)
        rewards = np.array([reward(t, g) for t, g in zip(comp_txt, golds)], dtype=np.float32)
        full = torch.cat([enc["input_ids"], comp_ids], dim=1)
        attn = (full != tok.pad_token_id).long()
        attn[:, :plen] = enc["attention_mask"]
        return full, attn, plen, comp_ids, rewards

    def fit_value(feats, rew_t):
        for _ in range(V_ITERS):
            vopt.zero_grad()
            loss = 0.5 * ((vhead(feats).squeeze(-1) - rew_t) ** 2).mean()
            loss.backward(); vopt.step()
        return float(loss.item())

    G = GROUP[arm]
    n_prompts = N_GEN // G
    t0 = time.time()
    pre_correct, pre_trunc, pre_texts = heldout()
    t_eval = time.time() - t0

    # Critic warm-up: rollouts from the initial policy, value head only.
    warm_log = []
    for w in range(w_warm if is_ppo else 0):
        prompts, golds = next_batch(n_prompts, G)
        full, attn, plen, _, rewards = rollout(prompts, golds)
        rew_t = torch.tensor(rewards, device=device)
        with torch.no_grad():
            feats = torch.cat([forward(full, attn, plen, cs, min(cs + CHUNK, len(full)), True)[1]
                               for cs in range(0, len(full), CHUNK)])
            pre_mse = float(0.5 * ((vhead(feats).squeeze(-1) - rew_t) ** 2).mean())
        warm_log.append({"warm_step": w, "mean_reward": float(rewards.mean()),
                         "value_loss_before_fit": pre_mse, "value_loss_after_fit": fit_value(feats, rew_t)})

    step_log = []
    for step in range(n_steps):
        prompts, golds = next_batch(n_prompts, G)
        full, attn, plen, comp_ids, rewards = rollout(prompts, golds)
        rew_t = torch.tensor(rewards, device=device)
        B = full.shape[0]
        with torch.no_grad():
            olds, masks, feats = [], [], []
            for cs in range(0, B, CHUNK):
                ce = min(cs + CHUNK, B)
                lp, f = forward(full, attn, plen, cs, ce, is_ppo)
                olds.append(lp); masks.append((full[cs:ce][:, 1:] != tok.pad_token_id).float()[:, plen - 1:])
                if is_ppo:
                    feats.append(f)
            old_lp, mask = torch.cat(olds), torch.cat(masks)
        comp_len = float(mask.sum(1).mean().item())
        trunc_frac = float(np.mean(truncated(comp_ids)))

        vl_pre = vl_post = None
        if is_ppo:
            feats = torch.cat(feats)
            with torch.no_grad():
                V_old = vhead(feats).squeeze(-1)  # predicted before seeing this batch's rewards
            vl_pre = float(0.5 * ((V_old - rew_t) ** 2).mean())
            adv = rew_t - V_old
            adv = (adv - adv.mean()) / (adv.std() + EPS)
            zvf = float("nan")
        else:
            R = rewards.reshape(n_prompts, G)
            gstd = R.std(1, keepdims=True)
            zvf = float((gstd[:, 0] <= EPS).mean())
            A = (R - R.mean(1, keepdims=True)) / (gstd + EPS)
            adv = torch.tensor(A.reshape(-1), dtype=torch.float32, device=device)

        for _ep in range(K_EPOCHS):
            opt.zero_grad()
            for cs in range(0, B, CHUNK):
                ce = min(cs + CHUNK, B)
                lp, _ = forward(full, attn, plen, cs, ce, False)
                m = mask[cs:ce]; a_i = adv[cs:ce].unsqueeze(1)
                ratio = torch.exp(lp - old_lp[cs:ce])
                ptl = -torch.min(ratio * a_i, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * a_i) * m
                loss = (ptl.sum(1) / m.sum(1).clamp(min=1.0)).sum() / B
                loss.backward()
                del lp, ptl, loss
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            opt.step()
        if is_ppo:
            vl_post = fit_value(feats, rew_t)
        step_log.append({"step": step, "mean_reward": float(rewards.mean()),
                         "reward_var_half": float(0.5 * rewards.var()), "zvf": zvf,
                         "mean_comp_len": comp_len, "trunc_frac": trunc_frac, "grad_norm": gnorm,
                         "value_loss_pre_fit": vl_pre, "value_loss_post_fit": vl_post})
        print(f"[{arm} s{seed}] step {step} r={rewards.mean():.3f} len={comp_len:.0f} "
              f"trunc={trunc_frac:.2f} vl={vl_pre}", flush=True)

    post_correct, post_trunc, post_texts = heldout()
    pre, post = np.array(pre_correct), np.array(post_correct)
    res = {"protocol": protocol, "experiment": "samestack_gsm8k_cot_v2", "arm": arm, "seed": seed, "model": MODEL,
           "smoke": smoke, "group": G, "prompts_per_step": n_prompts, "n_gen": N_GEN,
           "n_steps": n_steps, "k_epochs": K_EPOCHS, "lr": LR, "max_new": MAX_NEW,
           "critic": {"v_lr": V_LR, "v_iters": V_ITERS, "warm_batches": len(warm_log),
                      "warm_generations": len(warm_log) * N_GEN} if is_ppo else None,
           "heldout_pre_acc": float(pre.mean()), "heldout_post_acc": float(post.mean()),
           "heldout_pre_trunc": float(np.mean(pre_trunc)), "heldout_post_trunc": float(np.mean(post_trunc)),
           "n_eval": int(len(pre)),
           "wrong_to_right": int(((pre == 0) & (post == 1)).sum()),
           "right_to_wrong": int(((pre == 1) & (post == 0)).sum()),
           "last10_avg": float(np.mean([s["mean_reward"] for s in step_log[-10:]])),
           "eval_seconds": t_eval, "elapsed_seconds": time.time() - t0,
           "pre_correct": pre_correct, "post_correct": post_correct,
           "pre_trunc": pre_trunc, "post_trunc": post_trunc,
           "warm_log": warm_log, "step_log": step_log}
    # Colab-only additions (not in the Modal output): provenance block.
    res["env"] = {**env_info(), "seed": seed, "wall_start_utc": wall_start_utc,
                  "wall_end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "source_script": "platform_hybrid/experiments/modal/modal_samestack_gsm8k_cot_v2.py (run_arm, verbatim)",
                  "concurrent_processes_on_gpu": int(os.environ.get("PF_CONCURRENCY", "1"))}
    os.makedirs(f"{out_dir}/adapters", exist_ok=True)
    model.save_pretrained(f"{out_dir}/adapters/{arm}_s{seed}")
    import gzip
    with gzip.open(f"{out_dir}/{arm}_s{seed}_texts.json.gz", "wt") as f:
        json.dump({"pre": pre_texts, "post": post_texts}, f)
    with open(out_path + ".tmp", "w") as f:
        json.dump(res, f)
    os.replace(out_path + ".tmp", out_path)
    print(f"[{arm} s{seed}] pre={pre.mean():.3f} post={post.mean():.3f} "
          f"trunc {np.mean(pre_trunc):.3f}->{np.mean(post_trunc):.3f} last10={res['last10_avg']:.3f}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(GROUP))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out_dir", default="/content/v2s")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    run_arm(a.arm, a.seed, a.out_dir, a.smoke)
