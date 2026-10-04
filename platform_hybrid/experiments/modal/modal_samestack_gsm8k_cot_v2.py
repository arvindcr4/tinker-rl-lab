"""Modal: same-stack PPO vs GRPO on GSM8K-CoT, v2 (fixes to the 2026-10-02 rerun).

v1 (modal_samestack_gsm8k_cot.py, results/samestack_gsm8k_cot.json) had three
design weaknesses that this version removes:

  1. Token cap. v1 used max_new=200; training completions averaged 190-198
     tokens and the base model scored 0.20 greedy, so the regime was
     truncation-bound. v2 uses max_new=512 and logs per-completion truncation
     (no end-of-turn token) in training and evaluation.
  2. Cold critic. v1's value head was a randomly initialised linear layer on
     raw hidden states at lr 1e-5 with no warm-up (initial value loss 0.5-13
     against a ~0.08 floor), so PPO advantages were mostly noise. v2 uses a
     zero-initialised linear probe on the detached prompt-end hidden state,
     its own Adam (lr 1e-3), W_WARM warm-up rollout batches from the initial
     policy before any policy update, and V_ITERS fitting steps per batch.
     The baseline for a batch is always predicted BEFORE the head sees that
     batch's rewards. Warm-up rollouts are extra PPO compute and are logged.
  3. Exposure and power. v1 had 5 seeds on 200 test items and PPO saw 8x the
     distinct prompts. v2 has 10 seeds, the full 1,319-item GSM8K test split,
     and an exposure-matched PPO arm.
  4. Reward. v1 fell back to the last number in the text when no \\boxed{}
     was present and compared raw strings. v2 scores the last \\boxed{}
     only, numerically normalised; greedy eval completions are stored
     (<arm>_s<seed>_texts.json.gz on the volume) so scoring can be audited.

Arms (64 generations/step, 30 steps, K=2 inner epochs, clip 0.2, per-response
length normalisation, LoRA r=16 q/k/v/o, policy lr 1e-5, grad clip 1.0, A10G):

  grpo_g8  : 8 prompts x 8,  A=(r-mean_g)/(std_g+eps)
  grpo_g2  : 32 prompts x 2, same estimator
  ppo      : 64 prompts x 1, A=r-V(prompt), batch-normalised (standard PPO shape)
  ppo_8x8  : 8 prompts x 8,  A=r-V(prompt), batch-normalised (exposure-matched)

Pre-specified analysis (fixed before launch, 2026-10-03): unit = seed;
primary contrasts are post-training accuracy grpo_g8-ppo, grpo_g8-ppo_8x8 and
grpo_g8-grpo_g2; paired t 95% CI, exact two-sided sign-flip p, Holm across
the three primary contrasts, TOST at +/-2.0 accuracy points.

Usage (from platform_hybrid/):
  modal run experiments/modal/modal_samestack_gsm8k_cot_v2.py --smoke   # 1 seed/arm, tiny
  modal run experiments/modal/modal_samestack_gsm8k_cot_v2.py
"""
import itertools
import json
import math
import os
import time

import modal

app = modal.App("tinkerrl-samestack-gsm8k-cot-v2")
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

SEEDS = [42, 123, 456, 789, 1024, 2048, 3141, 4096, 5555, 7777]
ARMS = ["grpo_g8", "grpo_g2", "ppo", "ppo_8x8"]
PRIMARY = [("grpo_g8", "ppo"), ("grpo_g8", "ppo_8x8"), ("grpo_g8", "grpo_g2")]
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
MARGIN = 0.02


@app.function(image=image, gpu="A10G", timeout=6 * 3600, volumes={RESULTS_DIR: results_vol},
              retries=1, secrets=[modal.Secret.from_name("huggingface-secret")])
def run_arm(arm: str, seed: int, smoke: bool = False) -> dict:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    sub = "samestack_gsm8k_v2s_smoke" if smoke else "samestack_gsm8k_v2s"  # v2s: strict boxed reward
    out_path = f"{RESULTS_DIR}/{sub}/{arm}_s{seed}.json"
    if os.path.exists(out_path):  # resume: a finished arm is not re-run
        with open(out_path) as f:
            return json.load(f)
    n_steps, n_eval, w_warm = (2, 32, 1) if smoke else (N_STEPS, N_EVAL, W_WARM)

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
    res = {"experiment": "samestack_gsm8k_cot_v2", "arm": arm, "seed": seed, "model": MODEL,
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
    os.makedirs(f"{RESULTS_DIR}/{sub}/adapters", exist_ok=True)
    model.save_pretrained(f"{RESULTS_DIR}/{sub}/adapters/{arm}_s{seed}")
    import gzip
    with gzip.open(f"{RESULTS_DIR}/{sub}/{arm}_s{seed}_texts.json.gz", "wt") as f:
        json.dump({"pre": pre_texts, "post": post_texts}, f)
    with open(out_path, "w") as f:
        json.dump(res, f)
    results_vol.commit()
    print(f"[{arm} s{seed}] pre={pre.mean():.3f} post={post.mean():.3f} "
          f"trunc {np.mean(pre_trunc):.3f}->{np.mean(post_trunc):.3f} last10={res['last10_avg']:.3f}")
    return res


def _t_cdf(t, df):
    """Student-t CDF via the regularized incomplete beta (scipy-free)."""
    if not math.isfinite(df) or df <= 0 or math.isnan(t):
        raise ValueError("Student-t requires positive finite df and non-NaN t")
    if t == 0:
        return 0.5
    if math.isinf(t):
        return 1.0 if t > 0 else 0.0
    # Compute df/(df+t*t) without overflowing t*t. Keep log(x) even
    # when x underflows: for heavy tails the probability may still be finite.
    log_ratio = math.log(abs(t)) - 0.5 * math.log(df)
    if log_ratio > 0:
        log_x = -2 * log_ratio - math.log1p(math.exp(-2 * log_ratio))
        x = math.exp(log_x)
    else:
        x = 1 / (1 + (t / math.sqrt(df)) ** 2)
    if x < 1e-100:
        a = df / 2.0
        # I_x(a, 1/2) = x**a / (a*B(a,1/2)) * (1 + O(x)).
        # In this branch the omitted relative correction is below 1e-100.
        log_tail = (math.lgamma(a + 0.5) - math.lgamma(a)
                    - math.lgamma(0.5) - math.log(a) - math.log(2) + a * log_x)
        tail = math.exp(log_tail)
        return 1 - tail if t > 0 else tail
    if x == 1:
        # Near zero, squaring t loses the complement of x to rounding.
        # The local linear expansion avoids log(0); cubic error is negligible.
        density = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
        return 0.5 + t * density
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
    if not 0 < p < 1 or not math.isfinite(df) or df <= 0:
        raise ValueError("Student-t quantile requires 0 < p < 1 and positive finite df")
    if p == 0.5:
        return 0.0
    # Invert the lower tail on both sides. Comparing an upper CDF near 1
    # loses relative accuracy in the requested tail through subtraction.
    q = min(p, 1 - p)
    sign = -1.0 if p < 0.5 else 1.0
    lo, hi = 0.0, 1.0
    largest = float.fromhex("0x1.fffffffffffffp+1023")
    while _t_cdf(-hi, df) > q:
        lo = hi
        if hi == largest:
            # The mathematical quantile exceeds the finite float range.
            return sign * float("inf")
        hi = min(2 * hi, largest)
    for _ in range(200):
        mid = lo / 2 + hi / 2
        if mid == lo or mid == hi:
            break
        lo, hi = (mid, hi) if _t_cdf(-mid, df) > q else (lo, mid)
    return sign * (lo / 2 + hi / 2)


def signflip_p(d):
    """Exact two-sided sign-flip permutation p on the mean of paired differences."""
    obs = abs(sum(d))
    hits = sum(abs(sum(s * x for s, x in zip(signs, d))) >= obs - 1e-12
               for signs in itertools.product([1, -1], repeat=len(d)))
    return hits / 2 ** len(d)


def paired(a: dict, b: dict) -> dict:
    seeds = sorted(set(a) & set(b))
    d = [a[s] - b[s] for s in seeds]
    n = len(d)
    if n < 2:
        raise ValueError("paired requires at least two common seeds")
    if not all(math.isfinite(x) for x in d):
        raise ValueError("paired requires finite differences")
    md = d[0] if all(x == d[0] for x in d) else math.fsum(d) / n
    sd = math.sqrt(sum((x - md) ** 2 for x in d) / (n - 1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n)
    df = n - 1
    t = md / se if se > 0 else (math.copysign(float("inf"), md) if md else 0.0)
    p = 2 * _t_cdf(-abs(t), df) if se > 0 else (0.0 if md else 1.0)
    tc = _t_ppf(0.975, df)
    # Use one-sided limiting probabilities at zero SE. At the equivalence
    # boundary the null is not rejected; outside it equivalence is false.
    p_lo = _t_cdf(-(md + MARGIN) / se, df) if se > 0 else (0.0 if md > -MARGIN else 0.5 if md == -MARGIN else 1.0)
    p_hi = _t_cdf((md - MARGIN) / se, df) if se > 0 else (0.0 if md < MARGIN else 0.5 if md == MARGIN else 1.0)
    return {"n_seeds": n, "seeds": seeds, "diffs": d, "mean_diff": md, "sd": sd,
            "ci95": [md - tc * se, md + tc * se], "t": t, "df": df, "p_two_sided": p,
            "signflip_p": signflip_p(d),
            "tost_margin": MARGIN, "tost_p": max(p_lo, p_hi),
            "equivalent_at_margin": max(p_lo, p_hi) < 0.05}


def holm(ps: dict) -> dict:
    order = sorted(ps, key=ps.get)
    out, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * ps[k]))
        out[k] = running
    return out


@app.local_entrypoint()
def main(smoke: bool = False):
    seeds = SEEDS[:1] if smoke else SEEDS
    jobs = [(a, s, smoke) for a in ARMS for s in seeds]
    print(f"Same-stack GSM8K-CoT v2{' SMOKE' if smoke else ''}: {len(jobs)} runs ({ARMS} x {seeds})")
    results, errors = [], []
    for r in run_arm.starmap(jobs, return_exceptions=True):
        (results if isinstance(r, dict) else errors).append(r)
    for e in errors:
        print("FAILED:", repr(e)[:500])
    by = {a: {r["seed"]: r for r in results if r["arm"] == a} for a in ARMS}

    summ = {}
    for a in ARMS:
        rs = list(by[a].values())
        if not rs:
            continue
        mean = lambda k: sum(r[k] for r in rs) / len(rs)
        summ[a] = {"n_seeds": len(rs), "heldout_pre_mean": mean("heldout_pre_acc"),
                   "heldout_post_mean": mean("heldout_post_acc"),
                   "delta_mean": mean("heldout_post_acc") - mean("heldout_pre_acc"),
                   "heldout_pre_trunc": mean("heldout_pre_trunc"),
                   "heldout_post_trunc": mean("heldout_post_trunc"),
                   "train_trunc_mean": sum(sum(s["trunc_frac"] for s in r["step_log"]) / len(r["step_log"])
                                           for r in rs) / len(rs),
                   "last10_mean": mean("last10_avg")}
        if rs[0]["arm"].startswith("ppo"):
            vl = [s["value_loss_pre_fit"] for r in rs for s in r["step_log"]]
            floor = [s["reward_var_half"] for r in rs for s in r["step_log"]]
            summ[a]["critic_value_loss_pre_fit_mean"] = sum(vl) / len(vl)
            summ[a]["critic_floor_half_reward_var_mean"] = sum(floor) / len(floor)
        d = {s: r["heldout_post_acc"] - r["heldout_pre_acc"] for s, r in by[a].items()}
        if len(d) >= 2:
            summ[a]["delta_vs_pre"] = paired(d, {s: 0.0 for s in d})
    post_of = {a: {s: r["heldout_post_acc"] for s, r in by[a].items()} for a in ARMS}
    contrasts = {}
    for x, y in PRIMARY:
        if len(set(post_of[x]) & set(post_of[y])) >= 2:
            contrasts[f"{x}_minus_{y}"] = paired(post_of[x], post_of[y])
    for k, h in holm({k: c["p_two_sided"] for k, c in contrasts.items()}).items():
        contrasts[k]["holm_p_t"] = h
    for k, h in holm({k: c["signflip_p"] for k, c in contrasts.items()}).items():
        contrasts[k]["holm_p_signflip"] = h

    out = {"config": {"model": MODEL, "n_steps": N_STEPS, "n_gen": N_GEN, "group": GROUP,
                      "k_epochs": K_EPOCHS, "lr": LR, "max_new": MAX_NEW,
                      "n_eval": "full test split (1319)" if N_EVAL is None else N_EVAL,
                      "critic": {"init": "zeros", "input": "detached prompt-end hidden state",
                                 "v_lr": V_LR, "v_iters": V_ITERS, "warm_batches": W_WARM},
                      "seeds": seeds, "tost_margin": MARGIN, "unit": "seed", "smoke": smoke,
                      "primary_contrasts": [f"{x}_minus_{y}" for x, y in PRIMARY],
                      "metric": "greedy accuracy on GSM8K test after training"},
           "summary": summ, "contrasts": contrasts, "n_failed": len(errors),
           "runs": [{k: v for k, v in r.items()
                     if k not in ("pre_correct", "post_correct", "pre_trunc", "post_trunc", "step_log", "warm_log")}
                    for r in results]}
    stem = "experiments/results/samestack_gsm8k_cot_v2" + ("_smoke" if smoke else "")
    with open(stem + ".json", "w") as f:
        json.dump(out, f, indent=2)
    with open(stem + "_full.json", "w") as f:
        json.dump({**out, "runs": results}, f)
    print("\n=== Same-stack GSM8K-CoT v2 (Qwen2.5-1.5B-Instruct, cap 512) ===")
    for a, s in summ.items():
        print(f"  {a:8s}: {s['heldout_pre_mean']:.3f} -> {s['heldout_post_mean']:.3f} "
              f"(Δ{s['delta_mean']:+.3f}, n={s['n_seeds']}) trunc {s['heldout_pre_trunc']:.3f}->"
              f"{s['heldout_post_trunc']:.3f} last10={s['last10_mean']:.3f}")
    for k, c in contrasts.items():
        print(f"  {k}: {c['mean_diff']:+.4f} CI [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] "
              f"p_t={c['p_two_sided']:.4f} signflip={c['signflip_p']:.4f} holm_t={c['holm_p_t']:.4f} "
              f"TOST p={c['tost_p']:.3f}")
    print(f"Saved {stem}.json")
