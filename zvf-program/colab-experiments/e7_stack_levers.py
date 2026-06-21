"""E7 (Pillar 4, MIN-REPORT-RL): how much does flipping ONE unreported "stack"
lever move the headline, and -- for the big bf16->fp32 effect -- IS IT REAL or a
training/numerics artifact?

Round-2 Codex fix: the first run's fp32 jump (train reward 0.23->0.95) was
confounded (dtype couples generation distribution, optimizer numerics, and
possible bf16 update instability). This version DECOUPLES it:
  * ZERO-UPDATE generation baseline per dtype (gen quality BEFORE any training).
  * Per-step diagnostics: total grad-norm, last-layer param-delta norm, and
    non-finite-grad counts (bf16 instability detector).
  * LR CONTRAST: bf16 and fp32 at lr in {1e-6, 2e-6} -> is bf16 just LR-sensitive?
  * PAIRED + replicated (same prompt stream/seed per arm, 2 seeds, mean+-std).
Headline = ZVF / ERF / reward trajectory (held-out over 10 toy steps is too noisy).

Reference = bf16 / sdpa / temp 1.0 / top_p 0.95 / lr 2e-6.

Run:  colab run --gpu T4 --timeout 1800 e7_stack_levers.py
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json, re, random, statistics, math
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SEEDS = [0, 1]
G, BATCH, MAX_NEW, STEPS = 6, 4, 128, 10    # MAX_NEW 128: avoid truncation -> spurious 0 reward
HELDOUT_N = 20
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# name -> (dtype, attn_impl, temperature, top_p, lr)
ARMS = {
    "reference":   (torch.bfloat16, "sdpa",  1.0, 0.95, 2e-6),
    "fp32":        (torch.float32,  "sdpa",  1.0, 0.95, 2e-6),
    "bf16_lr1e-6": (torch.bfloat16, "sdpa",  1.0, 0.95, 1e-6),
    "fp32_lr1e-6": (torch.float32,  "sdpa",  1.0, 0.95, 1e-6),
    "eager_attn":  (torch.bfloat16, "eager", 1.0, 0.95, 2e-6),
    "temp_0.7":    (torch.bfloat16, "sdpa",  0.7, 0.95, 2e-6),
}

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
PAD = tok.pad_token_id

FEWSHOT = [
    {"role": "user", "content": "Compute 3 + 4. Reason briefly, then put the final integer after '####'."},
    {"role": "assistant", "content": "3 + 4 = 7.\n#### 7"},
]

def problem(rng):
    # 3-digit addition: ~0.5-0.7 base accuracy -> headroom for precision/LR to matter
    # (2-digit addition saturates at ~0.9, masking any lever effect).
    a, b = rng.randint(100, 999), rng.randint(100, 999)
    return f"{a} + {b}", a + b

def prompt_of(q):
    msgs = FEWSHOT + [{"role": "user",
                       "content": f"Compute {q}. Reason briefly, then put the final integer after '####'."}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def parse(text):
    seg = text.split("####")
    if len(seg) < 2:
        return None, False
    m = re.findall(r"-?\d+", seg[-1])
    return (int(m[0]) if m else None), bool(m)

def gen_group(model, prompt, gold, temp, top_p):
    model.eval()
    enc = tok([prompt] * G, return_tensors="pt", padding=True).to(DEV)
    with torch.no_grad():
        out = model.generate(**enc, do_sample=True, temperature=temp, top_p=top_p,
                             max_new_tokens=MAX_NEW, pad_token_id=PAD)
    gens = out[:, enc.input_ids.shape[1]:]
    rewards, fmt = [], []
    for t in tok.batch_decode(gens, skip_special_tokens=True):
        ans, ok = parse(t)
        rewards.append(1.0 if ans == gold else 0.0); fmt.append(1.0 if ok else 0.0)
    return enc.input_ids[0], gens, rewards, fmt

def seq_logprob(model, pids, gen_row):
    gen_row = gen_row[gen_row != PAD]
    if gen_row.numel() == 0:
        return None
    ids = torch.cat([pids, gen_row]).unsqueeze(0)
    logits = model(ids).logits[:, :-1, :].float()
    tgt = ids[:, 1:]
    lp = F.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return lp[:, pids.shape[0] - 1:].sum()

@torch.no_grad()
def heldout_acc(model, evalset):
    model.eval(); c = 0
    for q, gold in evalset:
        enc = tok([prompt_of(q)], return_tensors="pt", padding=True).to(DEV)
        out = model.generate(**enc, do_sample=False, max_new_tokens=MAX_NEW, pad_token_id=PAD)
        ans, _ = parse(tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True))
        if ans == gold:
            c += 1
    return c / len(evalset)

def grad_diag(model):
    """Total grad L2 norm + count of non-finite grads. Non-finite grads PROPAGATE into
    the norm (no skipping) so gnorm blows up exactly when instability strikes."""
    sq, nonfinite = 0.0, 0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach().float()
            nonfinite += int((~torch.isfinite(g)).sum().item())
            sq += float((g * g).sum().item())
    gn = math.sqrt(sq) if math.isfinite(sq) and sq >= 0 else float("inf")
    return gn, nonfinite

def run(arm, seed):
    dtype, attn, temp, top_p, lr = ARMS[arm]
    rng = random.Random(seed); torch.manual_seed(seed)        # PAIRED stream per seed
    evalset = [problem(rng) for _ in range(HELDOUT_N)]
    train_stream = [[problem(rng) for _ in range(BATCH)] for _ in range(STEPS)]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, attn_implementation=attn).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    last = list(model.model.layers[-1].parameters())

    # ZERO-UPDATE generation baseline (gen quality before any training, this dtype)
    base_r, base_f = [], []
    for q, gold in train_stream[0]:
        _, _, rw, fm = gen_group(model, prompt_of(q), gold, temp, top_p)
        base_r += rw; base_f += fm
    pre = heldout_acc(model, evalset)

    zvfs, erfs, ps, gnorms, dnorms, nan_tot = [], [], [], [], [], 0
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        zv, rall, fall, n_terms = 0, [], [], 0
        for q, gold in train_stream[step]:
            pids, gens, rewards, fmt = gen_group(model, prompt_of(q), gold, temp, top_p)
            rall += rewards; fall += fmt
            m = sum(rewards) / G; v = statistics.pvariance(rewards); s = v ** 0.5
            if v == 0.0:
                zv += 1; continue
            for i in range(G):
                a = (rewards[i] - m) / (s + 1e-6)
                if a:
                    lp = seq_logprob(model, pids, gens[i])
                    if lp is not None:
                        (-a * lp).backward(); n_terms += 1
        gnorm, nf = grad_diag(model)
        nan_tot += nf
        before = [p.detach().float().clone() for p in last]
        if n_terms:
            opt.step()
        dnorm = math.sqrt(sum((p.detach().float() - b).pow(2).sum().item()
                              for p, b in zip(last, before)))
        zvfs.append(zv / BATCH); erfs.append(sum(fall) / len(fall)); ps.append(sum(rall) / len(rall))
        gnorms.append(gnorm); dnorms.append(dnorm)
        print(f"[e7:{arm:12s} s{seed}] step={step+1:2d} ZVF={zvfs[-1]:.2f} ERF={erfs[-1]:.2f} "
              f"p={ps[-1]:.2f} gnorm={gnorm:.1f} dnorm={dnorm:.3g} nan={nf}", flush=True)
    post = heldout_acc(model, evalset)
    out = {"arm": arm, "seed": seed,
           "gen_baseline_p": round(statistics.mean(base_r), 3),
           "gen_baseline_erf": round(statistics.mean(base_f), 3),
           "mean_zvf": round(statistics.mean(zvfs), 3), "mean_erf": round(statistics.mean(erfs), 3),
           "last3_p": round(statistics.mean(ps[-3:]), 3), "heldout_delta": round(post - pre, 3),
           "mean_grad_norm": round(statistics.mean(gnorms), 2),
           "mean_param_delta": round(statistics.mean(dnorms), 5), "nonfinite_grads": nan_tot}
    del model, opt; torch.cuda.empty_cache()
    return out

results = []
for arm in ARMS:
    for seed in SEEDS:
        results.append(run(arm, seed))

def ms(xs):
    m = statistics.mean(xs)
    if not math.isfinite(m):                      # non-finite gnorm -> JSON/W&B-safe sentinel
        return ["non-finite", 0.0]
    return [round(m, 3), round(statistics.pstdev(xs), 3)]
ref = {s: next(r for r in results if r["arm"] == "reference" and r["seed"] == s) for s in SEEDS}
by_lever = {}
for arm in ARMS:
    rs = [r for r in results if r["arm"] == arm]
    by_lever[arm] = {
        "gen_baseline_p": ms([r["gen_baseline_p"] for r in rs]),
        "gen_baseline_erf": ms([r["gen_baseline_erf"] for r in rs]),
        "mean_zvf": ms([r["mean_zvf"] for r in rs]), "mean_erf": ms([r["mean_erf"] for r in rs]),
        "last3_p": ms([r["last3_p"] for r in rs]),
        "mean_grad_norm": ms([r["mean_grad_norm"] for r in rs]),
        "mean_param_delta": ms([r["mean_param_delta"] for r in rs]),
        "nonfinite_grads": sum(r["nonfinite_grads"] for r in rs),
        "delta_p_vs_ref": ms([r["last3_p"] - ref[r["seed"]]["last3_p"] for r in rs]),
        "delta_zvf_vs_ref": ms([r["mean_zvf"] - ref[r["seed"]]["mean_zvf"] for r in rs]),
    }
print("E7_RESULT " + json.dumps({"experiment": "E7_stack_levers", "model": MODEL, "seeds": SEEDS,
                                "reference": "bf16/sdpa/temp1.0/top_p0.95/lr2e-6",
                                "by_lever": by_lever, "runs": results}), flush=True)
