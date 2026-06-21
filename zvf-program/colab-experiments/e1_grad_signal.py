"""E1 (corrected): empirically ground ZVF <-> GRPO gradient magnitude.

Colab-only: requires the OPEN backward pass to read per-step policy-gradient
norm -- Tinker's closed LoRA API never exposes this.

Fixes over the pilot:
  * SUMMED (not mean-normalized) gradient norm + signal-per-rollout.
  * Difficulty spans the full accuracy range (trivial->impossible) so batch ZVF
    sweeps [0,1]; ZVF->1 at BOTH ends (all-correct and all-wrong collapse).
  * ERF (Effective-Rollout Fraction): fraction emitting a parseable #### answer,
    to separate "wrong" from "format-gated" (the confound the paper flags).
  * Multi-seed aggregation.

Theory test (Pillar 2 / T3): grad signal should track S = p(1-p), an INVERTED-U,
not GU=1-ZVF (which is monotone in difficulty). We report both correlations.

Run:  colab run --gpu T4 --timeout 900 e1_grad_signal.py
"""
import json, re, random, statistics
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SEEDS = [0, 1, 2]
G, BATCH, MAX_NEW, LR = 6, 4, 40, 2e-6
# magnitude ranges chosen to span p from ~1 (trivial) to ~0 (impossible)
DIFFS = {
    "trivial":    (1, 9,    1, 9),
    "easy":       (5, 20,   5, 20),
    "medium":     (11, 60,  11, 60),
    "hard":       (50, 300, 50, 300),
    "impossible": (200, 999, 200, 999),
}
ROUNDS = 2  # passes over all difficulties per seed
DEV = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
PAD = tok.pad_token_id

def problem(diff):
    lo1, hi1, lo2, hi2 = DIFFS[diff]
    a, b = random.randint(lo1, hi1), random.randint(lo2, hi2)
    return f"{a} + {b}", a + b

def prompt_of(q):
    return tok.apply_chat_template(
        [{"role": "user", "content": f"Compute {q}. Reason briefly, then put the final integer after '####'."}],
        tokenize=False, add_generation_prompt=True)

def parse(text):
    m = re.findall(r"-?\d+", text.split("####")[-1])
    return (int(m[0]) if m else None), (1.0 if m else 0.0)  # (answer, format_ok)

def gen_group(model, prompt, gold):
    model.eval()
    enc = tok([prompt] * G, return_tensors="pt", padding=True).to(DEV)
    with torch.no_grad():
        out = model.generate(**enc, do_sample=True, temperature=1.0, top_p=0.95,
                             max_new_tokens=MAX_NEW, pad_token_id=PAD)
    gens = out[:, enc.input_ids.shape[1]:]
    rewards, fmt = [], []
    for t in tok.batch_decode(gens, skip_special_tokens=True):
        ans, ok = parse(t)
        rewards.append(1.0 if ans == gold else 0.0); fmt.append(ok)
    return enc.input_ids[0], gens, rewards, fmt

def seq_logprob(model, pids, gen_row):
    model.train()
    gen_row = gen_row[gen_row != PAD]
    if gen_row.numel() == 0:
        return None
    ids = torch.cat([pids, gen_row]).unsqueeze(0)
    logits = model(ids).logits[:, :-1, :].float()
    tgt = ids[:, 1:]
    lp = F.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return lp[:, pids.shape[0] - 1:].sum()

def grad_norm(model):
    return sum(p.grad.detach().float().pow(2).sum().item()
               for p in model.parameters() if p.grad is not None) ** 0.5

def pearson(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5; sy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (sx * sy) if sx and sy else float("nan")

all_rows = []
for seed in SEEDS:
    random.seed(seed); torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    step = 0
    for _ in range(ROUNDS):
        for diff in DIFFS:
            step += 1
            opt.zero_grad(set_to_none=True)
            rewards_all, fmt_all, losses, zv = [], [], [], 0
            for _ in range(BATCH):
                q, gold = problem(diff)
                pids, gens, rewards, fmt = gen_group(model, prompt_of(q), gold)
                rewards_all += rewards; fmt_all += fmt
                m = sum(rewards) / G; v = statistics.pvariance(rewards); s = v ** 0.5
                if v == 0.0:
                    zv += 1; continue
                for i in range(G):
                    adv = (rewards[i] - m) / (s + 1e-6)
                    if adv:
                        lp = seq_logprob(model, pids, gens[i])
                        if lp is not None:
                            losses.append(-adv * lp)
            if losses:
                torch.stack(losses).sum().backward()   # SUMMED, not averaged
                gn = grad_norm(model)
            else:
                gn = 0.0
            opt.step()
            p = sum(rewards_all) / len(rewards_all)
            zvf = zv / BATCH
            row = {"seed": seed, "step": step, "difficulty": diff, "p": round(p, 3),
                   "zvf": round(zvf, 3), "gu": round(1 - zvf, 3),
                   "erf": round(sum(fmt_all) / len(fmt_all), 3),
                   "p1mp": round(p * (1 - p), 4), "grad_norm": round(gn, 3),
                   "signal_per_rollout": round(gn / (BATCH * G), 4)}
            all_rows.append(row)
            print(f"[e1] s{seed} st{step:2d} {diff:10s} p={p:.2f} ZVF={zvf:.2f} "
                  f"ERF={row['erf']:.2f} |g|={gn:8.2f} spr={row['signal_per_rollout']:.3f}", flush=True)
    del model, opt; torch.cuda.empty_cache()

gn_all = [r["grad_norm"] for r in all_rows]
summary = {
    "experiment": "E1_grad_signal", "model": MODEL, "seeds": SEEDS,
    "n_steps": len(all_rows),
    "pearson_gradnorm_vs_p1mp": round(pearson([r["p1mp"] for r in all_rows], gn_all), 3),
    "pearson_gradnorm_vs_GU":   round(pearson([r["gu"] for r in all_rows], gn_all), 3),
    "pearson_gradnorm_vs_ERF":  round(pearson([r["erf"] for r in all_rows], gn_all), 3),
    "by_difficulty": {d: {
        "mean_p": round(statistics.mean([r["p"] for r in all_rows if r["difficulty"] == d]), 3),
        "mean_ZVF": round(statistics.mean([r["zvf"] for r in all_rows if r["difficulty"] == d]), 3),
        "mean_grad_norm": round(statistics.mean([r["grad_norm"] for r in all_rows if r["difficulty"] == d]), 2),
    } for d in DIFFS},
}
print("E1_RESULT " + json.dumps(summary), flush=True)
