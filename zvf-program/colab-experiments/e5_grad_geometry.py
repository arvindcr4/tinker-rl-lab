"""E5 v4 (Pillar 2, theory): does the GRPO gradient signal track p(1-p)?

agy (Gemini 3.1 Pro) hardening catches, all addressed here:
  * MAX_NEW=24 truncated reasoning -> spurious 0 reward -> low-p starved of live
    groups. -> MAX_NEW=128 + parser returns None when '####' is absent.
  * The v3 negative corr was a SIMPSON'S-PARADOX artifact: GRPO's empirical
    per-group std (r-m)/(s+eps) explodes the advantage of rare failures in nearly
    saturated groups. -> POPULATION-standardized advantage (r - rbar)/sd using the
    PILOT estimate of each prompt's reward mean/std (stable, not the tiny group's).
  * Binary exact-match can't produce LIVE groups at low exact-match p on a 0.5B
    model. -> CONTINUOUS partial-credit reward (1 - |ans-gold|/|gold|) so even hard
    prompts have reward variance -> live groups across the whole difficulty range.
  * Length-normalization is a possible confound -> kept (per-token) but mean
    completion length is reported per bin so the confound is visible.

x-axis = the THEORY's quantity, exact-match p(1-p) (continuous reward only provides
the contrast needed to measure). Per-LIVE-group regression of signal on x.

Run:  colab run --gpu T4 --timeout 1800 e5_grad_geometry.py
"""
import json, re, random, statistics
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 0
G, R_GROUPS, MAX_NEW = 6, 6, 128
N_CAND, N_PILOT, N_KEEP = 80, 24, 32
MIN_SD = 0.04          # keep prompts whose pilot reward std can yield live groups
DEV = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
PAD = tok.pad_token_id

FEWSHOT = [
    {"role": "user", "content": "Compute 3 + 4. Reason briefly, then put the final integer after '####'."},
    {"role": "assistant", "content": "3 + 4 = 7.\n#### 7"},
]

def candidate(rng):
    lvl = rng.randint(0, 4)
    if lvl == 0:
        a, b = rng.randint(1, 30), rng.randint(1, 30); op = "+"
    elif lvl == 1:
        a, b = rng.randint(10, 99), rng.randint(10, 99); op = "+"
    elif lvl == 2:
        a, b = rng.randint(100, 999), rng.randint(100, 999); op = "+"
    elif lvl == 3:
        a, b = rng.randint(11, 99), rng.randint(11, 99); op = "*"   # 2-digit mult
    else:
        a, b = rng.randint(100, 999), rng.randint(11, 99); op = "*"  # hard mult
    return f"{a} {op} {b}", (a + b if op == "+" else a * b)

def prompt_of(q):
    msgs = FEWSHOT + [{"role": "user",
                       "content": f"Compute {q}. Reason briefly, then put the final integer after '####'."}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def parse(text):
    if "####" not in text:
        return None, False
    m = re.findall(r"-?\d+", text.split("####")[-1])
    return (int(m[0]) if m else None), bool(m)

def reward(ans, gold):
    """Continuous partial credit -> contrast (variance) even when exact-match p ~ 0."""
    if ans is None:
        return 0.0
    if ans == gold:
        return 1.0
    return max(0.0, 1.0 - abs(ans - gold) / max(1, abs(gold)))

@torch.no_grad()
def pilot(model, prompt, gold, n):
    enc = tok([prompt] * n, return_tensors="pt", padding=True).to(DEV)
    out = model.generate(**enc, do_sample=True, temperature=1.0, top_p=0.95,
                         max_new_tokens=MAX_NEW, pad_token_id=PAD)
    gens = out[:, enc.input_ids.shape[1]:]
    rs, ex = [], []
    for t in tok.batch_decode(gens, skip_special_tokens=True):
        ans, _ = parse(t)
        rs.append(reward(ans, gold)); ex.append(1.0 if ans == gold else 0.0)
    return rs, statistics.mean(ex)

def calibrate(model, rng):
    kept = []
    for _ in range(N_CAND):
        q, gold = candidate(rng)
        rs, exact_p = pilot(model, prompt_of(q), gold, N_PILOT)
        rbar, sd = statistics.mean(rs), statistics.pstdev(rs)
        if sd >= MIN_SD:                          # has contrast -> will yield live groups
            kept.append({"q": q, "gold": gold, "exact_p": exact_p, "rbar": rbar, "sd": sd})
        if len(kept) >= N_KEEP:
            break
    kept.sort(key=lambda d: d["exact_p"])
    if kept:
        print(f"[e5] kept {len(kept)} prompts; exact_p in "
              f"[{kept[0]['exact_p']:.2f},{kept[-1]['exact_p']:.2f}]", flush=True)
    return kept

def gen_group(model, prompt, gold):
    model.eval()
    enc = tok([prompt] * G, return_tensors="pt", padding=True).to(DEV)
    with torch.no_grad():
        out = model.generate(**enc, do_sample=True, temperature=1.0, top_p=0.95,
                             max_new_tokens=MAX_NEW, pad_token_id=PAD)
    gens = out[:, enc.input_ids.shape[1]:]
    rewards = [reward(parse(t)[0], gold)
               for t in tok.batch_decode(gens, skip_special_tokens=True)]
    return enc.input_ids[0], gens, rewards

def rollout_grad(model, slice_params, pids, gen_row):
    gen_row = gen_row[gen_row != PAD]
    n = gen_row.numel()
    if n == 0:
        return None, 0
    ids = torch.cat([pids, gen_row]).unsqueeze(0)
    logits = model(ids).logits[:, :-1, :].float()
    tgt = ids[:, 1:]
    lp = F.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    seqlp = lp[:, pids.shape[0] - 1:].sum() / n        # length-normalized
    grads = torch.autograd.grad(seqlp, slice_params, allow_unused=True)
    return torch.cat([g.reshape(-1) for g in grads if g is not None]).detach(), int(n)

def group_signal(model, slice_params, pids, gens, rewards, rbar, sd):
    # POPULATION-standardized advantage (pilot rbar/sd) -> no small-group std explosion
    gs, advs, lens = [], [], []
    for i in range(G):
        adv = (rewards[i] - rbar) / (sd + 1e-6)
        g, n = rollout_grad(model, slice_params, pids, gens[i])
        if g is not None:
            gs.append(g); advs.append(adv); lens.append(n)
    if len(gs) < 2:
        return None
    Gmat = torch.stack(gs)
    a = torch.tensor(advs, device=Gmat.device, dtype=Gmat.dtype)
    wg = a.unsqueeze(1) * Gmat
    agg = wg.sum(0)
    per = Gmat.norm(dim=1)
    return {"signal": (agg.norm() / G).item(), "fisher": (per ** 2).mean().item(),
            "snr": (agg.norm() / wg.norm(dim=1).sum().clamp_min(1e-9)).item(),
            "mean_len": statistics.mean(lens)}

def pearson(xs, ys):
    if len(xs) < 3:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return round(num / den, 3) if den else None

def ols_slope(xs, ys):
    if len(xs) < 3:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return round(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den, 3) if den else None

def main():
    rng = random.Random(SEED); torch.manual_seed(SEED)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV)
    slice_params = [p for layer in model.model.layers[-2:] for p in layer.parameters() if p.requires_grad]
    prompts = calibrate(model, rng)

    rows = []
    for d in prompts:
        x = d["exact_p"] * (1 - d["exact_p"])        # theory's quantity
        for _ in range(R_GROUPS):
            pids, gens, rewards = gen_group(model, prompt_of(d["q"]), d["gold"])
            if statistics.pvariance(rewards) <= 1e-9:
                continue                              # dead group
            gsig = group_signal(model, slice_params, pids, gens, rewards, d["rbar"], d["sd"])
            if gsig is not None:
                rows.append({"x": x, "exact_p": d["exact_p"], **gsig})
        print(f"[e5] exact_p~{d['exact_p']:.2f} x={x:.3f} -> "
              f"{sum(1 for r in rows if r['exact_p']==d['exact_p'])} live groups", flush=True)

    xs = [r["x"] for r in rows]
    by_bin = {}
    for lo, hi, lab in [(0.0, 0.08, "p1mp<0.08"), (0.08, 0.16, "0.08-0.16"),
                        (0.16, 0.22, "0.16-0.22"), (0.22, 0.26, "p1mp>0.22")]:
        b = [r for r in rows if lo <= r["x"] < hi]
        if b:
            by_bin[lab] = {"n": len(b), "mean_exact_p": round(statistics.mean(r["exact_p"] for r in b), 3),
                           "mean_signal": round(statistics.mean(r["signal"] for r in b), 4),
                           "mean_fisher": round(statistics.mean(r["fisher"] for r in b), 2),
                           "mean_snr": round(statistics.mean(r["snr"] for r in b), 4),
                           "mean_len": round(statistics.mean(r["mean_len"] for r in b), 1)}
    result = {"experiment": "E5_grad_geometry", "model": MODEL, "seed": SEED,
              "method": "per-group regression; continuous reward; population-standardized advantage",
              "reward": "continuous_partial_credit", "advantage": "population_standardized_pilot",
              "max_new": MAX_NEW, "few_shot": True, "length_normalized": True,
              "grad_slice": "last_2_decoder_layers", "n_prompts": len(prompts),
              "n_live_groups": len(rows),
              "exact_p_range": [round(min(r["exact_p"] for r in rows), 3),
                                round(max(r["exact_p"] for r in rows), 3)] if rows else None,
              "corr_signal_p1mp": pearson(xs, [r["signal"] for r in rows]),
              "slope_signal_p1mp": ols_slope(xs, [r["signal"] for r in rows]),
              "corr_fisher_p1mp": pearson(xs, [r["fisher"] for r in rows]),
              "corr_snr_p1mp": pearson(xs, [r["snr"] for r in rows]),
              "by_p_bin": by_bin}
    print("E5_RESULT " + json.dumps(result), flush=True)

main()
