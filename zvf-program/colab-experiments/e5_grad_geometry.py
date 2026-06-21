"""E5 (Pillar 2, theory): GRADIENT GEOMETRY of GRPO vs ZVF / p(1-p), measured with
quantities that REQUIRE the open backward pass (not loggable on Tinker).

Round-2 Codex fix: the first run's p was compressed to 0.12-0.27 (format-gated),
so the p(1-p) correlation spanned almost no range. Now:
  * FEW-SHOT scaffold removes the format confound; ERF (format rate) reported per bin.
  * CALIBRATE prompts into 5 empirical p-bins ~[0.05,0.25,0.5,0.75,0.95] so p(1-p)
    actually sweeps its inverted-U.
  * Length-NORMALIZED completion log-prob for the gradient (removes the length
    confound in gradient magnitude).
  * Gradient slice = last TWO decoder layers (a wider proxy than one layer).

Per live group we measure mean ||grad_i|| (first moment), mean ||grad_i||^2
(Fisher-trace proxy), ||sum adv_i grad_i||/G (effective signal), advantage-weighted
SNR and cosine alignment. Theory T3: signal/Fisher track p(1-p) (inverted-U) more
than GU=1-ZVF (monotone).

Run:  colab run --gpu T4 --timeout 1500 e5_grad_geometry.py
"""
import json, re, random, statistics
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 0
G, MAX_NEW = 6, 24
TARGET_PS = [0.05, 0.25, 0.5, 0.75, 0.95]
TOL = 0.12            # bin assignment tolerance
N_PER_BIN = 8         # calibrated prompts kept per bin
N_PILOT = 20
N_CANDIDATES = 130
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
    r = rng.random()
    if r < 0.4:
        a, b = rng.randint(1, 50), rng.randint(1, 50)            # easy -> high p
    elif r < 0.7:
        a, b = rng.randint(10, 99), rng.randint(100, 999)        # mixed -> mid p
    else:
        a, b = rng.randint(200, 999), rng.randint(200, 999)      # hard -> low p
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

@torch.no_grad()
def pilot_p(model, prompt, gold, n):
    enc = tok([prompt] * n, return_tensors="pt", padding=True).to(DEV)
    out = model.generate(**enc, do_sample=True, temperature=1.0, top_p=0.95,
                         max_new_tokens=MAX_NEW, pad_token_id=PAD)
    gens = out[:, enc.input_ids.shape[1]:]
    return statistics.mean(1.0 if parse(t)[0] == gold else 0.0
                           for t in tok.batch_decode(gens, skip_special_tokens=True))

def calibrate(model, rng):
    bins = {t: [] for t in TARGET_PS}
    screened = 0
    for _ in range(N_CANDIDATES):
        q, gold = candidate(rng)
        ph = pilot_p(model, prompt_of(q), gold, N_PILOT)
        screened += 1
        t = min(TARGET_PS, key=lambda t: abs(t - ph))
        if abs(t - ph) <= TOL and len(bins[t]) < N_PER_BIN:
            bins[t].append((q, gold, ph))
        if all(len(v) >= N_PER_BIN for v in bins.values()):
            break
    for t in TARGET_PS:
        print(f"[e5] bin p~{t}: {len(bins[t])} prompts (screened {screened})", flush=True)
    return bins

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
        rewards.append(1.0 if ans == gold else 0.0); fmt.append(1.0 if ok else 0.0)
    return enc.input_ids[0], gens, rewards, fmt

def rollout_grad(model, slice_params, pids, gen_row):
    gen_row = gen_row[gen_row != PAD]
    n = gen_row.numel()
    if n == 0:
        return None
    ids = torch.cat([pids, gen_row]).unsqueeze(0)
    logits = model(ids).logits[:, :-1, :].float()
    tgt = ids[:, 1:]
    lp = F.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    seqlp = lp[:, pids.shape[0] - 1:].sum() / n         # LENGTH-NORMALIZED
    grads = torch.autograd.grad(seqlp, slice_params, allow_unused=True)
    return torch.cat([g.reshape(-1) for g in grads if g is not None]).detach()

def bin_geometry(model, slice_params, prompts):
    norms, fishers, signals, snrs, coss = [], [], [], [], []
    rewards_all, fmt_all, n_live, n_groups = [], [], 0, 0
    for q, gold, _ in prompts:
        pids, gens, rewards, fmt = gen_group(model, prompt_of(q), gold)
        rewards_all += rewards; fmt_all += fmt; n_groups += 1
        m = sum(rewards) / G; v = statistics.pvariance(rewards)
        if v == 0.0:
            continue
        n_live += 1; s = v ** 0.5
        gs, advs = [], []
        for i in range(G):
            adv = (rewards[i] - m) / (s + 1e-6)
            g = rollout_grad(model, slice_params, pids, gens[i])
            if g is not None:
                gs.append(g); advs.append(adv)
        if not gs:
            continue
        Gmat = torch.stack(gs)
        a = torch.tensor(advs, device=Gmat.device, dtype=Gmat.dtype)
        wg = a.unsqueeze(1) * Gmat
        agg = wg.sum(0)
        per = Gmat.norm(dim=1)
        norms.append(per.mean().item()); fishers.append((per ** 2).mean().item())
        signals.append((agg.norm() / G).item())
        snrs.append((agg.norm() / wg.norm(dim=1).sum().clamp_min(1e-9)).item())
        wn = F.normalize(wg, dim=1); cm = wn @ wn.t()
        coss.append(((cm.sum() - cm.diag().sum()) / (len(gs) * (len(gs) - 1) + 1e-9)).item())
    p = sum(rewards_all) / len(rewards_all)
    erf = sum(fmt_all) / len(fmt_all)
    zvf = 1 - n_live / n_groups
    ag = lambda xs: round(statistics.mean(xs), 5) if xs else None
    return {"p": round(p, 3), "erf": round(erf, 3), "zvf": round(zvf, 3),
            "p1mp": round(p * (1 - p), 4), "grad_norm": ag(norms), "fisher_trace": ag(fishers),
            "signal_per_roll": ag(signals), "snr": ag(snrs), "cos_align": ag(coss),
            "n_live": n_live, "n_groups": n_groups}

def pearson(xs, ys):
    pts = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pts) < 3:
        return None
    xs, ys = zip(*pts)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in pts)
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return round(num / den, 3) if den else None

def main():
    rng = random.Random(SEED); torch.manual_seed(SEED)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV)
    slice_params = [p for layer in model.model.layers[-2:] for p in layer.parameters() if p.requires_grad]
    bins = calibrate(model, rng)
    by_bin = {}
    for t in TARGET_PS:
        if not bins[t]:
            continue
        c = bin_geometry(model, slice_params, bins[t])
        by_bin[f"p~{t}"] = c
        print(f"[e5] p~{t}: p={c['p']:.2f} erf={c['erf']:.2f} ZVF={c['zvf']:.2f} "
              f"signal={c['signal_per_roll']} snr={c['snr']} fisher={c['fisher_trace']}", flush=True)
    cells = list(by_bin.values())
    p1mp = [c["p1mp"] for c in cells]
    gu = [1 - c["zvf"] for c in cells]
    signal = [c["signal_per_roll"] for c in cells]
    fisher = [c["fisher_trace"] for c in cells]
    result = {"experiment": "E5_grad_geometry", "model": MODEL, "seed": SEED, "few_shot": True,
              "length_normalized": True, "grad_slice": "last_2_decoder_layers",
              "p_range": [round(min(c['p'] for c in cells), 3), round(max(c['p'] for c in cells), 3)],
              "by_bin": by_bin,
              "corr_signal_p1mp": pearson(signal, p1mp), "corr_signal_gu": pearson(signal, gu),
              "corr_fisher_p1mp": pearson(fisher, p1mp)}
    print("E5_RESULT " + json.dumps(result), flush=True)

main()
