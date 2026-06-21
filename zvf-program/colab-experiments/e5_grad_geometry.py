"""E5 (Pillar 2, theory): the GRADIENT GEOMETRY of GRPO vs ZVF / p(1-p).

Codex-review fix: the first draft's "gradient efficiency = live-fraction" is just
GU = 1 - ZVF restated (tautological, and loggable on Tinker). Redesigned to read
quantities that REQUIRE the open backward pass and are NOT 1-ZVF:

  per difficulty cell, for each live (variance>0) group we backprop each rollout's
  sequence log-prob through the LAST decoder layer and measure
    * grad_norm        : mean ||grad_i||                       (cf. E1, first moment)
    * fisher_trace     : mean ||grad_i||^2  ~ tr(Fisher) proxy  (curvature)
    * signal_per_roll  : ||sum_i adv_i * grad_i|| / G           (effective update)
    * snr              : ||sum adv_i grad_i|| / sum ||adv_i grad_i||   (alignment)
    * cos_align        : mean pairwise cosine of advantage-weighted per-rollout grads

Theory (T3): effective signal S = p(1-p)(1-h_G(p)) is INVERTED-U in difficulty and
-> 0 as ZVF -> 1 at both ends; it should correlate with p(1-p) more than with GU.
None of these per-rollout/curvature quantities are exposed by Tinker's closed
LoRA API -> Colab-only.

Run:  colab run --gpu T4 --timeout 1200 e5_grad_geometry.py
"""
import json, re, random, statistics
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SEEDS = [0, 1]
G, BATCH, MAX_NEW = 6, 8, 32
DIFFS = {                      # span p from ~1 (trivial) to ~0 (hard)
    "trivial": (1, 9, 1, 9),
    "easy":    (5, 25, 5, 25),
    "medium":  (25, 70, 25, 70),
    "hard":    (120, 600, 120, 600),
}
DEV = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
PAD = tok.pad_token_id

def problem(rng, spec):
    a = rng.randint(spec[0], spec[1]); b = rng.randint(spec[2], spec[3])
    return f"{a} + {b}", a + b

def prompt_of(q):
    return tok.apply_chat_template(
        [{"role": "user", "content": f"Compute {q}. Reason briefly, then put the final integer after '####'."}],
        tokenize=False, add_generation_prompt=True)

def parse(text):
    m = re.findall(r"-?\d+", text.split("####")[-1])
    return int(m[0]) if m else None

@torch.no_grad()
def gen_group(model, prompt, gold):
    model.eval()
    enc = tok([prompt] * G, return_tensors="pt", padding=True).to(DEV)
    out = model.generate(**enc, do_sample=True, temperature=1.0, top_p=0.95,
                         max_new_tokens=MAX_NEW, pad_token_id=PAD)
    gens = out[:, enc.input_ids.shape[1]:]
    rewards = [1.0 if parse(t) == gold else 0.0
               for t in tok.batch_decode(gens, skip_special_tokens=True)]
    return enc.input_ids[0], gens, rewards

def rollout_grad(model, slice_params, pids, gen_row):
    """Flattened gradient of one rollout's seq log-prob w.r.t. the last-layer slice."""
    gen_row = gen_row[gen_row != PAD]
    if gen_row.numel() == 0:
        return None
    ids = torch.cat([pids, gen_row]).unsqueeze(0)
    logits = model(ids).logits[:, :-1, :].float()
    tgt = ids[:, 1:]
    lp = F.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    seqlp = lp[:, pids.shape[0] - 1:].sum()
    grads = torch.autograd.grad(seqlp, slice_params, retain_graph=False, allow_unused=True)
    flat = torch.cat([g.reshape(-1) for g in grads if g is not None])
    return flat.detach()

def cell(model, slice_params, prompts):
    """Aggregate gradient geometry over the live groups of one difficulty cell."""
    norms, fishers, signals, snrs, coss = [], [], [], [], []
    rewards_all, n_live, n_groups = [], 0, 0
    for q, gold in prompts:
        pids, gens, rewards = gen_group(model, prompt_of(q), gold)
        rewards_all += rewards
        n_groups += 1
        m = sum(rewards) / G; v = statistics.pvariance(rewards)
        if v == 0.0:
            continue                              # ZVF group: no advantage, no gradient
        n_live += 1
        s = v ** 0.5
        gs, advs = [], []
        for i in range(G):
            adv = (rewards[i] - m) / (s + 1e-6)
            g = rollout_grad(model, slice_params, pids, gens[i])
            if g is not None:
                gs.append(g); advs.append(adv)
        if not gs:
            continue
        Gmat = torch.stack(gs)                    # [g, d]
        a = torch.tensor(advs, device=Gmat.device, dtype=Gmat.dtype)
        wg = a.unsqueeze(1) * Gmat                # advantage-weighted per-rollout grads
        agg = wg.sum(0)
        per = Gmat.norm(dim=1)
        norms.append(per.mean().item())
        fishers.append((per ** 2).mean().item())
        signals.append((agg.norm() / G).item())
        denom = wg.norm(dim=1).sum().clamp_min(1e-9)
        snrs.append((agg.norm() / denom).item())
        wn = F.normalize(wg, dim=1)
        cm = wn @ wn.t()
        off = cm.sum() - cm.diag().sum()
        coss.append((off / (len(gs) * (len(gs) - 1) + 1e-9)).item())
    p = sum(rewards_all) / len(rewards_all)
    zvf = 1 - n_live / n_groups
    agg = lambda xs: round(statistics.mean(xs), 5) if xs else None
    return {"p": round(p, 3), "zvf": round(zvf, 3), "p1mp": round(p * (1 - p), 4),
            "grad_norm": agg(norms), "fisher_trace": agg(fishers),
            "signal_per_roll": agg(signals), "snr": agg(snrs),
            "cos_align": agg(coss), "n_live": n_live, "n_groups": n_groups}

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
    by_diff = {d: [] for d in DIFFS}
    for seed in SEEDS:
        rng = random.Random(seed); torch.manual_seed(seed)
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV)
        slice_params = [p for p in model.model.layers[-1].parameters() if p.requires_grad]
        for d, spec in DIFFS.items():
            prompts = [problem(rng, spec) for _ in range(BATCH)]
            c = cell(model, slice_params, prompts)
            by_diff[d].append(c)
            print(f"[e5 s{seed}] {d:8s} p={c['p']:.2f} ZVF={c['zvf']:.2f} "
                  f"signal={c['signal_per_roll']} snr={c['snr']} fisher={c['fisher_trace']}", flush=True)
        del model; torch.cuda.empty_cache()

    # average cells across seeds
    avg = {}
    for d, cs in by_diff.items():
        keys = ["p", "zvf", "p1mp", "grad_norm", "fisher_trace", "signal_per_roll", "snr", "cos_align"]
        avg[d] = {k: round(statistics.mean([c[k] for c in cs if c[k] is not None]), 4)
                  if any(c[k] is not None for c in cs) else None for k in keys}
    order = list(DIFFS)
    p1mp = [avg[d]["p1mp"] for d in order]
    gu = [1 - avg[d]["zvf"] for d in order]
    signal = [avg[d]["signal_per_roll"] for d in order]
    fisher = [avg[d]["fisher_trace"] for d in order]
    result = {"experiment": "E5_grad_geometry", "model": MODEL, "seeds": SEEDS,
              "by_difficulty": avg,
              "corr_signal_p1mp": pearson(signal, p1mp),
              "corr_signal_gu": pearson(signal, gu),
              "corr_fisher_p1mp": pearson(fisher, p1mp)}
    print("E5_RESULT " + json.dumps(result), flush=True)

main()
