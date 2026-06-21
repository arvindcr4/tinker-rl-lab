"""E7 (Pillar 4, MIN-REPORT-RL): how much does flipping ONE unreported "stack"
lever move the headline numbers, holding task/data/seed/compute/algorithm fixed?

Codex-review fixes vs the first draft:
  * PAIRED + REPLICATED: identical prompt stream and seed across arms, >=2 seeds,
    report mean +- std (a single 10-step toy run is too noisy to trust).
  * HEADLINE on ZVF / ERF / reward trajectory, NOT held-out delta (held-out over
    ~10 toy steps is the noisiest possible estimator).
  * Trim to the cleanly Tinker-blocked levers: precision (fp32) and attention
    backend (eager), plus one sampler lever (temp 0.7) as an illustrative contrast.

Reference config = bf16 / sdpa / temp 1.0 / top_p 0.95. Each arm flips exactly one
lever. ERF = fraction of rollouts emitting a parseable '####' answer (the
format-vs-wrong confound the parent paper flags).

Colab-only because: numerical precision and the attention backend are fixed by
Tinker; you cannot isolate them there.

Run:  colab run --gpu T4 --timeout 1500 e7_stack_levers.py
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json, re, random, statistics
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SEEDS = [0, 1]
G, BATCH, MAX_NEW, LR, STEPS = 6, 4, 32, 2e-6, 10
HELDOUT_N = 20
DEV = "cuda" if torch.cuda.is_available() else "cpu"

ARMS = {  # name -> (dtype, attn_impl, temperature, top_p)
    "reference": (torch.bfloat16, "sdpa",  1.0, 0.95),
    "fp32":      (torch.float32,  "sdpa",  1.0, 0.95),
    "eager_attn":(torch.bfloat16, "eager", 1.0, 0.95),
    "temp_0.7":  (torch.bfloat16, "sdpa",  0.7, 0.95),
}

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
PAD = tok.pad_token_id

def problem(rng):
    a, b = rng.randint(25, 90), rng.randint(25, 90)
    return f"{a} + {b}", a + b

def prompt_of(q):
    return tok.apply_chat_template(
        [{"role": "user", "content": f"Compute {q}. Reason briefly, then put the final integer after '####'."}],
        tokenize=False, add_generation_prompt=True)

def parse(text):
    seg = text.split("####")
    if len(seg) < 2:
        return None, False                 # no '####' emitted -> not format-compliant
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

def run(arm, seed):
    dtype, attn, temp, top_p = ARMS[arm]
    # PAIRED: same seed -> same prompt stream + eval set across all arms
    rng = random.Random(seed); torch.manual_seed(seed)
    evalset = [problem(rng) for _ in range(HELDOUT_N)]
    train_stream = [[problem(rng) for _ in range(BATCH)] for _ in range(STEPS)]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, attn_implementation=attn).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    pre = heldout_acc(model, evalset)
    zvfs, erfs, ps = [], [], []
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
                        (-a * lp).backward()      # accumulate grad, free graph (fp32-safe)
                        n_terms += 1
        if n_terms:
            opt.step()
        zvfs.append(zv / BATCH); erfs.append(sum(fall) / len(fall)); ps.append(sum(rall) / len(rall))
        print(f"[e7:{arm:10s} s{seed}] step={step+1:2d} ZVF={zvfs[-1]:.2f} ERF={erfs[-1]:.2f} p={ps[-1]:.2f}", flush=True)
    post = heldout_acc(model, evalset)
    out = {"arm": arm, "seed": seed, "mean_zvf": round(statistics.mean(zvfs), 3),
           "mean_erf": round(statistics.mean(erfs), 3), "last3_p": round(statistics.mean(ps[-3:]), 3),
           "heldout_delta": round(post - pre, 3)}
    del model, opt; torch.cuda.empty_cache()
    return out

results = []
for arm in ARMS:
    for seed in SEEDS:
        results.append(run(arm, seed))

ref = {s: next(r for r in results if r["arm"] == "reference" and r["seed"] == s) for s in SEEDS}
def ms(xs):
    return [round(statistics.mean(xs), 3), round(statistics.pstdev(xs), 3)]
by_lever = {}
for arm in ARMS:
    rs = [r for r in results if r["arm"] == arm]
    by_lever[arm] = {
        "mean_zvf": ms([r["mean_zvf"] for r in rs]),
        "mean_erf": ms([r["mean_erf"] for r in rs]),
        "last3_p": ms([r["last3_p"] for r in rs]),
        "delta_zvf_vs_ref": ms([r["mean_zvf"] - ref[r["seed"]]["mean_zvf"] for r in rs]),
        "delta_p_vs_ref": ms([r["last3_p"] - ref[r["seed"]]["last3_p"] for r in rs]),
        "delta_heldout_vs_ref": ms([r["heldout_delta"] - ref[r["seed"]]["heldout_delta"] for r in rs]),
    }
print("E7_RESULT " + json.dumps({"experiment": "E7_stack_levers", "model": MODEL, "seeds": SEEDS,
                                "reference": "bf16/sdpa/temp1.0/top_p0.95",
                                "by_lever": by_lever, "runs": results}), flush=True)
