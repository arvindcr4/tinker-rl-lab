"""Parallel Tinker experiment runner — runs multiple GRPO configs CONCURRENTLY.

Tinker is a remote training API, so N training loops run in parallel from one
process (threads release the GIL during network I/O). Each config -> its own W&B
run + its own LoRA training client (no shared state; loss uses a per-call closure).

First batch: P3 group-size sweep G in {2,4,8,16}, compute-matched by steps, tracking
ZVF (zero-gradient waste), reward, and held-out accuracy gain per G.

Usage: python parallel_sweep.py --model Qwen/Qwen3.5-4B --steps 6 --seeds 0 --groups 2,4,8,16
"""
import os, re, json, random, argparse, warnings, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore")
assert os.environ.get("TINKER_API_KEY"), "need TINKER_API_KEY"
import torch, tinker, tinker.types as T
from transformers import AutoTokenizer
from datasets import load_dataset
import wandb

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
ap.add_argument("--steps", type=int, default=6)
ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--rank", type=int, default=8)
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--groups", default="2,4,8,16")
ap.add_argument("--heldout", type=int, default=8)
ap.add_argument("--max_parallel", type=int, default=4)
args = ap.parse_args()
GROUPS = [int(g) for g in args.groups.split(",")]

SYS = "You are a math assistant. Solve step by step, then give the final numerical answer inside \\boxed{}."
ds = load_dataset("openai/gsm8k", "main", split="train")
_ex = []
for row in ds:
    m = re.search(r"####\s*([\-\d,\.]+)", row["answer"])
    if not m: continue
    _ex.append((f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{row['question']}<|im_end|>\n<|im_start|>assistant\n",
                m.group(1).replace(",", "").strip()))
random.seed(args.seed); random.shuffle(_ex)
HELDOUT = _ex[:args.heldout]; POOL = _ex[args.heldout:]
print(f"pool={len(POOL)} heldout={len(HELDOUT)} groups={GROUPS} model={args.model}", flush=True)

def reward(response, answer):
    r = response.strip()
    for b in re.findall(r"\\boxed\{([^}]+)\}", r):
        bc = b.strip().replace(",", "").replace(" ", "")
        try:
            if abs(float(bc) - float(answer)) < 0.01: return 1.0
        except:
            if bc == answer: return 1.0
    nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", r)
    if nums:
        try:
            if abs(float(nums[-1].replace(",", "")) - float(answer)) < 0.01: return 1.0
        except: pass
    return 0.0

def make_loss(advs):
    def _fn(data, lp):
        losses = [(-advs[i] * lp[i].sum()) for i in range(len(lp))]
        loss = torch.stack(losses).mean()
        return loss, {"loss": loss.item()}
    return _fn

svc = tinker.ServiceClient(base_url=None)
_tok_cache = {}
_tok_lock = threading.Lock()
def get_tok(model):
    with _tok_lock:
        if model not in _tok_cache:
            _tok_cache[model] = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        return _tok_cache[model]

def heldout_acc(sc, tok):
    c = 0
    for pt, ans in HELDOUT:
        pid = tok.encode(pt, add_special_tokens=False)[:1024]
        resp = sc.sample(T.ModelInput.from_ints(pid), num_samples=1,
                         sampling_params=T.SamplingParams(max_tokens=512, temperature=0.0, top_p=1.0)).result()
        c += reward(tok.decode(list(resp.sequences[0].tokens), skip_special_tokens=True), ans)
    return c / len(HELDOUT)

def run_config(G):
    tok = get_tok(args.model)
    tc = svc.create_lora_training_client(base_model=args.model, rank=args.rank)
    w0 = tc.save_weights_for_sampler(name=f"G{G}_s0").result()
    sc = tc.create_sampling_client(model_path=w0.path)
    run = wandb.init(project="rlvr-openings", name=f"p3-groupsize-G{G}-{args.model.split('/')[-1]}-s{args.seed}",
                     group="p3-groupsize-sweep", config={"G": G, **vars(args)},
                     tags=["P3","group-size","tinker","parallel"], reinit=True)
    ho_before = heldout_acc(sc, tok)
    zero_loss = 0; rewards_hist = []; tokens_used = 0
    for step in range(args.steps):
        data, advs_all, batch_r = [], [], []
        for _ in range(args.batch):
            pt, ans = random.choice(POOL)
            pid = tok.encode(pt, add_special_tokens=False)[:1024]
            resp = sc.sample(T.ModelInput.from_ints(pid), num_samples=G,
                             sampling_params=T.SamplingParams(max_tokens=512, temperature=0.8, top_p=0.95)).result()
            rews = [reward(tok.decode(list(r.tokens), skip_special_tokens=True), ans) for r in resp.sequences]
            tokens_used += sum(len(r.tokens) for r in resp.sequences)
            mr = sum(rews)/len(rews); sr = (sum((x-mr)**2 for x in rews)/len(rews))**0.5 + 1e-8
            batch_r.extend(rews)
            for rseq, a in zip(resp.sequences, [(x-mr)/sr for x in rews]):
                fid = pid + list(rseq.tokens); tid = fid[1:] + [0]
                data.append(T.Datum(model_input=T.ModelInput.from_ints(fid),
                    loss_fn_inputs={"target_tokens": T.TensorData(data=tid, dtype="int64", shape=[len(tid)])}))
                advs_all.append(a)
        res = tc.forward_backward_custom(data=data, loss_fn=make_loss(advs_all)).result()
        tc.optim_step(T.AdamParams(learning_rate=args.lr, beta1=0.9, beta2=0.95, eps=1e-8)).result()
        lv = res.metrics.get("loss", 0.0)
        if abs(lv) < 1e-6: zero_loss += 1
        avg = sum(batch_r)/len(batch_r)
        rewards_hist.append(avg)
        wandb.log({"reward": avg, "loss": lv, "collapsed": int(abs(lv)<1e-6), "tokens_used": tokens_used}, step=step)
        ck = tc.save_weights_for_sampler(name=f"G{G}_s{step+1}").result()
        sc = tc.create_sampling_client(model_path=ck.path)
        print(f"[G{G}] {step+1}/{args.steps} loss={lv:.3f} reward={avg:.3f} tokens={tokens_used}", flush=True)
    ho_after = heldout_acc(sc, tok)
    out = {"G": G, "zero_loss_frac": zero_loss/args.steps, "mean_reward": sum(rewards_hist)/len(rewards_hist),
           "heldout_before": ho_before, "heldout_after": ho_after, "heldout_gain": ho_after-ho_before,
           "tokens_used": tokens_used}
    for k, v in out.items():
        run.summary[k] = v
    wandb.finish()
    print(f"[G{G}] DONE {json.dumps(out)} | {run.url}", flush=True)
    return out

results = []
with ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
    futs = {ex.submit(run_config, G): G for G in GROUPS}
    for f in as_completed(futs):
        try: results.append(f.result())
        except Exception as e:
            print(f"[G{futs[f]}] FAILED: {type(e).__name__}: {e}", flush=True)
results.sort(key=lambda d: d["G"])
os.makedirs("experiments/results/p3_groupsize", exist_ok=True)
json.dump(results, open("experiments/results/p3_groupsize/sweep_results.json","w"), indent=2)
print("=== SWEEP COMPLETE ===")
print(json.dumps(results, indent=2))
