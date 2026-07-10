"""Ceiling-effect test — is the curriculum null just because GSM8K is too easy for 4B?

The earlier nulls (curriculum ties baseline) were on full GSM8K where the 4B model already
solves ~80% (few learnable groups). This pre-filters to HARD-LEARNABLE prompts (base pass-rate
in (0, 0.5]) where there is real headroom AND mixed-variance groups, then re-runs baseline vs
curriculum multi-seed. If curriculum wins HERE, the earlier null was a ceiling effect.

Stage 1 (per seed, shared): sample base model on N_PROBE prompts, keep those with 0<pass<=0.5.
Stage 2: baseline vs curriculum GRPO on that hard-learnable pool. Multi-seed, process-parallel.
"""
import os, re, json, random, argparse, warnings
warnings.filterwarnings("ignore")
import multiprocessing as mp

def run_one(cfg):
    import torch, tinker, tinker.types as T, wandb
    from transformers import AutoTokenizer
    from datasets import load_dataset
    name, mode, seed, steps, G, heldout_n, n_probe, model, rank, lr, batch = cfg
    random.seed(seed); torch.manual_seed(seed)
    SYS = "You are a math assistant. Solve step by step, then give the final numerical answer inside \\boxed{}."
    ds = load_dataset("openai/gsm8k", "main", split="train")
    ex = []
    for row in ds:
        m = re.search(r"####\s*([\-\d,\.]+)", row["answer"])
        if not m: continue
        ex.append((f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{row['question']}<|im_end|>\n<|im_start|>assistant\n",
                   m.group(1).replace(",", "").strip()))
    random.shuffle(ex)
    def reward(resp, ans):
        r = resp.strip()
        for b in re.findall(r"\\boxed\{([^}]+)\}", r):
            bc = b.strip().replace(",", "").replace(" ", "")
            try:
                if abs(float(bc)-float(ans)) < 0.01: return 1.0
            except:
                if bc == ans: return 1.0
        nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", r)
        if nums:
            try:
                if abs(float(nums[-1].replace(",",""))-float(ans)) < 0.01: return 1.0
            except: pass
        return 0.0
    def mkloss(advs):
        def _fn(data, lp):
            l = torch.stack([(-advs[i]*lp[i].sum()) for i in range(len(lp))]).mean()
            return l, {"loss": l.item()}
        return _fn
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    svc = tinker.ServiceClient(base_url=None)
    tc = svc.create_lora_training_client(base_model=model, rank=rank)
    sc = tc.create_sampling_client(model_path=tc.save_weights_for_sampler(name=f"{name}_0").result().path)
    def probe_passrate(pt, ans, k=6):
        pid = tok.encode(pt, add_special_tokens=False)[:1024]
        resp = sc.sample(T.ModelInput.from_ints(pid), num_samples=k,
                         sampling_params=T.SamplingParams(max_tokens=512, temperature=0.8, top_p=0.95)).result()
        return sum(reward(tok.decode(list(r.tokens), skip_special_tokens=True), ans) for r in resp.sequences)/k
    # Stage 1: difficulty pre-filter (shared probe order via the seeded shuffle)
    hard = []
    for pt, ans in ex[:n_probe]:
        p = probe_passrate(pt, ans)
        if 0.0 < p < 1.0:   # any non-collapsed (mixed-variance) prompt = has gradient signal
            hard.append((pt, ans))
        if len(hard) >= heldout_n + 40:
            break
    if len(hard) < heldout_n + 8:
        out = dict(name=name, mode=mode, seed=seed, error="too few hard-learnable prompts", n_hard=len(hard))
        print("DONE", json.dumps(out), flush=True); return out
    HO = hard[:heldout_n]; POOL = hard[heldout_n:]
    def held():
        c = 0
        for pt, ans in HO:
            pid = tok.encode(pt, add_special_tokens=False)[:1024]
            resp = sc.sample(T.ModelInput.from_ints(pid), num_samples=1,
                             sampling_params=T.SamplingParams(max_tokens=512, temperature=0.0, top_p=1.0)).result()
            c += reward(tok.decode(list(resp.sequences[0].tokens), skip_special_tokens=True), ans)
        return c/len(HO)
    run = wandb.init(project="rlvr-openings", name=name, group="hard-curriculum",
                     config=dict(mode=mode, seed=seed, steps=steps, G=G, model=model),
                     tags=["hard-curriculum", mode, f"seed{seed}"], reinit=True)
    ho0 = held(); zero_loss = 0; rewards = []
    for step in range(steps):
        data, advs, br = [], [], []
        for _ in range(batch):
            pt, ans = random.choice(POOL)
            pid = tok.encode(pt, add_special_tokens=False)[:1024]
            resp = sc.sample(T.ModelInput.from_ints(pid), num_samples=G,
                             sampling_params=T.SamplingParams(max_tokens=512, temperature=0.8, top_p=0.95)).result()
            rr = [reward(tok.decode(list(r.tokens), skip_special_tokens=True), ans) for r in resp.sequences]
            mr = sum(rr)/len(rr)
            if mode == "curriculum" and mr in (0.0, 1.0):
                continue
            sd = (sum((x-mr)**2 for x in rr)/len(rr))**0.5 + 1e-8; br.extend(rr)
            for rs, a in zip(resp.sequences, [(x-mr)/sd for x in rr]):
                fid = pid + list(rs.tokens); tid = fid[1:]+[0]
                data.append(T.Datum(model_input=T.ModelInput.from_ints(fid),
                    loss_fn_inputs={"target_tokens": T.TensorData(data=tid, dtype="int64", shape=[len(tid)])}))
                advs.append(a)
        if not data: continue
        res = tc.forward_backward_custom(data=data, loss_fn=mkloss(advs)).result()
        tc.optim_step(T.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)).result()
        lv = res.metrics.get("loss", 0.0)
        if abs(lv) < 1e-6: zero_loss += 1
        avg = sum(br)/len(br) if br else 0.0; rewards.append(avg)
        wandb.log({"reward": avg, "loss": lv}, step=step)
        sc = tc.create_sampling_client(model_path=tc.save_weights_for_sampler(name=f"{name}_{step+1}").result().path)
    ho1 = held()
    out = dict(name=name, mode=mode, seed=seed, n_hard_pool=len(POOL), zero_loss_frac=zero_loss/max(steps,1),
               heldout_before=ho0, heldout_after=ho1, heldout_gain=ho1-ho0)
    for k, v in out.items(): run.summary[k] = v
    wandb.finish(); print("DONE", json.dumps(out), flush=True)
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--G", type=int, default=6)
    ap.add_argument("--heldout", type=int, default=16)
    ap.add_argument("--n_probe", type=int, default=120)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    cfgs = [(f"{mode}-hard-s{s}", mode, s, a.steps, a.G, a.heldout, a.n_probe, a.model, a.rank, a.lr, a.batch)
            for s in seeds for mode in ["baseline", "curriculum"]]
    print(f"HARD-CURRICULUM: {len(cfgs)} runs (hard-learnable pool), {a.workers} parallel", flush=True)
    with mp.get_context("spawn").Pool(a.workers) as pool:
        results = pool.map(run_one, cfgs)
    os.makedirs("experiments/results/hard_curriculum", exist_ok=True)
    json.dump(results, open("experiments/results/hard_curriculum/results.json", "w"), indent=2)
    print("=== HARD-CURRICULUM COMPLETE ===", flush=True); print(json.dumps(results, indent=2))
