"""P2/P3 strengthener — multi-seed group-size sweep with per-group ZVF logging.

Two goals: (1) give P3 a POWERED group-size answer (the earlier single-seed sweep was
contradicted); (2) measure the P2 backbone live — does per-group ZVF (fraction of groups
with zero within-group reward variance) track held-out outcome across configs?

Logs per step: zvf = mean over the batch's groups of 1[var(rewards)==0]; reward; then
held-out gain. Sweep G in {2,4,8,16} x seeds {0,1,2}. Process-parallel.
"""
import os, re, json, random, argparse, warnings
warnings.filterwarnings("ignore")
import multiprocessing as mp

def run_one(cfg):
    import torch, tinker, tinker.types as T, wandb
    from transformers import AutoTokenizer
    from datasets import load_dataset
    name, G, seed, steps, heldout_n, model, rank, lr, batch = cfg
    random.seed(seed); torch.manual_seed(seed)
    SYS = "You are a math assistant. Solve step by step, then give the final numerical answer inside \\boxed{}."
    ds = load_dataset("openai/gsm8k", "main", split="train")
    ex = []
    for row in ds:
        m = re.search(r"####\s*([\-\d,\.]+)", row["answer"])
        if not m: continue
        ex.append((f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{row['question']}<|im_end|>\n<|im_start|>assistant\n",
                   m.group(1).replace(",", "").strip()))
    random.shuffle(ex); HO = ex[:heldout_n]; POOL = ex[heldout_n:]
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
    def held():
        c = 0
        for pt, ans in HO:
            pid = tok.encode(pt, add_special_tokens=False)[:1024]
            resp = sc.sample(T.ModelInput.from_ints(pid), num_samples=1,
                             sampling_params=T.SamplingParams(max_tokens=512, temperature=0.0, top_p=1.0)).result()
            c += reward(tok.decode(list(resp.sequences[0].tokens), skip_special_tokens=True), ans)
        return c/len(HO)
    run = wandb.init(project="rlvr-openings", name=name, group="groupsize-zvf",
                     config=dict(G=G, seed=seed, steps=steps, model=model),
                     tags=["P3","P2","zvf",f"G{G}",f"seed{seed}"], reinit=True)
    ho0 = held(); zvf_hist = []; tokens = 0
    for step in range(steps):
        data, advs, br, group_zv = [], [], [], []
        for _ in range(batch):
            pt, ans = random.choice(POOL)
            pid = tok.encode(pt, add_special_tokens=False)[:1024]
            resp = sc.sample(T.ModelInput.from_ints(pid), num_samples=G,
                             sampling_params=T.SamplingParams(max_tokens=512, temperature=0.8, top_p=0.95)).result()
            tokens += sum(len(r.tokens) for r in resp.sequences)
            rr = [reward(tok.decode(list(r.tokens), skip_special_tokens=True), ans) for r in resp.sequences]
            mr = sum(rr)/len(rr); var = sum((x-mr)**2 for x in rr)/len(rr)
            group_zv.append(1.0 if var == 0.0 else 0.0)   # per-group zero-variance indicator
            if var == 0.0:
                continue
            sd = var**0.5 + 1e-8; br.extend(rr)
            for rs, a in zip(resp.sequences, [(x-mr)/sd for x in rr]):
                fid = pid + list(rs.tokens); tid = fid[1:]+[0]
                data.append(T.Datum(model_input=T.ModelInput.from_ints(fid),
                    loss_fn_inputs={"target_tokens": T.TensorData(data=tid, dtype="int64", shape=[len(tid)])}))
                advs.append(a)
        zvf = sum(group_zv)/len(group_zv) if group_zv else 0.0; zvf_hist.append(zvf)
        if data:
            res = tc.forward_backward_custom(data=data, loss_fn=mkloss(advs)).result()
            tc.optim_step(T.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)).result()
            sc = tc.create_sampling_client(model_path=tc.save_weights_for_sampler(name=f"{name}_{step+1}").result().path)
        avg = sum(br)/len(br) if br else 0.0
        wandb.log({"zvf": zvf, "reward": avg, "tokens": tokens}, step=step)
    ho1 = held()
    out = dict(name=name, G=G, seed=seed, mean_zvf=sum(zvf_hist)/len(zvf_hist),
               heldout_before=ho0, heldout_after=ho1, heldout_gain=ho1-ho0, tokens=tokens)
    for k, v in out.items(): run.summary[k] = v
    wandb.finish(); print("DONE", json.dumps(out), flush=True)
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--groups", default="2,4,8,16")
    ap.add_argument("--heldout", type=int, default=16)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]; Gs = [int(g) for g in a.groups.split(",")]
    cfgs = [(f"G{G}-s{s}", G, s, a.steps, a.heldout, a.model, a.rank, a.lr, a.batch) for s in seeds for G in Gs]
    print(f"GROUPSIZE-ZVF: {len(cfgs)} runs (G x seed), {a.workers} parallel", flush=True)
    with mp.get_context("spawn").Pool(a.workers) as pool:
        results = pool.map(run_one, cfgs)
    os.makedirs("experiments/results/groupsize_zvf", exist_ok=True)
    json.dump(results, open("experiments/results/groupsize_zvf/results.json", "w"), indent=2)
    # correlation of mean_zvf with heldout_gain (the P2 diagnostic claim)
    import statistics as st
    zv=[r["mean_zvf"] for r in results if r]; hg=[r["heldout_gain"] for r in results if r]
    if len(zv)>2:
        mz,mh=st.mean(zv),st.mean(hg)
        cov=sum((a-mz)*(b-mh) for a,b in zip(zv,hg))/len(zv)
        sz=st.pstdev(zv)+1e-9; sh=st.pstdev(hg)+1e-9
        print(f"ZVF-vs-gain Pearson r = {cov/(sz*sh):.3f} (n={len(zv)})", flush=True)
    print("=== GROUPSIZE-ZVF COMPLETE ===", flush=True); print(json.dumps(results, indent=2))
