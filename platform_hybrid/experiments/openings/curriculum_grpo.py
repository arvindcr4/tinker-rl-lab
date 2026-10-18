"""P2/P3 opening experiment — Difficulty-Curriculum GRPO vs baseline.

Claim under test: baseline GRPO wastes most gradient steps on collapsed (all-correct/
all-wrong) groups; a curriculum that only keeps MIXED-variance groups eliminates the
waste and learns more per gradient update. Both arms: same base model, same #optim
steps, same seed, per-step on-policy sampler refresh, held-out eval before/after.
Logs both arms to W&B (rlvr-openings).
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "platform_hybrid").is_dir())))
from platform_hybrid.experiments.research_contracts import completion_alignment, gsm8k_reward

import os, re, json, random, argparse, warnings
warnings.filterwarnings("ignore")
assert os.environ.get("TINKER_API_KEY"), "need TINKER_API_KEY"
import torch, tinker, tinker.types as T
from transformers import AutoTokenizer
from datasets import load_dataset
import wandb

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
ap.add_argument("--steps", type=int, default=8)
ap.add_argument("--group", type=int, default=4)
ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--rank", type=int, default=8)
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--oversample_cap", type=int, default=8)
ap.add_argument("--heldout", type=int, default=20)
args = ap.parse_args()
random.seed(args.seed); torch.manual_seed(args.seed)

SYS = "You are a math assistant. Solve step by step, then give the final numerical answer inside \\boxed{}."
ds = load_dataset("openai/gsm8k", "main", split="train")
examples = []
for row in ds:
    m = re.search(r"####\s*([\-\d,\.]+)", row["answer"])
    if not m: continue
    ans = m.group(1).replace(",", "").strip()
    prompt = (f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{row['question']}<|im_end|>\n<|im_start|>assistant\n")
    examples.append((prompt, ans))
random.shuffle(examples)
heldout = examples[:args.heldout]; train_pool = examples[args.heldout:]
print(f"train_pool={len(train_pool)} heldout={len(heldout)} model={args.model}")

def reward(response, answer):
    return gsm8k_reward(response, answer)

_adv = []
def loss_fn(data, lp):
    losses = [(-_adv[i] * (lp[i] * torch.tensor(data[i].loss_fn_inputs["completion_mask"].data, device=lp[i].device)).sum()) for i in range(len(lp))]
    loss = torch.stack(losses).mean()
    return loss, {"loss": loss.item()}

svc = tinker.ServiceClient(base_url=None)
tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

def sample_group(sc, prompt_text, ans, g):
    pid = tok.encode(prompt_text, add_special_tokens=False)[:1024]
    sp = T.SamplingParams(max_tokens=512, temperature=0.8, top_p=0.95)
    resp = sc.sample(T.ModelInput.from_ints(pid), num_samples=g, sampling_params=sp).result()
    rews = [reward(tok.decode(list(r.tokens), skip_special_tokens=True), ans) for r in resp.sequences]
    return pid, resp, rews

def heldout_acc(sc):
    correct = 0
    for prompt_text, ans in heldout:
        pid = tok.encode(prompt_text, add_special_tokens=False)[:1024]
        sp = T.SamplingParams(max_tokens=512, temperature=0.0, top_p=1.0)
        resp = sc.sample(T.ModelInput.from_ints(pid), num_samples=1, sampling_params=sp).result()
        correct += reward(tok.decode(list(resp.sequences[0].tokens), skip_special_tokens=True), ans)
    return correct / len(heldout)

def run_arm(mode):
    global _adv
    tc = svc.create_lora_training_client(base_model=args.model, rank=args.rank)
    w0 = tc.save_weights_for_sampler(name=f"{mode}_s0").result()
    sc = tc.create_sampling_client(model_path=w0.path)
    ho_before = heldout_acc(sc)
    zero_loss = 0; sampled_total = 0; accepted_total = 0; step_rewards = []
    for step in range(args.steps):
        all_data, all_advs, batch_r = [], [], []
        slots = 0; attempts = 0
        while slots < args.batch and attempts < args.batch * args.oversample_cap:
            prompt_text, ans = random.choice(train_pool); attempts += 1; sampled_total += 1
            pid, resp, rews = sample_group(sc, prompt_text, ans, args.group)
            batch_r.extend(rews)
            mr = sum(rews)/len(rews)
            mixed = 0.0 < mr < 1.0
            if mode == "curriculum" and not mixed:
                continue  # skip collapsed group, resample
            slots += 1; accepted_total += 1
            sr = (sum((r-mr)**2 for r in rews)/len(rews))**0.5 + 1e-8
            advs = [(r-mr)/sr for r in rews]
            for rseq, a in zip(resp.sequences, advs):
                fid, tid, cmask = completion_alignment(pid, list(rseq.tokens))
                all_data.append(T.Datum(model_input=T.ModelInput.from_ints(fid),
                    loss_fn_inputs={"target_tokens": T.TensorData(data=tid, dtype="int64", shape=[len(tid)]), "completion_mask": T.TensorData(data=cmask, dtype="float32", shape=[len(cmask)])}))
                all_advs.append(a)
        if not all_data:
            zero_loss += 1
            step_rewards.append(sum(batch_r)/len(batch_r) if batch_r else None); continue
        _adv = all_advs
        res = tc.forward_backward_custom(data=all_data, loss_fn=loss_fn).result()
        tc.optim_step(T.AdamParams(learning_rate=args.lr, beta1=0.9, beta2=0.95, eps=1e-8)).result()
        lv = res.metrics.get("loss", 0.0)
        if abs(lv) < 1e-6: zero_loss += 1
        avg = sum(batch_r)/len(batch_r) if batch_r else 0.0
        step_rewards.append(avg)
        # per-step on-policy refresh
        ck = tc.save_weights_for_sampler(name=f"{mode}_s{step+1}").result()
        sc = tc.create_sampling_client(model_path=ck.path)
        print(f"[{mode}] {step+1}/{args.steps} loss={lv:.3f} reward={avg:.3f} sampled={sampled_total} accepted={accepted_total}")
    ho_after = heldout_acc(sc)
    return dict(mode=mode, near_zero_scalar_loss_fraction=zero_loss/args.steps,
                oversample_factor=sampled_total/max(accepted_total,1),
                heldout_before=ho_before, heldout_after=ho_after,
                heldout_gain=ho_after-ho_before, step_rewards=step_rewards)

run = wandb.init(project="rlvr-openings", name=f"curriculum-vs-baseline-{args.model.split('/')[-1]}-s{args.seed}",
                 config=vars(args), tags=["curriculum","P2","P3","tinker"])
results = {}
for mode in ["baseline", "curriculum"]:
    r = run_arm(mode); results[mode] = r
    for k in ["near_zero_scalar_loss_fraction","oversample_factor","heldout_before","heldout_after","heldout_gain"]:
        run.summary[f"{mode}/{k}"] = r[k]
    for s, rw in enumerate(r["step_rewards"]):
        wandb.log({f"{mode}/reward": rw}, step=s)
    print(f"=== {mode}: {json.dumps({k:v for k,v in r.items() if k!='step_rewards'})}")
os.makedirs("experiments/results/curriculum_opening", exist_ok=True)
json.dump(results, open("experiments/results/curriculum_opening/results.json","w"), indent=2)
run.summary["heldout_gain_delta(curr-base)"] = results["curriculum"]["heldout_gain"] - results["baseline"]["heldout_gain"]
wandb.finish()
print("DONE. W&B:", run.url)
