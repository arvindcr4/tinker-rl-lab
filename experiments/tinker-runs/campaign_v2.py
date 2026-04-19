#!/usr/bin/env python3
"""
BITTER LESSON CAMPAIGN v2: Correct Tinker SDK API usage.
Uses forward_backward_custom() + optim_step() pattern.
"""

import os, re, json, random, sys, time, warnings, traceback, torch
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

API_KEY = os.environ["TINKER_API_KEY"]
WANDB_KEY = os.environ["WANDB_API_KEY"]

os.environ["TINKER_API_KEY"] = API_KEY
os.environ["WANDB_API_KEY"] = WANDB_KEY

import tinker
import tinker.types as T
from transformers import AutoTokenizer
from datasets import load_dataset

# ── Load GSM8K once ──────────────────────────────────────────────────
print("Loading GSM8K dataset...", flush=True)
_ds = load_dataset("openai/gsm8k", "main", split="train")
GSM8K_EXAMPLES = []
for row in _ds:
    ans_match = re.search(r'####\s*([\-\d,\.]+)', row["answer"])
    if ans_match:
        answer = ans_match.group(1).replace(",", "").strip()
        GSM8K_EXAMPLES.append((row["question"], answer))
print(f"Loaded {len(GSM8K_EXAMPLES)} GSM8K examples", flush=True)

SYSTEM_PROMPT = "You are a math assistant. Solve the problem step by step, then give your final numerical answer inside \\boxed{}."
QUESTION_SUFFIX = " Provide a numerical answer without units, written inside \\boxed{}."


def reward_fn(response: str, answer: str) -> float:
    response = response.strip()
    boxed = re.findall(r'\\boxed\{([^}]+)\}', response)
    for b in boxed:
        b_clean = b.strip().replace(",", "").replace(" ", "")
        try:
            if abs(float(b_clean) - float(answer)) < 0.01:
                return 1.0
        except:
            if b_clean == answer:
                return 1.0
    all_nums = re.findall(r'[-+]?\d[\d,]*\.?\d*', response)
    if all_nums:
        last = all_nums[-1].replace(",", "")
        try:
            if abs(float(last) - float(answer)) < 0.01:
                return 1.0
        except:
            pass
    return 0.0


# ── Experiment Definitions ───────────────────────────────────────────
EXPERIMENTS = []

# WAVE 1: New Frontier Models
WAVE1 = [
    ("qwen35-397b", "Qwen/Qwen3.5-397B-A17B", "CRITICAL"),
    ("gpt-oss-120b", "openai/gpt-oss-120b", "HIGH"),
    ("llama33-70b-inst", "meta-llama/Llama-3.3-70B-Instruct", "HIGH"),
    ("llama31-70b-base", "meta-llama/Llama-3.1-70B", "HIGH"),
    ("kimi-k25", "moonshotai/Kimi-K2.5", "HIGH"),
    ("kimi-k2-thinking", "moonshotai/Kimi-K2-Thinking", "HIGH"),
    ("qwen35-35b-moe", "Qwen/Qwen3.5-35B-A3B", "MEDIUM"),
    ("qwen3-8b-base", "Qwen/Qwen3-8B-Base", "HIGH"),
    ("deepseek-v31-base", "deepseek-ai/DeepSeek-V3.1-Base", "HIGH"),
    ("nemotron3-nano-30b", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "MEDIUM"),
    ("llama32-1b", "meta-llama/Llama-3.2-1B", "MEDIUM"),
    ("llama32-3b", "meta-llama/Llama-3.2-3B", "MEDIUM"),
    ("llama31-8b-base", "meta-llama/Llama-3.1-8B", "HIGH"),
    ("qwen3-4b-inst-2507", "Qwen/Qwen3-4B-Instruct-2507", "MEDIUM"),
    ("qwen3-30b-inst-2507", "Qwen/Qwen3-30B-A3B-Instruct-2507", "MEDIUM"),
]
for tag, model, priority in WAVE1:
    EXPERIMENTS.append({"tag": f"w1_{tag}", "model": model, "wave": "1-frontier",
                        "priority": priority, "steps": 30, "group_size": 8, "lr": 1e-5, "seed": 42})

# WAVE 2: Group Size Sweep
for G in [2, 4, 16, 32]:
    EXPERIMENTS.append({"tag": f"w2_qwen3-8b_G{G}", "model": "Qwen/Qwen3-8B", "wave": "2-groupsize",
                        "priority": "HIGH", "steps": 30, "group_size": G, "lr": 1e-5, "seed": 42})

# WAVE 3: Multi-Seed Frontier
for tag, model in [("dsv31", "deepseek-ai/DeepSeek-V3.1"),
                    ("nem120b", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"),
                    ("q235b", "Qwen/Qwen3-235B-A22B-Instruct-2507")]:
    for seed in [123, 456, 789]:
        EXPERIMENTS.append({"tag": f"w3_{tag}_s{seed}", "model": model, "wave": "3-multiseed",
                            "priority": "HIGH", "steps": 30, "group_size": 8, "lr": 1e-5, "seed": seed})

# WAVE 4: LR Sweep
for lr in [5e-6, 3e-5, 1e-4]:
    EXPERIMENTS.append({"tag": f"w4_q8b_lr{lr}", "model": "Qwen/Qwen3-8B", "wave": "4-lr",
                        "priority": "MEDIUM", "steps": 30, "group_size": 8, "lr": lr, "seed": 42})

# WAVE 5: Long Training
EXPERIMENTS.append({"tag": "w5_q8b_100s", "model": "Qwen/Qwen3-8B", "wave": "5-long",
                    "priority": "HIGH", "steps": 100, "group_size": 8, "lr": 1e-5, "seed": 42})

print(f"\nTotal experiments: {len(EXPERIMENTS)}")
from collections import Counter
for w, c in sorted(Counter(e["wave"] for e in EXPERIMENTS).items()):
    print(f"  {w}: {c}")


def run_experiment(exp):
    """Run a single experiment using the correct Tinker SDK API."""
    tag = exp["tag"]
    model = exp["model"]
    steps = exp["steps"]
    group_size = exp["group_size"]
    lr = exp["lr"]
    seed = exp["seed"]

    print(f"  >> [{tag}] Starting: {model} G={group_size} LR={lr} steps={steps} seed={seed}", flush=True)

    try:
        random.seed(seed)
        torch.manual_seed(seed)

        # Load tokenizer
        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

        # W&B
        wb_run = None
        try:
            import wandb
            wandb.login(key=WANDB_KEY, relogin=True)
            wb_run = wandb.init(
                project="tinker-rl-lab-world-class",
                name=f"campaign_{tag}",
                config={"model": model, "method": "grpo", "task": "gsm8k", "seed": seed,
                        "group_size": group_size, "lr": lr, "steps": steps},
                reinit=True,
            )
        except Exception as e:
            print(f"  [{tag}] W&B init failed: {e}", flush=True)

        # Tinker client
        svc = tinker.ServiceClient()
        tc = svc.create_lora_training_client(base_model=model, rank=32, seed=seed)
        run_id = tc.model_id

        # Initial sampler
        w0 = tc.save_weights_for_sampler(name="s0").result()
        sc = tc.create_sampling_client(model_path=w0.path)

        _advs = []

        def loss_fn(data, lp):
            losses = [(-_advs[i] * lp[i].sum()) for i in range(len(lp))]
            loss = torch.stack(losses).mean()
            return loss, {"loss": loss.item()}

        step_rewards = []
        examples = list(GSM8K_EXAMPLES)
        random.shuffle(examples)

        for step in range(steps):
            batch = random.sample(examples, min(2, len(examples)))
            all_data, all_advs, batch_r = [], [], []

            for question, ans in batch:
                prompt = (f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                          f"<|im_start|>user\n{question + QUESTION_SUFFIX}<|im_end|>\n"
                          f"<|im_start|>assistant\n")
                pid = tok.encode(prompt, add_special_tokens=False)
                if len(pid) > 1024:
                    pid = pid[:1024]

                sp = T.SamplingParams(max_tokens=512, temperature=0.8, top_p=0.95)
                resp = sc.sample(T.ModelInput.from_ints(pid), num_samples=group_size, sampling_params=sp).result()

                rews = [reward_fn(tok.decode(list(r.tokens), skip_special_tokens=True), ans) for r in resp.sequences]
                mr = sum(rews) / len(rews)
                sr = (sum((r - mr)**2 for r in rews) / len(rews))**0.5 + 1e-8
                advs = [(r - mr) / sr for r in rews]
                batch_r.extend(rews)

                for r_seq, adv in zip(resp.sequences, advs):
                    rid = list(r_seq.tokens)
                    fid = pid + rid
                    tid = fid[1:] + [0]
                    all_data.append(T.Datum(
                        model_input=T.ModelInput.from_ints(fid),
                        loss_fn_inputs={"target_tokens": T.TensorData(data=tid, dtype="int64", shape=[len(tid)])},
                    ))
                    all_advs.append(adv)

            if not all_data:
                continue

            _advs.clear()
            _advs.extend(all_advs)
            fwdbwd = tc.forward_backward_custom(data=all_data, loss_fn=loss_fn).result()
            tc.optim_step(T.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)).result()

            avg = sum(batch_r) / len(batch_r) if batch_r else 0.0
            step_rewards.append(avg)

            if step % 5 == 0:
                print(f"  [{tag}] step {step+1}/{steps} reward={avg:.3f}", flush=True)

            if wb_run:
                try:
                    wb_run.log({"train/reward": avg, "train/step": step+1})
                except:
                    pass

            # Refresh sampler every 10 steps
            if (step + 1) % 10 == 0:
                ckpt = tc.save_weights_for_sampler(name=f"s{step+1}").result()
                sc = tc.create_sampling_client(model_path=ckpt.path)

        final_ckpt = tc.save_weights_for_sampler(name="final").result()

        peak = max(step_rewards) if step_rewards else 0
        last10 = sum(step_rewards[-10:]) / min(10, len(step_rewards)) if step_rewards else 0
        first5 = sum(step_rewards[:5]) / min(5, len(step_rewards)) if step_rewards else 0

        result = {
            "tag": tag, "model": model, "wave": exp["wave"], "status": "completed",
            "run_id": run_id, "seed": seed, "group_size": group_size, "lr": lr,
            "steps": steps, "peak_reward": peak, "last10_avg": last10, "first5_avg": first5,
            "reward_trace": step_rewards, "final_checkpoint": final_ckpt.path,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if wb_run:
            wb_run.summary.update({"peak_reward": peak, "last10_avg": last10})
            wb_run.finish()

        print(f"  ✓ [{tag}] DONE peak={peak:.3f} last10={last10:.3f}", flush=True)
        return result

    except Exception as e:
        print(f"  ✗ [{tag}] FAILED: {e}", flush=True)
        traceback.print_exc()
        try:
            if wb_run:
                wb_run.finish(exit_code=1)
        except:
            pass
        return {"tag": tag, "model": model, "wave": exp["wave"], "status": "failed",
                "error": str(e), "timestamp": datetime.utcnow().isoformat()}


def launch(max_parallel=6):
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    results_file = f"/home/user/workspace/tinker-rl-lab/experiments/tinker-runs/results/campaign_v2_{timestamp}.json"

    priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}
    sorted_exps = sorted(EXPERIMENTS, key=lambda e: priority_order.get(e.get("priority", "MEDIUM"), 2))

    print(f"\n{'='*70}")
    print(f"BITTER LESSON CAMPAIGN v2 — {len(EXPERIMENTS)} experiments")
    print(f"Max parallel: {max_parallel}")
    print(f"{'='*70}\n")

    results = []
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {executor.submit(run_experiment, exp): exp for exp in sorted_exps}
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                with open(results_file, "w") as f:
                    json.dump({"campaign": "bitter_lesson_v2", "timestamp": timestamp,
                               "total": len(EXPERIMENTS), "completed": len(results), "results": results}, f, indent=2)
                c = len([r for r in results if r.get("status") == "completed"])
                x = len([r for r in results if r.get("status") == "failed"])
                print(f"  Progress: {len(results)}/{len(EXPERIMENTS)} (✓{c} ✗{x})\n", flush=True)
            except Exception as e:
                print(f"  Future error: {e}", flush=True)

    c = len([r for r in results if r.get("status") == "completed"])
    x = len([r for r in results if r.get("status") == "failed"])
    print(f"\n{'='*70}")
    print(f"CAMPAIGN COMPLETE: ✓{c} ✗{x} / {len(EXPERIMENTS)}")
    print(f"{'='*70}")
    if c > 0:
        print("\nTop results:")
        for r in sorted([r for r in results if r.get("status")=="completed"],
                        key=lambda x: x.get("last10_avg",0), reverse=True)[:10]:
            print(f"  {r['tag']:35s} peak={r['peak_reward']:.3f} last10={r['last10_avg']:.3f}")
    print(f"\nSaved: {results_file}")


if __name__ == "__main__":
    if "--launch" in sys.argv:
        idx = sys.argv.index("--launch")
        mp = int(sys.argv[idx+1]) if len(sys.argv) > idx+1 else 6
        launch(max_parallel=mp)
    else:
        print("\nDry run. Use --launch [N] to start.")
