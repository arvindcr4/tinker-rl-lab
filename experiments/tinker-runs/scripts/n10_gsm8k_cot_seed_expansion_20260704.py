#!/usr/bin/env python3
"""
Experiment N10 — gsm8k_cot seed expansion (n=3 → 8) on the Tinker RL API.

This is a polite, low-concurrency pilot that replicates the GRPO vs Dr.GRPO
GSM8K-CoT protocol from the P4 analysis line (iters 128/132/136) on a
Tinker-supported <=8B model and expands the seed panel from 3 to 8 seeds.

Usage:
  python n10_gsm8k_cot_seed_expansion_20260704.py --smoke
  python n10_gsm8k_cot_seed_expansion_20260704.py --seeds 8 --steps 30
"""

import argparse
import json
import os
import random
import re
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import tinker
import tinker.types as T
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer

REPO = "/home/claude/tinker-rl-lab"
RESULTS_DIR = os.path.join(REPO, "experiments", "results", "n10_seed_expansion")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEFAULTS = {
    "model": "Qwen/Qwen3.5-4B",
    "rank": 16,
    "group": 8,
    "batch": 8,
    "steps": 30,
    "lr": 1e-5,
    "temp": 0.8,
    "top_p": 0.95,
    "max_tokens": 256,
    "max_prompt_tokens": 1024,
    "k_epochs": 2,
    "clip": 0.2,
    "n_eval": 128,
}

SYSTEM_PROMPT = "You are a math assistant. Solve the problem step by step, then give your final numerical answer inside \\boxed{}."


def load_gsm8k(seed: int):
    """Load GSM8K train/test and format prompts."""
    ds_train = load_dataset("openai/gsm8k", "main", split="train")
    ds_test = load_dataset("openai/gsm8k", "main", split="test")

    def fmt(split):
        out = []
        for row in split:
            m = re.search(r'####\s*([\-\d,\.]+)', row["answer"])
            if not m:
                continue
            ans = m.group(1).replace(",", "").strip()
            prompt = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{row['question']}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            out.append((prompt, ans))
        return out

    train = fmt(ds_train)
    test = fmt(ds_test)
    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def reward_fn(response: str, answer: str) -> float:
    response = response.strip()
    for b in re.findall(r'\\boxed\{([^}]+)\}', response):
        b_clean = b.strip().replace(",", "").replace(" ", "")
        try:
            if abs(float(b_clean) - float(answer)) < 0.01:
                return 1.0
        except Exception:
            if b_clean == answer:
                return 1.0
    nums = re.findall(r'[-+]?\d[\d,]*\.?\d*', response)
    if nums:
        last = nums[-1].replace(",", "")
        try:
            if abs(float(last) - float(answer)) < 0.01:
                return 1.0
        except Exception:
            pass
    return 0.0


def run_arm(algo: str, seed: int, args) -> dict:
    """Run one (algo, seed) cell on Tinker."""
    exp_tag = f"n10_{algo}_s{seed}"
    print(f"\n{'='*70}\n[{exp_tag}] START algo={algo} seed={seed}\n{'='*70}")

    random.seed(seed)
    np.random.seed(seed)

    train_examples, test_examples = load_gsm8k(seed)
    print(f"[{exp_tag}] train={len(train_examples)} test={len(test_examples)}")

    if args.no_wandb:
        wb = None
        wandb_run_path = "disabled"
    else:
        wb = wandb.init(
            project=args.wandb_project,
            name=f"{exp_tag}_20260704",
            config={
                "experiment": "N10",
                "algo": algo,
                "seed": seed,
                "model": args.model,
                "rank": args.rank,
                "group": args.group,
                "batch": args.batch,
                "steps": args.steps,
                "lr": args.lr,
                "max_tokens": args.max_tokens,
                "k_epochs": args.k_epochs,
                "clip": args.clip,
            },
            tags=["N10", "gsm8k_cot", "seed-expansion", algo, "20260704"],
            reinit=True,
        )
        wandb_run_path = f"{wb.entity}/{wb.project}/{wb.id}"
    print(f"[{exp_tag}] WANDB_RUN_PATH={wandb_run_path}", flush=True)

    svc = tinker.ServiceClient(base_url=None)
    tc = svc.create_lora_training_client(base_model=args.model, rank=args.rank)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Build tokenized train/test pools
    train_pool = []
    for prompt_text, ans in train_examples:
        pid = tok.encode(prompt_text, add_special_tokens=False)
        if len(pid) > args.max_prompt_tokens:
            pid = pid[:args.max_prompt_tokens]
        train_pool.append((pid, ans))

    test_pool = []
    for prompt_text, ans in test_examples[:args.n_eval]:
        pid = tok.encode(prompt_text, add_special_tokens=False)
        if len(pid) > args.max_prompt_tokens:
            pid = pid[:args.max_prompt_tokens]
        test_pool.append((pid, ans))

    w0 = tc.save_weights_for_sampler(name=f"{exp_tag}_s0").result()
    sc = tc.create_sampling_client(model_path=w0.path)

    def sample_responses(sc, pid, num_samples):
        sp = T.SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temp,
            top_p=args.top_p,
        )
        try:
            return sc.sample(T.ModelInput.from_ints(pid), num_samples=num_samples, sampling_params=sp).result()
        except Exception as e:
            print(f"[{exp_tag}] sample retry after: {e}")
            return sc.sample(T.ModelInput.from_ints(pid), num_samples=num_samples, sampling_params=sp).result()

    step_log = []
    t0 = time.time()
    train_iter = iter(train_pool)

    for step in range(args.steps):
        batch = []
        for _ in range(args.batch):
            try:
                batch.append(next(train_iter))
            except StopIteration:
                random.seed(seed + step)
                random.shuffle(train_pool)
                train_iter = iter(train_pool)
                batch.append(next(train_iter))

        all_data, batch_rewards = [], []
        group_zvf_count = 0
        all_lengths = []

        for pid, ans in batch:
            resp = sample_responses(sc, pid, args.group)
            rews = [reward_fn(tok.decode(list(r.tokens), skip_special_tokens=True), ans) for r in resp.sequences]
            lens = [len(list(r.tokens)) for r in resp.sequences]
            all_lengths.extend(lens)

            mr = sum(rews) / len(rews)
            sr = (sum((r - mr) ** 2 for r in rews) / len(rews)) ** 0.5 + 1e-8

            if algo == "dr_grpo":
                advs = [r - mr for r in rews]
            else:
                advs = [(r - mr) / sr for r in rews]

            if sr <= 1e-6:
                group_zvf_count += 1

            batch_rewards.extend(rews)

            for r, a in zip(resp.sequences, advs):
                if a == 0:
                    continue
                rid = list(r.tokens)
                full = pid + rid
                # Effective advantage implements per-response normalization:
                #   GRPO    : average over response tokens (constant adv -> -mean(adv * logp))
                #   Dr.GRPO : divide by MAX_NEW (adv * len(rid)/MAX_NEW -> -sum(adv*logp)/MAX_NEW)
                if algo == "dr_grpo":
                    a_eff = a * len(rid) / args.max_tokens
                else:
                    a_eff = a
                all_data.append(T.Datum(
                    model_input=T.ModelInput.from_ints(full[:-1]),
                    loss_fn_inputs={
                        "target_tokens": T.TensorData(data=full[1:], dtype="int64", shape=[len(full) - 1]),
                        "logprobs": T.TensorData(
                            data=[0.0] * (len(pid) - 1) + list(r.logprobs),
                            dtype="float32",
                            shape=[len(full) - 1],
                        ),
                        "advantages": T.TensorData(
                            data=[0.0] * (len(pid) - 1) + [a_eff] * len(rid),
                            dtype="float32",
                            shape=[len(full) - 1],
                        ),
                    }
                ))

        if not all_data:
            print(f"[{exp_tag}] no data at step {step}; skipping")
            continue

        loss_val = None
        for ep in range(args.k_epochs):
            try:
                result = tc.forward_backward(data=all_data, loss_fn="importance_sampling").result()
                tc.optim_step(T.AdamParams(learning_rate=args.lr, beta1=0.9, beta2=0.95, eps=1e-8)).result()
                metrics = result.metrics
                if hasattr(metrics, "get"):
                    loss_val = metrics.get("loss", None)
                    if loss_val is None:
                        for k, v in metrics.items():
                            if "loss" in str(k).lower():
                                loss_val = v
                                break
                elif isinstance(metrics, dict):
                    loss_val = metrics.get("loss", None)
                if loss_val is not None and not isinstance(loss_val, (int, float)):
                    loss_val = float(loss_val)
            except Exception as e:
                print(f"[{exp_tag}] train step {step} epoch {ep} failed: {e}")
                traceback.print_exc()
                if ep == 0:
                    break

        avg_reward = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0.0
        zvf = group_zvf_count / len(batch) if batch else 0.0
        mean_len = sum(all_lengths) / len(all_lengths) if all_lengths else 0.0

        step_log.append({
            "step": step + 1,
            "loss": loss_val,
            "reward": avg_reward,
            "zvf": zvf,
            "mean_len": mean_len,
        })

        if wb is not None:
            wb.log({
                "train/reward": avg_reward,
                "train/zvf": zvf,
                "train/mean_len": mean_len,
                "train/loss": loss_val,
            }, step=step + 1)

        print(f"[{exp_tag}] {step+1:3d}/{args.steps} | reward={avg_reward:.3f} | zvf={zvf:.3f} | len={mean_len:.1f}")

        if (step + 1) % max(args.steps // 3, 5) == 0:
            try:
                ckpt = tc.save_weights_for_sampler(name=f"{exp_tag}_s{step+1}").result()
                sc = tc.create_sampling_client(model_path=ckpt.path)
            except Exception as e:
                print(f"[{exp_tag}] checkpoint refresh failed: {e}")

    # Final held-out evaluation (greedy)
    def heldout_eval():
        try:
            final_ckpt = tc.save_weights_for_sampler(name=f"{exp_tag}_final").result()
            final_sc = tc.create_sampling_client(model_path=final_ckpt.path)
        except Exception:
            final_sc = sc
        sp_greedy = T.SamplingParams(max_tokens=args.max_tokens, temperature=0.0, top_p=1.0)
        correct = []
        for pid, ans in test_pool:
            try:
                resp = final_sc.sample(T.ModelInput.from_ints(pid), num_samples=1, sampling_params=sp_greedy).result()
                text = tok.decode(list(resp.sequences[0].tokens), skip_special_tokens=True)
                correct.append(1.0 if reward_fn(text, ans) == 1.0 else 0.0)
            except Exception as e:
                print(f"[{exp_tag}] eval sample failed: {e}")
                correct.append(0.0)
        return sum(correct) / len(correct) if correct else 0.0

    heldout_acc = heldout_eval()
    print(f"[{exp_tag}] heldout_acc@{args.n_eval} = {heldout_acc:.3f}")

    last10 = [s["reward"] for s in step_log[-10:]]
    first5 = [s["reward"] for s in step_log[:5]]
    summary = {
        "experiment": "N10",
        "algo": algo,
        "seed": seed,
        "model": args.model,
        "rank": args.rank,
        "group": args.group,
        "batch": args.batch,
        "steps": len(step_log),
        "lr": args.lr,
        "max_tokens": args.max_tokens,
        "heldout_acc": heldout_acc,
        "n_eval": len(test_pool),
        "first5_avg_reward": sum(first5) / len(first5) if first5 else 0.0,
        "last10_avg_reward": sum(last10) / len(last10) if last10 else 0.0,
        "mean_zvf": sum(s["zvf"] for s in step_log) / len(step_log) if step_log else 0.0,
        "mean_len_first5": sum(s["mean_len"] for s in step_log[:5]) / min(5, len(step_log)) if step_log else 0.0,
        "mean_len_last5": sum(s["mean_len"] for s in step_log[-5:]) / min(5, len(step_log)) if step_log else 0.0,
        "elapsed_seconds": time.time() - t0,
        "run_id": tc.model_id,
        "wandb_run_path": wandb_run_path,
        "step_log": step_log,
        "timestamp": datetime.now().isoformat(),
    }

    out_path = os.path.join(RESULTS_DIR, f"{exp_tag}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    if wb is not None:
        wb.summary.update({
            "final/heldout_acc": heldout_acc,
            "final/first5_avg_reward": summary["first5_avg_reward"],
            "final/last10_avg_reward": summary["last10_avg_reward"],
            "final/mean_zvf": summary["mean_zvf"],
            "final/mean_len_first5": summary["mean_len_first5"],
            "final/mean_len_last5": summary["mean_len_last5"],
        })
        wb.finish()

    print(f"[{exp_tag}] DONE last10={summary['last10_avg_reward']:.3f} heldout={heldout_acc:.3f} -> {out_path}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULTS["model"])
    parser.add_argument("--rank", type=int, default=DEFAULTS["rank"])
    parser.add_argument("--group", type=int, default=DEFAULTS["group"])
    parser.add_argument("--batch", type=int, default=DEFAULTS["batch"])
    parser.add_argument("--steps", type=int, default=DEFAULTS["steps"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--temp", type=float, default=DEFAULTS["temp"])
    parser.add_argument("--top-p", type=float, default=DEFAULTS["top_p"])
    parser.add_argument("--max-tokens", type=int, default=DEFAULTS["max_tokens"])
    parser.add_argument("--max-prompt-tokens", type=int, default=DEFAULTS["max_prompt_tokens"])
    parser.add_argument("--k-epochs", type=int, default=DEFAULTS["k_epochs"])
    parser.add_argument("--clip", type=float, default=DEFAULTS["clip"])
    parser.add_argument("--n-eval", type=int, default=DEFAULTS["n_eval"])
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--algos", default="grpo,dr_grpo")
    parser.add_argument("--wandb-project", default="tinker-new-research")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.seeds = 2
        args.steps = 3
        args.batch = 2
        args.n_eval = 8
        print("SMOKE MODE: seeds=2 steps=3 batch=2 n_eval=8")

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]

    # Fixed seed schedule expanding the original [42, 123, 456] panel
    seed_schedule = [args.seed_base + i * 137 for i in range(args.seeds)]

    manifest = {
        "experiment": "N10",
        "model": args.model,
        "seeds": seed_schedule,
        "algos": algos,
        "started_at": datetime.now().isoformat(),
        "runs": [],
    }

    for algo in algos:
        for seed in seed_schedule:
            try:
                summary = run_arm(algo, seed, args)
                manifest["runs"].append({
                    "algo": algo,
                    "seed": seed,
                    "status": "ok",
                    "last10_avg_reward": summary["last10_avg_reward"],
                    "heldout_acc": summary["heldout_acc"],
                    "wandb_run_path": summary["wandb_run_path"],
                })
            except Exception as e:
                print(f"[n10:{algo}:s{seed}] FAILED: {e}")
                traceback.print_exc()
                manifest["runs"].append({
                    "algo": algo,
                    "seed": seed,
                    "status": "failed",
                    "error": str(e),
                })

    manifest["finished_at"] = datetime.now().isoformat()
    manifest_path = os.path.join(RESULTS_DIR, "n10_manifest_20260704.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n{'='*70}\nN10 manifest written to {manifest_path}\n{'='*70}")


if __name__ == "__main__":
    main()
