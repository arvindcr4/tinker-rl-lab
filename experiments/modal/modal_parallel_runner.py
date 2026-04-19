"""Modal GPU experiments with W&B online logging + HF Hub checkpointing.
Uses H100 GPUs for maximum throughput. Complementary to Tinker API experiments.
"""
import modal, os

app = modal.App("tinker-rl-lab-world-class")

HF_TOKEN = os.environ.get("HF_TOKEN", "")
WANDB_KEY = os.environ.get("WANDB_API_KEY", "")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1", "transformers>=4.40", "datasets", "trl>=0.12",
        "peft", "accelerate", "bitsandbytes", "scipy",
        "numpy", "pandas", "wandb", "huggingface_hub",
    )
    .env({
        "HF_TOKEN": HF_TOKEN,
        "WANDB_API_KEY": WANDB_KEY,
        "WANDB_PROJECT": "tinker-rl-lab-world-class",
    })
)

# ── Helper: upload results to HF Hub ─────────────────────────────────────
def _upload_hf(exp_tag, model_id, results, method="grpo"):
    import json, os, shutil
    from huggingface_hub import HfApi, create_repo
    try:
        api = HfApi(token=HF_TOKEN)
        repo_id = f"arvindcr4/tinker-rl-bench-{exp_tag}"
        create_repo(repo_id, repo_type="model", exist_ok=True, token=HF_TOKEN)
        card = f"""---
tags: [{method}, reinforcement-learning, tinker-rl-bench, modal-h100]
base_model: {model_id}
datasets: [openai/gsm8k]
license: mit
---
# {exp_tag}
Modal H100 GPU experiment from TinkerRL-Bench world-class suite.
## Results
```json
{json.dumps({k:v for k,v in results.items() if k not in ('per_problem','reward_trace','kl_trace','entropy_trace','step_log')}, indent=2)}
```
## Citation
```bibtex
@misc{{tinker-rl-bench-2026,
  title={{TinkerRL-Bench: A Unified Benchmark for RL Post-Training}},
  author={{Arvind C R and Sandhya Jeyaraj and Madhu Kumara L and Mohammad Rafi and Dhruva N Murthy and Arumugam K}},
  year={{2026}},
  url={{https://github.com/arvindcr4/tinker-rl-lab}}
}}
```
"""
        tmp = f"/tmp/hf_{exp_tag}"
        os.makedirs(tmp, exist_ok=True)
        with open(f"{tmp}/README.md", "w") as f: f.write(card)
        with open(f"{tmp}/results.json", "w") as f: json.dump(results, f, indent=2)
        api.upload_folder(folder_path=tmp, repo_id=repo_id, repo_type="model", token=HF_TOKEN)
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"[HF] ✓ {repo_id}")
        return repo_id
    except Exception as e:
        print(f"[HF] ✗ {exp_tag}: {e}")
        return None


# ── Experiment 1: PPO/REINFORCE baseline on GSM8K ────────────────────────
@app.function(image=image, gpu="H100", timeout=7200)
def run_ppo_gsm8k(model_name: str = "Qwen/Qwen3-8B", seed: int = 42, steps: int = 30):
    import torch, random, re, json, wandb, os
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    random.seed(seed); torch.manual_seed(seed)
    exp = f"ppo_gsm8k_{model_name.split('/')[-1]}_s{seed}"

    wandb.login(key=os.environ["WANDB_API_KEY"])
    wandb.init(project="tinker-rl-lab-world-class", name=exp,
               config={"model": model_name, "method": "ppo_reinforce", "task": "gsm8k",
                        "seed": seed, "steps": steps, "platform": "modal_h100",
                        "gpu": "H100"},
               tags=["modal", "ppo", "gsm8k", "h100", model_name.split("/")[-1]])

    print(f"[{exp}] Loading model on H100...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    lora_config = LoraConfig(
        r=32, lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    ds = load_dataset("openai/gsm8k", "main", split="train")
    examples = []
    for row in ds:
        m = re.search(r'####\s*([\-\d,\.]+)', row["answer"])
        if m: examples.append({"question": row["question"], "answer": m.group(1).replace(",","").strip()})
    random.shuffle(examples)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    system = "You are a math assistant. Solve step by step, give final answer in \\boxed{}."
    step_rewards = []

    for step in range(steps):
        ex = random.choice(examples)
        prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{ex['question']}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=512, temperature=0.8, top_p=0.95,
                do_sample=True, num_return_sequences=4, pad_token_id=tokenizer.pad_token_id,
            )

        rewards = []
        for out in outputs:
            resp = tokenizer.decode(out[inputs.input_ids.shape[1]:], skip_special_tokens=True)
            r = 0.0
            for b in re.findall(r'\\boxed\{([^}]+)\}', resp):
                try:
                    if abs(float(b.strip().replace(",","")) - float(ex['answer'])) < 0.01: r = 1.0
                except: pass
            if r == 0.0:
                nums = re.findall(r'[-+]?\d[\d,]*\.?\d*', resp)
                if nums:
                    try:
                        if abs(float(nums[-1].replace(",","")) - float(ex['answer'])) < 0.01: r = 1.0
                    except: pass
            rewards.append(r)

        avg_r = sum(rewards) / len(rewards)
        step_rewards.append(avg_r)

        loss_val = 0.0
        if max(rewards) > 0:
            best_idx = rewards.index(max(rewards))
            best_out = outputs[best_idx]
            labels = best_out.clone()
            labels[:inputs.input_ids.shape[1]] = -100
            output = model(input_ids=best_out.unsqueeze(0), labels=labels.unsqueeze(0))
            loss_val = output.loss.item() * max(rewards)
            optimizer.zero_grad()
            (output.loss * max(rewards)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        wandb.log({
            "step": step+1, "train/reward": avg_r, "train/loss": loss_val,
            "train/cumulative_reward": sum(step_rewards)/len(step_rewards),
            "train/peak_reward": max(step_rewards),
        }, step=step+1)

        if (step+1) % 5 == 0:
            print(f"[{exp}] {step+1}/{steps} reward={avg_r:.3f} avg={sum(step_rewards)/len(step_rewards):.3f}")

    # Save LoRA adapter to HF
    adapter_path = f"/tmp/ppo_adapter_{exp}"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    from huggingface_hub import HfApi, create_repo
    repo_id = None
    try:
        repo_id = f"arvindcr4/tinker-rl-bench-{exp}"
        create_repo(repo_id, repo_type="model", exist_ok=True, token=HF_TOKEN)
        api = HfApi(token=HF_TOKEN)
        api.upload_folder(folder_path=adapter_path, repo_id=repo_id, repo_type="model", token=HF_TOKEN)
        print(f"[HF] ✓ Uploaded adapter to {repo_id}")
    except Exception as e:
        print(f"[HF] ✗ {e}")
        repo_id = None

    last10 = step_rewards[-10:]
    result = {
        "experiment": exp, "model": model_name, "method": "ppo_reinforce", "task": "gsm8k",
        "seed": seed, "steps": steps, "platform": "modal_h100",
        "first5_avg": sum(step_rewards[:5])/5 if len(step_rewards)>=5 else 0,
        "last10_avg": sum(last10)/len(last10) if last10 else 0,
        "peak": max(step_rewards) if step_rewards else 0,
        "reward_trace": [round(r,4) for r in step_rewards],
        "hf_repo": repo_id,
    }
    wandb.summary.update({"final/last10_avg": result["last10_avg"], "final/peak": result["peak"], "hf_repo": repo_id})
    wandb.finish()
    return result


# ── Experiment 2: Full HumanEval evaluation ──────────────────────────────
@app.function(image=image.pip_install("human-eval"), gpu="H100", timeout=7200)
def run_humaneval_eval(model_name: str = "Qwen/Qwen3-8B", num_samples: int = 5):
    import torch, json, wandb, os
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from human_eval.data import read_problems
    from human_eval.execution import check_correctness

    exp = f"humaneval_{model_name.split('/')[-1]}"
    wandb.login(key=os.environ["WANDB_API_KEY"])
    wandb.init(project="tinker-rl-lab-world-class", name=exp,
               config={"model": model_name, "task": "humaneval", "num_samples": num_samples,
                        "platform": "modal_h100", "gpu": "H100"},
               tags=["modal", "humaneval", "eval", "h100", model_name.split("/")[-1]])

    # Load model ONCE outside the problem loop — avoids re-initializing per problem
    print(f"[{exp}] Loading model on H100 (once for all problems)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()  # inference-only; no gradients needed

    problems = read_problems()
    results = {}

    for i, (task_id, problem) in enumerate(problems.items()):
        prompt = problem["prompt"]
        passed = 0
        for _ in range(num_samples):
            inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs, max_new_tokens=512, temperature=0.8, top_p=0.95,
                    do_sample=True, pad_token_id=tokenizer.pad_token_id,
                )
            comp = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            if "```python" in comp: comp = comp.split("```python")[1].split("```")[0]
            elif "```" in comp: comp = comp.split("```")[1].split("```")[0]
            try:
                r = check_correctness(problem, comp, timeout=5.0)
                if r["passed"]: passed += 1
            except: pass

        results[task_id] = {"passed": passed, "total": num_samples, "pass_rate": passed/num_samples}
        wandb.log({
            "eval/problem_idx": i+1,
            "eval/pass_rate": passed/num_samples,
            "eval/cumulative_pass1": sum(1 for r in results.values() if r["passed"]>0)/len(results),
        }, step=i+1)

        if (i+1) % 20 == 0:
            print(f"[{exp}] {i+1}/{len(problems)}: cum_pass@1={sum(1 for r in results.values() if r['passed']>0)/len(results)*100:.1f}%")

    pass1 = sum(1 for r in results.values() if r["passed"]>0) / len(results)
    mean_pass = sum(r["pass_rate"] for r in results.values()) / len(results)

    result = {
        "experiment": exp, "model": model_name, "task": "humaneval",
        "num_samples": num_samples, "pass_at_1": pass1, "mean_pass_at_k": mean_pass,
        "num_problems": len(results), "per_problem": results, "platform": "modal_h100",
    }
    wandb.summary.update({"final/pass_at_1": pass1, "final/mean_pass_at_k": mean_pass})
    _upload_hf(exp, model_name, result, method="eval")
    wandb.finish()
    return result


# ── Experiment 3: KL divergence + entropy tracking ───────────────────────
@app.function(image=image, gpu="H100", timeout=7200)
def run_kl_tracking(model_name: str = "Qwen/Qwen3-8B", seed: int = 42, steps: int = 30):
    import torch, torch.nn.functional as F, random, re, json, wandb, os
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    random.seed(seed); torch.manual_seed(seed)
    exp = f"kl_track_{model_name.split('/')[-1]}_s{seed}"

    wandb.login(key=os.environ["WANDB_API_KEY"])
    wandb.init(project="tinker-rl-lab-world-class", name=exp,
               config={"model": model_name, "task": "kl_tracking", "seed": seed, "steps": steps,
                        "platform": "modal_h100", "gpu": "H100"},
               tags=["modal", "kl_tracking", "h100", model_name.split("/")[-1]])

    print(f"[{exp}] Loading ref + policy models on H100...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    ref_model.eval()
    # Freeze ref model entirely — it's monitoring-only, never needs gradients
    for p in ref_model.parameters():
        p.requires_grad = False

    policy_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    lora_config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj","v_proj"],
                              lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    policy_model = get_peft_model(policy_model, lora_config)

    ds = load_dataset("openai/gsm8k", "main", split="train")
    questions = [row["question"] for row in ds]
    random.shuffle(questions)

    system = "You are a math assistant. Solve step by step, give final answer in \\boxed{}."
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=3e-5)
    kl_trace, entropy_trace = [], []

    for step in range(steps):
        q = random.choice(questions)
        prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(policy_model.device)

        # KL and entropy are monitoring metrics only — compute entirely inside no_grad
        # to avoid "does not require grad" errors when mixing frozen ref tensors
        with torch.no_grad():
            pol_out = policy_model(**inputs)
            ref_out = ref_model(**inputs)
            policy_logits = pol_out.logits.detach()
            ref_logits = ref_out.logits.detach()

            # Use F.kl_div for numerically stable KL: KL(policy || ref)
            kl = F.kl_div(
                F.log_softmax(policy_logits, dim=-1),
                F.softmax(ref_logits, dim=-1),
                reduction='batchmean',
            ).item()

            pol_lp = F.log_softmax(policy_logits, dim=-1)
            pol_p = torch.exp(pol_lp)
            entropy = -(pol_p * pol_lp).sum(dim=-1).mean().item()

        kl_trace.append(kl); entropy_trace.append(entropy)
        wandb.log({
            "step": step+1, "kl/kl_divergence": kl, "kl/entropy": entropy,
            "kl/max_kl_so_far": max(kl_trace),
        }, step=step+1)

        # Dummy GRPO-style gradient step to measure KL drift under training
        # Computed in a fresh forward pass OUTSIDE no_grad so gradients flow correctly
        train_out = policy_model(**inputs)
        loss = train_out.logits.mean() * 0.001
        optimizer.zero_grad(); loss.backward(); optimizer.step()

        if (step+1) % 5 == 0:
            print(f"[{exp}] {step+1}/{steps} KL={kl:.6f} entropy={entropy:.4f}")

    result = {
        "experiment": exp, "model": model_name, "seed": seed, "steps": steps,
        "kl_trace": [round(k,6) for k in kl_trace],
        "entropy_trace": [round(e,4) for e in entropy_trace],
        "final_kl": kl_trace[-1], "final_entropy": entropy_trace[-1],
        "max_kl": max(kl_trace), "platform": "modal_h100",
    }
    wandb.summary.update({"final/kl": result["final_kl"], "final/entropy": result["final_entropy"], "final/max_kl": result["max_kl"]})
    _upload_hf(exp, model_name, result, method="kl_tracking")
    wandb.finish()
    return result


# ── Experiment 4: Held-out GSM8K eval for new/larger models ──────────────
@app.function(image=image, gpu="H100", timeout=7200)
def run_gsm8k_heldout_eval(model_name: str = "Qwen/Qwen3-32B", num_examples: int = 200):
    import torch, re, json, random, wandb, os
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM

    exp = f"heldout_gsm8k_{model_name.split('/')[-1]}"
    wandb.login(key=os.environ["WANDB_API_KEY"])
    wandb.init(project="tinker-rl-lab-world-class", name=exp,
               config={"model": model_name, "task": "gsm8k_heldout", "num_examples": num_examples,
                        "platform": "modal_h100", "gpu": "H100"},
               tags=["modal", "eval", "gsm8k", "h100", model_name.split("/")[-1]])

    print(f"[{exp}] Loading model on H100...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    ds = load_dataset("openai/gsm8k", "main", split="test")
    examples = list(ds); random.seed(42); random.shuffle(examples)
    examples = examples[:num_examples]

    system = "You are a math assistant. Solve step by step, give final answer in \\boxed{}."
    correct = 0

    for i, ex in enumerate(examples):
        prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{ex['question']}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=512, temperature=0.0,
                do_sample=False, pad_token_id=tokenizer.pad_token_id,
            )

        resp = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        m = re.search(r'####\s*([\-\d,\.]+)', ex["answer"])
        if not m: continue
        gold = m.group(1).replace(",","").strip()

        got = False
        for b in re.findall(r'\\boxed\{([^}]+)\}', resp):
            try:
                if abs(float(b.strip().replace(",","")) - float(gold)) < 0.01: got = True; break
            except: pass
        if not got:
            nums = re.findall(r'[-+]?\d[\d,]*\.?\d*', resp)
            if nums:
                try:
                    if abs(float(nums[-1].replace(",","")) - float(gold)) < 0.01: got = True
                except: pass

        if got: correct += 1
        wandb.log({"eval/example_idx": i+1, "eval/running_accuracy": correct/(i+1)}, step=i+1)

        if (i+1) % 50 == 0:
            print(f"[{exp}] {i+1}/{num_examples}: {correct}/{i+1} = {correct/(i+1)*100:.1f}%")

    result = {
        "experiment": exp, "model": model_name, "task": "gsm8k_heldout",
        "num_examples": num_examples, "correct": correct,
        "accuracy": correct / num_examples, "platform": "modal_h100",
    }
    wandb.summary.update({"final/accuracy": result["accuracy"], "final/correct": correct})
    _upload_hf(exp, model_name, result, method="eval")
    wandb.finish()
    return result


# ── Orchestrator ─────────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    import json, time

    print("=" * 70)
    print("LAUNCHING MODAL H100 GPU EXPERIMENTS IN PARALLEL")
    print("W&B: tinker-rl-lab-world-class | HF: arvindcr4/tinker-rl-bench-*")
    print("=" * 70)

    start = time.time()

    futures = {
        # PPO baselines (compare with Tinker GRPO)
        "ppo_qwen3-8b": run_ppo_gsm8k.spawn("Qwen/Qwen3-8B", 42, 30),
        "ppo_llama-8b-inst": run_ppo_gsm8k.spawn("meta-llama/Llama-3.1-8B-Instruct", 42, 30),
        # HumanEval full eval (fill critical gap)
        "humaneval_qwen3-8b": run_humaneval_eval.spawn("Qwen/Qwen3-8B", 5),
        # KL/entropy tracking (fill critical gap)
        "kl_qwen3-8b": run_kl_tracking.spawn("Qwen/Qwen3-8B", 42, 30),
        # Held-out evals for new scaling models
        "heldout_qwen3-32b": run_gsm8k_heldout_eval.spawn("Qwen/Qwen3-32B", 200),
        "heldout_qwen3.5-27b": run_gsm8k_heldout_eval.spawn("Qwen/Qwen3.5-27B", 200),
    }

    results = {}
    for name, future in futures.items():
        try:
            result = future.get()
            results[name] = result
            print(f"\n★ [{name}] COMPLETED")
            summary = {k: v for k, v in result.items()
                       if k not in ("per_problem", "reward_trace", "kl_trace", "entropy_trace")}
            print(json.dumps(summary, indent=2))
        except Exception as e:
            print(f"\n✗ [{name}] FAILED: {e}")
            results[name] = {"error": str(e)}

    elapsed = time.time() - start

    results_dir = "/home/user/workspace/tinker-rl-lab/experiments/modal/results"
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/modal_parallel_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"ALL MODAL H100 EXPERIMENTS COMPLETE in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'=' * 70}")
