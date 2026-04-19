"""Relaunch just the KL tracking experiment (fixed gradient bug)."""
import modal, os, json

app = modal.App("tinker-rl-kl-fix")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "datasets", "wandb", "huggingface_hub", "accelerate")
)

@app.function(image=image, gpu="H100", timeout=7200)
def run_kl_tracking(model_name: str = "Qwen/Qwen3-8B", seed: int = 42, steps: int = 30):
    import torch, torch.nn.functional as F, random, wandb, os, json
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import HfApi

    exp = f"kl_track_{model_name.split('/')[-1]}_s{seed}_fixed"
    os.environ["WANDB_API_KEY"] = os.environ.get("WANDB_API_KEY", "")
    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    wandb.init(project="tinker-rl-lab-world-class", entity="arvindcr4-pes-university",
               name=exp, tags=["kl_tracking", "modal", "h100", "fixed"], config={
                   "model": model_name, "seed": seed, "steps": steps, "platform": "modal_h100"
               })

    print(f"[{exp}] Loading model on H100...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    policy_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                         device_map="auto", trust_remote_code=True)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                      device_map="auto", trust_remote_code=True)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    system = "You are a math tutor. Solve step by step."
    questions = [
        "What is 15% of 240?", "Solve: 3x + 7 = 22",
        "A train travels 180km in 3 hours. What is its speed?",
        "Calculate the area of a circle with radius 5.",
        "If 8 workers can build a wall in 10 days, how many days for 5 workers?",
    ]
    random.seed(seed)
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=3e-5)
    kl_trace, entropy_trace = [], []

    for step in range(steps):
        q = random.choice(questions)
        prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(policy_model.device)

        # KL and entropy are monitoring metrics only — wrap entirely in no_grad
        # and use detached tensors so we never mix frozen ref tensors with grad tracking
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

        # Gradient step uses a fresh forward pass OUTSIDE no_grad so gradients flow correctly
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

    # Upload to HF
    api = HfApi(token=os.environ["HF_TOKEN"])
    repo_id = "arvindcr4/tinker-rl-bench-kl-tracking"
    try:
        api.create_repo(repo_id, exist_ok=True)
    except Exception:
        pass
    api.upload_file(path_or_fileobj=json.dumps(result, indent=2).encode(),
                    path_in_repo=f"results/{exp}.json", repo_id=repo_id)

    wandb.finish()
    return result


@app.local_entrypoint()
def main():
    print("Relaunching KL tracking with gradient fix...")
    result = run_kl_tracking.remote("Qwen/Qwen3-8B", 42, 30)
    print(f"KL tracking completed: final_kl={result['final_kl']:.6f}, max_kl={result['max_kl']:.6f}")

    # Save result locally
    os.makedirs("/home/user/workspace/tinker-rl-lab/experiments/modal/results", exist_ok=True)
    with open("/home/user/workspace/tinker-rl-lab/experiments/modal/results/kl_tracking_fixed.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Saved to results/kl_tracking_fixed.json")
