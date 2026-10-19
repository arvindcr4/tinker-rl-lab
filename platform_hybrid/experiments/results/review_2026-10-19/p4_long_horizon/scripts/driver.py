"""Modal: Dr. GRPO vs GRPO on GSM8K with chain-of-thought (the long-output
regime where Dr. GRPO's length-bias fix actually applies) + a measured pre->post
held-out McNemar generalization test (pillar 4).

Two arms share everything (Qwen2.5-1.5B-Instruct, GSM8K boxed-answer correctness
reward, token-level clipped surrogate, group size, compute, eval) and differ ONLY
in the two Dr. GRPO fixes:
  GRPO    : A=(r-mean_g)/(std_g+eps) ; loss normalized per-response length
  Dr.GRPO : A= r-mean_g             ; loss normalized by a constant (max_new)

Unlike the 5-token arithmetic probe, CoT answers are long and variable-length, so
the length-bias fix has something to act on. We log the mean completion-length
TRAJECTORY (Dr. GRPO is designed to curb GRPO's response-length inflation).

Each run also evaluates the SAME held-out GSM8K slice greedily BEFORE (pre-RL,
base) and AFTER training, storing per-item correctness -> exact McNemar pre->post.

Usage:
  modal run experiments/modal/modal_drgrpo_gsm8k_cot.py
"""

import modal
import json
import os
import time

# WARNING (security): HF/W&B tokens previously were baked into the image via .env()/Secret.from_dict(); they now come from named Modal secrets. Rotate tokens if old images exist.

app = modal.App("tinkerrl-drgrpo-gsm8k-cot-lenstrat")
results_vol = modal.Volume.from_name("tinkerrl-results", create_if_missing=True)
RESULTS_DIR = "/results"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.3.0",
        "transformers>=4.46.0",
        "peft>=0.13.0",
        "datasets>=3.0.0",
        "numpy>=1.26.0,<2.0.0",
        "accelerate>=1.0.0",
        "safetensors>=0.4.0",
        "huggingface-hub>=0.26.0",
        "wandb>=0.16.0",
    )
    .env(
        {
            "WANDB_PROJECT": "tinkerrl-drgrpo-lenstrat",
        }
    )
)

SEEDS = [42, 123, 456]
ALGOS = ["grpo", "dr_grpo"]
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
N_STEPS = 30
N_PROMPTS = 8
GROUP = 8
K_EPOCHS = 2
CLIP = 0.2
MAX_NEW = 200
N_EVAL = 200
EPS = 1e-6
CHUNK = 4


@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,
    volumes={RESULTS_DIR: results_vol},
    retries=1,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_dict(
            {
                "WANDB_PROJECT": "tinkerrl-drgrpo-lenstrat",
            }
        ),
    ],
)
def run_arm(algo: str, seed: int, max_new: int) -> dict:
    MAX_NEW = max_new
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import random
    import re
    import numpy as np
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = get_peft_model(
        AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.bfloat16, trust_remote_code=True
        ).to(device),
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-5)

    ds = load_dataset("openai/gsm8k", "main")
    train = ds["train"].shuffle(seed=seed)
    test = ds["test"].shuffle(seed=0).select(range(N_EVAL))

    def gold_of(ans):
        return ans.split("####")[-1].strip().replace(",", "")

    def extract(t):
        m = re.findall(r"\\boxed\{([^}]+)\}", t)
        cand = m[-1] if m else (re.findall(r"-?\d[\d,]*", t) or [None])[-1]
        return cand.replace(",", "").replace("$", "").strip() if cand else None

    def build(q):
        msgs = [
            {
                "role": "user",
                "content": q
                + "\nThink step by step, then give the final answer as \\boxed{ANSWER}.",
            }
        ]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def reward(text, gold):
        p = extract(text)
        return 1.0 if (p is not None and p == gold) else 0.0

    def gen_batch(prompts, greedy):
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        plen = enc["input_ids"].shape[1]
        with torch.inference_mode():
            g = model.generate(
                **enc,
                max_new_tokens=MAX_NEW,
                do_sample=not greedy,
                temperature=(1.0 if not greedy else None),
                top_p=(1.0 if not greedy else None),
                pad_token_id=tok.pad_token_id,
            )
        return enc, plen, g[:, plen:]

    def heldout():
        model.eval()
        corr = []
        for i in range(0, len(test), 16):
            ch = list(test)[i : i + 16]
            _, _, c = gen_batch([build(x["question"]) for x in ch], greedy=True)
            for x, t in zip(ch, tok.batch_decode(c, skip_special_tokens=True), strict=False):
                corr.append(int(reward(t, gold_of(x["answer"])) == 1.0))
        model.train()
        return corr

    protocol = dict(model=MODEL, algo=algo, seed=seed, max_new=MAX_NEW, n_eval=N_EVAL,
                    n_prompts=N_PROMPTS, group=GROUP, k_epochs=K_EPOCHS, chunk=CHUNK,
                    clip=CLIP, eps=EPS, lr=1e-5, lora_r=16, lora_alpha=32)
    checkpoint = None
    if os.environ.get("P4_RESUME_FROM"):
        # Only load our own checkpoint files, never an untrusted pickle.
        checkpoint = torch.load(os.environ["P4_RESUME_FROM"], map_location="cpu", weights_only=False)
        assert checkpoint["protocol"] == protocol, "Checkpoint protocol mismatch"
        assert 0 <= len(checkpoint["step_log"]) <= N_STEPS
        assert [s["step"] for s in checkpoint["step_log"]] == list(range(len(checkpoint["step_log"])))
        loaded = set_peft_model_state_dict(model, checkpoint["adapter"])
        assert not loaded.unexpected_keys, loaded.unexpected_keys
        opt.load_state_dict(checkpoint["optimizer"])
    pre_correct = checkpoint["pre_correct"] if checkpoint else heldout()  # base / pre-RL

    def token_logps_grad(full, attn, plen):
        outs_lp, outs_m = [], []
        for cs in range(0, full.shape[0], CHUNK):
            ce = min(cs + CHUNK, full.shape[0])
            logits = (
                model(input_ids=full[cs:ce], attention_mask=attn[cs:ce]).logits[:, :-1, :].float()
            )
            tgt = full[cs:ce][:, 1:]
            lp = (
                F.log_softmax(logits, dim=-1)
                .gather(-1, tgt.unsqueeze(-1))
                .squeeze(-1)[:, plen - 1 :]
            )
            cm = (tgt != tok.pad_token_id).float()[:, plen - 1 :]
            outs_lp.append(lp)
            outs_m.append(cm)
            del logits
        return torch.cat(outs_lp), torch.cat(outs_m)

    import wandb

    run_name = f"{algo}_s{seed}"
    wandb.init(
        project="tinkerrl-drgrpo-lenstrat",
        name=run_name,
        config={"algo": algo, "seed": seed, "model": MODEL, "max_new": MAX_NEW},
    )

    step_log = checkpoint["step_log"] if checkpoint else []
    t0 = time.time() - (checkpoint["elapsed_seconds"] if checkpoint else 0)
    train_iter = iter(train)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    progress_path = f"{RESULTS_DIR}/{algo}_s{seed}_progress.jsonl"
    if checkpoint:
        # Replay only dataset iteration, not generation or optimizer updates.
        for skipped_step in range(len(step_log)):
            for _ in range(N_PROMPTS):
                try:
                    next(train_iter)
                except StopIteration:
                    train_iter = iter(train.shuffle(seed=seed + skipped_step))
                    next(train_iter)
        with open(progress_path, "w") as f:
            f.write(checkpoint["progress_jsonl"])
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        print(f"RESUMED {algo} s{seed} after {len(step_log)} complete steps", flush=True)
    checkpoint_every = int(os.environ.get("P4_CHECKPOINT_EVERY", "10"))
    for step in range(len(step_log), N_STEPS):
        batch = []
        for _ in range(N_PROMPTS):
            try:
                batch.append(next(train_iter))
            except StopIteration:
                train_iter = iter(train.shuffle(seed=seed + step))
                batch.append(next(train_iter))
        prompts, golds = [], []
        for ex in batch:
            for _ in range(GROUP):
                prompts.append(build(ex["question"]))
                golds.append(gold_of(ex["answer"]))
        enc, plen, comp_ids = gen_batch(prompts, greedy=False)
        comp_txt = tok.batch_decode(comp_ids, skip_special_tokens=True)
        rewards = np.array(
            [reward(t, g) for t, g in zip(comp_txt, golds, strict=False)], dtype=np.float32
        )

        R = rewards.reshape(N_PROMPTS, GROUP)
        gmean = R.mean(1, keepdims=True)
        gstd = R.std(1, keepdims=True)
        zvf = float((gstd[:, 0] <= EPS).mean())
        A = (R - gmean) / (gstd + EPS) if algo == "grpo" else (R - gmean)
        adv = torch.tensor(A.reshape(-1), dtype=torch.float32, device=device)

        full = torch.cat([enc["input_ids"], comp_ids], dim=1)
        attn = (full != tok.pad_token_id).long()
        B = full.shape[0]
        with torch.no_grad():
            old_lp, mask = token_logps_grad(full, attn, plen)
        comp_len = float(mask.sum(1).mean().item())
        _L = mask.sum(1).float().cpu().numpy()
        _c = np.asarray(rewards) > 0.5
        lenstrat = {
            "len_correct_mean": float(_L[_c].mean()) if _c.any() else None,
            "len_incorrect_mean": float(_L[~_c].mean()) if (~_c).any() else None,
            "n_correct": int(_c.sum()),
            "n_incorrect": int((~_c).sum()),
            "frac_at_cap": float((_L >= MAX_NEW).mean()),
            "frac_at_cap_correct": float((_L[_c] >= MAX_NEW).mean()) if _c.any() else None,
            "frac_at_cap_incorrect": float((_L[~_c] >= MAX_NEW).mean()) if (~_c).any() else None,
            "lengths": _L.astype(int).tolist(),
            "rewards": _c.astype(int).tolist(),
        }

        for _ep in range(K_EPOCHS):
            opt.zero_grad()
            for cs in range(0, B, CHUNK):
                ce = min(cs + CHUNK, B)
                logits = (
                    model(input_ids=full[cs:ce], attention_mask=attn[cs:ce])
                    .logits[:, :-1, :]
                    .float()
                )
                tgt = full[cs:ce][:, 1:]
                lp = (
                    F.log_softmax(logits, dim=-1)
                    .gather(-1, tgt.unsqueeze(-1))
                    .squeeze(-1)[:, plen - 1 :]
                )
                m = mask[cs:ce]
                old = old_lp[cs:ce]
                a_i = adv[cs:ce].unsqueeze(1)
                ratio = torch.exp(lp - old)
                ptl = -torch.min(ratio * a_i, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * a_i) * m
                if algo == "grpo":
                    loss_chunk = (ptl.sum(1) / m.sum(1).clamp(min=1.0)).sum() / B
                else:
                    loss_chunk = ptl.sum() / (B * MAX_NEW)
                loss_chunk.backward()
                del logits, lp, ptl
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        metrics = {
            "step": step,
            "mean_reward": float(rewards.mean()),
            "zvf": zvf,
            "mean_comp_len": comp_len,
                **lenstrat,
        }
        step_log.append(metrics)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(f"{RESULTS_DIR}/{algo}_s{seed}_progress.jsonl", "a") as f:
            f.write(json.dumps({**metrics, "elapsed_seconds": time.time() - t0}) + "\n")
        print(f"STEP {step + 1}/{N_STEPS} {algo} s{seed} "
              f"elapsed={time.time() - t0:.1f}s reward={rewards.mean():.3f} "
              f"length={comp_len:.1f}", flush=True)
        wandb.log({k: v for k, v in metrics.items() if not isinstance(v, list)})
        if (step + 1) % checkpoint_every == 0 or step + 1 == N_STEPS:
            with open(progress_path) as f:
                progress_jsonl = f.read()
            path = f"{RESULTS_DIR}/{algo}_s{seed}_checkpoint.pt"
            torch.save(dict(
                protocol=protocol, adapter=get_peft_model_state_dict(model),
                optimizer=opt.state_dict(), pre_correct=pre_correct,
                step_log=step_log, progress_jsonl=progress_jsonl,
                elapsed_seconds=time.time() - t0,
                python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            ), path + ".tmp")
            os.replace(path + ".tmp", path)
            print(f"CHECKPOINT {step + 1} {path}", flush=True)

    post_correct = heldout()  # post-GRPO

    import numpy as _np

    pre = _np.array(pre_correct)
    post = _np.array(post_correct)
    res = {
        "experiment": "drgrpo_gsm8k_cot",
        "algo": algo,
        "seed": seed,
        "model": MODEL,
        "max_new": MAX_NEW,
        "heldout_pre_acc": float(pre.mean()),
        "heldout_post_acc": float(post.mean()),
        "n_eval": int(len(pre)),
        "improved_wrong_to_right": int(((pre == 0) & (post == 1)).sum()),
        "regressed_right_to_wrong": int(((pre == 1) & (post == 0)).sum()),
        "last10_avg": float(_np.mean([s["mean_reward"] for s in step_log[-10:]])),
        "mean_comp_len_first5": float(_np.mean([s["mean_comp_len"] for s in step_log[:5]])),
        "mean_comp_len_last5": float(_np.mean([s["mean_comp_len"] for s in step_log[-5:]])),
        "mean_zvf": float(_np.mean([s["zvf"] for s in step_log])),
        "elapsed_seconds": time.time() - t0,
        "pre_correct": pre_correct,
        "post_correct": post_correct,
        "step_log": step_log,
    }
    os.makedirs(f"{RESULTS_DIR}/drgrpo_gsm8k_lenstrat_cap{MAX_NEW}", exist_ok=True)
    with open(f"{RESULTS_DIR}/drgrpo_gsm8k_lenstrat_cap{MAX_NEW}/{algo}_s{seed}.json", "w") as f:
        json.dump(res, f)
    results_vol.commit()
    print(
        f"[{algo} s{seed}] pre={pre.mean():.3f} post={post.mean():.3f} "
        f"len {res['mean_comp_len_first5']:.0f}->{res['mean_comp_len_last5']:.0f} last10={res['last10_avg']:.3f}"
    )

    if False:  # review rerun: no HF push
        repo_id = f"drgrpo-gsm8k-cot-{algo}-s{seed}"
        print(f"Pushing adapter to Hub: {repo_id}")
        try:
            model.push_to_hub(repo_id, token=os.environ["HF_TOKEN"], private=True)
            tok.push_to_hub(repo_id, token=os.environ["HF_TOKEN"], private=True)
        except Exception as e:
            print(f"Failed to push to hub: {e}")

    wandb.finish()

    return res


@app.local_entrypoint()
def main(max_new: int = 0, out: str = ""):
    mn = max_new or MAX_NEW
    jobs = [(al, s, mn) for al in ALGOS for s in SEEDS]
    print(f"{len(jobs)} runs ({ALGOS} x {SEEDS}), MAX_NEW={mn}")
    results = [r for r in run_arm.starmap(jobs, return_exceptions=True) if isinstance(r, dict)]
    with open(out, "w") as f:
        json.dump({"max_new": mn, "model": MODEL, "runs": results}, f)
    print("saved", out, len(results), "runs")
