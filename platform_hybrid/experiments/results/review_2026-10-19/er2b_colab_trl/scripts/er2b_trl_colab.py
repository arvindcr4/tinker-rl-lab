"""E-R2b matched-budget GRPO arm on Colab with TRL (adapted from
platform_colab/publication_fixes/matched_trl_colab.py; that file is not modified).

July objective (scripts/../er2b_heldout_eval/scripts/live_zvf_probe_dd6c231.py, the runner that produced
platform_hybrid/experiments/tinker-runs/results/er2b_g{2,16}_s{123,456}.json):
  Qwen/Qwen3-8B, LoRA rank 4, Adam(lr 3e-5, b1 0.9, b2 0.95, eps 1e-8), no KL, no clipping,
  temperature 1.0, top_p 0.95, 512 completion tokens, 8 prompts x G completions per optimizer step,
  prompts = rng.sample(load_gsm8k_examples(256, seed), 8) per step with rng = Random(seed),
  raw ChatML prompt (thinking on, no think stub), reward_fn = boxed match else last-number,
  advantage A = (r - mean_g) / (pop_std_g + 1e-8), loss = mean over sequences of -A * sum(logprobs),
  sampler weights refreshed every 5 optimizer steps (save_every=5), no importance ratio.

TRL mapping:
  * loss_type="luspo" with importance_sampling_level="token": (per_token_loss*mask).sum(-1).mean(),
    i.e. sum over completion tokens, mean over sequences. old_per_token_logps is dropped so the
    ratio is exactly 1 and the gradient is -A * grad(sum log p), the July REINFORCE form.
  * advantages are overwritten with the July formula (population std + 1e-8) computed from the
    same rewards TRL sees (TRL's own uses unbiased std + 1e-4).
  * steps_per_generation = 5 * grad_accum: completions for 5 optimizer steps are sampled from the
    policy at the start of the block (July sampler refresh every 5 steps). TRL's generation-batch
    shuffle is disabled so each optimizer step holds exactly the July step's 8 prompt groups.
  * HF generate (no vLLM) in chunks of GEN_CHUNK sequences to fit a 40 GB A100.

  python er2b_trl_colab.py g2_s123 OUT_DIR [--steps N] [--block B]
"""

import argparse, json, math, os, platform, random, re, subprocess, sys, time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL = "Qwen/Qwen3-8B"
RANK, LR, TEMP, TOP_P, MAX_TOK = 4, 3e-5, 1.0, 0.95, 512
BATCH = 8  # prompts per optimizer step
ROLLOUTS = 2560
EXAMPLE_LIMIT = 256
SAVE_EVERY = 5  # July sampler refresh period
MICRO = 8  # completions per forward/backward micro-batch
GEN_CHUNK = 160  # sequences per HF generate call
LORA = dict(
    r=RANK,
    lora_alpha=32,  # Tinker adapter_config uses alpha 32 (probed at r32, see matched_common.py)
    lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# ---- verbatim from live_zvf_probe_dd6c231.py ----
SYSTEM_PROMPT = (
    "You are a math assistant. Solve the problem step by step, then give "
    "your final numerical answer inside \\boxed{}."
)
QUESTION_SUFFIX = " Provide the final numerical answer inside \\boxed{}."


def reward_fn(response: str, answer: str) -> float:
    response = response.strip()
    boxed = re.findall(r"\\boxed\{([^}]+)\}", response)
    for item in boxed:
        cleaned = item.strip().replace(",", "").replace(" ", "")
        try:
            if abs(float(cleaned) - float(answer)) < 0.01:
                return 1.0
        except ValueError:
            if cleaned == answer:
                return 1.0
    nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", response)
    if nums:
        try:
            if abs(float(nums[-1].replace(",", "")) - float(answer)) < 0.01:
                return 1.0
        except ValueError:
            pass
    return 0.0


def load_gsm8k_examples(limit: int, seed: int):
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    examples: list[tuple[str, str]] = []
    for row in ds:
        match = re.search(r"####\s*([\-\d,\.]+)", row["answer"])
        if not match:
            continue
        answer = match.group(1).replace(",", "").strip()
        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{row['question']}{QUESTION_SUFFIX}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        examples.append((prompt, answer))
    rng = random.Random(seed)
    rng.shuffle(examples)
    return examples[:limit]
# ---- end verbatim ----


def july_advantages(rewards, G):
    out = []
    for k in range(0, len(rewards), G):
        g = rewards[k : k + G]
        m = sum(g) / len(g)
        var = sum((r - m) ** 2 for r in g) / len(g)
        s = math.sqrt(var) + 1e-8
        out.extend((r - m) / s for r in g)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")  # e.g. g2_s123
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--block", type=int, default=SAVE_EVERY)
    a = ap.parse_args()
    G, seed = map(int, re.fullmatch(r"g(\d+)_s(\d+)", a.arm).groups())
    steps = a.steps or ROLLOUTS // (BATCH * G)
    block = min(a.block, steps)
    assert steps % block == 0
    tag = f"er2b_trl_{a.arm}" + ("" if a.steps is None else f"_smoke{steps}")
    os.makedirs(f"{a.out}/train", exist_ok=True)
    os.makedirs(f"{a.out}/adapters", exist_ok=True)
    out_json = f"{a.out}/train/{tag}.json"
    t0 = time.time()

    import torch, transformers, trl, peft, datasets
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer
    import trl.trainer.grpo_trainer as gt

    gt.shuffle_sequence_dict = lambda d: d  # keep July step composition inside a generation block

    random.seed(seed)
    torch.manual_seed(seed)
    examples = load_gsm8k_examples(EXAMPLE_LIMIT, seed)
    rng = random.Random(seed)
    stream = [rng.sample(examples, min(BATCH, len(examples))) for _ in range(steps)]
    rows = [{"prompt": p, "answer": ans, "step": s} for s, b in enumerate(stream) for p, ans in b]
    ds = Dataset.from_list(rows)

    calls, pending_adv = [], []
    state = {"tok": None, "eos": None}

    def reward(completions, answer, step, completion_ids, **kw):
        r = [reward_fn(c, x) for c, x in zip(completions, answer, strict=True)]
        eos = set(state["eos"] or [])
        lens = [len(ids) for ids in completion_ids]
        trunc = [int(len(ids) >= MAX_TOK and (not ids or ids[-1] not in eos)) for ids in completion_ids]
        calls.append({"step": list(step), "rewards": r, "lens": lens, "trunc": trunc})
        pending_adv.append(july_advantages(r, G))
        partial = {"tag": tag, "status": "running", "group_size": G, "seed": seed, "steps": steps,
                   "step_log": build_log(), "wall_clock_sec": time.time() - t0}
        json.dump(partial, open(out_json, "w"), indent=1)
        return r

    def build_log():
        per = {}
        for c in calls:
            for k in range(0, len(c["rewards"]), G):
                ss = set(c["step"][k : k + G])
                assert len(ss) == 1, f"group spans steps {ss}"
                s = ss.pop()
                d = per.setdefault(s, {"groups": [], "lens": [], "trunc": []})
                d["groups"].append(c["rewards"][k : k + G])
                d["lens"] += c["lens"][k : k + G]
                d["trunc"] += c["trunc"][k : k + G]
        log = []
        for s in sorted(per):
            d = per[s]
            flat = [x for g in d["groups"] for x in g]
            zv = sum(1 for g in d["groups"] if sum((x - sum(g) / len(g)) ** 2 for x in g) / len(g) < 1e-10)
            zvf = zv / len(d["groups"])
            log.append({"step": s + 1, "reward": sum(flat) / len(flat), "zvf": zvf, "gu": 1 - zvf,
                        "n_groups": len(d["groups"]), "mean_comp_len": sum(d["lens"]) / len(d["lens"]),
                        "trunc_frac": sum(d["trunc"]) / len(d["trunc"])})
        return log

    class JulyGRPO(GRPOTrainer):
        def _generate_single_turn(self, prompt_ids, images, multimodal_fields, num_generations, has_tool_images=False):
            if len(prompt_ids) <= GEN_CHUNK:
                return super()._generate_single_turn(prompt_ids, images, multimodal_fields, num_generations, has_tool_images)
            assert images is None and not multimodal_fields
            comp = []
            for i in range(0, len(prompt_ids), GEN_CHUNK):
                c, _ = super()._generate_single_turn(prompt_ids[i : i + GEN_CHUNK], None, {}, num_generations, has_tool_images)
                comp += c
            return comp, None

        def _generate_and_score_completions(self, inputs):
            saved = self.args.steps_per_generation
            # TRL only computes old_per_token_logps when grad_accum % steps_per_generation != 0; the July
            # loss has no ratio, so skip that extra forward pass.
            self.args.steps_per_generation = self.args.gradient_accumulation_steps
            try:
                out = super()._generate_and_score_completions(inputs)
            finally:
                self.args.steps_per_generation = saved
            out.pop("old_per_token_logps", None)
            adv = pending_adv.pop(0)
            trl_adv = out["advantages"]
            assert trl_adv.numel() == len(adv), (trl_adv.shape, len(adv))
            new = torch.tensor(adv, dtype=trl_adv.dtype, device=trl_adv.device).view_as(trl_adv)
            # same zero pattern and sign as TRL's own advantage (scale differs: unbiased std + 1e-4)
            assert torch.equal(torch.sign(new.round(decimals=3)), torch.sign(trl_adv.round(decimals=3))), "adv mismatch"
            out["advantages"] = new
            return out

        def _compute_loss(self, model, inputs):
            inputs.pop("old_per_token_logps", None)  # ratio == 1 exactly
            return super()._compute_loss(model, inputs)

    ga = BATCH * G // MICRO
    args = GRPOConfig(
        output_dir=f"/tmp/trl-er2b-{a.arm}",
        seed=seed,
        max_steps=steps,
        per_device_train_batch_size=MICRO,
        gradient_accumulation_steps=ga,
        steps_per_generation=block * ga,
        num_generations=G,
        num_iterations=1,
        shuffle_dataset=False,
        max_completion_length=MAX_TOK,
        temperature=TEMP,
        top_p=TOP_P,
        top_k=0,
        repetition_penalty=1.0,
        learning_rate=LR,
        lr_scheduler_type="constant",
        warmup_steps=0,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        weight_decay=0.0,
        max_grad_norm=0.0,
        beta=0.0,
        loss_type="luspo",
        importance_sampling_level="token",
        scale_rewards="group",
        mask_truncated_completions=False,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        report_to="none",
        save_strategy="no",
        use_vllm=False,
        log_completions=False,
        model_init_kwargs={"dtype": "bfloat16"},
    )
    tr = JulyGRPO(
        model=MODEL,
        args=args,
        train_dataset=ds,
        reward_funcs=reward,
        peft_config=LoraConfig(task_type="CAUSAL_LM", bias="none", **LORA),
    )
    state["eos"] = list(tr.eos_token_ids)
    tr.model.print_trainable_parameters()
    print("model dtype", next(tr.model.parameters()).dtype, "cuda GB", torch.cuda.memory_allocated() / 2**30, flush=True)
    tr.train()

    ad = f"{a.out}/adapters/{tag}"
    tr.model.save_pretrained(ad)
    step_log = build_log()
    hist = [h for h in tr.state.log_history if "loss" in h and "train_runtime" not in h]
    for i, e in enumerate(step_log):
        if i < len(hist):
            e["trl_log"] = {k: v for k, v in hist[i].items() if not isinstance(v, (list, dict))}
            e["loss"] = hist[i].get("loss")
    rewards = [e["reward"] for e in step_log]
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    res = {
        "tag": tag,
        "status": "completed" if len(step_log) == steps else "incomplete",
        "backend": "trl_colab",
        "model": MODEL,
        "task": "gsm8k",
        "seed": seed,
        "rank": RANK,
        "group_size": G,
        "batch": BATCH,
        "lr": LR,
        "temperature": TEMP,
        "top_p": TOP_P,
        "max_tokens": MAX_TOK,
        "steps": steps,
        "sampler_refresh_every": block,
        "rollouts": steps * BATCH * G,
        "reward_trace": rewards,
        "step_log": step_log,
        "first5_avg": sum(rewards[:5]) / len(rewards[:5]),
        "last10_avg": sum(rewards[-10:]) / len(rewards[-10:]),
        "zvf_final20": sum(e["zvf"] for e in step_log[-20:]) / len(step_log[-20:]),
        "stream_order_ok": [e["step"] for e in step_log] == list(range(1, steps + 1))
        and all(e["n_groups"] == BATCH for e in step_log),
        "adapter_dir": ad,
        "trl_generation_kwargs": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in tr.generation_kwargs.items()},
        "trl_eos_token_ids": list(tr.eos_token_ids),
        "config": {k: getattr(args, k) for k in [
            "max_steps", "per_device_train_batch_size", "gradient_accumulation_steps", "steps_per_generation",
            "generation_batch_size", "num_generations", "temperature", "top_p", "top_k", "learning_rate",
            "lr_scheduler_type", "adam_beta1", "adam_beta2", "adam_epsilon", "weight_decay", "max_grad_norm",
            "beta", "loss_type", "importance_sampling_level", "scale_rewards", "num_iterations", "shuffle_dataset",
            "optim"]},
        "lora": LORA,
        "micro_batch": MICRO,
        "gen_chunk": GEN_CHUNK,
        "env": {"platform": "google_colab", "gpu": smi, "python": platform.python_version(),
                "torch": torch.__version__, "transformers": transformers.__version__, "trl": trl.__version__,
                "peft": peft.__version__, "datasets": datasets.__version__},
        "wall_clock_sec": time.time() - t0,
    }
    res["config"] = {k: (str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v)
                     for k, v in res["config"].items()}
    json.dump(res, open(out_json, "w"), indent=1)
    open(f"{ad}/DONE", "w").write(tag)
    print("DONE", tag, "last10", res["last10_avg"], "zvf20", res["zvf_final20"], "wall", res["wall_clock_sec"], flush=True)


if __name__ == "__main__":
    main()
