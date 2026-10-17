"""TRL side of the matched TRL-vs-Tinker run (Colab GPU).

Matches matched_tinker.py: Qwen3-8B, LoRA r32/alpha32/dropout0 on q,k,v,o,gate,up,down + lm_head
(= Tinker's adapter), byte-identical hand-built prompt (non-conversational, so TRL applies no
chat template), campaign_v2 reward, temp 0.8 / top_p 0.95 / top_k off / 512 tok, eos = TRL default
for Qwen3-8B, 2 prompts x 8 per step in the campaign_v2 stream order (shuffle_dataset=False),
30 steps, constant lr 1e-5, Adam (0.9, 0.95, 1e-8), wd 0, no grad clipping, beta 0, one
optimizer step per generation batch (on-policy), loss_type dr_grpo (sum over tokens; constant
1/(N*512) scale, which Adam removes).
Known residual differences, not removable without patching TRL: TRL scores logprobs at
logits/temperature; group std is unbiased (ddof 1) + 1e-4 vs Tinker population std + 1e-8
(a constant sqrt(8/7) scale); bf16 HF forward vs Tinker's server numerics.

  python matched_trl_colab.py SEED OUT_DIR
"""
import json, os, platform, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matched_common import *  # noqa


def main(seed, out):
    t0 = time.time()
    import torch, transformers, trl, peft
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer
    stream = prompt_stream(seed, gsm8k("train"))
    rows = [{"prompt": build_prompt(q), "answer": a, "step": s} for s, b in enumerate(stream) for q, a in b]
    ds = Dataset.from_list(rows)
    calls = []

    def reward(completions, answer, step, **kw):
        r = [reward_fn(c, a) for c, a in zip(completions, answer)]
        calls.append({"steps": sorted(set(step)), "rewards": r,
                      "n_unique_answers": len(set(answer))})
        return r

    args = GRPOConfig(output_dir=f"/tmp/trl-matched-{seed}", seed=seed, max_steps=STEPS,
                      per_device_train_batch_size=PROMPTS_PER_STEP * GROUP, num_generations=GROUP,
                      gradient_accumulation_steps=1, steps_per_generation=1, num_iterations=1,
                      shuffle_dataset=False, max_completion_length=MAX_TOK,
                      temperature=TEMP, top_p=TOP_P, top_k=0, repetition_penalty=1.0,
                      learning_rate=LR, lr_scheduler_type="constant", warmup_steps=0,
                      adam_beta1=ADAM["beta1"], adam_beta2=ADAM["beta2"], adam_epsilon=ADAM["eps"],
                      weight_decay=ADAM["weight_decay"], max_grad_norm=0.0, beta=0.0,
                      loss_type="dr_grpo", scale_rewards="group", mask_truncated_completions=False,
                      bf16=True, gradient_checkpointing=True, logging_steps=1, report_to="none",
                      save_strategy="no", use_vllm=False, log_completions=False)
    tr = GRPOTrainer(model=MODEL, args=args, train_dataset=ds, reward_funcs=reward,
                     peft_config=LoraConfig(task_type="CAUSAL_LM", bias="none", **LORA))
    tr.train()
    hist = [h for h in tr.state.log_history if "reward" in h]
    step_log = []
    for i, c in enumerate(calls):
        g = [c["rewards"][k:k + GROUP] for k in range(0, len(c["rewards"]), GROUP)]
        step_log.append({"step": i, "dataset_steps_seen": c["steps"], "mean_reward": sum(c["rewards"]) / len(c["rewards"]),
                         "zvf": sum(len(set(x)) == 1 for x in g) / len(g),
                         "trl_log": {k: hist[i][k] for k in hist[i] if not isinstance(hist[i][k], (list, dict))} if i < len(hist) else None})
    ad = f"{out}/trl_s{seed}_adapter"
    tr.model.save_pretrained(ad)
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    res = {"side": "trl", "seed": seed, "model": MODEL, "step_log": step_log, "adapter_dir": ad,
           "stream_order_ok": all(s["dataset_steps_seen"] == [s["step"]] for s in step_log),
           "trl_generation_kwargs": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in tr.generation_kwargs.items()},
           "trl_eos_token_ids": list(tr.eos_token_ids),
           "config": {k: getattr(args, k) for k in ["max_steps", "per_device_train_batch_size", "num_generations", "temperature",
                                                     "top_p", "top_k", "learning_rate", "lr_scheduler_type", "adam_beta1",
                                                     "adam_beta2", "adam_epsilon", "weight_decay", "max_grad_norm", "beta",
                                                     "loss_type", "scale_rewards", "num_iterations", "shuffle_dataset"]},
           "lora": LORA,
           "env": {"platform": "google_colab", "gpu": smi, "python": platform.python_version(), "torch": torch.__version__,
                   "transformers": transformers.__version__, "trl": trl.__version__, "peft": peft.__version__,
                   "wallclock_s": time.time() - t0, "seeds": {"stream": seed, "trainer": seed}}}
    json.dump(res, open(f"{out}/trl_s{seed}.json", "w"), indent=1)
    print("DONE", seed, sum(s["mean_reward"] for s in step_log[-10:]) / 10, time.time() - t0)


if __name__ == "__main__":
    os.makedirs(sys.argv[2], exist_ok=True)
    main(int(sys.argv[1]), sys.argv[2])
