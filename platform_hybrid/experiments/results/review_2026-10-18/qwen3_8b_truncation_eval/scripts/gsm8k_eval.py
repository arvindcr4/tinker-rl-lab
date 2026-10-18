"""Held-out GSM8K eval for Qwen3-8B (+ Tinker LoRA adapters) under training-matched vs non-truncating decoding.

Usage (on the GPU VM):
  python gsm8k_eval.py merge NAME ADAPTER_DIR            # CPU merge -> /content/merged/NAME
  python gsm8k_eval.py gen MODEL_PATH NAME OUT_DIR COND[,COND...]

Prompts (verbatim from repo):
  wc      = world_class_experiments.py (P8 scale_gsm8k_qwen3-8b): system + raw question, no suffix
  matched = platform_colab/publication_fixes/matched_common.py (P6 sec 5.2): system + question + boxed suffix
Conditions: <prompt>_<think|nothink>_<max_tokens>, e.g. wc_think_512, matched_nothink_4096.
  think   = raw ChatML "<|im_start|>assistant\\n" exactly as in training (Qwen3 thinks by default)
  nothink = same + "<think>\\n\\n</think>\\n\\n" (what the Qwen3 chat template emits for enable_thinking=False)
Items: openai/gsm8k main test, first 200 (same as matched_eval_colab.py). Greedy, stop on [151645, 151643].
"""
import json, math, os, re, sys, time

MODEL = "Qwen/Qwen3-8B"
N_EVAL = int(os.environ.get("N_EVAL", "200"))
EOS = [151645, 151643]
SYS = "You are a math assistant. Solve the problem step by step, then give your final numerical answer inside \\boxed{}."
SUFFIX = " Provide a numerical answer without units, written inside \\boxed{}."


def build(q, prompt, think):
    user = q + SUFFIX if prompt == "matched" else q
    p = f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    return p if think else p + "<think>\n\n</think>\n\n"


def campaign_reward(response, answer):  # verbatim training reward (campaign_v2 / world_class reward_math)
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
        try:
            if abs(float(nums[-1].replace(",", "")) - float(answer)) < 0.01:
                return 1.0
        except Exception:
            pass
    return 0.0


def boxed_reward(text, expected):
    for b in re.findall(r'\\boxed\{([^}]+)\}', text):
        b_clean = b.strip().replace(",", "")
        try:
            if abs(float(b_clean) - float(expected)) < 0.01:
                return 1.0
        except Exception:
            if b_clean == expected:
                return 1.0
    return 0.0


def gsm8k_test():
    from datasets import load_dataset
    out = []
    for row in load_dataset("openai/gsm8k", "main", split="test"):
        m = re.search(r'####\s*([\-\d,\.]+)', row["answer"])
        if m:
            out.append((row["question"], m.group(1).replace(",", "").strip()))
    return out[:N_EVAL]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def merge(name, ad):
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    t0 = time.time()
    sd = load_file(f"{ad}/adapter_model.safetensors")
    sd = {k.replace("base_model.model.model.unembed_tokens", "base_model.model.lm_head"): v for k, v in sd.items()}
    mods = sorted({k.split(".lora_")[0].split(".")[-1] for k in sd})
    fixed = f"/content/peft_fixed/{name}"
    os.makedirs(fixed, exist_ok=True)
    save_file(sd, f"{fixed}/adapter_model.safetensors")
    cfg = json.load(open(f"{ad}/adapter_config.json"))
    cfg.update(target_modules=mods, base_model_name_or_path=MODEL)
    json.dump(cfg, open(f"{fixed}/adapter_config.json", "w"))
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    m = PeftModel.from_pretrained(base, fixed)
    zero_b = sum(1 for n, p in m.named_parameters() if "lora_B" in n and float(p.abs().max()) == 0)
    n_lora = sum(1 for n, _ in m.named_parameters() if "lora_" in n)
    m = m.merge_and_unload()
    out = f"/content/merged/{name}"
    m.save_pretrained(out)
    AutoTokenizer.from_pretrained(MODEL).save_pretrained(out)
    info = {"name": name, "adapter_dir": ad, "target_modules": mods, "lora_alpha": cfg.get("lora_alpha"), "r": cfg.get("r"),
            "n_lora_params_loaded": n_lora, "zero_lora_B_tensors": zero_b, "merge_s": time.time() - t0}
    json.dump(info, open(f"{out}/merge_info.json", "w"), indent=1)
    print(json.dumps(info), flush=True)


def gen(model_path, name, out_dir, conds):
    from vllm import LLM, SamplingParams
    import vllm, torch
    os.makedirs(out_dir, exist_ok=True)
    test = gsm8k_test()
    maxlen = max(int(c.split("_")[2]) for c in conds) + 512
    t0 = time.time()
    llm = LLM(model=model_path, tokenizer=MODEL, dtype="bfloat16", max_model_len=maxlen,
              gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.92")), seed=0, enable_prefix_caching=True)
    load_s = time.time() - t0
    summ = []
    for cond in conds:
        prompt, think, mt = cond.split("_")
        mt = int(mt)
        prompts = [build(q, prompt, think == "think") for q, _ in test]
        sp = SamplingParams(temperature=0.0, max_tokens=mt, stop_token_ids=EOS, seed=0)
        t1 = time.time()
        outs = llm.generate(prompts, sp)
        wall = time.time() - t1
        rows = []
        for (q, a), o in zip(test, outs):
            c = o.outputs[0]
            txt = c.text
            post = txt.split("</think>")[-1] if "</think>" in txt else ("" if think == "think" else txt)
            rows.append({"q": q, "gold": a, "text": txt, "n_tokens": len(c.token_ids), "finish": c.finish_reason,
                         "truncated": c.finish_reason == "length",
                         "has_boxed": bool(re.search(r'\\boxed\{([^}]+)\}', txt)),
                         "think_closed": "</think>" in txt,
                         "r_campaign": campaign_reward(txt, a), "r_boxed": boxed_reward(txt, a),
                         "r_boxed_final": boxed_reward(post, a)})
        n = len(rows)
        k = sum(r["r_campaign"] for r in rows)
        kb = sum(r["r_boxed"] for r in rows)
        s = {"model": name, "cond": cond, "n": n, "acc_campaign": k / n, "acc_campaign_wilson95": wilson(k, n),
             "acc_boxed": kb / n, "acc_boxed_wilson95": wilson(kb, n),
             "trunc_rate": sum(r["truncated"] for r in rows) / n,
             "parse_fail_rate_no_boxed": sum(not r["has_boxed"] for r in rows) / n,
             "think_unclosed_rate": (sum(not r["think_closed"] for r in rows) / n) if think == "think" else None,
             "mean_tokens": sum(r["n_tokens"] for r in rows) / n,
             "acc_campaign_among_truncated": (lambda t: sum(r["r_campaign"] for r in t) / len(t) if t else None)([r for r in rows if r["truncated"]]),
             "acc_campaign_among_finished": (lambda t: sum(r["r_campaign"] for r in t) / len(t) if t else None)([r for r in rows if not r["truncated"]]),
             "gen_wall_s": wall, "model_load_s": load_s, "gpu": torch.cuda.get_device_name(0), "vllm": vllm.__version__}
        with open(f"{out_dir}/gen_{name}__{cond}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        with open(f"{out_dir}/summary.jsonl", "a") as f:
            f.write(json.dumps(s) + "\n")
        print(json.dumps({k2: v for k2, v in s.items() if k2 not in ("gpu",)}), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "merge":
        merge(sys.argv[2], sys.argv[3])
    else:
        gen(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5].split(","))
