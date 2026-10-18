"""Claim B: is observed ZVF just the binomial identity? (inference only, vLLM)

Qwen2.5-1.5B-Instruct, 200 GSM8K-train prompts (fixed by seed 0), v2 prompt and
v2 strict boxed reward (copied from modal_samestack_gsm8k_cot_v2.py), temp 1.0,
top_p 1.0, 512 tokens.
  Stage 1: 32 samples/prompt -> k_i. Predicted ZVF(G) = mean_i [C(k,G)+C(32-k,G)]/C(32,G)
           (exact hypergeometric: G drawn without replacement from the 32).
  Stage 2: 3 fresh independent groups per prompt per G in {4,8,16}, separate sampling
           calls with different seeds. Observed ZVF(G).
Raw per-sample rewards are saved; analysis is in zvf_analyze.py (runs locally).
"""

import json, os, platform, random, re, subprocess, sys, time

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
N_PROMPTS, N_REF, GS, N_GROUPS, MAX_NEW = 200, 32, (4, 8, 16), 3, 512
OUT = sys.argv[1] if len(sys.argv) > 1 else "/content/zvf"


def norm_num(s):
    s = re.sub(r"\\[!,;: ]|\\\$|\$|,|\\text\{[^}]*\}|\\%|%", "", s).strip().rstrip(".")
    try:
        return f"{float(s):.6g}"
    except ValueError:
        return s


def extract(t):
    m = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})+)\}", t)
    return norm_num(m[-1]) if m else None


def reward(text, gold):
    p = extract(text)
    return 1.0 if (p is not None and p == norm_num(gold)) else 0.0


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    from datasets import load_dataset
    from transformers import AutoTokenizer
    import vllm
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(MODEL)
    train = load_dataset("openai/gsm8k", "main", split="train")
    idx = sorted(random.Random(0).sample(range(len(train)), N_PROMPTS))
    rows = [train[i] for i in idx]
    gold = [r["answer"].split("####")[-1].strip().replace(",", "") for r in rows]
    prompts = [
        tok.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": r["question"]
                    + "\nThink step by step, then give the final answer as \\boxed{ANSWER}.",
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for r in rows
    ]
    llm = LLM(MODEL, dtype="bfloat16", seed=0, gpu_memory_utilization=0.85, max_model_len=1536)

    def draw(n, seed):
        sp = SamplingParams(n=n, temperature=1.0, top_p=1.0, max_tokens=MAX_NEW, seed=seed)
        outs = llm.generate(prompts, sp, use_tqdm=False)
        R = [[reward(c.text, g) for c in o.outputs] for o, g in zip(outs, gold, strict=False)]
        T = [[int(c.finish_reason == "length") for c in o.outputs] for o in outs]
        return R, T

    res = {"model": MODEL, "train_indices": idx, "gold": gold}
    # Per-request seeds differ per call, so stage-2 groups are independent draws from stage 1.
    res["ref_rewards"], res["ref_trunc"] = draw(N_REF, seed=1000)
    res["groups"] = {}
    for G in GS:
        for j in range(N_GROUPS):
            R, T = draw(G, seed=2000 + 100 * G + j)
            res["groups"][f"G{G}_rep{j}"] = {"rewards": R, "trunc": T}
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    import torch, transformers

    res["env"] = {
        "gpu": smi,
        "python": platform.python_version(),
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "seeds": {
            "prompt_selection": 0,
            "llm": 0,
            "ref_draw": 1000,
            "group_draw": "2000+100*G+rep",
        },
        "decoding": {"temperature": 1.0, "top_p": 1.0, "max_tokens": MAX_NEW},
        "reward": "v2 strict boxed (modal_samestack_gsm8k_cot_v2.py)",
        "wallclock_s": time.time() - t0,
        "platform": "google_colab",
    }
    json.dump(res, open(f"{OUT}/zvf_raw.json", "w"))
    print("DONE", time.time() - t0)


if __name__ == "__main__":
    main()
