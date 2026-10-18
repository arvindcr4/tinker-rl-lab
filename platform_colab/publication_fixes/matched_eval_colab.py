"""Common held-out evaluator for the matched TRL-vs-Tinker adapters (Colab GPU, HF + PEFT).

Greedy, 512 tokens, campaign_v2 prompt, first 200 GSM8K test items, stop on Qwen3-8B eos ids.
Scores with campaign_v2 reward (primary) and the TRL boxed-only reward.
Tinker adapters store the unembedding LoRA as `model.unembed_tokens`; it is renamed to `lm_head`.

  python matched_eval_colab.py OUT_DIR NAME=ADAPTER_DIR [NAME=ADAPTER_DIR ...]   (NAME=base for no adapter)
"""

import json, os, platform, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matched_common import (
    LORA,
    MODEL,
    N_EVAL,
    MAX_TOK,
    gsm8k,
    build_prompt,
    reward_fn,
    boxed_only_reward,
)

EOS = [151645, 151643]


def fix_tinker(ad):
    from safetensors.torch import load_file, save_file

    p = f"{ad}/adapter_model.safetensors"
    sd = load_file(p)
    if not any("unembed_tokens" in k for k in sd):
        return ad
    out = ad + "_peft"
    os.makedirs(out, exist_ok=True)
    save_file(
        {
            k.replace("base_model.model.model.unembed_tokens", "base_model.model.lm_head"): v
            for k, v in sd.items()
        },
        f"{out}/adapter_model.safetensors",
    )
    cfg = json.load(open(f"{ad}/adapter_config.json"))
    cfg["target_modules"] = LORA["target_modules"]
    cfg["base_model_name_or_path"] = MODEL
    json.dump(cfg, open(f"{out}/adapter_config.json", "w"))
    return out


def main(out, specs):
    import torch, transformers, peft
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    test = gsm8k("test")[:N_EVAL]
    prompts = [build_prompt(q) for q, _ in test]
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    res = {}
    os.makedirs(out, exist_ok=True)
    for index, spec in enumerate(specs):
        name, ad = spec.split("=", 1)
        if ad == "base":
            model = base
        else:
            ad = fix_tinker(ad)
            model = PeftModel.from_pretrained(base, ad).eval()
            missing = [
                n
                for n, p in model.named_parameters()
                if "lora_B" in n and float(p.abs().max()) == 0
            ]
        texts, ntok = [], []
        for i in range(0, len(prompts), 50):
            enc = tok(
                prompts[i : i + 50], return_tensors="pt", padding=True, add_special_tokens=False
            ).to("cuda")
            with torch.no_grad():
                g = model.generate(
                    **enc,
                    max_new_tokens=MAX_TOK,
                    do_sample=False,
                    eos_token_id=EOS,
                    pad_token_id=tok.pad_token_id,
                )
            c = g[:, enc["input_ids"].shape[1] :]
            texts += tok.batch_decode(c, skip_special_tokens=True)
            ntok += [int((row != tok.pad_token_id).sum()) for row in c]
        r1 = [reward_fn(t, a) for t, (_, a) in zip(texts, test, strict=False)]
        r2 = [boxed_only_reward(t, a) for t, (_, a) in zip(texts, test, strict=False)]
        res[name] = {
            "adapter": ad,
            "acc_campaign_reward": sum(r1) / len(r1),
            "acc_boxed_only": sum(r2) / len(r2),
            "r_campaign": r1,
            "r_boxed": r2,
            "n_tokens": ntok,
            "trunc_frac": sum(n >= MAX_TOK for n in ntok) / len(ntok),
            "texts": texts,
            "zero_lora_B_tensors": (len(missing) if ad != "base" else None),
        }
        print(
            name,
            res[name]["acc_campaign_reward"],
            res[name]["acc_boxed_only"],
            res[name]["trunc_frac"],
            flush=True,
        )
        with open(f"{out}/eval_heldout.json.tmp", "w") as handle:
            json.dump(res, handle)
        os.replace(f"{out}/eval_heldout.json.tmp", f"{out}/eval_heldout.json")
        if ad != "base":
            del model, base
            if "g" in locals():
                del g, c, enc
            import gc

            gc.collect()
            torch.cuda.empty_cache()
            if index + 1 < len(specs):
                base = (
                    AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
                    .to("cuda")
                    .eval()
                )
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    res["_env"] = {
        "platform": "google_colab",
        "gpu": smi,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "python": platform.python_version(),
        "wallclock_s": time.time() - t0,
        "decoding": "greedy, 512 new tokens, eos [151645,151643]",
        "items": "openai/gsm8k main test, first 200 (regex-parsed)",
    }
    json.dump(res, open(f"{out}/eval_heldout.json", "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
