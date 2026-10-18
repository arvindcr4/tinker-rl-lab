"""Claim C harness check (inference only): checkpoint x harness on 500 GSM8K test items.

Checkpoints: Qwen/Qwen3-8B-Base, Qwen/Qwen3-8B.
Harnesses (reproduced from code):
  trl    : platform_hybrid/experiments/modal/modal_grpo_trl.py -- conversational prompt
           (its SYSTEM_PROMPT + user suffix) rendered by TRL's own GRPOTrainer path
           (apply_chat_template, add_generation_prompt, chat_template_kwargs={}),
           TRL GRPOConfig default decoding as built by GRPOTrainer (captured at runtime),
           512 tokens, TRL's eos ids, reward = boxed-match only (verbatim).
  tinker : platform_hybrid/experiments/tinker-runs/campaign_v2.py -- hand-built ChatML
           string, temp 0.8 / top_p 0.95, 512 tokens, reward = boxed-or-last-number (verbatim).
Sensitivity cell trl_modelgen: TRL prompt + the checkpoint's own generation_config.json
sampling values (what older transformers/TRL merges produced, see transformers#42762).

Modes (separate processes so the HF model and vLLM never share the GPU):
  python harness_check_colab.py probe MODEL OUT_DIR     # TRL GRPOTrainer -> prompt ids + gen kwargs
  python harness_check_colab.py gen   OUT_DIR PROBE     # vLLM sampling for one probe file
"""

import json, os, platform, re, subprocess, sys, time

N_ITEMS, N_SAMPLES, MAX_TOK = 500, 4, 512
TRL_SYSTEM = (
    "You are a math assistant. Solve step by step, then give your final answer inside \\boxed{}."
)
TRL_SUFFIX = " Give a numerical answer inside \\boxed{}."
TK_SYSTEM = "You are a math assistant. Solve the problem step by step, then give your final numerical answer inside \\boxed{}."
TK_SUFFIX = " Provide a numerical answer without units, written inside \\boxed{}."


def gold_of(ans):
    m = re.search(r"####\s*([\-\d,\.]+)", ans)
    return m.group(1).replace(",", "").strip()


def trl_reward(text, expected):
    """modal_grpo_trl.py reward_function body for one completion (verbatim logic)."""
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    reward = 0.0
    for b in boxed:
        b_clean = b.strip().replace(",", "")
        try:
            if expected and abs(float(b_clean) - float(expected)) < 0.01:
                reward = 1.0
                break
        except:
            if b_clean == expected:
                reward = 1.0
                break
    return reward


def tinker_reward(response, answer):
    """campaign_v2.py reward_fn (verbatim)."""
    response = response.strip()
    boxed = re.findall(r"\\boxed\{([^}]+)\}", response)
    for b in boxed:
        b_clean = b.strip().replace(",", "").replace(" ", "")
        try:
            if abs(float(b_clean) - float(answer)) < 0.01:
                return 1.0
        except:
            if b_clean == answer:
                return 1.0
    all_nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", response)
    if all_nums:
        last = all_nums[-1].replace(",", "")
        try:
            if abs(float(last) - float(answer)) < 0.01:
                return 1.0
        except:
            pass
    return 0.0


def items():
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=f"test[:{N_ITEMS}]")
    return [r["question"] for r in ds], [gold_of(r["answer"]) for r in ds]


def probe(model_id, out):
    import torch, transformers, trl
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    qs, golds = items()
    ds = Dataset.from_list(
        [
            {
                "prompt": [
                    {"role": "system", "content": TRL_SYSTEM},
                    {"role": "user", "content": q + TRL_SUFFIX},
                ]
            }
            for q in qs[:16]
        ]
    )
    # Same GRPOConfig values as modal_grpo_trl.py; hub/wandb/eval/checkpoint I/O removed (no effect on sampling).
    args = GRPOConfig(
        output_dir="/tmp/trl-probe",
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        num_generations=8,
        generation_batch_size=8,
        max_completion_length=512,
        num_train_epochs=1,
        max_steps=30,
        logging_steps=1,
        report_to="none",
        bf16=True,
        gradient_accumulation_steps=1,
    )
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    tr = GRPOTrainer(
        model=model_id,
        args=args,
        train_dataset=ds,
        reward_funcs=lambda completions, **k: [0.0] * len(completions),
        peft_config=peft_config,
    )
    tok = tr.processing_class
    conv = [
        [{"role": "system", "content": TRL_SYSTEM}, {"role": "user", "content": q + TRL_SUFFIX}]
        for q in qs
    ]
    # Exactly the non-environment branch of GRPOTrainer._tokenize_prompts in TRL 1.16.
    tk = tok.apply_chat_template(
        conversation=conv,
        tools=tr.tools or None,
        chat_template=tr.chat_template,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        **tr.chat_template_kwargs,
    )
    trl_ids = [list(map(int, x)) for x in tk["input_ids"]]
    tk_prompts = [
        (
            f"<|im_start|>system\n{TK_SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{q + TK_SUFFIX}<|im_end|>\n<|im_start|>assistant\n"
        )
        for q in qs
    ]
    tk_ids = [tok.encode(p, add_special_tokens=False)[:1024] for p in tk_prompts]
    gk = {
        k: (v if not hasattr(v, "tolist") else v.tolist()) for k, v in tr.generation_kwargs.items()
    }
    mg = tr.model.generation_config.to_dict() if hasattr(tr.model, "generation_config") else {}
    from transformers import GenerationConfig

    try:
        model_gen = GenerationConfig.from_pretrained(model_id).to_dict()
    except Exception as e:
        model_gen = {"error": repr(e)}
    resp_parse = (
        getattr(tok, "response_template", None) is not None
        or getattr(tok, "response_schema", None) is not None
    )
    res = {
        "model": model_id,
        "trl_prompt_ids": trl_ids,
        "tinker_prompt_ids": tk_ids,
        "golds": golds,
        "trl_prompt_example": tok.decode(trl_ids[0]),
        "tinker_prompt_example": tok.decode(tk_ids[0]),
        "trl_generation_kwargs": gk,
        "trl_eos_token_ids": list(tr.eos_token_ids),
        "model_generation_config": model_gen,
        "trainer_model_generation_config": mg,
        "chat_template_kwargs": tr.chat_template_kwargs,
        "tokenizer_has_chat_template": tok.chat_template is not None,
        "trl_uses_parse_response": resp_parse,
        "trl_config_defaults": {
            k: getattr(args, k)
            for k in [
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "repetition_penalty",
                "beta",
                "loss_type",
                "scale_rewards",
            ]
        },
        "versions": {
            "trl": trl.__version__,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
        },
    }
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(f"{out}/probe_{model_id.split('/')[-1]}.json", "w"))
    print(
        "PROBE",
        model_id,
        json.dumps(
            {
                k: res[k]
                for k in [
                    "trl_generation_kwargs",
                    "trl_eos_token_ids",
                    "trl_uses_parse_response",
                    "tokenizer_has_chat_template",
                    "chat_template_kwargs",
                ]
            },
            default=str,
        ),
    )
    print(res["trl_prompt_example"])
    print(res["tinker_prompt_example"])


def gen(out, pf):
    import vllm, torch, transformers
    from vllm import LLM, SamplingParams, TokensPrompt
    from transformers import AutoTokenizer

    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if True:
        P = json.load(open(pf))
        mid = P["model"]
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(mid)
        llm = LLM(
            mid,
            dtype="bfloat16",
            seed=0,
            gpu_memory_utilization=0.92,
            max_model_len=1024,
            generation_config="vllm",
        )  # vllm: do not inherit generation_config.json defaults
        gk, eos = P["trl_generation_kwargs"], P["trl_eos_token_ids"]
        mg = P["model_generation_config"]
        cells = {
            "trl": (
                P["trl_prompt_ids"],
                dict(
                    temperature=gk["temperature"],
                    top_p=gk["top_p"],
                    top_k=(gk["top_k"] or -1),
                    min_p=(gk.get("min_p") or 0.0),
                    repetition_penalty=gk["repetition_penalty"],
                ),
            ),
            "tinker": (P["tinker_prompt_ids"], dict(temperature=0.8, top_p=0.95)),
            "trl_modelgen": (
                P["trl_prompt_ids"],
                dict(
                    temperature=(
                        mg.get("temperature") if mg.get("temperature") is not None else 1.0
                    ),
                    top_p=(mg.get("top_p") if mg.get("top_p") is not None else 1.0),
                    top_k=(mg.get("top_k") or -1),
                ),
            ),
        }
        res = {
            "model": mid,
            "golds": P["golds"],
            "cells": {},
            "probe": {k: v for k, v in P.items() if not k.endswith("prompt_ids") and k != "golds"},
        }
        for cell, (ids, kw) in cells.items():
            # Stop on TRL's eos ids for every cell; Tinker's server-side stop set is checked separately on Tinker.
            sp = SamplingParams(
                n=N_SAMPLES,
                max_tokens=MAX_TOK,
                seed=1234,
                stop_token_ids=eos,
                skip_special_tokens=True,
                **kw,
            )
            outs = llm.generate([TokensPrompt(prompt_token_ids=x) for x in ids], sp, use_tqdm=False)
            texts = [[c.text for c in o.outputs] for o in outs]
            fin = [[c.finish_reason for c in o.outputs] for o in outs]
            ntok = [[len(c.token_ids) for c in o.outputs] for o in outs]
            res["cells"][cell] = {
                "sampling": {
                    **kw,
                    "n": N_SAMPLES,
                    "max_tokens": MAX_TOK,
                    "stop_token_ids": eos,
                    "seed": 1234,
                },
                "texts": texts,
                "finish": fin,
                "n_tokens": ntok,
                "r_trl": [
                    [trl_reward(t, g) for t in ts] for ts, g in zip(texts, P["golds"], strict=False)
                ],
                "r_tinker": [
                    [tinker_reward(t, g) for t in ts]
                    for ts, g in zip(texts, P["golds"], strict=False)
                ],
            }
            print(
                mid,
                cell,
                "acc_trl_fn",
                sum(map(sum, res["cells"][cell]["r_trl"])) / (N_ITEMS * N_SAMPLES),
                "acc_tinker_fn",
                sum(map(sum, res["cells"][cell]["r_tinker"])) / (N_ITEMS * N_SAMPLES),
                flush=True,
            )
        res["env"] = {
            "gpu": smi,
            "python": platform.python_version(),
            "vllm": vllm.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "wallclock_s": time.time() - t0,
            "platform": "google_colab",
            "items": f"openai/gsm8k main test[:{N_ITEMS}]",
        }
        json.dump(res, open(f"{out}/gen_{mid.split('/')[-1]}.json", "w"))
        del llm
        import gc

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    if sys.argv[1] == "probe":
        probe(sys.argv[2], sys.argv[3])
    else:
        gen(sys.argv[2], sys.argv[3])
