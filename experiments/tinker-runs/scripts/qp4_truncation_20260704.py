#!/usr/bin/env python3
"""qp4-truncation: held-out GSM8K accuracy vs generation cap (SAMPLING-ONLY).

Feeds paper P4 (length bias). Evaluates base model at max_tokens in
{64,128,256,512} on 200 held-out GSM8K test problems. No training.

Usage:
  python3 qp4_truncation_20260704.py --smoke     # 1 cap, 4 problems
  python3 qp4_truncation_20260704.py             # full run
"""
import os, re, json, argparse, time, warnings
warnings.filterwarnings("ignore")

import tinker
import tinker.types as T
from transformers import AutoTokenizer
from datasets import load_dataset
import wandb

EXP = "qp4-truncation"
DATE = "20260704"
RESULTS_DIR = "/home/claude/tinker-rl-lab/experiments/results/quick_20260704"
TSV_PATH = os.path.join(RESULTS_DIR, "qp4_truncation.tsv")
MANIFEST_PATH = os.path.join(RESULTS_DIR, "qp4_truncation_manifest.json")

SYSTEM_PROMPT = ("You are a math assistant. Solve the problem step by step, "
                 "then give your final numerical answer inside \\boxed{}.")


def grade(response: str, answer: str) -> float:
    """Binary exact-match on \\boxed{} or last number (same as grpo_gsm8k_base)."""
    response = response.strip()
    boxed = re.findall(r'\\boxed\{([^}]+)\}', response)
    for b in boxed:
        b_clean = b.strip().replace(",", "").replace(" ", "")
        try:
            if abs(float(b_clean) - float(answer)) < 0.01:
                return 1.0
        except Exception:
            if b_clean == answer:
                return 1.0
    all_nums = re.findall(r'[-+]?\d[\d,]*\.?\d*', response)
    if all_nums:
        last = all_nums[-1].replace(",", "")
        try:
            if abs(float(last) - float(answer)) < 0.01:
                return 1.0
        except Exception:
            pass
    return 0.0


def load_heldout(n: int):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    examples = []
    for row in ds:
        m = re.search(r'####\s*([\-\d,\.]+)', row["answer"])
        if not m:
            continue
        examples.append((row["question"], m.group(1).replace(",", "").strip()))
        if len(examples) >= n:
            break
    return examples


def build_prompt_ids(tok, question: str):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}]
    try:
        out = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        if hasattr(out, "input_ids"):
            out = out["input_ids"]
        if out and isinstance(out[0], list):
            out = out[0]
        return list(out)
    except Exception:
        text = (f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n")
        return tok.encode(text, add_special_tokens=False)


def append_tsv(model, cap, n, acc, mean_len):
    new = not os.path.exists(TSV_PATH)
    with open(TSV_PATH, "a") as f:
        if new:
            f.write("model\tcap\tn\taccuracy\tmean_len\n")
        f.write(f"{model}\t{cap}\t{n}\t{acc:.4f}\t{mean_len:.1f}\n")


def write_manifest(args, model):
    manifest = {
        "exp": EXP,
        "date": DATE,
        "model": model,
        "loss_form": "none — sampling-only evaluation, no training loss",
        "ref_policy_kl_handling": "n/a — no training, no reference policy or KL term",
        "sampler_backend_precision": "unknown/closed-stack (Tinker hosted sampler; precision not exposed)",
        "per_step_zvf_path": "n/a — no RL steps; TSV rows are cumulative eval batches per cap",
        "group_size_schedule": "n/a — fixed num_samples=1 per problem at every cap",
        "heldout_split": f"openai/gsm8k main test[:{args.n}] (prior RL runs trained on train split only)",
        "decontamination_notes": ("GSM8K test is public; pretraining contamination of the base model is "
                                  "unknown/closed-stack. Zero overlap with our RL training prompts, which "
                                  "come exclusively from the GSM8K train split."),
        "sampling": {"temperature": args.temperature, "top_p": args.top_p,
                     "caps": args.caps, "n_problems": args.n},
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--caps", type=int, nargs="+", default=[64, 128, 256, 512])
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--chunk", type=int, default=25)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.n, args.caps, args.chunk = 4, [64], 4

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"[{EXP}] loading tokenizer + data for {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    examples = load_heldout(args.n)
    prompt_ids = [build_prompt_ids(tok, q) for q, _ in examples]
    print(f"[{EXP}] {len(examples)} held-out problems", flush=True)

    svc = tinker.ServiceClient(base_url=None)
    sc = svc.create_sampling_client(base_model=args.model)

    run = None
    if not args.smoke:
        run = wandb.init(project="tinker-new-research", name=f"{EXP}-{DATE}",
                         config=vars(args))
        print(f"[{EXP}] wandb: {run.entity}/{run.project}/{run.id}", flush=True)
        write_manifest(args, args.model)

    t0 = time.time()
    for cap in args.caps:
        sp = T.SamplingParams(max_tokens=cap, temperature=args.temperature,
                              top_p=args.top_p)
        correct, total, tok_sum = 0.0, 0, 0
        for start in range(0, len(examples), args.chunk):
            chunk_ids = prompt_ids[start:start + args.chunk]
            chunk_ans = [a for _, a in examples[start:start + args.chunk]]
            futs = [sc.sample(T.ModelInput.from_ints(ids), num_samples=1,
                              sampling_params=sp) for ids in chunk_ids]
            for fut, ans in zip(futs, chunk_ans):
                seq = fut.result().sequences[0]
                toks = list(seq.tokens)
                text = tok.decode(toks, skip_special_tokens=True)
                correct += grade(text, ans)
                tok_sum += len(toks)
                total += 1
            acc = correct / total
            mean_len = tok_sum / total
            append_tsv(args.model, cap, total, acc, mean_len)
            if run:
                run.log({"cap": cap, "n": total, "accuracy": acc,
                         "mean_len": mean_len, f"acc_cap{cap}": acc,
                         "elapsed_s": time.time() - t0})
            print(f"[{EXP}] cap={cap} n={total}/{len(examples)} "
                  f"acc={acc:.3f} mean_len={mean_len:.1f}", flush=True)
        print(f"[{EXP}] DONE cap={cap}: acc={correct/total:.4f}", flush=True)

    if run:
        run.finish()
    print(f"[{EXP}] all caps done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
