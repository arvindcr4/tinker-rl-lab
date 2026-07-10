#!/usr/bin/env python3
"""
GSM8K Test Set Evaluation Script
=================================
This script evaluates trained checkpoints on the held-out GSM8K test set.

Usage:
    python evaluate_gsm8k_test.py --checkpoint_path <path> --model_name <model>
    
Or with Tinker:
    python evaluate_gsm8k_test.py --use_tinker --run_id <run_id>

Results will be saved to gsm8k_test_results.json
"""

import argparse
import json
import random
import re
import time
from typing import List, Tuple, Optional
from tqdm import tqdm

def bootstrap_accuracy_ci(correct: int, total: int, seed: int, n_bootstrap: int = 2000) -> Tuple[float, float]:
    """Bootstrap a 95% CI for exact-match accuracy from Bernoulli outcomes."""
    if total <= 0:
        return 0.0, 0.0

    rng = random.Random(seed)
    outcomes = [1] * correct + [0] * (total - correct)
    samples = []
    for _ in range(n_bootstrap):
        draw = [outcomes[rng.randrange(total)] for _ in range(total)]
        samples.append(sum(draw) / total)
    samples.sort()
    lower = samples[int(0.025 * (n_bootstrap - 1))]
    upper = samples[int(0.975 * (n_bootstrap - 1))]
    return lower, upper

def setup_argparse():
    parser = argparse.ArgumentParser(description="Evaluate on GSM8K test set")
    parser.add_argument("--checkpoint_path", type=str, help="Local path to checkpoint")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B", help="HuggingFace model name")
    parser.add_argument("--use_tinker", action="store_true", help="Use Tinker API")
    parser.add_argument("--run_id", type=str, help="Tinker run ID for checkpoint")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation")
    parser.add_argument("--n_samples", type=int, default=1, help="Samples per question")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: greedy evaluation)")
    parser.add_argument("--do_sample", action="store_true", help="Enable stochastic sampling instead of greedy decoding")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens to generate (2048 for CoT models)")
    parser.add_argument("--output", type=str, default="gsm8k_test_results.json", help="Output file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test examples")
    parser.add_argument("--split", type=str, default="test", choices=["test"], help="Dataset split to evaluate (locked to held-out test for integrity)")
    return parser

def extract_answer(text: str) -> Optional[str]:
    """Extract final numeric answer from response.

    Handles multiple output formats:
    - GSM8K standard: #### <number>
    - LaTeX boxed: \\boxed{<number>}
    - Explicit statement: "the answer is <number>"
    - Qwen3 <think> reasoning: looks for the last numeric conclusion
    """
    # Strip <think>...</think> wrapper if present — answer is usually restated after
    # or at the end of reasoning
    clean = text
    # If there's content after </think>, prefer that
    think_end = text.find('</think>')
    if think_end != -1:
        after_think = text[think_end + 8:].strip()
        if after_think:
            clean = after_think  # Use post-thinking answer

    # GSM8K format: answer ends with "#### <number>"
    match = re.search(r'####\s*(-?\d[\d,]*\.?\d*)', clean)
    if match:
        return match.group(1).replace(',', '')

    boxed = re.search(r'\\boxed\{\s*(-?\d[\d,]*\.?\d*)\s*\}', clean)
    if boxed:
        return boxed.group(1).replace(',', '')

    # "the answer is X" / "answer: X" patterns
    explicit = re.search(r'(?i)(?:the\s+)?(?:final\s+)?answer\s*(?:is|:|=)\s*\$?\s*(-?\d[\d,]*\.?\d*)', clean)
    if explicit:
        return explicit.group(1).replace(',', '')

    # "= X cups/dollars/etc" at end of reasoning — take the last "= <number>"
    equals_matches = re.findall(r'=\s*(-?\d[\d,]*\.?\d*)\s*(?:cups?|dollars?|\$|%|items?|people|hours?|minutes?|days?|miles?|meters?|kg|lbs?|pounds?|gallons?|liters?|feet|inches|years?|months?|weeks?|seconds?|cents?|\.?\s*$)', clean, re.I)
    if equals_matches:
        return equals_matches[-1].replace(',', '')

    # Last resort: search the FULL text (including <think>) for the patterns above
    if clean != text:
        # Try "the answer is X" in the thinking section
        explicit_full = re.search(r'(?i)(?:the\s+)?(?:final\s+)?answer\s*(?:is|:|=|would be)\s*\$?\s*(-?\d[\d,]*\.?\d*)', text)
        if explicit_full:
            return explicit_full.group(1).replace(',', '')

        # "So, X cups/dollars" pattern common in Qwen3 reasoning
        so_pattern = re.findall(r'(?i)(?:so|therefore|thus|hence),?\s*(?:the\s+)?(?:answer\s+is\s+)?(?:it\s+(?:is|would be)\s+)?\$?\s*(-?\d[\d,]*\.?\d*)\s*(?:cups?|dollars?|\$|%|items?|people|hours?|minutes?|days?|miles?|meters?|kg|lbs?|pounds?|gallons?|liters?|feet|inches|years?|months?|weeks?|seconds?|cents?)', text)
        if so_pattern:
            return so_pattern[-1].replace(',', '')

    return None

def normalize_number(s: str) -> str:
    """Normalize numbers for comparison."""
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return str(f)
    except:
        return s

def load_model(args):
    """Load model and tokenizer based on arguments."""
    from transformers import AutoTokenizer
    
    if args.use_tinker:
        import tinker as T
        import os
        os.environ["TINKER_API_KEY"] = os.environ.get("TINKER_API_KEY", "")
        
        svc = T.ServiceClient()
        if args.run_id:
            # Run IDs from training logs include :train:0 suffix for the training shard
            run_id = args.run_id
            if ":train:" not in run_id:
                run_id = f"{run_id}:train:0"
            checkpoint_path = f"tinker://{run_id}/sampler_weights/final"
            client = svc.create_sampling_client(model_path=checkpoint_path)
        else:
            client = svc.create_sampling_client(base_model=args.model_name)
        # Load tokenizer for encoding/decoding
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        return client, tokenizer, True  # (model_or_client, tokenizer, is_tinker)
    else:
        from transformers import AutoModelForCausalLM
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model on {device}...")
        
        model_source = args.checkpoint_path or args.model_name
        tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        
        return model, tokenizer, False

def generate_with_tinker(client, tokenizer, prompt: str, args) -> str:
    """Generate using Tinker API."""
    from tinker import ModelInput, SamplingParams

    input_ids = tokenizer.encode(prompt)
    
    model_input = ModelInput.from_ints(input_ids)
    sp = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    
    result = client.sample(model_input, num_samples=args.n_samples, sampling_params=sp).result()
    
    if result.sequences:
        tokens = list(result.sequences[0].tokens)
        response = tokenizer.decode(tokens, skip_special_tokens=True)
        return response
    return ""

def generate_with_hf(model, tokenizer, prompt: str, args) -> str:
    """Generate using HuggingFace model."""
    inputs = tokenizer(prompt, return_tensors="pt")
    if hasattr(model, 'device'):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    generation_kwargs = {
        **inputs,
        "max_new_tokens": args.max_tokens,
        "do_sample": args.do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if args.do_sample:
        generation_kwargs["temperature"] = args.temperature if args.temperature > 0 else 0.7

    outputs = model.generate(**generation_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][prompt_len:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response


def set_seed(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def extract_question_and_answer(example):
    """Extract question and ground truth from GSM8K example."""
    question = example["question"]
    answer = example["answer"]
    
    gt_match = re.search(r'####\s*(-?\d+\.?\d*)', answer)
    ground_truth = gt_match.group(1) if gt_match else None
    
    return question, ground_truth

def evaluate_model(args):
    """Main evaluation loop."""
    from datasets import load_dataset
    
    print("Loading GSM8K dataset...")
    gsm8k = load_dataset("openai/gsm8k", "main")
    test_data = gsm8k[args.split]
    
    if args.limit:
        test_data = test_data.select(range(args.limit))
    
    print(f"Evaluating on {len(test_data)} test examples...")
    
    # Load model
    model_or_client, tokenizer, is_tinker = load_model(args)
    
    results = {
        "schema_version": 2,
        "evaluation_status": "completed",
        "config": {
            "model": args.model_name if not args.run_id else f"tinker://{args.run_id}",
            "model_source": args.checkpoint_path or (f"tinker://{args.run_id}/sampler_weights/final" if args.run_id else args.model_name),
            "dataset": "openai/gsm8k",
            "dataset_config": "main",
            "dataset_split": args.split,
            "n_samples": args.n_samples,
            "temperature": args.temperature,
            "do_sample": args.do_sample,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "test_size": len(test_data),
        },
        "summary": {
            "correct": 0,
            "incorrect": 0,
            "errors": 0,
            "attempted": 0,
            "accuracy": 0.0,
            "accuracy_percent": "0.0%",
        },
        "examples": [],
    }
    
    prompt_template = "<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    
    for i, example in enumerate(tqdm(test_data, desc="Evaluating")):
        question, ground_truth = extract_question_and_answer(example)
        
        if ground_truth is None:
            results["summary"]["errors"] += 1
            continue
        
        try:
            if is_tinker:
                prompt = prompt_template.format(question=question)
                response = generate_with_tinker(model_or_client, tokenizer, prompt, args)
            else:
                prompt = prompt_template.format(question=question)
                response = generate_with_hf(model_or_client, tokenizer, prompt, args)
            
            predicted = extract_answer(response)
            
            if predicted and normalize_number(predicted) == normalize_number(ground_truth):
                results["summary"]["correct"] += 1
                status = "correct"
            else:
                results["summary"]["incorrect"] += 1
                status = "incorrect"
            
            results["examples"].append({
                "idx": i,
                "question": question[:100] + "..." if len(question) > 100 else question,
                "predicted": predicted,
                "ground_truth": ground_truth,
                "status": status,
            })
            
        except Exception as e:
            results["summary"]["errors"] += 1
            results["examples"].append({
                "idx": i,
                "error": str(e),
                "status": "error",
            })

    # Calculate metrics
    total = results["summary"]["correct"] + results["summary"]["incorrect"]
    results["summary"]["attempted"] = total
    if total > 0:
        results["summary"]["accuracy"] = results["summary"]["correct"] / total
        ci_low, ci_high = bootstrap_accuracy_ci(results["summary"]["correct"], total, args.seed)
        results["summary"]["accuracy_ci_95"] = [ci_low, ci_high]
        results["summary"]["accuracy_ci_95_percent"] = [f"{ci_low:.1%}", f"{ci_high:.1%}"]
    else:
        results["summary"]["accuracy"] = 0.0
        results["summary"]["accuracy_ci_95"] = [0.0, 0.0]
        results["summary"]["accuracy_ci_95_percent"] = ["0.0%", "0.0%"]
        if results["summary"]["errors"] > 0:
            results["evaluation_status"] = "failed"
            results["failure_reason"] = "No held-out examples were successfully scored; inspect the recorded errors."
    results["summary"]["accuracy_percent"] = f"{results['summary']['accuracy']:.1%}"
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"RESULTS: GSM8K Test Set Evaluation")
    print(f"{'='*50}")
    print(f"Model: {results['config']['model']}")
    print(f"Test Set Size: {len(test_data)}")
    print(f"Status: {results['evaluation_status']}")
    print(f"Correct: {results['summary']['correct']}")
    print(f"Incorrect: {results['summary']['incorrect']}")
    print(f"Errors: {results['summary']['errors']}")
    print(f"Accuracy: {results['summary']['accuracy']:.2%}")
    print(f"95% bootstrap CI: [{results['summary']['accuracy_ci_95_percent'][0]}, {results['summary']['accuracy_ci_95_percent'][1]}]")
    print(f"{'='*50}")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")
    
    return results

def main():
    parser = setup_argparse()
    args = parser.parse_args()
    set_seed(args.seed)
    
    results = evaluate_model(args)

    # Print for easy copying
    print(f"\n## GSM8K Test Accuracy: {results['summary']['accuracy']:.2%}")
    print(f"## 95% bootstrap CI: [{results['summary']['accuracy_ci_95_percent'][0]}, {results['summary']['accuracy_ci_95_percent'][1]}]")
    print(f"# Correct: {results['summary']['correct']}/{results['summary']['attempted']}")

if __name__ == "__main__":
    main()
