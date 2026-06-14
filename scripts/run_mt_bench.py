#!/usr/bin/env python3
"""
MT-Bench Evaluation Harness Integration for TinkerRL.
This script runs MT-Bench on an RL-trained model using FastChat's llm_judge.
It automates answer generation, LLM judgment, and result summarization.
"""

import argparse
import subprocess
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Run MT-Bench on RL-trained models")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the RL-trained model (HuggingFace format)")
    parser.add_argument("--model-id", type=str, required=True, help="A unique ID for the model (used for saving results)")
    parser.add_argument("--judge-model", type=str, default="gpt-4", help="Judge model to use for evaluation (default: gpt-4)")
    parser.add_argument("--parallel", type=int, default=4, help="Number of parallel API calls for judgment")
    parser.add_argument("--max-new-token", type=int, default=1024, help="Max new tokens for answer generation")
    parser.add_argument("--num-gpus-per-model", type=int, default=1, help="Number of GPUs to use for answer generation")
    parser.add_argument("--num-gpus-total", type=int, default=1, help="Total number of GPUs to use for parallel generation")
    args = parser.parse_args()

    print(f"=== Starting MT-Bench Evaluation for {args.model_id} ===")

    # Step 1: Generate answers
    print(f"\n[1/3] Generating answers for MT-Bench questions...")
    cmd_gen = [
        sys.executable, "-m", "fastchat.llm_judge.gen_model_answer",
        "--model-path", args.model_path,
        "--model-id", args.model_id,
        "--max-new-token", str(args.max_new_token),
        "--num-gpus-per-model", str(args.num_gpus_per_model),
        "--num-gpus-total", str(args.num_gpus_total)
    ]
    try:
        subprocess.run(cmd_gen, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during answer generation: {e}")
        sys.exit(1)

    # Step 2: Generate judgments
    print(f"\n[2/3] Generating judgments using {args.judge_model}...")
    # NOTE: To use GPT-4 as a judge, OPENAI_API_KEY must be set in the environment.
    if "OPENAI_API_KEY" not in os.environ and args.judge_model.startswith("gpt-"):
        print("Warning: OPENAI_API_KEY environment variable is not set. The judgment step may fail.")

    cmd_judge = [
        sys.executable, "-m", "fastchat.llm_judge.gen_judgment",
        "--model-list", args.model_id,
        "--judge-model", args.judge_model,
        "--parallel", str(args.parallel)
    ]
    try:
        subprocess.run(cmd_judge, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during judgment generation: {e}")
        sys.exit(1)

    # Step 3: Show results
    print(f"\n[3/3] Displaying MT-Bench results for {args.model_id}...")
    cmd_show = [
        sys.executable, "-m", "fastchat.llm_judge.show_result",
        "--model-list", args.model_id,
        "--judge-model", args.judge_model
    ]
    try:
        subprocess.run(cmd_show, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error displaying results: {e}")
        sys.exit(1)

    print("\n=== MT-Bench Evaluation Complete ===")

if __name__ == "__main__":
    main()
