"""
Upload Model Checkpoints and Datasets to Hugging Face Hub
==========================================================
Uploads trained model checkpoints with proper model cards.

Usage:
    python huggingface/upload_to_hub.py \
        --model-path ./results/trl_grpo_math/seed_42/final \
        --repo-id pes-llm-research/tinker-rl-grpo-llama3-1b-arithmetic \
        --base-model meta-llama/Llama-3.2-1B \
        --method GRPO \
        --dataset arithmetic

    python huggingface/upload_to_hub.py \
        --model-path ./results/trl_gsm8k_math/seed_42/final \
        --repo-id pes-llm-research/tinker-rl-grpo-llama3-8b-gsm8k \
        --base-model meta-llama/Llama-3.1-8B-Instruct \
        --method GRPO \
        --dataset gsm8k
"""

import os
import json
import argparse
from pathlib import Path


def create_model_card(args, metrics: dict = None) -> str:
    """Generate a model card from template and arguments."""
    template_path = os.path.join(os.path.dirname(__file__), "MODEL_CARD_TEMPLATE.md")

    with open(template_path, "r") as f:
        template = f.read()

    replacements = {
        "{MODEL_NAME}": args.repo_id.split("/")[-1],
        "{BASE_MODEL}": args.base_model,
        "{DATASET}": args.dataset,
        "{METHOD}": args.method,
        "{FRAMEWORK}": args.framework or "TRL",
        "{LORA_RANK}": str(args.lora_rank),
        "{LEARNING_RATE}": str(args.learning_rate),
        "{BATCH_SIZE}": str(args.batch_size),
        "{NUM_STEPS}": str(args.num_steps),
        "{GPU_TYPE}": args.gpu_type or "NVIDIA A100",
        "{TRAINING_TIME}": args.training_time or "N/A",
        "{TOTAL_COMPUTE}": args.total_compute or "N/A",
        "{PREPROCESSING_DESCRIPTION}": args.preprocessing or "Standard tokenization with chat template",
        "{ACCURACY}": str(metrics.get("accuracy", "N/A")) if metrics else "N/A",
        "{BENCHMARK_1}": args.dataset.upper(),
        "{METRIC_1}": "Accuracy",
        "{SCORE_1}": str(metrics.get("score_mean_se", "N/A")) if metrics else "N/A",
        "{CI_1}": str(metrics.get("ci_95", "N/A")) if metrics else "N/A",
    }

    card = template
    for key, value in replacements.items():
        card = card.replace(key, value)

    return card


def upload_model(args):
    """Upload model to Hugging Face Hub."""
    from huggingface_hub import HfApi, create_repo

    api = HfApi()

    # Create repo if it doesn't exist
    try:
        create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            private=not args.public,
            exist_ok=True,
        )
        print(f"Repository {args.repo_id} ready")
    except Exception as e:
        print(f"Note: {e}")

    # Load metrics if available
    metrics = None
    metrics_path = os.path.join(args.model_path, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

    # Generate and save model card
    model_card = create_model_card(args, metrics)
    readme_path = os.path.join(args.model_path, "README.md")
    with open(readme_path, "w") as f:
        f.write(model_card)
    print("Model card generated")

    # Upload all files
    api.upload_folder(
        folder_path=args.model_path,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=f"Upload {args.method} model trained on {args.dataset}",
    )

    print(f"Model uploaded to https://huggingface.co/{args.repo_id}")


def upload_dataset(args):
    """Upload benchmark dataset to Hugging Face Hub."""
    from huggingface_hub import HfApi, create_repo

    api = HfApi()

    create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=not args.public,
        exist_ok=True,
    )

    api.upload_folder(
        folder_path=args.dataset_path,
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Upload benchmark dataset",
    )

    print(f"Dataset uploaded to https://huggingface.co/datasets/{args.repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload to Hugging Face Hub")
    subparsers = parser.add_subparsers(dest="command")

    # Model upload
    model_parser = subparsers.add_parser("model", help="Upload a model")
    model_parser.add_argument("--model-path", required=True)
    model_parser.add_argument("--repo-id", required=True)
    model_parser.add_argument("--base-model", required=True)
    model_parser.add_argument("--method", required=True, choices=["GRPO", "DPO", "SFT", "Distillation"])
    model_parser.add_argument("--dataset", required=True)
    model_parser.add_argument("--framework", default="TRL")
    model_parser.add_argument("--lora-rank", type=int, default=32)
    model_parser.add_argument("--learning-rate", type=float, default=1e-4)
    model_parser.add_argument("--batch-size", type=int, default=4)
    model_parser.add_argument("--num-steps", type=int, default=20)
    model_parser.add_argument("--gpu-type", default="NVIDIA A100")
    model_parser.add_argument("--training-time", default=None)
    model_parser.add_argument("--total-compute", default=None)
    model_parser.add_argument("--preprocessing", default=None)
    model_parser.add_argument("--public", action="store_true")

    # Dataset upload
    dataset_parser = subparsers.add_parser("dataset", help="Upload a dataset")
    dataset_parser.add_argument("--dataset-path", required=True)
    dataset_parser.add_argument("--repo-id", required=True)
    dataset_parser.add_argument("--public", action="store_true")

    args = parser.parse_args()

    if args.command == "model":
        upload_model(args)
    elif args.command == "dataset":
        upload_dataset(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
