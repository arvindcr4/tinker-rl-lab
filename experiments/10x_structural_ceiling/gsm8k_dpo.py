"""
Block E: DPO baseline on GSM8K using tinker_cookbook preference recipe.

Creates preference pairs from GSM8K:
- Chosen: ground-truth chain-of-thought + correct answer
- Rejected: same CoT with corrupted final answer

Calls train_dpo.main directly with a custom GSM8K comparison builder.
"""

import re
import random
import datasets
import chz

from tinker_cookbook.preference.preference_datasets import (
    ComparisonDatasetBuilder,
    Comparison,
    LabeledComparison,
)
from tinker_cookbook.preference.dpo_datasets import DPODatasetBuilderFromComparisons
from tinker_cookbook.preference import train_dpo
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook import checkpoint_utils


def corrupt_answer(answer: str) -> str:
    """Create a plausible but wrong answer by perturbing the number."""
    num_match = re.search(r"[\d,]+\.?\d*", answer.replace(",", ""))
    if not num_match:
        return answer + " (wrong)"
    num = float(num_match.group().replace(",", ""))
    offset = random.choice([1, 2, 3, 5, 10, -1, -2, -3, -5])
    wrong_num = int(num + offset) if num == int(num) else round(num + offset, 2)
    if wrong_num < 0:
        wrong_num = abs(wrong_num)
    return str(wrong_num)


def extract_gsm8k_answer(solution: str) -> str:
    """Extract the final numeric answer from a GSM8K solution."""
    match = re.search(r"####\s*(.+)", solution)
    return match.group(1).strip() if match else ""


@chz.chz
class GSM8KComparisonBuilder(ComparisonDatasetBuilder):
    """GSM8K preference pair builder for DPO training."""

    test_size: int = 200
    seed: int = 42

    def get_train_and_test_datasets(self):
        dataset = datasets.load_dataset("openai/gsm8k", "main")
        train = dataset["train"].shuffle(seed=self.seed)
        test = dataset["test"].shuffle(seed=self.seed).select(
            range(min(self.test_size, len(dataset["test"])))
        )
        return train, test

    def example_to_labeled_comparison(self, example: dict) -> LabeledComparison | None:
        question = example["question"]
        solution = example["answer"]

        correct_answer = extract_gsm8k_answer(solution)
        if not correct_answer:
            return None

        wrong_answer = corrupt_answer(correct_answer)
        rejected_text = re.sub(r"####\s*.+", f"#### {wrong_answer}", solution)

        system_prompt = (
            "You are a helpful math tutor. Solve the problem step by step "
            "and give the final answer after ####."
        )

        prompt_conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        completion_a = [{"role": "assistant", "content": solution}]
        completion_b = [{"role": "assistant", "content": rejected_text}]

        return LabeledComparison(
            comparison=Comparison(
                prompt_conversation=prompt_conversation,
                completion_A=completion_a,
                completion_B=completion_b,
            ),
            label="A",
        )


def main():
    model_name = "Qwen/Qwen3-8B"
    renderer_name = checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default(
        model_name=model_name,
        explicit_renderer_name="qwen3",
        load_checkpoint_path=None,
        base_url=None,
    )

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=2048,
        batch_size=256,
    )

    dataset_builder = DPODatasetBuilderFromComparisons(
        common_config=common_config,
        comparison_builder=GSM8KComparisonBuilder(),
    )

    config = train_dpo.Config(
        log_path="./checkpoints/10x/gsm8k_qwen8b_dpo/",
        model_name=model_name,
        dataset_builder=dataset_builder,
        renderer_name=renderer_name,
        learning_rate=2e-5,
        lr_schedule="linear",
        dpo_beta=0.1,
        lora_rank=32,
        max_steps=50,
        evaluator_builders=[],
        wandb_project="tinker-structural-ceiling",
        wandb_name="block_e_dpo_qwen8b_gsm8k",
    )

    print("=" * 60)
    print("  Block E: DPO Baseline — Qwen3-8B on GSM8K")
    print(f"  Model: {model_name}")
    print(f"  LR: {config.learning_rate}, Beta: {config.dpo_beta}")
    print(f"  Max steps: {config.max_steps}, LoRA rank: {config.lora_rank}")
    print("=" * 60)

    train_dpo.main(config)


if __name__ == "__main__":
    main()
