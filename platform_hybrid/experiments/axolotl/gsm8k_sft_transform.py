"""
GSM8K SFT data transform for Axolotl.

Formats GSM8K examples as question → chain-of-thought + \\boxed{answer}
to match the GRPO experiment's evaluation format.
"""


def gsm8k_sft_transform(cfg, *args, **kwargs):
    """Return (transform_fn, extra_kwargs) for Axolotl dataset loading."""

    question_suffix = " Provide a numerical answer without units, written inside \\boxed{}."

    def transform_fn(example, tokenizer=None):
        question = example["question"]
        full_solution = example["answer"]

        # GSM8K format: "reasoning steps\n#### numeric_answer"
        parts = full_solution.split("####")
        reasoning = parts[0].strip()
        numeric_answer = parts[-1].strip().replace(",", "")

        # Format to match GRPO eval: reasoning + \boxed{answer}
        formatted = f"{reasoning}\n\\boxed{{{numeric_answer}}}"

        return {
            "text": (
                f"Question: {question}{question_suffix}\n\n"
                f"Answer: {formatted}"
            ),
        }

    return transform_fn, {"remove_columns": ["question", "answer"]}
