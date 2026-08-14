from platform_tinker.tinkerrl.grpo import TrainingExample
from platform_tinker.tinkerrl.pavlov_portfolio_eval import select_examples


def test_select_examples_is_balanced_deterministic_and_exact():
    rows = [
        TrainingExample(prompt=f"a{i}", metadata={"suite_id": "api_bank_rlvr_train"})
        for i in range(4)
    ] + [
        TrainingExample(prompt=f"s{i}", metadata={"suite_id": "swe_gym_train"})
        for i in range(4)
    ]

    selected = select_examples(rows, per_suite=2)

    assert [row.prompt for row in selected] == ["a0", "a1", "s0", "s1"]
