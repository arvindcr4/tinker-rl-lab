"""Final-number fallback must not award an expression's component as its value."""

from platform_tinker.tinkerrl.grpo import (
    ExactMathReward,
    MathReward,
    TrainingExample,
    _exact_math_box_spans,
)


def score(response, target):
    return ExactMathReward().score(response, TrainingExample(prompt="q", target=target))


def test_fraction_expression_and_identifier_fragments_are_not_answers():
    for response, target in (
        (r"\boxed{1/2}", "2"),
        (r"\boxed{\frac{1}{2}}", "2"),
        (r"\frac{1}{2}", "2"),
        ("1/2", "2"),
        ("1 / 2", "2"),
        ("10^2", "2"),
        ("10 ** 2", "2"),
        ("1 + 2", "2"),
        ("1 -2", "-2"),
        ("(3, 2)", "2"),
        ("[3, 2]", "2"),
        ("sqrt(2)", "2"),
        (r"\boxed{5} for Q2", "2"),
        ("version2", "2"),
        ("x_2", "2"),
        ("2x", "2"),
        ("1.2.3", "0.3"),
        ("12,34", "1234"),
        (r"\boxed{12,34}", "1234"),
    ):
        assert score(response, target) == 0.0, response


def test_incomplete_boxes_cannot_leak_numbers_or_rescue_earlier_reasoning():
    for response in (r"\boxed{2", r"2 then \boxed{99", r"\boxed{\frac{1}{2}"):
        assert score(response, "2") == 0.0, response
    assert score(r"42 then \boxed{99}", "42") == 0.0
    assert score(r"42 then \boxed{}", "42") == 0.0
    assert score(r"42 then \boxed{1/2}", "42") == 0.0


def test_boxed_and_later_bare_answer_policy_is_preserved():
    example = TrainingExample(prompt="q", target="42")
    response = r"\boxed{99} and the final answer is 42"
    assert ExactMathReward().score(response, example) == 0.0
    assert MathReward().score(response, example) == 0.3
    assert score(r"\boxed{42} then Q99", "42") == 1.0
    assert score(r"\boxed{99} then Q42", "42") == 0.0


def test_common_bare_numeric_formats_still_match():
    for response, target in (
        ("answer = 42", "42"),
        ("42.", "42"),
        (r"\boxed{{42}}", "42"),
        ("answer -2.", "-2"),
        ("(-2)", "-2"),
        ("answer ( 2 )", "2"),
        (r"answer \(42\)", "42"),
        (r"answer \[42\]", "42"),
        ("$42$", "42"),
        ("1e-3", "0.001"),
        (".5", "0.5"),
        ("1,234.5", "1234.5"),
        (r"\boxed{1,234}", "1234"),
        (r"\boxed {42}", "42"),
    ):
        assert score(response, target) == 1.0, response


def test_balanced_nested_boxes_are_complete_without_symbolic_evaluation():
    response = r"\boxed{\frac{1}{2}}"
    assert _exact_math_box_spans(response) == [(0, len(response), r"\frac{1}{2}")]
    assert score(response, r"\frac{1}{2}") == 1.0
    assert score(response, "0.5") == 1.0  # v3 exact rational normalization
    escaped = r"\boxed{\text{a\}b}}"
    assert _exact_math_box_spans(escaped)[0][2] == r"\text{a\}b}"


def test_deep_or_many_boxes_are_scanned_without_recursion():
    deep = r"\boxed{" + "{" * 10000 + "2" + "}" * 10001
    assert len(_exact_math_box_spans(deep)) == 1
    assert score(deep, "2") == 1.0
    repeated = r"\boxed{99} " * 1000 + "42"
    assert score(repeated, "42") == 0.0  # final box remains authoritative


def test_ratio_comparison_and_operator_components_are_not_bare_answers():
    for response, target in (
        ("12:30", "30"),
        ("12 : 30", "30"),
        ("1<2", "2"),
        ("1<=2", "2"),
        ("1>=2", "2"),
        ("1!=2", "2"),
        ("1==2", "2"),
        ("1≤2", "2"),
        ("2×3", "3"),
        ("12%5", "5"),
        ("3!", "3"),
        ("2|3", "3"),
    ):
        assert score(response, target) == 0.0, response
    # A label colon, assignment/equation equals sign, and percent unit retain
    # their established numeric-answer interpretation.
    for response in ("answer: 42", "answer = 42"):
        assert score(response, "42") == 1.0


def test_nested_numeric_wrappers_and_sentence_punctuation_remain_valid():
    for response in ("(42).", "((42))", "( (42) ).", r"\((42)\)."):
        assert score(response, "42") == 1.0, response
    assert score("sqrt((2))", "2") == 0.0
    assert score("(42).3", "0.3") == 0.0
    assert score("(" * 10000 + "42" + ")" * 10000 + ".", "42") == 1.0


def test_incidental_suffixes_are_not_authoritative_answers():
    for response in ("42, thanks", "(42). Done", "42%", "Reasoning took 42 steps"):
        assert score(response, "42") == 0.0
