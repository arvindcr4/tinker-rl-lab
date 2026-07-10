"""Deepened GRPO run module.

Consolidates the copy-pasted GRPO training loop that lived across
``grpo_exp_*.py``, ``grpo_100_*.py``, ``grpo_gsm8k_base.py``, and
``grpo_tooluse_tinker.py``.  The module exposes a small interface:

* :class:`GRPOConfig` — all experiment knobs in one value object.
* :class:`TrainingExample` — one prompt + target pair.
* :class:`DatasetAdapter` / :class:`RewardAdapter` — the two seams.
* :func:`run_grpo` — the deepened loop that wires them together.

The caller (CLI, test, or notebook) supplies the adapters; the module
owns the seed loop, sampling, advantage computation, loss call, optimizer
step, and checkpoint cadence.
"""

from __future__ import annotations

import json
import os
import random
import re
import warnings
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

import torch

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TrainingExample:
    """One training example: a prompt plus a task-specific target."""

    prompt: str
    target: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class InMemoryDataset:
    """A minimal dataset adapter backed by two lists of examples."""

    train: Sequence[TrainingExample]
    test: Sequence[TrainingExample] = ()

    def train_examples(self) -> Sequence[TrainingExample]:
        return self.train

    def test_examples(self) -> Sequence[TrainingExample]:
        return self.test


@dataclass(slots=True)
class GRPOConfig:
    """All knobs for a GRPO experiment in one value object."""

    name: str
    model: str = "Qwen/Qwen3-8B"
    lora_rank: int = 32
    steps: int = 200
    group_size: int = 8
    batch_size: int = 4
    lr: float = 3e-5
    temperature: float = 0.8
    top_p: float = 0.95
    max_prompt_tokens: int = 1024
    max_response_tokens: int = 512
    save_every: Optional[int] = None
    seed: int = 42
    num_seeds: int = 1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    evaluate_heldout: bool = False
    base_url: Optional[str] = None

    def effective_save_every(self) -> int:
        return self.save_every or max(self.steps // 4, 10)


@dataclass(slots=True)
class GRPORunResult:
    """Outcome of one seed inside :func:`run_grpo`."""

    seed: int
    run_id: Optional[str] = None
    sampler_path: Optional[str] = None
    reward_trace: List[float] = field(default_factory=list)
    avg_first5: float = 0.0
    avg_last10: float = 0.0
    peak_reward: float = 0.0
    zero_loss_steps: int = 0
    zero_reward_steps: int = 0
    heldout_reward: Optional[float] = None


# ---------------------------------------------------------------------------
# Protocols — the two seams
# ---------------------------------------------------------------------------


class DatasetAdapter(Protocol):
    """Supplies training and held-out examples."""

    def train_examples(self) -> Sequence[TrainingExample]: ...
    def test_examples(self) -> Sequence[TrainingExample]: ...


class RewardAdapter(Protocol):
    """Scores one completion against the example's target."""

    def score(self, response: str, example: TrainingExample) -> float: ...


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def normalize_rewards(
    rewards: Sequence[float], epsilon: float = 1e-8
) -> List[float]:
    """Group-relative advantage normalization (mean 0, std 1)."""
    n = len(rewards)
    if n == 0:
        return []
    mean_r = sum(rewards) / n
    std_r = (sum((r - mean_r) ** 2 for r in rewards) / n) ** 0.5 + epsilon
    return [(r - mean_r) / std_r for r in rewards]


def make_grpo_loss_fn(
    advantages: Sequence[float],
) -> Callable:
    """Return a Tinker-compatible loss closure bound to ``advantages``."""

    def _loss_fn(data: Any, logprobs_list: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
        losses = []
        for i, logprobs in enumerate(logprobs_list):
            losses.append(-advantages[i] * logprobs.sum())
        if not losses:
            return torch.tensor(0.0), {"grpo_loss": 0.0}
        loss = torch.stack(losses).mean()
        return loss, {"grpo_loss": loss.item()}

    return _loss_fn


def _decode_response(tokenizer: Any, resp: Any) -> str:
    return tokenizer.decode(list(resp.tokens), skip_special_tokens=True)


def _build_datum(prompt_ids: List[int], response_ids: List[int]) -> Any:
    """Build a ``T.Datum`` from prompt + response token ids."""
    import tinker.types as T

    full_ids = prompt_ids + response_ids
    target_ids = full_ids[1:] + [0]
    return T.Datum(
        model_input=T.ModelInput.from_ints(full_ids),
        loss_fn_inputs={
            "target_tokens": T.TensorData(
                data=target_ids, dtype="int64", shape=[len(target_ids)]
            )
        },
    )


def _metric(result: Any, names: Sequence[str], default: float = float("nan")) -> float:
    metrics = getattr(result, "metrics", {}) or {}
    for name in names:
        if name in metrics:
            return metrics[name]
    return default


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------


def _run_one_seed(
    config: GRPOConfig,
    dataset: DatasetAdapter,
    reward: RewardAdapter,
    tokenizer: Any,
    logger: Callable[[str], Any] = print,
) -> GRPORunResult:
    """Execute the GRPO loop for one seed.  Pure enough to unit-test with fakes."""
    import tinker
    import tinker.types as T
    from transformers import AutoTokenizer as _AT  # noqa: F811

    seed = config.seed
    random.seed(seed)
    torch.manual_seed(seed)

    train_examples = list(dataset.train_examples())
    if not train_examples:
        raise ValueError(f"[{config.name}] dataset returned 0 training examples")

    logger(f"[{config.name}] Connecting to Tinker...")
    svc = tinker.ServiceClient(base_url=config.base_url)
    tc = svc.create_lora_training_client(base_model=config.model, rank=config.lora_rank)
    tok = tokenizer if tokenizer is not None else _AT.from_pretrained(
        config.model, trust_remote_code=True
    )
    w0 = tc.save_weights_for_sampler(name=f"{config.name}_seed{seed}_step_0").result()
    sc = tc.create_sampling_client(model_path=w0.path)
    logger(f"[{config.name}] Run: {tc.model_id}")

    loss_fn_template = make_grpo_loss_fn([])
    save_every = config.effective_save_every()
    step_rewards: List[float] = []
    zero_loss_steps = 0
    zero_reward_steps = 0

    for step in range(config.steps):
        batch = random.sample(train_examples, min(config.batch_size, len(train_examples)))
        all_data: List[Any] = []
        all_advs: List[float] = []
        batch_rewards: List[float] = []

        for example in batch:
            prompt_ids = tok.encode(example.prompt, add_special_tokens=False)
            if len(prompt_ids) > config.max_prompt_tokens:
                prompt_ids = prompt_ids[: config.max_prompt_tokens]

            sp = T.SamplingParams(
                max_tokens=config.max_response_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
            )
            responses = sc.sample(
                T.ModelInput.from_ints(prompt_ids),
                num_samples=config.group_size,
                sampling_params=sp,
            ).result()

            rewards = [
                reward.score(_decode_response(tok, resp), example)
                for resp in responses.sequences
            ]
            advs = normalize_rewards(rewards)
            batch_rewards.extend(rewards)

            for resp, adv in zip(responses.sequences, advs):
                resp_ids = list(resp.tokens)
                all_data.append(_build_datum(prompt_ids, resp_ids))
                all_advs.append(adv)

        if not all_data:
            continue

        loss_fn = make_grpo_loss_fn(all_advs)
        result = tc.forward_backward_custom(data=all_data, loss_fn=loss_fn).result()
        tc.optim_step(
            T.AdamParams(
                learning_rate=config.lr,
                beta1=config.beta1,
                beta2=config.beta2,
                eps=config.eps,
            )
        ).result()

        avg = sum(batch_rewards) / len(batch_rewards)
        step_rewards.append(avg)
        loss_val = _metric(result, ["grpo_loss", "loss"])
        if abs(loss_val) < 1e-6:
            zero_loss_steps += 1
        if avg == 0:
            zero_reward_steps += 1

        logger(
            f"[{config.name}] Step {step + 1:3d}/{config.steps}"
            f" | loss={loss_val:.4f} | reward={avg:.3f}"
        )

        if (step + 1) % save_every == 0:
            tc.save_state(name=f"state_seed{seed}_{step + 1}")
            ckpt = tc.save_weights_for_sampler(
                name=f"step_seed{seed}_{step + 1}"
            ).result()
            sc = tc.create_sampling_client(model_path=ckpt.path)
            logger(f"[{config.name}]   -> Checkpoint step_{step + 1}")

    tc.save_state(name=f"seed{seed}_final")
    final = tc.save_weights_for_sampler(name=f"seed{seed}_final").result()

    last10 = step_rewards[-10:] if step_rewards else []
    first5 = step_rewards[:5] if step_rewards else []
    heldout_reward: Optional[float] = None

    if config.evaluate_heldout:
        test_examples = list(dataset.test_examples())
        if test_examples:
            test_rewards: List[float] = []
            for ex in test_examples:
                pid = tok.encode(ex.prompt, add_special_tokens=False)
                if len(pid) > config.max_prompt_tokens:
                    pid = pid[: config.max_prompt_tokens]
                sp = T.SamplingParams(
                    max_tokens=config.max_response_tokens,
                    temperature=0.1,
                    top_p=0.95,
                )
                try:
                    resp = sc.sample(
                        T.ModelInput.from_ints(pid), num_samples=1, sampling_params=sp
                    ).result()
                    text = _decode_response(tok, resp.sequences[0])
                    test_rewards.append(reward.score(text, ex))
                except Exception:
                    continue
            if test_rewards:
                heldout_reward = sum(test_rewards) / len(test_rewards)

    return GRPORunResult(
        seed=seed,
        run_id=getattr(tc, "model_id", None),
        sampler_path=getattr(final, "path", None),
        reward_trace=step_rewards,
        avg_first5=(sum(first5) / len(first5)) if first5 else 0.0,
        avg_last10=(sum(last10) / len(last10)) if last10 else 0.0,
        peak_reward=max(step_rewards) if step_rewards else 0.0,
        zero_loss_steps=zero_loss_steps,
        zero_reward_steps=zero_reward_steps,
        heldout_reward=heldout_reward,
    )


def run_grpo(
    config: GRPOConfig,
    dataset: DatasetAdapter,
    reward: RewardAdapter,
    tokenizer: Any = None,
    logger: Callable[[str], Any] = print,
) -> List[GRPORunResult]:
    """Run GRPO for ``config.num_seeds`` seeds and return all results."""
    results: List[GRPORunResult] = []
    for seed_idx in range(config.num_seeds):
        cfg = GRPOConfig(
            **{
                **config.__dict__,
                "seed": config.seed + seed_idx,
            }
        )
        result = _run_one_seed(cfg, dataset, reward, tokenizer, logger)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Built-in adapters
# ---------------------------------------------------------------------------


def make_synthetic_tool_use_dataset(
    system_prompt: str = (
        'You are a tool-calling assistant. Respond ONLY with a valid JSON object:\n'
        '{"tool": "<name>", "arguments": {<key>: <value>}}\n'
        "No prose. Only JSON."
    ),
) -> InMemoryDataset:
    """Build the 5-tool synthetic dataset used by ``grpo_exp_a/b/c`` and ``grpo_100_synthetic``."""
    tools = [
        {"name": "calculator", "description": "Arithmetic", "parameters": {"expression": "string"}},
        {"name": "get_weather", "description": "Weather for a city", "parameters": {"city": "string", "units": "string"}},
        {"name": "web_search", "description": "Web search", "parameters": {"query": "string"}},
        {"name": "get_time", "description": "Time in timezone", "parameters": {"timezone": "string"}},
        {"name": "set_reminder", "description": "Set a reminder", "parameters": {"task": "string", "time": "string"}},
    ]
    tool_schema = json.dumps(tools)

    raw: List[Tuple[str, str, Dict[str, str]]] = [
        ("What is 245 * 37?", "calculator", {"expression": "245 * 37"}),
        ("Calculate sqrt(144)", "calculator", {"expression": "sqrt(144)"}),
        ("15% of 980?", "calculator", {"expression": "0.15 * 980"}),
        ("Divide 1024 by 32", "calculator", {"expression": "1024 / 32"}),
        ("2 to the power of 10", "calculator", {"expression": "2 ** 10"}),
        ("Weather in Tokyo?", "get_weather", {"city": "Tokyo", "units": "metric"}),
        ("Is it raining in London?", "get_weather", {"city": "London", "units": "metric"}),
        ("Temperature in New York", "get_weather", {"city": "New York", "units": "imperial"}),
        ("How hot is Dubai right now?", "get_weather", {"city": "Dubai", "units": "metric"}),
        ("Search for GPT-5 news", "web_search", {"query": "GPT-5 news"}),
        ("Capital of Australia?", "web_search", {"query": "capital of Australia"}),
        ("Find Python asyncio tutorial", "web_search", {"query": "Python asyncio tutorial"}),
        ("What time is it in Singapore?", "get_time", {"timezone": "Asia/Singapore"}),
        ("Current time in Los Angeles?", "get_time", {"timezone": "America/Los_Angeles"}),
        ("Time in Berlin?", "get_time", {"timezone": "Europe/Berlin"}),
        ("Remind me to call mom at 6pm", "set_reminder", {"task": "call mom", "time": "6pm"}),
        ("Set a reminder for team meeting 10am", "set_reminder", {"task": "team meeting", "time": "10am"}),
        ("Remind me to take medicine at 8pm", "set_reminder", {"task": "take medicine", "time": "8pm"}),
    ]

    def _mkp(q: str) -> str:
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\nAvailable tools:\n{tool_schema}\n\nUser: {q}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    examples = [
        TrainingExample(prompt=_mkp(q), target={"tool": t, "arguments": a})
        for q, t, a in raw
    ]
    return InMemoryDataset(train=examples * 28)


def make_synthetic_math_dataset(
    system_prompt: str = (
        "You are a math assistant. Solve the problem step by step, "
        "then give your final answer inside \\boxed{}."
    ),
) -> InMemoryDataset:
    """Build the synthetic MATH dataset used by ``grpo_100_math``."""
    problems: List[Tuple[str, str]] = [
        ("What is 17 * 23?", "391"),
        ("What is 256 / 16?", "16"),
        ("What is 2^8?", "256"),
        ("Solve: 3x + 7 = 22", "5"),
        ("What is sqrt(625)?", "25"),
        ("What is 15! / 14!?", "15"),
        ("What is the sum of the first 10 positive integers?", "55"),
        ("What is 7^3?", "343"),
        ("Solve: 2x - 5 = 13", "9"),
        ("What is 144 / 12?", "12"),
        ("What is the GCD of 48 and 36?", "12"),
        ("What is 3^4 + 4^3?", "145"),
        ("Solve: x^2 = 49, x > 0", "7"),
        ("What is 1000 - 37 * 27?", "1"),
        ("What is the LCM of 12 and 18?", "36"),
        ("How many prime numbers are less than 20?", "8"),
        ("What is 5! (5 factorial)?", "120"),
        ("Solve: |x - 3| = 7, find positive x", "10"),
        ("What is 99 * 101?", "9999"),
        ("What is the 10th Fibonacci number?", "55"),
        ("What is 2^10 - 1?", "1023"),
        ("Solve: x + x/2 + x/4 = 14", "8"),
        ("What is 13^2 - 12^2?", "25"),
        ("What is the area of a circle with radius 7? (use pi=22/7)", "154"),
        ("What is 111 * 111?", "12321"),
        ("Solve: 2^x = 64", "6"),
        ("What is the sum of angles in a pentagon?", "540"),
        ("What is 17^2?", "289"),
        ("How many ways to choose 2 items from 5?", "10"),
        ("What is log_2(256)?", "8"),
        ("Solve: 5x + 3 = 2x + 18", "5"),
        ("What is 37 + 48 + 65 + 50?", "200"),
        ("What is the remainder when 100 is divided by 7?", "2"),
        ("What is 25% of 480?", "120"),
        ("Solve: x^2 - 5x + 6 = 0, find the larger root", "3"),
        ("What is 8 * 7 * 6 / (3 * 2 * 1)?", "56"),
        ("What is 1/2 + 1/3 + 1/6? Express as integer.", "1"),
        ("How many diagonals does a hexagon have?", "9"),
        ("What is the cube root of 27?", "3"),
        ("What is 50^2 - 49^2?", "99"),
    ]

    def _mkp(q: str) -> str:
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{q}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    examples = [
        TrainingExample(prompt=_mkp(q), target=answer)
        for q, answer in problems
    ]
    return InMemoryDataset(train=examples * 20)


class ToolCallReward:
    """Scores tool-call completions the way the original ``grpo_exp_*.py`` scripts did."""

    def score(self, response: str, example: TrainingExample) -> float:
        target = example.target or {}
        tool_name = target.get("tool", target.get("name", ""))
        arguments = target.get("arguments", target.get("parameters", {}))

        m = re.search(r"\{.*\}", response.strip(), re.DOTALL)
        if not m:
            return 0.0
        try:
            parsed = json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            return 0.1
        score = 0.3
        if parsed.get("tool") == tool_name or parsed.get("name") == tool_name:
            score += 0.4
        pred_args = parsed.get("arguments", parsed.get("parameters", {}))
        if isinstance(pred_args, dict) and arguments:
            score += 0.3 * sum(1 for k in arguments if k in pred_args) / len(arguments)
        return min(score, 1.0)


class MathReward:
    """Scores math completions: boxed answer > last number > partial credit."""

    def score(self, response: str, example: TrainingExample) -> float:
        answer = str(example.target or "")
        response = response.strip()

        boxed = re.findall(r"\\boxed\{([^}]+)\}", response)
        for b in boxed:
            b_clean = b.strip().replace(",", "").replace(" ", "")
            if b_clean == answer:
                return 1.0
            try:
                if abs(float(b_clean) - float(answer)) < 0.01:
                    return 1.0
            except (ValueError, TypeError):
                pass
        if boxed:
            return 0.3

        nums = re.findall(r"\b" + re.escape(answer) + r"\b", response)
        if nums:
            return 0.5

        all_nums = re.findall(r"[-+]?\d*\.?\d+", response)
        if all_nums:
            last = all_nums[-1].replace(",", "")
            try:
                if abs(float(last) - float(answer)) < 0.01:
                    return 1.0
            except (ValueError, TypeError):
                pass

        if any(c in response for c in "+-*/="):
            return 0.1
        return 0.0
