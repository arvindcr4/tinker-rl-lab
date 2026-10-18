"""Versioned, provider-free contracts for prospective cross-framework recipes.

v3 does not reclassify historical runs. Recipe equivalence is never inferred from
an algorithm label or model name. Hashes bind the full requested specification;
resolved provider settings must be supplied separately before comparison.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from fractions import Fraction

PROTOCOL_VERSION = "framework-comparison-v3"
SCORER_ID = "gsm8k-final-visible-box-or-number-v3"
ZVF_ID = "mean-prompt-group-zero-population-variance-v1"


def content_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def experiment_spec(requested_config):
    # JSON round trip detaches mutable caller state and rejects non-JSON values.
    specification = json.loads(json.dumps(requested_config, allow_nan=False))
    return {
        "protocol_version": PROTOCOL_VERSION,
        "requested_config": specification,
        "config_hash": content_hash({"version": PROTOCOL_VERSION, "config": specification}),
    }


def validate_resume(expected, saved):
    for spec in (expected, saved):
        if spec.get("protocol_version") != PROTOCOL_VERSION or spec.get(
            "config_hash"
        ) != content_hash({"version": PROTOCOL_VERSION, "config": spec.get("requested_config")}):
            raise ValueError("Invalid or legacy immutable specification")
    if not saved.get("config_hash") or saved.get("config_hash") != expected["config_hash"]:
        raise ValueError("Immutable experiment specification mismatch; start a new attempt")
    if saved.get("requested_config") != expected["requested_config"]:
        raise ValueError("Specification content does not match its hash")


def stable_seed(*identifiers):
    return int(content_hash(identifiers)[:8], 16) % (2**31 - 1)


def completion_text(value):
    if isinstance(value, str):
        return value
    if (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], dict)
        and isinstance(value[0].get("content"), str)
    ):
        return value[0]["content"]
    raise TypeError("Expected text or a single conversational completion")


def final_answer(text):
    """Ignore closed thinking spans; last box is authoritative, even if malformed.

    Otherwise the last #### marker, then the last numeric literal wins. A wrong
    or malformed final box cannot be rescued by an incidental later number.
    """
    text = completion_text(text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "<think>" in text:  # unterminated thinking is not an answer
        text = text.split("<think>", 1)[0]
    starts = list(re.finditer(r"\\boxed\s*\{", text))
    if starts:
        start, depth = starts[-1].end(), 1
        for i in range(start, len(text)):
            depth += (text[i] == "{") - (text[i] == "}")
            if not depth:
                return text[start:i].strip()
        return None
    if "\\boxed" in text:
        return None
    text = text.rsplit("####", 1)[-1]
    values = re.findall(r"[-+]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:/\d+)?", text)
    return values[-1] if values else None


def score_gsm8k(response, answer):
    try:
        predicted, gold = final_answer(response), final_answer(str(answer))
        if predicted is None or gold is None:
            return 0.0
        p, g = float(Fraction(predicted.replace(",", ""))), float(Fraction(gold.replace(",", "")))
        return float(
            math.isfinite(p) and math.isfinite(g) and math.isclose(p, g, rel_tol=1e-9, abs_tol=1e-9)
        )
    except (ValueError, ZeroDivisionError, OverflowError):
        return 0.0


def score_batch(completions, answers):
    if answers is None or len(completions) != len(answers):
        raise ValueError("One answer is required per completion")
    return [score_gsm8k(completion_text(c), a) for c, a in zip(completions, answers, strict=True)]


def group_zvf(groups):
    groups = [list(g) for g in groups]
    if not groups or any(not g or any(not math.isfinite(x) for x in g) for g in groups):
        raise ValueError("Nonempty finite prompt groups required")
    return sum(len(set(g)) == 1 for g in groups) / len(groups)


def aligned_datum(T, prompt, response):
    if not prompt or not response:
        raise ValueError("Nonempty prompt and sampled response required")
    full = list(prompt) + list(response)
    return T.Datum(
        model_input=T.ModelInput.from_ints(full[:-1]),
        loss_fn_inputs={
            "target_tokens": T.TensorData(data=full[1:], dtype="int64", shape=[len(full) - 1]),
            "weights": T.TensorData(
                data=[0.0] * (len(prompt) - 1) + [1.0] * len(response),
                dtype="float32",
                shape=[len(full) - 1],
            ),
        },
    )


def response_logprobs(datum, logprobs):
    # TensorData is a provider DTO; only its serializable values are needed.
    mask = datum.loss_fn_inputs["weights"].data
    if len(mask) != len(logprobs):
        raise ValueError("Response mask/log-probability length mismatch")
    return logprobs[[i for i, weight in enumerate(mask) if weight]]


def token_advantage(advantage, response_length, max_tokens, algorithm, batch_samples=1):
    if response_length <= 0 or max_tokens <= 0 or batch_samples <= 0:
        raise ValueError("Positive response length, token budget and batch size required")
    if algorithm not in {"grpo", "dr_grpo"}:
        raise ValueError("Unknown N10 objective")
    denominator = max_tokens if algorithm == "dr_grpo" else response_length
    # Built-in IS sums both tokens and datums. Keep zero-advantage samples in
    # the declared rollout-batch denominator even if no update is necessary.
    return advantage / denominator / batch_samples


def objective_identity(requested):
    identities = {
        "grpo": "group_standardized_pg_response_sum_v3",
        "importance_sampling": "group_standardized_pg_response_sum_v3",
        "dapo": "group_standardized_pg_advantage_clip_v3",
        "gspo": "group_standardized_pg_response_mean_v3",
        "drgrpo": "group_centered_pg_response_sum_v3",
    }
    return {
        "requested_algorithm": requested,
        "objective_id": identities[requested],
        "canonical_algorithm_equivalent": False,
        "advantage_clip": [-2.0, 3.0] if requested == "dapo" else None,
        "ratio_clip": None,
        "dynamic_sampling": False,
        "sampling_policy": "current-policy-after-each-update",
    }


def validated_sample_sequences(response, expected_count):
    """Reject malformed fixed-G responses before grading or updating a model."""
    if not isinstance(expected_count, int) or expected_count <= 0:
        raise ValueError("Expected sample count must be positive")
    sequences = list(getattr(response, "sequences", None) or [])
    if len(sequences) != expected_count:
        raise ValueError(
            f"Incomplete fixed sample population: expected {expected_count}, got {len(sequences)}"
        )
    if any(not getattr(sequence, "tokens", None) for sequence in sequences):
        raise ValueError("Empty sampled completion in fixed sample population")
    return sequences


def completion_receipt(
    requested_updates, successful_updates, trainer_run_id, final_checkpoint, *, error=None
):
    complete = (
        not error
        and successful_updates == requested_updates
        and bool(trainer_run_id)
        and bool(final_checkpoint)
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "completed" if complete else ("partial" if successful_updates else "failed"),
        "requested_updates": requested_updates,
        "successful_updates": successful_updates,
        "trainer_run_id": trainer_run_id,
        "final_checkpoint": final_checkpoint,
        "completion_contract_satisfied": bool(complete),
        "evidence_error": error,
        "comparison_contract_satisfied": False,
        "claim_supported": False,
    }


def require_completed(record):
    if not (
        record.get("status") == "completed"
        and record.get("completion_contract_satisfied") is True
        and record.get("successful_updates") == record.get("requested_updates")
        and record.get("trainer_run_id")
        and record.get("final_checkpoint")
    ):
        raise ValueError("Verified completion receipt required")
    return record


def export_branch(tag):
    label = re.sub(r"[^a-z0-9.-]+", "-", str(tag).lower()).strip("-.")[:60] or "run"
    return f"{label}-{content_hash(str(tag))[:12]}"


def export_identity(record):
    required = ("config_hash", "checkpoint", "model_revision")
    if any(not record.get(k) for k in required):
        raise ValueError("Export requires exact checkpoint, model revision and config hash")
    if record["checkpoint"] != record.get("final_checkpoint"):
        raise ValueError("Export checkpoint differs from completed training receipt")
    return {k: record[k] for k in required}


def observed_updates(config, summary):
    for key in (
        "successful_updates",
        "train/global_step",
        "global_step",
        "trainer/global_step",
        "step",
    ):
        value = summary.get(key)
        if value is not None:
            return int(value)
    return None


def validate_comparison(requested, resolved, receipt):
    """Admit only fully observed, controlled scientific specifications.

    Each field includes exact values, not a requested-name equivalence claim.
    Callers with managed/unresolved internals retain native-recipe status.
    """
    require_completed(receipt)
    required = (
        "model_revision",
        "tokenizer_revision",
        "template_hash",
        "data_revision",
        "train_item_ids",
        "dev_item_ids",
        "test_item_ids",
        "scorer_id",
        "objective_id",
        "trainable_modules",
        "loss_reduction",
        "kl",
        "sampling",
        "optimizer",
        "budget",
        "selection_rule",
        "environment_digest",
        "code_revision",
    )

    def unavailable(value):
        if (
            value is None
            or isinstance(value, str)
            and value.lower() in {"unavailable", "unknown", "unverified"}
        ):
            return True
        if isinstance(value, dict):
            return "unavailable" in value or any(unavailable(v) for v in value.values())
        if isinstance(value, list):
            return any(unavailable(v) for v in value)
        return False

    for field in required:
        if (
            field not in requested
            or field not in resolved
            or unavailable(requested[field])
            or unavailable(resolved[field])
        ):
            raise ValueError(f"Comparison evidence unavailable: {field}")
        if requested[field] != resolved[field]:
            raise ValueError(f"Effective scientific treatment mismatch: {field}")
    train, dev, test = (set(resolved[f"{split}_item_ids"]) for split in ("train", "dev", "test"))
    if train & dev or train & test or dev & test:
        raise ValueError("Train/dev/test exposure must be disjoint")
    for counter in ("observed_prompts", "observed_completions", "observed_tokens"):
        if not isinstance(receipt.get(counter), int) or receipt[counter] < 0:
            raise ValueError(f"Observed work count unavailable: {counter}")
    return {"comparison_contract_satisfied": True, "comparison_hash": content_hash(resolved)}


def wave6_spec(exp):
    """Resolved prospective Wave 6 treatment; display tag/sweep are not science."""
    return experiment_spec(
        {
            "model": "Qwen/Qwen3-8B",
            "seed": exp.get("seed", 42),
            "rank": exp.get("rank", 32),
            "temperature": exp.get("temperature", 0.8),
            "batch": exp.get("batch", 2),
            "group_size": exp.get("group_size", 8),
            "lr": exp.get("lr", 1e-5),
            "steps": exp["steps"],
            "max_tokens": 512,
            "max_prompt_tokens": 1024,
            "top_p": 0.95,
            "scorer_id": SCORER_ID,
            "objective_id": "group_standardized_pg_response_sum_v3",
            "sampler_refresh_every": 1,
            "train_mlp": True,
            "train_attn": True,
            "train_unembed": True,
        }
    )


def wave6_failure_receipt(
    exp,
    error,
    *,
    status="failed",
    successful_updates=None,
    run_id=None,
    reward_trace=None,
    step_log=None,
):
    """Bind retryable failures to the same resolved treatment as successful runs.

    A worker-process crash has unknown progress, not an observed zero updates.
    """
    return {
        **wave6_spec(exp),
        **completion_receipt(exp["steps"], successful_updates, run_id, None, error=str(error)),
        "tag": exp["tag"],
        "sweep": exp["sweep"],
        "status": status,
        "error": str(error),
        "reward_trace": reward_trace,
        "step_log": step_log,
    }


def read_tinker_preset(path, name="gsm8k"):
    """Read a literal preset from source without importing/launching its trainer."""
    import ast
    from pathlib import Path

    source = Path(path).read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "PRESETS"
        ):
            for key, value in zip(node.value.keys, node.value.values, strict=True):
                if isinstance(key, ast.Constant) and key.value == name:
                    return {
                        "preset": ast.literal_eval(value),
                        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    }
    raise ValueError(f"Literal preset unavailable: {name}")


def read_skyrl_recipe(path, expected_model="Qwen/Qwen3-8B"):
    """Reject duplicate YAML keys and a differently labelled recipe before launch."""
    import yaml
    from pathlib import Path

    class UniqueLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in result:
                raise ValueError(f"Duplicate YAML key: {key}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    source = Path(path).read_text()
    config = yaml.load(source, Loader=UniqueLoader)
    model = config.get("trainer", {}).get("policy", {}).get("model", {}).get("path")
    if model != expected_model:
        raise ValueError(f"Recipe model {model!r} does not match launcher model {expected_model!r}")
    return {
        "requested_config": config,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }
