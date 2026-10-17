"""Strict run schema and artifact-completion contract shared by both launchers.

No library import, installation, training, service access, or file write occurs
on import. A valid legacy record can remain *unverified*: absent provenance is
never invented. A completion marker means artifacts were preserved and checked;
it does not mean optimizer/RNG state exists or interrupted training can resume.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from datetime import datetime

ARMS = ("grpo_g8", "grpo_g2", "ppo", "ppo_8x8")
SEEDS = (42, 123, 456, 789, 1024, 2048, 3141, 4096, 5555, 7777)
GROUPS = {"grpo_g8": 8, "grpo_g2": 2, "ppo": 1, "ppo_8x8": 8}
SCHEMA_VERSION = 1
HEX = re.compile(r"[0-9a-f]{64}\Z")
REVISION = re.compile(r"[0-9a-f]{40,64}\Z")
NAME = re.compile(r"(grpo_g8|grpo_g2|ppo_8x8|ppo)_s(\d+)\.json\Z")
ROOT_CONFIG = ("model", "arm", "seed", "smoke", "n_steps", "n_eval", "n_gen", "group",
               "prompts_per_step", "k_epochs", "lr", "max_new", "critic")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON numeric constant: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key}")
        result[key] = value
    return result


def strict_json_loads(text):
    result = json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
    if _nonfinite(result):
        raise ValueError("Non-finite JSON numeric value")
    return result


def strict_json_load(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return strict_json_loads(stream.read())
    return strict_json_loads(path.read_text(encoding="utf-8"))


def _nonfinite(value):
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_nonfinite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_nonfinite(item) for item in value)
    return False


def _hash(value):
    return isinstance(value, str) and bool(HEX.fullmatch(value))


def _binary(values, n):
    return (isinstance(values, list) and len(values) == n
            and all(type(value) is int and value in (0, 1) for value in values))


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_run(record, *, filename=None, texts=None, expected=None, seen=None,
                 require_texts=True):
    """Return structured errors/unverified reasons without mutating the record.

    `expected` locks manifest values (e.g. environment/protocol/config/run IDs).
    Optional `seen` is a caller-owned set of logical identities for batch checks.
    Missing provenance yields unverified, explicit conflicting data is invalid.
    """
    errors, unknown, metrics = [], [], {}
    identity = None
    if not isinstance(record, dict):
        return {"valid": False, "verified": False, "errors": ["run must be an object"],
                "unverified": [], "identity": None, "metrics": {}}
    if _nonfinite(record):
        return {"valid": False, "verified": False, "errors": ["non-finite numeric value in run"],
                "unverified": [], "identity": None, "metrics": {}}
    arm, seed = record.get("arm"), record.get("seed")
    arm = arm if isinstance(arm, str) else None
    if arm not in ARMS:
        errors.append("arm is not in the prespecified arm set")
    if type(seed) is not int or seed not in SEEDS:
        errors.append("seed is not in the prespecified seed set")
    if filename is not None:
        match = NAME.fullmatch(Path(filename).name)
        if not match or (match[1], int(match[2])) != (arm, seed):
            errors.append("filename arm/seed contradict record identity")
    n = record.get("n_eval")
    if type(n) is not int or n <= 0:
        errors.append("n_eval must be a positive integer")
        n = 0
    for key in ("pre_correct", "post_correct", "pre_trunc", "post_trunc"):
        if not _binary(record.get(key), n):
            errors.append(f"{key} must contain exactly n_eval binary integers")
    smoke = record.get("smoke")
    if type(smoke) is not bool:
        errors.append("smoke must be boolean")
    recipe = {"model": "Qwen/Qwen2.5-1.5B-Instruct", "n_steps": 2 if smoke else 30,
              "n_eval": 32 if smoke else 1319, "n_gen": 64, "k_epochs": 2,
              "lr": 1e-5, "max_new": 512}
    if arm in GROUPS:
        recipe.update(group=GROUPS[arm], prompts_per_step=64 // GROUPS[arm])
    for key, value in recipe.items():
        if record.get(key) != value or type(record.get(key)) is bool or (type(value) is int and type(record.get(key)) is not int):
            errors.append(f"effective config mismatch: {key}")
    logs = record.get("step_log")
    if not isinstance(logs, list) or len(logs) != recipe["n_steps"]:
        errors.append("step_log length must match n_steps")
    elif any(not isinstance(row, dict) or row.get("step") != i for i, row in enumerate(logs)):
        errors.append("step_log must have ordered consecutive step IDs")
    if isinstance(logs, list) and any(not isinstance(row, dict) or not _number(row.get("mean_reward"))
                                     or not 0 <= row["mean_reward"] <= 1 for row in logs):
        errors.append("step_log mean_reward must be finite probabilities")
    warm = record.get("warm_log")
    warm_count = (1 if smoke else 4) if isinstance(arm, str) and arm.startswith("ppo") else 0
    if not isinstance(warm, list) or len(warm) != warm_count:
        errors.append("warm_log length does not match arm critic warm-up")
    elif any(not isinstance(row, dict) or row.get("warm_step") != i for i, row in enumerate(warm)):
        errors.append("warm_log must have ordered consecutive warm_step IDs")
    critic = record.get("critic")
    if arm in ("ppo", "ppo_8x8"):
        if not isinstance(critic, dict) or any(critic.get(k) != v for k, v in {
                "v_lr": 1e-3, "v_iters": 50, "warm_batches": warm_count,
                "warm_generations": warm_count * 64}.items()):
            errors.append("critic config contradicts PPO recipe")
    elif critic is not None:
        errors.append("GRPO critic must be null")
    for array, metric in (("pre_correct", "heldout_pre_acc"), ("post_correct", "heldout_post_acc"),
                          ("pre_trunc", "heldout_pre_trunc"), ("post_trunc", "heldout_post_trunc")):
        if n and _binary(record.get(array), n):
            metrics[metric] = sum(record[array]) / n
    if n and _binary(record.get("pre_correct"), n) and _binary(record.get("post_correct"), n):
        pairs = list(zip(record["pre_correct"], record["post_correct"]))
        metrics["wrong_to_right"] = sum(a == 0 and b == 1 for a, b in pairs)
        metrics["right_to_wrong"] = sum(a == 1 and b == 0 for a, b in pairs)
    if isinstance(logs, list) and logs and all(isinstance(row, dict) and _number(row.get("mean_reward")) for row in logs):
        metrics["last10_avg"] = sum(row["mean_reward"] for row in logs[-10:]) / len(logs[-10:])
    for key, value in metrics.items():
        if key in ("wrong_to_right", "right_to_wrong") and type(record.get(key)) is not int:
            errors.append(f"reported {key} must be an integer count")
        if not _number(record.get(key)) or not math.isclose(record[key], value, rel_tol=0, abs_tol=1e-12):
            errors.append(f"reported {key} disagrees with recomputed metric")
    if texts is not None:
        if not isinstance(texts, dict):
            errors.append("texts must be an object")
        else:
            for key in ("pre", "post"):
                values = texts.get(key)
                if not isinstance(values, list) or len(values) != n or not all(isinstance(v, str) for v in values):
                    errors.append(f"texts.{key} must contain exactly n_eval strings")
    elif require_texts:
        unknown.append("completion texts unavailable")
    manifest = record.get("manifest")
    if not isinstance(manifest, dict) or not manifest:
        unknown.append("legacy run has no complete immutable manifest")
        if "execution_environment" not in record or not record["execution_environment"]:
            unknown.append("execution environment unavailable")
    else:
        if "schema_version" not in manifest:
            unknown.append("manifest.schema_version unavailable")
        elif manifest["schema_version"] != SCHEMA_VERSION:
            errors.append("unsupported manifest schema_version")
        for key in ("protocol_id", "analysis_plan_id", "run_id", "environment"):
            if not isinstance(manifest.get(key), str) or not manifest[key]:
                unknown.append(f"manifest.{key} unavailable")
        if "environment" in manifest and manifest["environment"] not in ("modal", "colab"):
            errors.append("manifest environment must be modal or colab")
        spec = manifest.get("protocol_spec")
        if not isinstance(spec, dict) or not spec:
            unknown.append("protocol spec unavailable")
        elif "protocol_hash" in manifest and manifest["protocol_hash"] != canonical_hash(spec):
            errors.append("protocol_hash mismatch")
        if not _hash(manifest.get("protocol_hash")):
            unknown.append("protocol_hash unavailable")
        try:
            from .samestack_protocol import PROTOCOL_ID, PROTOCOL_HASH, ANALYSIS_PLAN_ID, prompt_schedules
        except ImportError:
            from samestack_protocol import PROTOCOL_ID, PROTOCOL_HASH, ANALYSIS_PLAN_ID, prompt_schedules
        for key, required in (("protocol_id", PROTOCOL_ID), ("protocol_hash", PROTOCOL_HASH),
                              ("analysis_plan_id", ANALYSIS_PLAN_ID)):
            if key in manifest and manifest[key] != required:
                errors.append(f"manifest {key} is not the current prespecified protocol")
        config = manifest.get("effective_config")
        if not isinstance(config, dict) or not config:
            unknown.append("effective_config unavailable")
        else:
            if "config_hash" not in manifest:
                unknown.append("config_hash unavailable")
            elif manifest["config_hash"] != canonical_hash(config):
                errors.append("config_hash mismatch")
            for key in set(ROOT_CONFIG) | set(config):
                if key not in config or config[key] != record.get(key):
                    errors.append(f"manifest effective_config contradicts root {key}")
            for key in ("model_revision", "tokenizer_revision", "dataset_revision"):
                if not isinstance(config.get(key), str) or not REVISION.fullmatch(config[key]):
                    unknown.append(f"immutable {key} unavailable")
            for key, required in {"clip": 0.2, "eval_bs": 64, "eps": 1e-6, "chunk": 4}.items():
                if key not in config:
                    unknown.append(f"effective_config.{key} unavailable")
                elif config[key] != required:
                    errors.append(f"effective_config.{key} contradicts recipe")
            required_full = {
                "policy_optimizer": {"name": "Adam", "betas": [0.9, 0.999], "eps": 1e-8,
                                     "weight_decay": 0, "max_grad_norm": 1.0},
                "lora": {"r": 16, "alpha": 32, "dropout": 0.0,
                         "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
                "loss_normalization": "per-response mean over first-EOS-inclusive tokens",
                "reference_kl": None, "policy_dropout_enabled": False, "attention_backend": "sdpa",
                "precision": {"model": "torch.bfloat16", "logprobs": "torch.float32", "value_head": "torch.float32"},
                "advantage_normalization": "batch sample-std" if arm in ("ppo", "ppo_8x8") else "group population-std",
            }
            for key, required in required_full.items():
                if key not in config:
                    unknown.append(f"effective_config.{key} unavailable")
                elif config[key] != required:
                    errors.append(f"effective_config.{key} contradicts recipe")
            if config.get("generation") != manifest.get("generation"):
                errors.append("effective generation config disagrees with manifest")
            generations = manifest.get("generation", {})
            if not isinstance(generations, dict):
                errors.append("generation must be an object")
            else:
                for mode in ("train", "eval"):
                    generation = generations.get(mode)
                    if not isinstance(generation, dict):
                        unknown.append(f"generation.{mode} unavailable")
                        continue
                    required_generation = {"temperature": 1.0, "top_p": 1.0, "top_k": 0,
                                           "repetition_penalty": 1.0, "num_beams": 1,
                                           "do_sample": mode == "train", "max_new_tokens": recipe["max_new"]}
                    for key, required in required_generation.items():
                        if generation.get(key) != required:
                            errors.append(f"generation.{mode}.{key} contradicts policy")
                    eos = generation.get("eos_token_id")
                    if (not isinstance(eos, list) or not eos
                            or not all(type(token) is int and token >= 0 for token in eos)
                            or type(generation.get("pad_token_id")) is not int):
                        errors.append(f"generation.{mode} has invalid EOS/PAD token IDs")
                    for key in ("bad_words_ids", "forced_bos_token_id", "forced_eos_token_id",
                                "suppress_tokens", "begin_suppress_tokens", "constraints", "force_words_ids",
                                "sequence_bias", "min_p", "top_h"):
                        if generation.get(key) is not None:
                            errors.append(f"generation.{mode}.{key} unexpectedly changes policy")
        if not _hash(manifest.get("chat_template_hash")):
            unknown.append("immutable chat_template_hash unavailable")
        ids = manifest.get("test_item_ids")
        if ids is None:
            unknown.append("ordered test item IDs unavailable")
        elif not isinstance(ids, list) or len(ids) != n or not all(isinstance(v, str) and v for v in ids):
            errors.append("ordered test_item_ids must have n_eval nonempty strings")
        elif len(set(ids)) != n:
            errors.append("duplicate test item identities")
        elif "ordered_test_item_hash" not in manifest:
            unknown.append("ordered test item hash unavailable")
        elif manifest["ordered_test_item_hash"] != canonical_hash(ids):
            errors.append("ordered test item identity/hash mismatch")
        if ids is not None and isinstance(texts, dict) and "test_item_ids" in texts and texts["test_item_ids"] != ids:
            errors.append("text item order contradicts manifest")
        train_ids = manifest.get("train_item_ids")
        if not isinstance(train_ids, list) or not train_ids or not all(isinstance(v, str) and v for v in train_ids):
            unknown.append("ordered train item IDs unavailable")
        elif len(set(train_ids)) != len(train_ids):
            errors.append("duplicate train item IDs")
        dependencies, hardware = manifest.get("dependencies"), manifest.get("hardware")
        if not isinstance(dependencies, dict) or not dependencies:
            unknown.append("dependency versions unavailable")
        else:
            try:
                try:
                    from .samestack_dependencies import verify_runtime, lock_manifest
                except ImportError:
                    from samestack_dependencies import verify_runtime, lock_manifest
                verify_runtime(dependencies)
                if "dependency_lock" not in manifest:
                    unknown.append("dependency lock manifest unavailable")
                elif manifest["dependency_lock"] != lock_manifest():
                    errors.append("dependency lock manifest mismatch")
            except ValueError as exc:
                errors.append(str(exc))
        if not isinstance(hardware, dict) or not hardware:
            unknown.append("hardware environment unavailable")
        if isinstance(hardware, dict):
            for key in ("python", "gpu_names", "driver_versions", "cuda_runtime", "cudnn",
                        "attention_backend", "model_dtype", "logprob_dtype", "value_head_dtype",
                        "tf32_matmul", "deterministic_algorithms"):
                if key not in hardware:
                    unknown.append(f"hardware.{key} unavailable")
        environment_payload = {"environment": manifest.get("environment"), "dependencies": dependencies, "hardware": hardware}
        if "environment_hash" not in manifest:
            unknown.append("environment_hash unavailable")
        elif manifest["environment_hash"] != canonical_hash(environment_payload):
            errors.append("environment_hash mismatch")
        for key in ("source", "revisions", "generation", "schedules", "timestamps"):
            if not isinstance(manifest.get(key), dict) or not manifest[key]:
                unknown.append(f"manifest.{key} unavailable")
        source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
        if source.get("status") != "observed" or not isinstance(source.get("commit"), str) or not REVISION.fullmatch(source["commit"]):
            unknown.append("observed immutable source commit unavailable")
        if not _hash(source.get("dirty_hash")) or type(source.get("dirty")) is not bool:
            unknown.append("source dirty-tree state/hash unavailable")
        source_files = source.get("source_files")
        if not isinstance(source_files, dict) or not source_files or not all(isinstance(k, str) and _hash(v) for k, v in source_files.items()):
            unknown.append("source file hashes unavailable")
        revisions = manifest.get("revisions", {})
        if isinstance(revisions, dict) and isinstance(config, dict):
            for key in ("model", "tokenizer", "dataset"):
                revision = revisions.get(key)
                if not isinstance(revision, dict) or "revision" not in revision:
                    unknown.append(f"manifest {key} revision unavailable")
                elif revision["revision"] != config.get(key + "_revision"):
                    errors.append(f"manifest {key} revision contradicts config")
        schedules = manifest.get("schedules")
        if (isinstance(schedules, dict) and isinstance(train_ids, list) and train_ids
                and arm in GROUPS and type(seed) is int):
            required_schedule = prompt_schedules(seed, len(train_ids), recipe["n_steps"], warm_count, 64 // GROUPS[arm])
            for key, required in required_schedule.items():
                if schedules.get(key) != required:
                    errors.append(f"schedule {key} contradicts deterministic seed schedule")
            for key, indices_key in (("policy_item_ids", "policy"), ("critic_warmup_item_ids", "critic_warmup")):
                required = [[train_ids[i] for i in row] for row in required_schedule[indices_key]]
                if schedules.get(key) != required:
                    errors.append(f"schedule {key} does not match ordered train item IDs")
        timestamps = manifest.get("timestamps", {})
        try:
            start, finish = (datetime.fromisoformat(timestamps[key]) for key in ("started_at", "finished_at"))
            if start.tzinfo is None or finish.tzinfo is None or finish < start:
                raise ValueError("unordered or timezone-naive timestamps")
        except (ValueError, TypeError, KeyError, AttributeError):
            unknown.append("ordered timezone-aware run start/finish timestamps unavailable")
        run_payload = {"protocol_hash": manifest.get("protocol_hash"), "config_hash": manifest.get("config_hash"),
                       "environment_hash": manifest.get("environment_hash"), "environment": manifest.get("environment"), "source_commit": source.get("commit"),
                       "source_dirty_hash": source.get("dirty_hash")}
        if "run_id" in manifest and manifest["run_id"] != canonical_hash(run_payload):
            errors.append("run_id does not match immutable protocol/config/environment/source identity")
        for key, value in (expected or {}).items():
            actual = source.get(key.removeprefix("source_")) if key in ("source_commit", "source_dirty_hash") else manifest.get(key)
            if actual is None:
                unknown.append(f"expected manifest {key} unavailable")
            elif actual != value:
                errors.append(f"manifest {key} does not match expected run")
        identity = (manifest.get("environment"), manifest.get("protocol_hash"),
                    manifest.get("ordered_test_item_hash"), arm, seed)
        if not all(isinstance(value, str) for value in identity[:-1]) or type(seed) is not int:
            identity = None
        if seen is not None and identity is not None:
            if identity in seen:
                errors.append("duplicate run identity")
            else:
                seen.add(identity)
    return {"valid": not errors, "verified": not errors and not unknown,
            "errors": errors, "unverified": unknown, "identity": identity, "metrics": metrics}


def completion_path(result_path):
    return Path(result_path).with_suffix(".complete.json")


def artifact_files(result_path):
    """Return required artifact files relative to the run's containing directory."""
    result_path = Path(result_path)
    base, name = result_path.parent, result_path.stem
    texts_path = base / f"{name}_texts.json.gz"
    adapter = base / "adapters" / name
    required = [result_path, texts_path, adapter / "adapter_config.json", adapter / "tokenizer_config.json",
                adapter / "train_generation_config.json", adapter / "eval_generation_config.json"]
    weights = [path for pattern in ("adapter_model*.safetensors", "adapter_model*.bin") for path in adapter.glob(pattern)]
    tokens = [path for pattern in ("tokenizer.json", "tokenizer.model", "vocab.json") for path in adapter.glob(pattern)]
    if not weights or not tokens:
        raise ValueError("required adapter weights and tokenizer vocabulary are missing")
    required += weights + tokens
    for path in (adapter / "adapter_config.json", adapter / "tokenizer_config.json",
                 adapter / "train_generation_config.json", adapter / "eval_generation_config.json"):
        if not isinstance(strict_json_load(path), dict):
            raise ValueError(f"artifact config must be a strict JSON object: {path.name}")
    if any(not path.is_file() or path.is_symlink() or path.stat().st_size == 0 for path in required):
        raise ValueError("required run artifact missing, empty, or symlinked")
    # Include auxiliary adapter/tokenizer files too, so downloads are complete.
    files = sorted({*required, *(p for p in adapter.rglob("*") if p.is_file())})
    if any(path.is_symlink() or not path.resolve().is_relative_to(base.resolve()) for path in files):
        raise ValueError("artifact path leaves run directory")
    return {str(path.relative_to(base)): {"sha256": sha256_file(path), "size": path.stat().st_size} for path in files}


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _validate_generation_artifacts(result_path, record):
    adapter = Path(result_path).parent / "adapters" / Path(result_path).stem
    for mode in ("train", "eval"):
        if strict_json_load(adapter / f"{mode}_generation_config.json") != record["manifest"]["generation"][mode]:
            raise ValueError(f"saved {mode} generation config contradicts effective manifest")


def write_completion(result_path, *, expected=None):
    result_path = Path(result_path)
    record = strict_json_load(result_path)
    texts = strict_json_load(result_path.with_name(result_path.stem + "_texts.json.gz"))
    result = validate_run(record, filename=result_path.name, texts=texts, expected=expected)
    if not result["verified"]:
        raise ValueError("cannot mark unverified run complete: " + "; ".join(result["errors"] + result["unverified"]))
    _validate_generation_artifacts(result_path, record)
    manifest = record["manifest"]
    marker = {"schema_version": SCHEMA_VERSION, "run_id": manifest["run_id"],
              "protocol_hash": manifest["protocol_hash"], "config_hash": manifest["config_hash"],
              "environment_hash": manifest["environment_hash"], "training_resume_supported": False,
              "files": artifact_files(result_path)}
    _atomic_json(completion_path(result_path), marker)
    return marker


def validate_completion(result_path, *, expected=None):
    result_path = Path(result_path)
    marker = strict_json_load(completion_path(result_path))
    if not isinstance(marker, dict) or marker.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported completion marker")
    record = strict_json_load(result_path)
    texts = strict_json_load(result_path.with_name(result_path.stem + "_texts.json.gz"))
    result = validate_run(record, filename=result_path.name, texts=texts, expected=expected)
    if not result["verified"]:
        raise ValueError("run schema/provenance failed: " + "; ".join(result["errors"] + result["unverified"]))
    for key in ("run_id", "protocol_hash", "config_hash", "environment_hash"):
        if marker.get(key) != record["manifest"][key]:
            raise ValueError(f"completion {key} mismatch")
    _validate_generation_artifacts(result_path, record)
    if marker.get("files") != artifact_files(result_path):
        raise ValueError("completion artifact inventory/hash mismatch")
    if marker.get("training_resume_supported") is not False:
        raise ValueError("completion marker must not claim training resume support")
    return record


def load_completed(result_path, *, expected=None):
    """Skip only a fully verified matching artifact bundle, never a lone JSON."""
    try:
        return validate_completion(result_path, expected=expected)
    except (OSError, ValueError, KeyError, TypeError):
        return None
