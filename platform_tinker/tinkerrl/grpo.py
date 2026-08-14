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

import ast
import difflib
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

import torch


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


# These are identifiers, not dataset contents.  Keeping them in the runner's
# immutable metadata makes every receipt auditable without coupling the generic
# loop to one particular xLAM data adapter.
PAVLOV_TRAINING_SUITE_IDS: Tuple[str, ...] = (
    "agentdojo_train",
    "api_bank_rlvr_train",
    "bfcl_train",
    "browsergym_train",
    "crafter_train",
    "openr1_math_train",
    "openreward_train",
    "rtlcoder_train",
    "scienceworld_train",
    "swe_gym_train",
    "unix_ctf_train",
    "visual_app_train",
)
PAVLOV_PRIMARY_EVALUATION_SUITE_IDS: Tuple[str, ...] = (
    "agentharm_eval",
    "apex_agents_eval",
    "appbench_eval",
    "banker_toolbench_eval",
    "binaryaudit_eval",
    "frontier_swe_eval",
    "frontiermath_eval",
    "lifescibench_eval",
    "mle_bench_eval",
    "openreward_games_eval",
    "sdab_eval",
    "swe_bench_pro_eval",
    "verilog_eval",
    "webbench_eval",
)
# Only these six primary suites currently have an explicit held-out/private
# split description.  The other eight remain primary-evaluation IDs without a
# held-out claim until their split, hash, license, runtime, and decontamination
# receipts are frozen.
PAVLOV_HELDOUT_SUITE_IDS: Tuple[str, ...] = (
    "agentharm_eval",
    "apex_agents_eval",
    "appbench_eval",
    "banker_toolbench_eval",
    "frontiermath_eval",
    "openreward_games_eval",
)
PAVLOV_DOMAIN_TAGS: Tuple[str, ...] = (
    "code",
    "browser",
    "computer_use",
    "finance",
    "enterprise",
    "science",
    "security",
    "chip_design",
    "design",
    "ml",
    "games",
    "alignment",
    "tools",
    "long_horizon",
)
PAVLOV_DECLARED_DOMAINS: Tuple[str, ...] = (
    "alignment",
    "browser",
    "chip_design",
    "code",
    "computer_use",
    "design",
    "enterprise",
    "finance",
    "games",
    "long_horizon",
    "math",
    "ml",
    "multi_domain",
    "science",
    "security",
    "tool_use",
)
PAVLOV_TRAINING_DOMAIN_UNION = PAVLOV_DECLARED_DOMAINS
PAVLOV_PRIMARY_EVALUATION_DOMAIN_UNION = PAVLOV_DECLARED_DOMAINS

PAVLOV_NON_XLAM_SOURCE_REVISIONS: Dict[str, str] = {
    "Simu-Env/API-Bank-RLVR": "bf67c42626f02c305514b1df16dcabc5fc616333",
    "SWE-Gym/SWE-Gym": "bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb",
}
PAVLOV_NON_XLAM_TRANSFORM_VERSION = "api-target-tool-swegym-compact-qwen-reasoning-v2"
PAVLOV_NON_XLAM_DATASET_REVISION = hashlib.sha256(
    json.dumps(
        {
            "sources": PAVLOV_NON_XLAM_SOURCE_REVISIONS,
            "transform": PAVLOV_NON_XLAM_TRANSFORM_VERSION,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()


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
    checkpoint_dir: str = "checkpoints/grpo"
    resume: bool = True
    wandb_project: Optional[str] = "tinker-rl-lab"
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = "grpo"
    wandb_tags: Tuple[str, ...] = ("grpo", "tinker")
    wandb_mode: str = "online"
    wandb_enabled: bool = True
    require_wandb: bool = True
    hf_owner: Optional[str] = None
    hf_public: bool = False
    hf_repo_prefix: Optional[str] = None
    checkpoint_name_prefix: Optional[str] = None
    hf_enabled: bool = True
    require_hf: bool = True
    # Optional campaign launch-gate metadata.  Generic experiments leave these
    # unset; a campaign that supplies them is rejected unless its contract and
    # budget receipts agree on an explicitly launchable state.
    campaign_status: Optional[str] = None
    budget_status: Optional[str] = None
    paid_jobs_may_launch: Optional[bool] = None
    authorized_budget_usd: Optional[float] = None
    maximum_usd: Optional[float] = None
    training_suite_ids: Tuple[str, ...] = ()
    heldout_suite_ids: Tuple[str, ...] = ()
    held_out_suite_ids: Tuple[str, ...] = ()
    # A separate field is reserved for genuinely held-out subsets; the Pavlov
    # campaign's 14 primary-evaluation suites are carried below without this
    # label.
    primary_evaluation_suite_ids: Tuple[str, ...] = ()
    domain_tags: Tuple[str, ...] = ()
    declared_domains: Tuple[str, ...] = ()
    training_domain_union: Tuple[str, ...] = ()
    primary_evaluation_domain_union: Tuple[str, ...] = ()
    # Optional immutable source revisions.  Generic runs may leave them unset;
    # campaign presets pin them so W&B/checkpoint receipts identify the exact
    # dataset and model sources used by the run.
    dataset_revision: Optional[str] = None
    model_revision: Optional[str] = None

    def __post_init__(self) -> None:
        # JSON configuration commonly supplies lists.  Convert all metadata
        # sequences to tuples at construction so later caller mutation cannot
        # change the W&B/config/checkpoint receipt.
        object.__setattr__(self, "wandb_tags", tuple(self.wandb_tags or ()))
        object.__setattr__(self, "training_suite_ids", tuple(self.training_suite_ids or ()))
        object.__setattr__(self, "heldout_suite_ids", tuple(self.heldout_suite_ids or ()))
        object.__setattr__(self, "held_out_suite_ids", tuple(self.held_out_suite_ids or ()))
        object.__setattr__(
            self,
            "primary_evaluation_suite_ids",
            tuple(self.primary_evaluation_suite_ids or ()),
        )
        object.__setattr__(self, "domain_tags", tuple(self.domain_tags or ()))
        object.__setattr__(self, "declared_domains", tuple(self.declared_domains or ()))
        object.__setattr__(
            self, "training_domain_union", tuple(self.training_domain_union or ())
        )
        object.__setattr__(
            self,
            "primary_evaluation_domain_union",
            tuple(self.primary_evaluation_domain_union or ()),
        )

    def effective_save_every(self) -> int:
        return self.save_every or max(self.steps // 4, 10)

    def validate_tracking(self) -> None:
        """Reject configurations that would silently skip campaign receipts."""
        if not self.wandb_enabled or not self.require_wandb:
            raise ValueError("W&B tracking is mandatory; it cannot be disabled")
        if not isinstance(self.wandb_project, str) or not self.wandb_project.strip():
            raise ValueError("W&B tracking is mandatory; wandb_project is required")
        if not isinstance(self.wandb_group, str) or not self.wandb_group.strip():
            raise ValueError("W&B tracking requires a configured wandb_group")
        if not isinstance(self.wandb_mode, str) or self.wandb_mode.strip().lower() != "online":
            raise ValueError("W&B tracking must use online mode")
        if not self.hf_enabled or not self.require_hf:
            raise ValueError("Hugging Face checkpoint tracking is mandatory; it cannot be disabled")
        if not self.wandb_tags:
            raise ValueError("W&B tags must be configured")

    def validate_campaign_gate(self) -> None:
        """Reject contradictory or incomplete paid-campaign launch metadata.

        The consolidated runner is also used by small generic experiments, so
        this gate is opt-in: configurations without campaign/budget metadata
        retain their existing behavior.  Once a campaign declares a status or
        paid budget, every supplied receipt must be internally consistent and
        the contract must explicitly be launchable.
        """
        has_gate_metadata = any(
            value is not None
            for value in (
                self.campaign_status,
                self.budget_status,
                self.paid_jobs_may_launch,
                self.authorized_budget_usd,
                self.maximum_usd,
            )
        )
        if not has_gate_metadata:
            return

        status = self.campaign_status
        if status is not None and (
            not isinstance(status, str) or not status.strip()
        ):
            raise ValueError("campaign contract status is required when supplied")
        normalized_status = status.strip().lower().replace("_", "-") if status else None
        launchable_statuses = {"ready", "authorized", "approved", "pass"}
        if normalized_status is not None and normalized_status not in launchable_statuses:
            raise ValueError(
                f"campaign contract is not launchable: status={normalized_status}"
            )

        budget_status = self.budget_status
        if budget_status is not None and (
            not isinstance(budget_status, str) or not budget_status.strip()
        ):
            raise ValueError("campaign budget status is required when supplied")
        normalized_budget_status = (
            budget_status.strip().lower().replace("_", "-")
            if budget_status
            else None
        )
        budget_launchable_statuses = {
            "ready",
            "authorized",
            "authorized-tinker-only",
            "approved",
            "pass",
        }
        if (
            normalized_budget_status is not None
            and normalized_budget_status not in budget_launchable_statuses
        ):
            raise ValueError("paid jobs require an explicitly authorized budget status")

        if self.paid_jobs_may_launch is not None and not isinstance(
            self.paid_jobs_may_launch, bool
        ):
            raise ValueError("paid_jobs_may_launch must be a boolean")
        if self.paid_jobs_may_launch is False:
            raise ValueError("campaign budget gate disables paid jobs")
        if self.authorized_budget_usd is not None:
            if (
                isinstance(self.authorized_budget_usd, bool)
                or not isinstance(self.authorized_budget_usd, (int, float))
                or not math.isfinite(float(self.authorized_budget_usd))
            ):
                raise ValueError("authorized_budget_usd must be numeric")
            if self.authorized_budget_usd <= 0:
                raise ValueError("authorized_budget_usd must be positive")
        if self.maximum_usd is not None:
            if (
                isinstance(self.maximum_usd, bool)
                or not isinstance(self.maximum_usd, (int, float))
                or not math.isfinite(float(self.maximum_usd))
            ):
                raise ValueError("maximum_usd must be numeric")
            if self.maximum_usd <= 0:
                raise ValueError("maximum_usd must be positive")
        if (
            self.authorized_budget_usd is not None
            and self.maximum_usd is not None
            and self.authorized_budget_usd > self.maximum_usd
        ):
            raise ValueError("authorized budget exceeds the configured maximum_usd")

        if self.paid_jobs_may_launch:
            if normalized_status not in launchable_statuses:
                raise ValueError(
                    "paid jobs require an explicitly launchable campaign status"
                )
            if self.authorized_budget_usd is None or self.maximum_usd is None:
                raise ValueError(
                    "paid jobs require explicit authorized_budget_usd and maximum_usd"
                )


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
    resumed_from_step: int = 0
    checkpoint_path: Optional[str] = None
    checkpoint_urls: List[str] = field(default_factory=list)
    checkpoint_commit_shas: List[str] = field(default_factory=list)
    checkpoint_receipts: List[Dict[str, Any]] = field(default_factory=list)
    campaign_metadata: Dict[str, List[str]] = field(default_factory=dict)


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


def normalize_rewards(rewards: Sequence[float], epsilon: float = 1e-8) -> List[float]:
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
    # next-token alignment: input positions 0..L-2 predict 1..L-1; the old
    # `full_ids[1:] + [0]` trained a spurious token-0 target at the final
    # position (fixed 2026-07-11)
    target_ids = full_ids[1:]
    return T.Datum(
        model_input=T.ModelInput.from_ints(full_ids[:-1]),
        loss_fn_inputs={
            "target_tokens": T.TensorData(data=target_ids, dtype="int64", shape=[len(target_ids)])
        },
    )


def _metric(result: Any, names: Sequence[str], default: float = float("nan")) -> float:
    metrics = getattr(result, "metrics", {}) or {}
    for name in names:
        if name in metrics:
            return metrics[name]
    return default


def _checkpoint_path(config: GRPOConfig, seed: int) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", config.name).strip("-")
    return Path(config.checkpoint_dir) / f"{safe_name}_seed{seed}.json"


def _config_fingerprint(config: GRPOConfig, seed: int) -> Dict[str, Any]:
    values = asdict(config)
    values["seed"] = seed
    values.pop("resume", None)
    heldout = tuple(config.heldout_suite_ids or config.held_out_suite_ids)
    primary_eval = tuple(config.primary_evaluation_suite_ids)
    values["training_suite_ids"] = list(config.training_suite_ids)
    values["heldout_suite_ids"] = list(heldout)
    values["held_out_suite_ids"] = list(heldout)
    values["primary_evaluation_suite_ids"] = list(primary_eval)
    values["domain_tags"] = list(config.domain_tags)
    values["declared_domains"] = list(config.declared_domains)
    values["training_domain_union"] = list(config.training_domain_union)
    values["primary_evaluation_domain_union"] = list(
        config.primary_evaluation_domain_union
    )
    return values


def _campaign_metadata(config: GRPOConfig) -> Dict[str, List[str]]:
    """Return detached campaign metadata suitable for logs and receipts."""
    heldout = tuple(config.heldout_suite_ids or config.held_out_suite_ids)
    primary_eval = tuple(config.primary_evaluation_suite_ids)
    return {
        "training_suite_ids": list(config.training_suite_ids),
        "heldout_suite_ids": list(heldout),
        "primary_evaluation_suite_ids": list(primary_eval),
        "domain_tags": list(config.domain_tags),
        "declared_domains": list(config.declared_domains),
        "training_domain_union": list(config.training_domain_union),
        "primary_evaluation_domain_union": list(config.primary_evaluation_domain_union),
    }


def _write_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def _load_checkpoint(config: GRPOConfig, seed: int) -> Dict[str, Any] | None:
    path = _checkpoint_path(config, seed)
    if not config.resume or not path.exists():
        return None
    payload = json.loads(path.read_text())
    expected = _config_fingerprint(config, seed)
    if payload.get("config") != expected:
        raise ValueError(f"[{config.name}] incompatible checkpoint: {path}")
    return payload


def _redact_error(exc: BaseException) -> str:
    """Return an error string with credential values removed."""
    message = str(exc)
    for env_name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "WANDB_API_KEY"):
        secret = os.environ.get(env_name)
        if secret:
            message = message.replace(secret, "[redacted]")
    message = re.sub(
        r"(?i)\b(?:hf|wandb|sk|ghp|github_pat)[_-][A-Za-z0-9][A-Za-z0-9._-]{7,}\b",
        "[redacted]",
        message,
    )
    message = re.sub(r"(?i)\bBearer\s+[A-Za-z0-9._-]+", "Bearer [redacted]", message)
    return message or exc.__class__.__name__


def _immutable_config(config: GRPOConfig, seed: int) -> Dict[str, Any]:
    """Create a detached, JSON-only config snapshot for W&B and receipts."""
    # ``asdict`` already recursively copies dataclass fields; the JSON round-trip
    # also normalizes tuples and protects the object passed to W&B from later
    # mutation by either side of the integration.
    return json.loads(json.dumps(_config_fingerprint(config, seed), sort_keys=True))


def _finish_wandb(run: Any, *, success: bool) -> None:
    """Mark and finish a W&B run, failing if the receipt cannot be finalized."""
    if run is None:
        raise RuntimeError("W&B run is missing; final status is inadmissible")
    summary = _wandb_summary(run)
    try:
        summary["status"] = "success" if success else "failed"
        summary["final_status"] = "success" if success else "failed"
    except Exception as exc:
        raise RuntimeError("W&B summary status update failed") from exc
    finish = getattr(run, "finish", None)
    if not callable(finish):
        raise RuntimeError("W&B run has no finish method; final status is inadmissible")
    try:
        finish(exit_code=0 if success else 1)
    except Exception as exc:
        raise RuntimeError("W&B finish failed; final status is inadmissible") from exc


def _mark_wandb_failure(run: Any, exc: BaseException) -> None:
    """Record a failure without ever exposing credential-bearing exceptions."""
    if run is None:
        raise RuntimeError("W&B run is missing; failure status is inadmissible")
    summary = _wandb_summary(run)
    try:
        summary["status"] = "failed"
        summary["final_status"] = "failed"
        summary["error"] = _redact_error(exc)
    except Exception as status_exc:
        raise RuntimeError("W&B failure status update failed") from status_exc


def _wandb_summary(run: Any) -> Any:
    summary = getattr(run, "summary", None)
    if summary is None:
        summary = {}
        try:
            setattr(run, "summary", summary)
        except Exception as exc:
            raise RuntimeError("W&B run summary is unavailable") from exc
    return summary


def _wandb_log(run: Any, payload: Dict[str, Any]) -> None:
    """Log one payload and fail closed if W&B rejects it."""
    log = getattr(run, "log", None)
    if not callable(log):
        raise RuntimeError("W&B run has no log method; receipt is inadmissible")
    try:
        accepted = log(payload)
    except Exception as exc:
        raise RuntimeError("W&B log failed; receipt is inadmissible") from exc
    if accepted is False:
        raise RuntimeError("W&B log was rejected; receipt is inadmissible")


def _start_wandb(config: GRPOConfig, seed: int) -> Any:
    """Initialize an online W&B run before any Tinker client is constructed."""
    config.validate_tracking()
    env_mode = os.environ.get("WANDB_MODE")
    if env_mode and env_mode.strip().lower() != "online":
        raise RuntimeError("W&B tracking requires WANDB_MODE=online")
    if os.environ.get("WANDB_DISABLED", "").strip().lower() in {"1", "true", "yes", "on"}:
        raise RuntimeError("W&B tracking is disabled by WANDB_DISABLED")
    try:
        import wandb
    except Exception as exc:
        raise RuntimeError(f"W&B dependency is unavailable: {_redact_error(exc)}") from exc

    try:
        run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            group=config.wandb_group,
            name=f"{config.name}_seed{seed}",
            tags=list(config.wandb_tags),
            mode="online",
            config=_immutable_config(config, seed),
            reinit=True,
        )
    except Exception as exc:
        raise RuntimeError(f"W&B online initialization failed: {_redact_error(exc)}") from exc

    if run is None:
        raise RuntimeError("W&B online initialization returned no live run")
    if not getattr(run, "id", None):
        _finish_wandb(run, success=False)
        raise RuntimeError("W&B online initialization returned no live run ID")
    run_mode = getattr(run, "mode", None)
    if run_mode is not None and str(run_mode).lower() != "online":
        _finish_wandb(run, success=False)
        raise RuntimeError("W&B online initialization returned a non-online run")
    if bool(getattr(run, "disabled", False)) or bool(getattr(run, "offline", False)):
        _finish_wandb(run, success=False)
        raise RuntimeError("W&B online initialization returned a disabled run")
    return run


def _hf_token() -> Optional[str]:
    """Read the process credential without ever including it in a receipt."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def _make_hf_api() -> tuple[Any, Optional[str]]:
    """Construct an authenticated Hub API without exposing its credential."""
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        raise RuntimeError(f"Hugging Face dependency is unavailable: {_redact_error(exc)}") from exc
    token = _hf_token()
    try:
        try:
            api = HfApi(token=token)
        except TypeError:
            api = HfApi()
    except Exception as exc:
        raise RuntimeError(f"Hugging Face API initialization failed: {_redact_error(exc)}") from exc
    return api, token


def _preflight_hf(config: GRPOConfig) -> str:
    """Verify Hub authentication and resolve the owner before paid Tinker work."""
    config.validate_tracking()
    push_flag = os.environ.get("HF_PUSH")
    if push_flag and push_flag.strip().lower() in {"0", "false", "no", "off"}:
        raise RuntimeError("Hugging Face checkpoint tracking cannot be disabled")
    api, token = _make_hf_api()
    try:
        try:
            identity = api.whoami(token=token)
        except TypeError:
            identity = api.whoami()
    except Exception as exc:
        raise RuntimeError(f"Hugging Face authentication preflight failed: {_redact_error(exc)}") from exc

    if not isinstance(identity, dict) or not identity:
        raise RuntimeError("Hugging Face authentication preflight returned no identity")
    nested_user = identity.get("user")
    authenticated_owner = identity.get("name")
    if not authenticated_owner and isinstance(nested_user, dict):
        authenticated_owner = nested_user.get("name")
    if not authenticated_owner and isinstance(nested_user, str):
        authenticated_owner = nested_user
    configured_owner = config.hf_owner or os.environ.get("HF_REPO_OWNER")
    owner = configured_owner or authenticated_owner
    if not isinstance(owner, str) or not owner.strip():
        raise RuntimeError("Hugging Face authentication preflight returned no owner")
    return owner.strip()


def _verify_hf_revision(repo_id: str, revision: str) -> str:
    """Verify the pushed revision and return its immutable Hub commit SHA."""
    api, _token = _make_hf_api()
    try:
        info = api.model_info(repo_id, revision=revision)
    except Exception as exc:
        raise RuntimeError(
            f"Hugging Face checkpoint verification failed for revision {revision}: "
            f"{_redact_error(exc)}"
        ) from exc
    if isinstance(info, dict):
        commit_sha = info.get("sha") or info.get("commit_hash")
    else:
        commit_sha = getattr(info, "sha", None) or getattr(info, "commit_hash", None)
    if not isinstance(commit_sha, str) or not commit_sha.strip():
        raise RuntimeError(
            f"Hugging Face checkpoint verification returned no commit SHA for revision {revision}"
        )
    return commit_sha.strip()


def _prepare_hf_revision(config: GRPOConfig, repo_id: str, revision: str) -> None:
    """Create the target repo and branch before the official Tinker upload.

    ``huggingface_hub.upload_folder`` requires a named revision to exist.  The
    Tinker 0.22 CLI creates the repository but does not create a requested
    branch, which otherwise makes a fully downloaded multi-gigabyte checkpoint
    fail at the final upload step with ``RevisionNotFoundError``.
    """
    api, _token = _make_hf_api()
    try:
        api.create_repo(
            repo_id=repo_id,
            private=not config.hf_public,
            exist_ok=True,
        )
        api.create_branch(repo_id=repo_id, branch=revision, exist_ok=True)
    except Exception as exc:
        raise RuntimeError(
            f"Hugging Face checkpoint revision preparation failed for {revision}: "
            f"{_redact_error(exc)}"
        ) from exc


def _safe_component(value: Any, *, limit: int = 48) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-._").lower()
    return (text or "checkpoint")[:limit].rstrip("-._")


def _checkpoint_repo_id(
    config: GRPOConfig,
    owner: str,
    seed: int,
    step: int | str,
    tinker_path: str,
) -> str:
    """Build a unique Hub repo ID while retaining the exact source path in the CLI call."""
    base = config.hf_repo_prefix or config.checkpoint_name_prefix or config.name
    if "/" in base:
        # A slash in the prefix is treated as a name separator, never as an
        # implicit owner override; ownership comes from the authenticated Hub
        # identity unless ``hf_owner``/``HF_REPO_OWNER`` is explicit.
        _, base = base.split("/", 1)
    repo_owner = config.hf_owner or os.environ.get("HF_REPO_OWNER") or owner
    source_slug = _safe_component(tinker_path)
    source_hash = hashlib.sha256(str(tinker_path).encode("utf-8")).hexdigest()[:12]
    step_slug = _safe_component(step)
    repo_name = f"{_safe_component(base, limit=32)}-seed{seed}-step{step_slug}-{source_slug[:24]}-{source_hash}"
    return f"{repo_owner}/{repo_name}"


def _checkpoint_revision(seed: int, step: int | str, tinker_path: str) -> str:
    """Create a deterministic branch name unique to this sampler checkpoint."""
    source_hash = hashlib.sha256(str(tinker_path).encode("utf-8")).hexdigest()[:12]
    return f"checkpoint-seed{seed}-step{_safe_component(step)}-{source_hash}"


def _require_checkpoint_receipt(value: Any, *, step: int | str) -> Dict[str, Any]:
    """Require a complete detached Hub receipt before training may continue."""
    if not isinstance(value, dict):
        raise RuntimeError(f"Hugging Face checkpoint receipt missing for step {step}")
    required = (
        "repo_id",
        "revision",
        "commit_sha",
        "repo_url",
        "revision_url",
        "commit_url",
        "source_path",
    )
    if any(not isinstance(value.get(key), str) or not value[key] for key in required):
        raise RuntimeError(f"Hugging Face checkpoint receipt incomplete for step {step}")
    if "step" not in value or value["step"] != step:
        raise RuntimeError(f"Hugging Face checkpoint receipt step mismatch for step {step}")
    expected_repo_url = f"https://huggingface.co/{value['repo_id']}"
    if value["repo_url"] != expected_repo_url:
        raise RuntimeError(f"Hugging Face checkpoint receipt has an invalid repo URL for step {step}")
    if value["revision_url"] != f"{expected_repo_url}/tree/{value['revision']}":
        raise RuntimeError(
            f"Hugging Face checkpoint receipt has an invalid revision URL for step {step}"
        )
    if value["commit_url"] != f"{expected_repo_url}/commit/{value['commit_sha']}":
        raise RuntimeError(
            f"Hugging Face checkpoint receipt has an invalid commit URL for step {step}"
        )
    # A JSON round-trip ensures no mutable object owned by the Hub client or a
    # test double can alter a completed checkpoint's receipt later.
    return json.loads(json.dumps(value, sort_keys=True))


def _publish_checkpoint(
    config: GRPOConfig,
    seed: int,
    folder_path: str | None,
    logger: Callable[[str], Any],
    *,
    step: int | str = "final",
    hf_owner: Optional[str] = None,
    return_receipt: bool = False,
) -> Any:
    """Push one Tinker sampler checkpoint to a unique public/private Hub repo."""
    config.validate_tracking()
    push_flag = os.environ.get("HF_PUSH")
    if push_flag and push_flag.strip().lower() in {"0", "false", "no", "off"}:
        raise RuntimeError("Hugging Face checkpoint tracking cannot be disabled")
    if not folder_path:
        raise RuntimeError("Cannot publish an empty Tinker sampler checkpoint path")
    owner = hf_owner or _preflight_hf(config)
    repo_id = _checkpoint_repo_id(config, owner, seed, step, str(folder_path))
    revision = _checkpoint_revision(seed, step, str(folder_path))
    _prepare_hf_revision(config, repo_id, revision)
    command = [
        sys.executable,
        "-m",
        "tinker.cli",
        "checkpoint",
        "push-hf",
        str(folder_path),
        "--repo",
        repo_id,
        "--revision",
        revision,
    ]
    if config.hf_public:
        command.append("--public")
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("Tinker CLI is unavailable; Hugging Face checkpoint export failed") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Hugging Face checkpoint export failed for step {step} (exit code {exc.returncode})"
        ) from exc
    except Exception as exc:
        # Do not surface subprocess stderr/stdout: either stream can contain a
        # credential-bearing CLI diagnostic.  The exception remains chained for
        # local debugging without becoming part of a W&B/log/receipt message.
        raise RuntimeError(f"Hugging Face checkpoint export failed for step {step}") from exc
    commit_sha = _verify_hf_revision(repo_id, revision)
    repo_url = f"https://huggingface.co/{repo_id}"
    revision_url = f"{repo_url}/tree/{revision}"
    commit_url = f"{repo_url}/commit/{commit_sha}"
    receipt = _require_checkpoint_receipt(
        {
            "step": step,
            "repo_id": repo_id,
            "revision": revision,
            "hf_revision": revision,
            "commit_sha": commit_sha,
            "hf_commit_sha": commit_sha,
            "repo_url": repo_url,
            "hf_repo_url": repo_url,
            "revision_url": revision_url,
            "hf_revision_url": revision_url,
            "commit_url": commit_url,
            "source_path": str(folder_path),
        },
        step=step,
    )
    logger(f"[{config.name}] Published checkpoint step={step}: {revision_url}")
    return receipt if return_receipt else receipt["revision_url"]


def _log_wandb_checkpoint(
    run: Any,
    *,
    receipt: Dict[str, Any],
    receipts: Sequence[Dict[str, Any]],
) -> None:
    """Record an immutable Hub revision receipt in W&B immediately."""
    receipt_step = receipt.get("step", "unknown") if isinstance(receipt, dict) else "unknown"
    receipt = _require_checkpoint_receipt(receipt, step=receipt_step)
    detached_receipts = [
        _require_checkpoint_receipt(
            item,
            step=item.get("step", "unknown") if isinstance(item, dict) else "unknown",
        )
        for item in receipts
    ]
    _wandb_log(
        run,
        {
            "checkpoint/step": receipt["step"],
            "checkpoint/url": receipt["revision_url"],
            "checkpoint/repo_url": receipt["repo_url"],
            "checkpoint/revision_url": receipt["revision_url"],
            "checkpoint/revision": receipt["revision"],
            "checkpoint/commit_sha": receipt["commit_sha"],
            "checkpoint/hf_revision_url": receipt["revision_url"],
            "checkpoint/hf_commit_sha": receipt["commit_sha"],
        }
    )
    summary = _wandb_summary(run)
    detached = [json.loads(json.dumps(item, sort_keys=True)) for item in detached_receipts]
    summary["checkpoint_urls"] = [item["revision_url"] for item in detached]
    summary["checkpoint_commit_shas"] = [item["commit_sha"] for item in detached]
    summary["checkpoint_receipts"] = detached


def _log_campaign_metadata(run: Any, config: GRPOConfig) -> None:
    metadata = _campaign_metadata(config)
    payload = {
        "campaign/training_suite_ids": list(metadata["training_suite_ids"]),
        "campaign/primary_evaluation_suite_ids": list(
            metadata["primary_evaluation_suite_ids"]
        ),
        "campaign/domain_tags": list(metadata["domain_tags"]),
        "campaign/declared_domains": list(metadata["declared_domains"]),
        "campaign/training_domain_union": list(metadata["training_domain_union"]),
        "campaign/primary_evaluation_domain_union": list(
            metadata["primary_evaluation_domain_union"]
        ),
    }
    if config.campaign_status is not None:
        payload["campaign/status"] = config.campaign_status
    if config.budget_status is not None:
        payload["campaign/budget_status"] = config.budget_status
    if config.authorized_budget_usd is not None:
        payload["campaign/authorized_budget_usd"] = config.authorized_budget_usd
    if config.maximum_usd is not None:
        payload["campaign/maximum_usd"] = config.maximum_usd
    if metadata["heldout_suite_ids"]:
        payload["campaign/heldout_suite_ids"] = list(metadata["heldout_suite_ids"])
    _wandb_log(run, payload)
    summary = _wandb_summary(run)
    summary["training_suite_ids"] = list(metadata["training_suite_ids"])
    summary["primary_evaluation_suite_ids"] = list(
        metadata["primary_evaluation_suite_ids"]
    )
    if metadata["heldout_suite_ids"]:
        summary["heldout_suite_ids"] = list(metadata["heldout_suite_ids"])
    summary["domain_tags"] = list(metadata["domain_tags"])
    summary["declared_domains"] = list(metadata["declared_domains"])
    summary["training_domain_union"] = list(metadata["training_domain_union"])
    summary["primary_evaluation_domain_union"] = list(
        metadata["primary_evaluation_domain_union"]
    )
    if config.campaign_status is not None:
        summary["campaign_status"] = config.campaign_status
    if config.budget_status is not None:
        summary["budget_status"] = config.budget_status
    if config.authorized_budget_usd is not None:
        summary["authorized_budget_usd"] = config.authorized_budget_usd
    if config.maximum_usd is not None:
        summary["maximum_usd"] = config.maximum_usd


def _tinker_run_id(training_client: Any) -> Optional[str]:
    for attr in ("model_id", "run_id", "id"):
        value = getattr(training_client, attr, None)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    return None


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
    seed = config.seed
    config.validate_tracking()
    rng = random.Random(seed)
    torch.manual_seed(seed)

    train_examples = list(dataset.train_examples())
    if not train_examples:
        raise ValueError(f"[{config.name}] dataset returned 0 training examples")

    checkpoint_path = _checkpoint_path(config, seed)
    prior = _load_checkpoint(config, seed)
    resume_step = int((prior or {}).get("step", 0))
    resume_state_path = (prior or {}).get("train_state_path")
    for _ in range(resume_step):
        rng.sample(train_examples, min(config.batch_size, len(train_examples)))

    wb = _start_wandb(config, seed)
    try:
        # Hub identity is checked while the run is still local.  This is
        # intentionally before importing/constructing any paid Tinker client.
        _log_campaign_metadata(wb, config)
        # A contradictory campaign receipt is a local, fail-closed decision;
        # it must stop before the Hub check and before any paid Tinker client.
        try:
            config.validate_campaign_gate()
        except ValueError as exc:
            raise RuntimeError(f"campaign launch gate failed: {exc}") from exc
        hf_owner = _preflight_hf(config)

        if prior and prior.get("status") == "completed":
            completed = GRPORunResult(**prior["result"])
            if not completed.run_id:
                raise RuntimeError("Completed receipt has no nonempty Tinker run ID")
            _wandb_log(wb, {"tinker/run_id": completed.run_id})
            _wandb_summary(wb)["tinker_run_id"] = completed.run_id
            prior_receipts = list(getattr(completed, "checkpoint_receipts", []))
            if prior_receipts:
                receipts = [
                    _require_checkpoint_receipt(item, step=item.get("step", "unknown"))
                    for item in prior_receipts
                ]
                for receipt in receipts:
                    _log_wandb_checkpoint(wb, receipt=receipt, receipts=receipts)
            elif completed.checkpoint_urls:
                # Legacy completed receipts predate Hub revision metadata.  No
                # new sampler checkpoint is saved on this compatibility path;
                # retain their URLs without inventing a commit SHA.
                _wandb_summary(wb)["checkpoint_urls"] = list(completed.checkpoint_urls)
            _finish_wandb(wb, success=True)
            return completed

        try:
            import tinker
            import tinker.types as T
        except Exception as exc:
            raise RuntimeError(f"Tinker dependency is unavailable: {_redact_error(exc)}") from exc
        _AT = None
        if tokenizer is None:
            try:
                from transformers import AutoTokenizer as _AT  # noqa: F811
            except Exception as exc:
                raise RuntimeError(
                    f"Tokenizer dependency is unavailable: {_redact_error(exc)}"
                ) from exc

        tokenizer_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if config.model_revision is not None:
            tokenizer_kwargs["revision"] = config.model_revision
        tok = (
            tokenizer
            if tokenizer is not None
            else _AT.from_pretrained(config.model, **tokenizer_kwargs)
        )
        logger(f"[{config.name}] Connecting to Tinker...")
        svc = tinker.ServiceClient(base_url=config.base_url)
        if resume_state_path:
            tc = svc.create_training_client_from_state_with_optimizer(resume_state_path)
            logger(f"[{config.name}] Resuming from step {resume_step}: {resume_state_path}")
        else:
            user_metadata = {
                "experiment_name": config.name,
                "seed": str(config.seed),
            }
            if config.model_revision is not None:
                user_metadata["model_revision"] = config.model_revision
            if config.dataset_revision is not None:
                user_metadata["dataset_revision"] = config.dataset_revision
            tc = svc.create_lora_training_client(
                base_model=config.model,
                rank=config.lora_rank,
                seed=config.seed,
                user_metadata=user_metadata,
            )
        run_id = _tinker_run_id(tc)
        if not run_id:
            raise RuntimeError("Tinker training client returned no nonempty run ID")
        _wandb_log(wb, {"tinker/run_id": run_id})
        _wandb_summary(wb)["tinker_run_id"] = run_id
        logger(f"[{config.name}] Run: {run_id}")

        checkpoint_receipts: List[Dict[str, Any]] = [
            _require_checkpoint_receipt(item, step=item.get("step", "unknown"))
            for item in (prior or {}).get("checkpoint_receipts", [])
        ]
        checkpoint_urls: List[str] = [
            item["revision_url"] for item in checkpoint_receipts
        ]
        checkpoint_commit_shas: List[str] = [
            item["commit_sha"] for item in checkpoint_receipts
        ]
        initial_name = (
            config.checkpoint_name_prefix or config.name
        ) + f"_seed{seed}_step_{resume_step}"
        w0 = tc.save_weights_for_sampler(name=initial_name).result()
        initial_receipt = _publish_checkpoint(
            config,
            seed,
            w0.path,
            logger,
            step=resume_step,
            hf_owner=hf_owner,
            return_receipt=True,
        )
        initial_receipt = _require_checkpoint_receipt(initial_receipt, step=resume_step)
        checkpoint_receipts.append(initial_receipt)
        checkpoint_urls.append(initial_receipt["revision_url"])
        checkpoint_commit_shas.append(initial_receipt["commit_sha"])
        _log_wandb_checkpoint(
            wb, receipt=initial_receipt, receipts=checkpoint_receipts
        )
        sc = tc.create_sampling_client(model_path=w0.path)

        save_every = config.effective_save_every()
        step_rewards: List[float] = list((prior or {}).get("reward_trace", []))[:resume_step]
        zero_loss_steps = int((prior or {}).get("zero_loss_steps", 0))
        zero_reward_steps = int((prior or {}).get("zero_reward_steps", 0))

        for step in range(resume_step, config.steps):
            batch = rng.sample(train_examples, min(config.batch_size, len(train_examples)))
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
                    reward.score(_decode_response(tok, resp), example) for resp in responses.sequences
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
            train_result = tc.forward_backward_custom(
                data=all_data,
                loss_fn=loss_fn,
                loss_type_input="logprobs",
            ).result()
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
            loss_val = _metric(train_result, ["grpo_loss", "loss"])
            if abs(loss_val) < 1e-6:
                zero_loss_steps += 1
            if avg == 0:
                zero_reward_steps += 1

            logger(
                f"[{config.name}] Step {step + 1:3d}/{config.steps}"
                f" | loss={loss_val:.4f} | reward={avg:.3f}"
            )
            _wandb_log(
                wb,
                {
                    "train/loss": loss_val,
                    "train/reward": avg,
                    "train/step": step + 1,
                }
            )

            if (step + 1) % save_every == 0:
                state = tc.save_state(name=f"state_seed{seed}_{step + 1}", overwrite=True).result()
                ckpt = tc.save_weights_for_sampler(name=f"step_seed{seed}_{step + 1}").result()
                checkpoint_receipt = _publish_checkpoint(
                    config,
                    seed,
                    ckpt.path,
                    logger,
                    step=step + 1,
                    hf_owner=hf_owner,
                    return_receipt=True,
                )
                checkpoint_receipt = _require_checkpoint_receipt(
                    checkpoint_receipt, step=step + 1
                )
                checkpoint_receipts.append(checkpoint_receipt)
                checkpoint_urls.append(checkpoint_receipt["revision_url"])
                checkpoint_commit_shas.append(checkpoint_receipt["commit_sha"])
                _log_wandb_checkpoint(
                    wb, receipt=checkpoint_receipt, receipts=checkpoint_receipts
                )
                sc = tc.create_sampling_client(model_path=ckpt.path)
                _write_checkpoint(
                    checkpoint_path,
                    {
                        "status": "started",
                        "config": _config_fingerprint(config, seed),
                        "step": step + 1,
                        "train_state_path": state.path,
                        "sampler_path": ckpt.path,
                        "run_id": run_id,
                        "reward_trace": step_rewards,
                        "zero_loss_steps": zero_loss_steps,
                        "zero_reward_steps": zero_reward_steps,
                        "checkpoint_urls": checkpoint_urls,
                        "checkpoint_commit_shas": checkpoint_commit_shas,
                        "checkpoint_receipts": checkpoint_receipts,
                        "campaign_metadata": _campaign_metadata(config),
                    },
                )
                logger(f"[{config.name}]   -> Checkpoint step_{step + 1}")

        tc.save_state(name=f"seed{seed}_final", overwrite=True).result()
        final = tc.save_weights_for_sampler(name=f"seed{seed}_final").result()
        final_receipt = _publish_checkpoint(
            config,
            seed,
            final.path,
            logger,
            step="final",
            hf_owner=hf_owner,
            return_receipt=True,
        )
        final_receipt = _require_checkpoint_receipt(final_receipt, step="final")
        checkpoint_receipts.append(final_receipt)
        checkpoint_urls.append(final_receipt["revision_url"])
        checkpoint_commit_shas.append(final_receipt["commit_sha"])
        _log_wandb_checkpoint(wb, receipt=final_receipt, receipts=checkpoint_receipts)
        sc = tc.create_sampling_client(model_path=final.path)

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
                    _wandb_log(wb, {"test/reward": heldout_reward})

        result = GRPORunResult(
            seed=seed,
            run_id=run_id,
            sampler_path=getattr(final, "path", None),
            reward_trace=step_rewards,
            avg_first5=(sum(first5) / len(first5)) if first5 else 0.0,
            avg_last10=(sum(last10) / len(last10)) if last10 else 0.0,
            peak_reward=max(step_rewards) if step_rewards else 0.0,
            zero_loss_steps=zero_loss_steps,
            zero_reward_steps=zero_reward_steps,
            heldout_reward=heldout_reward,
            resumed_from_step=resume_step,
            checkpoint_path=str(checkpoint_path),
            checkpoint_urls=checkpoint_urls,
            checkpoint_commit_shas=checkpoint_commit_shas,
            checkpoint_receipts=checkpoint_receipts,
            campaign_metadata=_campaign_metadata(config),
        )
        # The local completion receipt is written before W&B is marked
        # successful.  Any receipt failure therefore leaves a failed run.
        _write_checkpoint(
            checkpoint_path,
            {
                "status": "completed",
                "final_status": "success",
                "config": _config_fingerprint(config, seed),
                "step": config.steps,
                "run_id": run_id,
                "checkpoint_urls": checkpoint_urls,
                "checkpoint_commit_shas": checkpoint_commit_shas,
                "checkpoint_receipts": checkpoint_receipts,
                "campaign_metadata": _campaign_metadata(config),
                "result": asdict(result),
            },
        )
        _wandb_summary(wb)["tinker_run_id"] = run_id
        _wandb_summary(wb)["checkpoint_urls"] = list(checkpoint_urls)
        _wandb_summary(wb)["checkpoint_commit_shas"] = list(checkpoint_commit_shas)
        _wandb_summary(wb)["checkpoint_receipts"] = [
            json.loads(json.dumps(item, sort_keys=True)) for item in checkpoint_receipts
        ]
        _finish_wandb(wb, success=True)
        return result
    except Exception as exc:
        status_exc: Optional[BaseException] = None
        try:
            _mark_wandb_failure(wb, exc)
        except Exception as failure_status_exc:
            status_exc = failure_status_exc
        try:
            _finish_wandb(wb, success=False)
        except Exception as finish_exc:
            status_exc = finish_exc
        if status_exc is not None:
            raise RuntimeError("W&B failure receipt could not be finalized") from status_exc
        raise


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
        cfg = replace(config, seed=config.seed + seed_idx)
        result = _run_one_seed(cfg, dataset, reward, tokenizer, logger)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Built-in adapters
# ---------------------------------------------------------------------------


def make_synthetic_tool_use_dataset(
    system_prompt: str = (
        "You are a tool-calling assistant. Respond ONLY with a valid JSON object:\n"
        '{"tool": "<name>", "arguments": {<key>: <value>}}\n'
        "No prose. Only JSON."
    ),
) -> InMemoryDataset:
    """Build the 5-tool synthetic dataset used by ``grpo_exp_a/b/c`` and ``grpo_100_synthetic``."""
    tools = [
        {"name": "calculator", "description": "Arithmetic", "parameters": {"expression": "string"}},
        {
            "name": "get_weather",
            "description": "Weather for a city",
            "parameters": {"city": "string", "units": "string"},
        },
        {"name": "web_search", "description": "Web search", "parameters": {"query": "string"}},
        {
            "name": "get_time",
            "description": "Time in timezone",
            "parameters": {"timezone": "string"},
        },
        {
            "name": "set_reminder",
            "description": "Set a reminder",
            "parameters": {"task": "string", "time": "string"},
        },
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
        (
            "Set a reminder for team meeting 10am",
            "set_reminder",
            {"task": "team meeting", "time": "10am"},
        ),
        (
            "Remind me to take medicine at 8pm",
            "set_reminder",
            {"task": "take medicine", "time": "8pm"},
        ),
    ]

    def _mkp(q: str) -> str:
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\nAvailable tools:\n{tool_schema}\n\nUser: {q}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    examples = [
        TrainingExample(prompt=_mkp(q), target={"tool": t, "arguments": a}) for q, t, a in raw
    ]
    heldout_raw: List[Tuple[str, str, Dict[str, str]]] = [
        ("3 to the power of 4", "calculator", {"expression": "3 ** 4"}),
        ("What is the weather in Paris?", "get_weather", {"city": "Paris", "units": "metric"}),
        (
            "Search for Python 3.12 release notes",
            "web_search",
            {"query": "Python 3.12 release notes"},
        ),
        ("Current time in Tokyo", "get_time", {"timezone": "Asia/Tokyo"}),
        (
            "Remind me to buy groceries tomorrow",
            "set_reminder",
            {"task": "buy groceries", "time": "tomorrow"},
        ),
    ]
    heldout = [
        TrainingExample(prompt=_mkp(q), target={"tool": t, "arguments": a})
        for q, t, a in heldout_raw
    ]
    return InMemoryDataset(train=examples * 28, test=heldout)


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

    split = int(len(problems) * 0.8)
    train = [TrainingExample(prompt=_mkp(q), target=answer) for q, answer in problems[:split]]
    test = [TrainingExample(prompt=_mkp(q), target=answer) for q, answer in problems[split:]]
    return InMemoryDataset(train=train * 20, test=test)


def make_gsm8k_dataset(seed: int = 42) -> InMemoryDataset:
    """Load GSM8K through the dataset adapter seam."""
    from datasets import load_dataset

    system_prompt = (
        "You are a math assistant. Solve the problem step by step, then give "
        "your final numerical answer inside \\boxed{}."
    )

    def convert(split: str) -> List[TrainingExample]:
        rows: List[TrainingExample] = []
        for row in load_dataset("openai/gsm8k", "main", split=split):
            match = re.search(r"####\s*([\-\d,\.]+)", row["answer"])
            if not match:
                continue
            prompt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{row['question']}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            rows.append(
                TrainingExample(
                    prompt=prompt,
                    target=match.group(1).replace(",", "").strip(),
                )
            )
        random.Random(seed).shuffle(rows)
        return rows

    return InMemoryDataset(train=convert("train"), test=convert("test"))


def make_xlam_dataset(
    seed: int = 42, revision: Optional[str] = None
) -> InMemoryDataset:
    """Load xLAM function-calling records through the dataset adapter seam."""
    from datasets import load_dataset

    system_prompt = (
        "You are a tool-calling assistant. Respond ONLY with a valid JSON object:\n"
        '{"tool": "<name>", "arguments": {<key>: <value>}}\n'
        "No prose. Only JSON."
    )
    examples: List[TrainingExample] = []
    dataset_kwargs: Dict[str, Any] = {"split": "train"}
    if revision is not None:
        dataset_kwargs["revision"] = revision
    for row in load_dataset("Salesforce/xlam-function-calling-60k", **dataset_kwargs):
        try:
            tools = (
                json.loads(row.get("tools", "[]"))
                if isinstance(row.get("tools"), str)
                else row.get("tools", [])
            )
            answers = (
                json.loads(row.get("answers", "[]"))
                if isinstance(row.get("answers"), str)
                else row.get("answers", [])
            )
            if not isinstance(answers, list) or not answers:
                continue
            answer = answers[0]
            tool = answer.get("name", answer.get("tool", ""))
            arguments = answer.get("arguments", answer.get("parameters", {}))
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not tool:
                continue
            prompt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\nAvailable tools:\n{json.dumps(tools[:8])}\n\n"
                f"User: {row.get('query', row.get('instruction', ''))}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            examples.append(
                TrainingExample(
                    prompt=prompt,
                    target={"tool": tool, "arguments": arguments},
                )
            )
        except (json.JSONDecodeError, TypeError, AttributeError):
            continue
    random.Random(seed).shuffle(examples)
    return InMemoryDataset(train=examples[:3000], test=examples[3000:3500])


def _api_bank_prompt(raw_prompt: str, tool_name: str) -> str:
    """Convert one API-Bank dialogue into an unambiguous JSON-call prompt."""
    try:
        messages = ast.literal_eval(raw_prompt)
    except (SyntaxError, ValueError) as exc:
        raise ValueError("API-Bank prompt is not a literal message list") from exc
    if not isinstance(messages, list) or len(messages) != 1 or not isinstance(messages[0], dict):
        raise ValueError("API-Bank prompt must contain exactly one message object")
    content = messages[0].get("content")
    if not isinstance(content, str) or "**Available Tools**" not in content or "[USER]" not in content:
        raise ValueError("API-Bank prompt is missing tools or dialogue history")
    tools_section = re.split(
        r"\*\*(?:Steps|Output Format)",
        content.split("**Available Tools**", 1)[1],
        maxsplit=1,
    )[0]
    tool_blocks = re.split(r"(?m)(?=^\d+\. Name:)", tools_section)
    matching_blocks = [
        block.strip()
        for block in tool_blocks
        if re.search(rf"(?m)^\d+\. Name:\s*{re.escape(tool_name)}\s*$", block)
    ]
    if len(matching_blocks) != 1:
        raise ValueError(f"API-Bank prompt does not define target tool {tool_name!r} exactly once")
    tool_contract = "Available tool:\n" + matching_blocks[0][:1800]
    dialogue = content[content.rfind("[USER]") :].strip()[-3500:]
    system = (
        "You are an enterprise tool-calling assistant. Use the available tool contract and "
        "dialogue history. Return ONLY one strict JSON object with keys 'tool' and "
        "'arguments'. Do not emit reasoning, XML, Markdown, or prose."
    )
    return (
        f"<|im_start|>system\n{system}\n\n{tool_contract}<|im_end|>\n"
        f"<|im_start|>user\n{dialogue}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _api_bank_target(raw_target: str) -> Dict[str, Any]:
    """Reduce API-Bank's dense nullable schema to the actual tool arguments."""
    try:
        payload = json.loads(raw_target)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("API-Bank ground truth is not JSON") from exc
    name = payload.get("name") if isinstance(payload, dict) else None
    parameters = payload.get("parameters") if isinstance(payload, dict) else None
    if not isinstance(name, str) or not name or not isinstance(parameters, dict):
        raise ValueError("API-Bank ground truth is missing name or parameters")
    return {
        "tool": name,
        "arguments": {str(key): value for key, value in parameters.items() if value is not None},
    }


def _swe_gym_prompt(row: Dict[str, Any]) -> str:
    """Build a patch-only prompt from one pinned SWE-Gym task."""
    problem = str(row.get("problem_statement") or "").strip()
    repo = str(row.get("repo") or "").strip()
    base_commit = str(row.get("base_commit") or "").strip()
    if not problem or not repo or not re.fullmatch(r"[0-9a-f]{40}", base_commit):
        raise ValueError("SWE-Gym task is missing repo, problem statement, or base commit")
    hints = str(row.get("hints_text") or "").strip()
    tests = sorted(
        str(item)
        for key in ("FAIL_TO_PASS", "PASS_TO_PASS")
        for item in (row.get(key) or [])
    )
    context = [
        f"Repository: {repo}",
        f"Base commit: {base_commit}",
        "Issue:",
        problem[:4000],
    ]
    if hints:
        context.extend(("Hints:", hints[:1000]))
    if tests:
        context.extend(("Tests that must pass:", "\n".join(tests)[:1200]))
    system = (
        "You are a software repair agent. Produce ONLY a unified diff that implements the "
        "requested fix. Start with 'diff --git'. Do not include Markdown fences or prose."
    )
    context_text = "\n".join(context)
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{context_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def make_pavlov_non_xlam_dataset(
    seed: int = 809, *, repo_root: Optional[Path] = None
) -> InMemoryDataset:
    """Load the pinned, locally decontaminated API-Bank + SWE-Gym training mix.

    This adapter deliberately excludes xLAM and OpenR1.  OpenR1 is held back
    until FrontierMath decontamination can be proven.  SWE-Gym is rejected if
    it overlaps the locally materialized E1/E2 identifiers; API-Bank is
    rejected if it overlaps the locally materialized E4 identifiers.
    """
    from datasets import load_dataset

    repo_root = repo_root or Path(__file__).resolve().parents[2]
    e1_path = repo_root / "outputs/e1_swe_bench_pro/hf_dataset/data/test-00000-of-00001.parquet"
    e2_path = repo_root / "outputs/e2_frontier_swe/frontier-swe/tasks"
    e4_path = repo_root / "outputs/e4_banker_toolbench/official_repo_ff6db552/native-data/tasks.jsonl"
    for required in (e1_path, e2_path, e4_path):
        if not required.exists():
            raise RuntimeError(f"required decontamination input is missing: {required}")

    e1 = load_dataset("parquet", data_files=str(e1_path), split="train")
    e1_ids = {str(value) for value in e1["instance_id"]}
    e1_pairs = {
        (str(repo), str(commit)) for repo, commit in zip(e1["repo"], e1["base_commit"])
    }
    e2_ids = {path.name for path in e2_path.iterdir() if path.is_dir()}

    swe = load_dataset(
        "SWE-Gym/SWE-Gym",
        split="train",
        revision=PAVLOV_NON_XLAM_SOURCE_REVISIONS["SWE-Gym/SWE-Gym"],
    )
    swe_rows: List[Dict[str, Any]] = []
    for raw in swe:
        row = dict(raw)
        task_id = str(row.get("instance_id") or "")
        pair = (str(row.get("repo") or ""), str(row.get("base_commit") or ""))
        if task_id in e1_ids or pair in e1_pairs or task_id in e2_ids:
            raise RuntimeError(f"SWE-Gym contamination detected for task {task_id}")
        swe_rows.append(row)

    e4_ids: set[str] = set()
    for line in e4_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        value = row.get("task_id") or row.get("id") or row.get("instance_id")
        if value is not None:
            e4_ids.add(str(value))

    api = load_dataset(
        "Simu-Env/API-Bank-RLVR",
        revision=PAVLOV_NON_XLAM_SOURCE_REVISIONS["Simu-Env/API-Bank-RLVR"],
    )
    api_train: List[TrainingExample] = []
    api_validation: List[TrainingExample] = []
    for split_name, destination in (("train", api_train), ("validation", api_validation)):
        for row in api[split_name]:
            extra = json.loads(row["extra_info"])
            task_id = str(extra.get("index"))
            if task_id in e4_ids:
                raise RuntimeError(f"API-Bank contamination detected for task {task_id}")
            target = _api_bank_target(row["ground_truth"])
            try:
                prompt = _api_bank_prompt(row["prompt"], str(target["tool"]))
            except ValueError:
                # The pinned source contains a small number of rows whose
                # label names a tool that is absent from the supplied contract.
                # Those rows cannot be trained safely and are excluded.
                continue
            destination.append(
                TrainingExample(
                    prompt=prompt,
                    target=target,
                    metadata={
                        "suite_id": "api_bank_rlvr_train",
                        "source_id": task_id,
                        "reward_kind": "tool_call",
                        "source_revision": PAVLOV_NON_XLAM_SOURCE_REVISIONS[
                            "Simu-Env/API-Bank-RLVR"
                        ],
                    },
                )
            )

    rng = random.Random(seed)
    rng.shuffle(api_train)
    rng.shuffle(api_validation)
    rng.shuffle(swe_rows)
    swe_train_rows = swe_rows[:256]
    swe_validation_rows = swe_rows[256:320]
    swe_train = [
        TrainingExample(
            prompt=_swe_gym_prompt(row),
            target=str(row["patch"]),
            metadata={
                "suite_id": "swe_gym_train",
                "source_id": str(row["instance_id"]),
                "reward_kind": "patch",
                "source_revision": PAVLOV_NON_XLAM_SOURCE_REVISIONS["SWE-Gym/SWE-Gym"],
            },
        )
        for row in swe_train_rows
    ]
    swe_validation = [
        TrainingExample(
            prompt=_swe_gym_prompt(row),
            target=str(row["patch"]),
            metadata={
                "suite_id": "swe_gym_train",
                "source_id": str(row["instance_id"]),
                "reward_kind": "patch",
                "source_revision": PAVLOV_NON_XLAM_SOURCE_REVISIONS["SWE-Gym/SWE-Gym"],
            },
        )
        for row in swe_validation_rows
    ]
    train = api_train[:256] + swe_train
    test = api_validation[:64] + swe_validation
    rng.shuffle(train)
    rng.shuffle(test)
    if len(train) != 512 or len(test) != 128:
        raise RuntimeError("non-xLAM dataset mix has unexpected train/test counts")
    return InMemoryDataset(train=train, test=test)


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


class _DuplicateJSONKey(ValueError):
    """Raised when duplicate JSON keys make a tool call ambiguous."""


_MISSING_TOOL_FIELD = object()
_CONFLICTING_TOOL_FIELD = object()


def _strict_json_object(response: str) -> Optional[Dict[str, Any]]:
    """Parse exactly one JSON object, rejecting prose, duplicates, and arrays."""
    if not isinstance(response, str) or not response.strip():
        return None

    def object_pairs_hook(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
        keys = [key for key, _ in pairs]
        if len(keys) != len(set(keys)):
            raise _DuplicateJSONKey("duplicate JSON key")
        return dict(pairs)

    def reject_nonstandard_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    try:
        parsed = json.loads(
            response.strip(),
            object_pairs_hook=object_pairs_hook,
            parse_constant=reject_nonstandard_constant,
        )
    except (json.JSONDecodeError, _DuplicateJSONKey, TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _tool_alias_value(payload: Dict[str, Any], primary: str, alias: str) -> Any:
    """Resolve an alias pair while rejecting conflicting values."""
    present = [payload[key] for key in (primary, alias) if key in payload]
    if not present:
        return _MISSING_TOOL_FIELD
    if len(present) == 2 and present[0] != present[1]:
        return _CONFLICTING_TOOL_FIELD
    return present[0]


def _canonical_tool_value(value: Any) -> Any:
    """Canonicalize JSON structure while preserving case and string whitespace."""
    if isinstance(value, dict):
        return {str(k): _canonical_tool_value(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_canonical_tool_value(v) for v in value]
    return value


class StrictToolCallReward:
    """Verifier-backed reward for one exact, unambiguous JSON tool call."""

    def score(self, response: str, example: TrainingExample) -> float:
        target = example.target if isinstance(example.target, dict) else {}
        tool_name = _tool_alias_value(target, "tool", "name")
        arguments = _tool_alias_value(target, "arguments", "parameters")
        if (
            tool_name is _MISSING_TOOL_FIELD
            or tool_name is _CONFLICTING_TOOL_FIELD
            or not isinstance(tool_name, str)
        ):
            return 0.0
        if arguments is _MISSING_TOOL_FIELD:
            arguments = {}
        if arguments is _CONFLICTING_TOOL_FIELD or not isinstance(arguments, dict):
            return 0.0

        parsed = _strict_json_object(response)
        if parsed is None:
            return 0.0
        allowed_keys = {"tool", "name", "arguments", "parameters"}
        if set(parsed) - allowed_keys:
            return 0.0
        predicted_tool = _tool_alias_value(parsed, "tool", "name")
        predicted_args = _tool_alias_value(parsed, "arguments", "parameters")
        if (
            predicted_tool is _MISSING_TOOL_FIELD
            or predicted_tool is _CONFLICTING_TOOL_FIELD
            or not isinstance(predicted_tool, str)
            or predicted_args is _MISSING_TOOL_FIELD
            or predicted_args is _CONFLICTING_TOOL_FIELD
            or not isinstance(predicted_args, dict)
        ):
            return 0.0

        score = 0.1
        if predicted_tool != tool_name:
            return score
        score += 0.4

        if not arguments:
            return 1.0 if not predicted_args else score

        expected_keys = set(arguments)
        predicted_keys = set(predicted_args)
        key_recall = len(expected_keys & predicted_keys) / len(expected_keys)
        key_precision = len(expected_keys & predicted_keys) / max(len(predicted_keys), 1)
        key_f1 = (
            2 * key_precision * key_recall / (key_precision + key_recall)
            if key_precision + key_recall
            else 0.0
        )
        score += 0.2 * key_f1
        exact_values = sum(
            _canonical_tool_value(predicted_args.get(key))
            == _canonical_tool_value(arguments[key])
            for key in expected_keys
            if key in predicted_args
        )
        score += 0.3 * exact_values / len(expected_keys)
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


class ExactMathReward:
    """Binary boxed-or-final-number reward used by held-out math benchmarks."""

    def score(self, response: str, example: TrainingExample) -> float:
        answer = str(example.target or "")
        boxed = re.findall(r"\\boxed\{([^}]+)\}", response.strip())
        candidates = [item.strip().replace(",", "").replace(" ", "") for item in boxed]
        all_numbers = re.findall(r"[-+]?\d[\d,]*\.?\d*", response)
        if all_numbers:
            candidates.append(all_numbers[-1].replace(",", ""))
        for candidate in candidates:
            try:
                if abs(float(candidate) - float(answer)) < 0.01:
                    return 1.0
            except ValueError:
                if candidate == answer:
                    return 1.0
        return 0.0


class PatchReward:
    """Dense, format-aware reward for a SWE-Gym unified diff candidate."""

    @staticmethod
    def _normalise(value: str) -> str:
        return "\n".join(line.rstrip() for line in value.strip().splitlines() if line.strip())

    def score(self, response: str, example: TrainingExample) -> float:
        expected = self._normalise(str(example.target or ""))
        candidate = self._normalise(response)
        if not expected or not candidate:
            return 0.0
        if candidate == expected:
            return 1.0
        if not candidate.startswith("diff --git"):
            return 0.0
        expected_files = set(re.findall(r"^diff --git a/(\S+) b/(\S+)$", expected, re.MULTILINE))
        candidate_files = set(re.findall(r"^diff --git a/(\S+) b/(\S+)$", candidate, re.MULTILINE))
        file_overlap = len(expected_files & candidate_files) / max(len(expected_files), 1)
        similarity = difflib.SequenceMatcher(None, expected, candidate, autojunk=False).ratio()
        return min(0.2 + 0.3 * file_overlap + 0.5 * similarity, 0.99)


class PavlovNonXLAMReward:
    """Route verifier-backed rewards for the non-xLAM training portfolio."""

    def __init__(self) -> None:
        self._tool = StrictToolCallReward()
        self._patch = PatchReward()

    def score(self, response: str, example: TrainingExample) -> float:
        # Qwen's chat template exposes its reasoning channel in decoded token
        # text.  Score only the answer after a completed reasoning block, just
        # as the E11 verifier extracts HDL after ``</think>``.  An unclosed
        # block remains invalid and receives no special treatment.
        if "</think>" in response:
            response = response.rsplit("</think>", 1)[1].lstrip()
        reward_kind = example.metadata.get("reward_kind")
        if reward_kind == "tool_call":
            return self._tool.score(response, example)
        if reward_kind == "patch":
            return self._patch.score(response, example)
        return 0.0
