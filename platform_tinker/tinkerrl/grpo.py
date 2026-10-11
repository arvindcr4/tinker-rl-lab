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
import io
import json
import math
import os
import random
import re
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import MISSING, asdict, dataclass, field, fields, replace
from decimal import Context, Decimal, InvalidOperation, ROUND_FLOOR, ROUND_HALF_EVEN, localcontext
from numbers import Real
from pathlib import Path
from urllib.parse import urlsplit
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

from .critic import (
    PromptValueCritic,
    explained_variance,
    require_finite_critic_state,
    train_critic_step,
)


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
# split description.  Of the other eight, four carry frozen public pins and
# four stay pending (see _PUBLIC_SUITE_RECEIPTS).
PAVLOV_HELDOUT_SUITE_IDS: Tuple[str, ...] = (
    "agentharm_eval",
    "apex_agents_eval",
    "appbench_eval",
    "banker_toolbench_eval",
    "frontiermath_eval",
    "openreward_games_eval",
)


def _pending_suite_receipt() -> Dict[str, Any]:
    return {
        "frozen": False,
        "split": None,
        "hash": None,
        "license": None,
        "runtime": None,
        "decontamination": None,
        "source": None,
    }


# Public pins for the eight non-held-out suites, researched 2026-10-03 from
# canonical sources (paper, official repo, HuggingFace dataset — every value
# read from the live source).  Four suites freeze here; four stay pending
# with their partial findings recorded.  ``None`` is a verified absence
# (field does not exist publicly), never a skipped lookup.
_PUBLIC_SUITE_RECEIPTS: Dict[str, Dict[str, Any]] = {
    "mle_bench_eval": {
        "frozen": True,
        "split": "Full 75 comps (split75.txt); Lite 21 (low.txt); pkg v1.0.0",
        "hash": "507f92e1138bb6e40dac5c6ee7a6758e6424bf97 "
        "(openai/mle-bench HEAD 2026-10-03; no tags)",
        "license": "MIT code only; datasets per-competition Kaggle terms",
        "runtime": "mlebench CLI + Docker mlebench-env grading server",
        "decontamination": "No pretraining decon; effects measured instead "
        "(familiarity + obfuscation ablations, arXiv:2410.07095 §4)",
        "source": "https://github.com/openai/mle-bench",
    },
    "swe_bench_pro_eval": {
        "frozen": True,
        "split": "test (default V2 642 tasks v2.0.0; hard 51; v1 731/tag v1.0)",
        "hash": "2d52cb3df914a3fcf80c7f66738b3a88ae37fc50 (HF ScaleAI/SWE-bench_Pro 2026-09-22)",
        "license": "MIT harness; task content per upstream repo licenses",
        "runtime": "Harbor task dirs + ghcr.io/scaleapi/swe-bench_pro-v2 "
        "images; hidden F2P/P2P re-grade",
        "decontamination": "GPL/copyleft-only repos as legal barrier; "
        "held-out set retained; V2 sanitised bundles",
        "source": "https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro",
    },
    "verilog_eval": {
        "frozen": True,
        "split": "v2: spec-to-rtl + code-complete-iccad2023 (156 HDLBits, "
        "Human only); v1: Human/Machine JSONLs",
        "hash": "v2.0.0=c498220d0a52248f8e3fdffe279075215bde2da6; "
        "v1.0.0=4b9b16e92f1d9cc520afbfa3ecd5a2f20a350fd5",
        "license": "MIT (NVIDIA 2023-2024 + OpenAI human-eval portion)",
        "runtime": "v1 Docker+pip (iverilog pinned); v2 bare-metal make, iverilog v12",
        "decontamination": "Partial: MinHash dedup on synthetic SFT data "
        "only; eval set itself not deconed",
        "source": "https://github.com/NVlabs/verilog-eval",
    },
    "webbench_eval": {
        "frozen": True,
        "split": "webbenchfinal.csv: 2454 tasks / 452 live sites; single set, no splits",
        "hash": "repo ea7a1628443321363989f354401f0653e0cba6f4; HF "
        "Halluminate/WebBench 48b35267b72544a6cc7fb13fab4ad947be22c3b0",
        "license": "MIT (repo + HF card)",
        "runtime": "No public harness: live-site agents graded by human "
        "annotators; leaderboard webbench.ai",
        "decontamination": "None documented (full-text searched; unreleased holdouts exist)",
        "source": "https://github.com/Halluminate/WebBench",
    },
    "binaryaudit_eval": {
        "frozen": False,
        "split": "tasks/ (full suite; 46 task dirs; no named split/version)",
        "hash": "cbd86c7cd8519f01ae6b7ad7db7fdb653ea54f23 (main HEAD "
        "2026-02-27; re-verified 2026-10-03: HEAD unchanged, still no tags)",
        "license": "Apache-2.0 claimed in README; no LICENSE file, GitHub "
        "API license=null (re-verified 2026-10-03)",
        "runtime": "Harbor harness + Docker per-task containers (Ghidra/Radare2)",
        "decontamination": None,
        "source": "https://github.com/QuesmaOrg/BinaryAudit",
    },
    "frontier_swe_eval": {
        "frozen": False,
        "split": "17 tasks @422b9bb (task.toml + instruction.md + job.yaml "
        "+ tests/test.sh per task; task_id_disjoint; exact eval over named tasks)",
        "hash": "422b9bb95deb8efe436becb0ed3c44be23611e10 "
        "(Proximal-Labs/frontier-swe main @2026-08-07; commit + 17/17 task "
        "IDs verified live 2026-10-03)",
        "license": "No LICENSE/COPYING file @pinned rev; GitHub API "
        "license=null; no license statement in README (verified 2026-10-03)",
        "runtime": "tests/test.sh in published per-task Docker image; "
        "reward.json scored by scripts/score_from_reward.py (SCORING.md); "
        "leaderboard frontierswe.com",
        "decontamination": None,
        "source": "https://github.com/Proximal-Labs/frontier-swe",
    },
    "lifescibench_eval": {
        "frozen": False,
        "split": "750 expert tasks, single eval set (7 workflows x 7 "
        "domains); rubric score + pass rate",
        "hash": None,
        "license": None,
        "runtime": "Single-turn + browsing; expert-rubric grading; no public harness",
        "decontamination": None,
        "source": "bioRxiv DOI 10.64898/2026.08.13.744657 v1 (paper-only; "
        "DOI verified via Crossref 2026-10-03; 0 GitHub repos, 0 HF "
        "datasets, no data/code links on bioRxiv page)",
    },
    "sdab_eval": {
        "frozen": False,
        "split": None,
        "hash": None,
        "license": None,
        "runtime": "Provider-run (Emulated, Inc.): Docker + cloudboxes; no "
        "public harness; gated access (samples upon request)",
        "decontamination": None,
        "source": "https://emulated.so/sdab (80-task SDAB, Apr 2026; page "
        "re-verified live 2026-10-03; no public paper/repo/dataset/license)",
    },
}


# Machine-checkable per-suite receipts for all 14 primary-evaluation IDs.
# The six held-out suites carry campaign split descriptions; four more
# suites carry frozen public pins; four stay pending with partial findings.
# No hashes are invented here: ``None`` means unpinned, full stop.
PAVLOV_SUITE_RECEIPTS: Dict[str, Dict[str, Any]] = {
    suite_id: (
        dict(_PUBLIC_SUITE_RECEIPTS[suite_id])
        if suite_id in _PUBLIC_SUITE_RECEIPTS
        else (
            {
                "frozen": True,
                "split": "held-out/private split described with suite",
                "hash": None,
                "license": None,
                "runtime": None,
                "decontamination": None,
                "source": None,
            }
            if suite_id in PAVLOV_HELDOUT_SUITE_IDS
            else _pending_suite_receipt()
        )
    )
    for suite_id in PAVLOV_PRIMARY_EVALUATION_SUITE_IDS
}


def pavlov_suite_receipt(suite_id: str) -> Dict[str, Any]:
    """Return a copy of the receipt record for a primary-evaluation suite.

    Raises ``KeyError`` for unknown suite IDs: lookups fail closed.
    """
    return dict(PAVLOV_SUITE_RECEIPTS[suite_id])


def require_frozen_suite_receipt(suite_id: str) -> Dict[str, Any]:
    """Return the receipt, refusing suites without frozen receipts.

    Adapter work (BFCL, SWE-bench, LiveCodeBench, ...) must gate on this so
    a pending suite can never silently pass as evaluated.
    """
    receipt = pavlov_suite_receipt(suite_id)
    if not receipt["frozen"] or not receipt["split"]:
        raise ValueError(
            f"suite {suite_id!r} has no frozen receipt "
            "(split/hash/license/runtime/decontamination pending)"
        )
    return receipt


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


_METADATA_SEQUENCE_FIELDS = (
    "wandb_tags",
    "training_suite_ids",
    "heldout_suite_ids",
    "held_out_suite_ids",
    "primary_evaluation_suite_ids",
    "domain_tags",
    "declared_domains",
    "training_domain_union",
    "primary_evaluation_domain_union",
)


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
    # Pass a per-request seed to the sampler. Off by default: whether Tinker
    # seeds each sample of a num_samples=G request independently is unverified,
    # and a shared seed would make every group identical (zero advantages).
    seed_sampling: bool = False
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
    # Optional declared source revisions. Dataset adapters must honor their
    # revision. model_revision is forwarded to the local tokenizer loader and
    # run metadata; it does not attest the managed provider's weight commit.
    dataset_revision: Optional[str] = None
    model_revision: Optional[str] = None
    # DAPO / Dr. GRPO training-shaping flags.  All default off so existing
    # presets reproduce their current behavior exactly.
    debias_advantages: bool = False
    dynamic_sampling: bool = False
    dynamic_sampling_max_resamples: int = 16
    mask_truncated_responses: bool = False
    # REINFORCE++ batch-global advantage normalization (Hu, 2025).  Off by
    # default; per-group path stays byte-identical.
    global_advantage_normalization: bool = False
    global_norm_exclude_truncated: bool = True
    # VAPO-style NLL auxiliary loss on correct responses (VAPO §4.3
    # "Positive Example LM Loss", Yue et al. 2025; paper uses μ=0.1).
    # Disabled and zero-weighted by default, so the loss is unchanged
    # unless opted in.
    nll_aux_enabled: bool = False
    nll_aux_coef: float = 0.0
    nll_aux_min_reward: float = 1.0
    # GSPO sequence-ratio loss mode (Zheng et al. 2025, arXiv:2507.18071
    # Eq. 5-7; paper clips 3e-4/4e-4, §4.1).  Off by default; the loop
    # keeps the plain GRPO loss unless opted in.
    gspo_enabled: bool = False
    gspo_epsilon_low: float = 3e-4
    gspo_epsilon_high: float = 4e-4
    gspo_update_epochs: int = 1
    # Prompt-conditioned value baseline (critic).  Off by default.  When on,
    # advantages are R - V(x) normalized batch-wide (implies the batch-wide
    # advantage path; per-group centering would cancel the baseline).
    critic_enabled: bool = False
    critic_lr: float = 1e-3
    critic_hidden_dim: int = 64
    critic_updates_per_step: int = 4
    critic_pretrain_batches: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.checkpoint_dir, os.PathLike):
            object.__setattr__(self, "checkpoint_dir", os.fspath(self.checkpoint_dir))
        self._validate_metadata_sequences()
        # Detach caller lists without treating a string as one label per character.
        for name in _METADATA_SEQUENCE_FIELDS:
            object.__setattr__(self, name, tuple(getattr(self, name) or ()))
        self.validate_numerics()

    def _validate_metadata_sequences(self) -> None:
        for name in _METADATA_SEQUENCE_FIELDS:
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, (list, tuple))
                or any(not isinstance(item, str) or not item.strip() for item in value)
            ):
                raise ValueError(f"{name} must be a list or tuple of non-empty strings, or None")

    def _validate_boundary_inputs(self) -> None:
        self._validate_metadata_sequences()
        for name in ("name", "model", "checkpoint_dir"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in (
            "model_revision",
            "dataset_revision",
            "hf_owner",
            "hf_repo_prefix",
            "checkpoint_name_prefix",
            "wandb_entity",
        ):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-empty string or None")
        # Never let strings such as hf_public="false" become public uploads.
        # Optional campaign flags retain their separate fail-closed gate.
        for spec in fields(self):
            if isinstance(spec.default, bool) and type(getattr(self, spec.name)) is not bool:
                raise ValueError(f"{spec.name} must be a boolean")
        value = self.base_url
        if value is None:
            return
        # Endpoint overrides are persisted in tracking/receipt configs. Reject
        # URL userinfo, and characters URL parsers may silently reinterpret.
        if (
            not isinstance(value, str)
            or not value
            or "\\" in value
            or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value)
        ):
            raise ValueError("base_url must be an HTTP(S) URL without whitespace or controls")
        try:
            url = urlsplit(value)
            hostname, port = url.hostname, url.port
        except ValueError:
            raise ValueError("base_url has an invalid host or port") from None
        if url.scheme not in {"http", "https"} or not hostname:
            raise ValueError("base_url must name an HTTP(S) service")
        if "@" in url.netloc:
            raise ValueError("base_url must not contain userinfo; endpoint config is logged")
        if (port is not None and not 1 <= port <= 65535) or url.netloc.endswith(":"):
            raise ValueError("base_url has an invalid port")
        if "#" in value:
            raise ValueError("base_url must not contain a fragment")

    def validate_numerics(self) -> None:
        """Reject malformed budgets before datasets, tracking, or provider work.

        Zero-step runs and rank zero retain their existing meaning. Single-item
        groups and zero temperature remain valid; this is not a provider-specific
        model-capability validator.
        """
        self._validate_boundary_inputs()
        minima = {
            "steps": 0,
            "lora_rank": 0,
            "group_size": 1,
            "batch_size": 1,
            "max_prompt_tokens": 1,
            "max_response_tokens": 1,
            "num_seeds": 1,
            "dynamic_sampling_max_resamples": 0,
            "gspo_update_epochs": 1,
            "critic_hidden_dim": 1,
            "critic_updates_per_step": 0,
            "critic_pretrain_batches": 0,
        }
        for name, minimum in minima.items():
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if self.save_every is not None and (
            type(self.save_every) is not int or self.save_every < 1
        ):
            raise ValueError("save_every must be None or an integer >= 1")
        for name in (
            "lr",
            "temperature",
            "top_p",
            "beta1",
            "beta2",
            "eps",
            "nll_aux_coef",
            "nll_aux_min_reward",
            "critic_lr",
            "gspo_epsilon_low",
            "gspo_epsilon_high",
        ):
            value = getattr(self, name)
            try:
                finite = type(value) in (int, float) and math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError(
                    f"{name}: gspo clip epsilons must be finite and >= 0"
                    if name.startswith("gspo_epsilon_")
                    else f"{name} must be a finite number"
                )
        # Necessary arithmetic/probability domains, not a guarantee that a
        # particular remote model accepts every otherwise valid setting.
        for name in ("lr", "eps", "temperature"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        for name in ("beta1", "beta2"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        if not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be in [0, 1]")
        if self.gspo_epsilon_low < 0 or self.gspo_epsilon_high < 0:
            raise ValueError("gspo clip epsilons must be finite and >= 0")
        if self.gspo_enabled and self.nll_aux_enabled:
            raise ValueError("gspo_enabled and nll_aux_enabled are mutually exclusive")
        if self.critic_lr <= 0:
            raise ValueError("critic_lr must be > 0")

    def effective_save_every(self) -> int:
        return self.save_every if self.save_every is not None else max(self.steps // 4, 10)

    def validate_tracking(self) -> None:
        """Reject configurations that would silently skip campaign receipts."""
        self._validate_boundary_inputs()
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
        if status is not None and (not isinstance(status, str) or not status.strip()):
            raise ValueError("campaign contract status is required when supplied")
        normalized_status = status.strip().lower().replace("_", "-") if status else None
        launchable_statuses = {"ready", "authorized", "approved", "pass"}
        if normalized_status is not None and normalized_status not in launchable_statuses:
            raise ValueError(f"campaign contract is not launchable: status={normalized_status}")

        budget_status = self.budget_status
        if budget_status is not None and (
            not isinstance(budget_status, str) or not budget_status.strip()
        ):
            raise ValueError("campaign budget status is required when supplied")
        normalized_budget_status = (
            budget_status.strip().lower().replace("_", "-") if budget_status else None
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
        for field_name in ("authorized_budget_usd", "maximum_usd"):
            value = getattr(self, field_name)
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{field_name} must be numeric")
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
        if (
            self.authorized_budget_usd is not None
            and self.maximum_usd is not None
            and self.authorized_budget_usd > self.maximum_usd
        ):
            raise ValueError("authorized budget exceeds the configured maximum_usd")

        if self.paid_jobs_may_launch:
            if normalized_status not in launchable_statuses:
                raise ValueError("paid jobs require an explicitly launchable campaign status")
            if self.authorized_budget_usd is None or self.maximum_usd is None:
                raise ValueError("paid jobs require explicit authorized_budget_usd and maximum_usd")


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
    # Empty means unrecorded (legacy), never inferred weight attestation.
    model_source: Dict[str, Any] = field(default_factory=dict)
    # Empty means legacy-unrecorded, not a disabled or successful evaluation.
    heldout_evaluation: Dict[str, Any] = field(default_factory=dict)
    dataset_source: Dict[str, Any] = field(default_factory=dict)


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


def _reward_mean(values: Sequence[float]) -> float:
    """Keep historical arithmetic unless its intermediate sum overflows."""
    result = sum(values) / len(values)
    if math.isfinite(result):
        return result
    context = Context(
        prec=1600 + len(str(len(values))), rounding=ROUND_HALF_EVEN, Emin=-999999, Emax=999999
    )
    with localcontext(context):
        return float(sum((Decimal.from_float(float(v)) for v in values), Decimal(0)) / len(values))


def _normalize_extreme_rewards(
    rewards: Sequence[float], basis: Sequence[float], epsilon: float, unbiased: bool
) -> List[float]:
    """Rare fallback when binary64 moments fail, without clamping advantages."""
    # Exact finite binary64 values span fewer than 1,400 decimal positions.
    # Preserve even tiny terms between cancelling extreme values; a usual
    # 28/80-digit Decimal context would lose those terms before centering.
    context = Context(
        prec=1600 + len(str(len(basis))),
        rounding=ROUND_HALF_EVEN,
        Emin=-999999,
        Emax=999999,
    )
    with localcontext(context):
        reference = [Decimal.from_float(float(value)) for value in basis]
        mean = sum(reference, Decimal(0)) / len(reference)
        centered = [Decimal.from_float(float(value)) - mean for value in rewards]
        if unbiased:
            result = [float(value) for value in centered]
        else:
            variance = sum(((value - mean) ** 2 for value in reference), Decimal(0)) / len(
                reference
            )
            denominator = variance.sqrt() + Decimal.from_float(float(epsilon))
            result = [float(value / denominator) for value in centered]
    if not all(math.isfinite(value) for value in result):
        raise ValueError("advantages cannot be represented as finite floats")
    return result


def normalize_rewards(
    rewards: Sequence[float], epsilon: float = 1e-8, unbiased: bool = False
) -> List[float]:
    """Group-relative advantages.

    Default matches historical behavior (mean 0, std 1).  With
    ``unbiased=True`` (Dr. GRPO: Liu et al., COLM 2025), return
    ``R - mean(R)`` with no std division, so near-uniform groups cannot
    explode.  This is the std-division half of the Dr. GRPO fix; this
    loss never length-normalizes, so the paper's length-norm removal has
    no counterpart here.
    """
    if not all(math.isfinite(r) for r in rewards):
        raise ValueError("rewards must be finite")
    if not unbiased and (not math.isfinite(epsilon) or epsilon <= 0):
        raise ValueError("epsilon must be finite and positive for standardized advantages")
    n = len(rewards)
    if n == 0:
        return []
    # Exactly tied rewards carry no policy signal, including fractional ties
    # whose naive sum/mean can otherwise introduce an epsilon-amplified residue.
    if all(r == rewards[0] for r in rewards):
        return [0.0] * n
    try:
        mean_r = math.fsum(rewards) / n
        centered = [r - mean_r for r in rewards]
        if unbiased:
            if all(math.isfinite(value) for value in centered):
                return centered
        else:
            variance = sum((r - mean_r) ** 2 for r in rewards) / n
            std_r = variance**0.5 + epsilon
            if math.isfinite(std_r) and (variance != 0 or not any(centered)):
                result = [(r - mean_r) / std_r for r in rewards]
                if all(math.isfinite(value) for value in result):
                    return result
    except OverflowError:
        pass
    return _normalize_extreme_rewards(rewards, rewards, epsilon, unbiased)


def is_degenerate_group(rewards: Sequence[float], epsilon: float = 1e-8) -> bool:
    """True when a group carries no learnable contrast (DAPO, Yu et al. 2025, §3.2).

    All-correct / all-wrong groups center to zero advantage, so dynamic
    sampling skips them instead of spending forward/backward on noise.
    Groups with fewer than two samples are degenerate by definition.
    """
    if len(rewards) < 2:
        return True
    first = rewards[0]
    return all(abs(r - first) <= epsilon for r in rewards[1:])


def apply_truncation_mask(
    advantages: Sequence[float], token_lengths: Sequence[int], max_tokens: int
) -> List[float]:
    """Zero advantages for responses that hit the sampling token budget.

    A truncated response may be sound-but-incomplete; scoring it as a full
    negative injects reward noise (DAPO §3.4).  ``token_lengths`` must pair
    1:1 with ``advantages``.
    """
    if len(advantages) != len(token_lengths):
        raise ValueError(
            "advantages and token_lengths must pair 1:1, got "
            f"{len(advantages)} advantages and {len(token_lengths)} lengths"
        )
    return [0.0 if length >= max_tokens else adv for adv, length in zip(advantages, token_lengths)]


def normalize_advantages_global(
    rewards: Sequence[float],
    epsilon: float = 1e-8,
    unbiased: bool = False,
    exclude: Optional[Sequence[bool]] = None,
) -> List[float]:
    """Batch-global advantage normalization (REINFORCE++, Hu 2025).

    Same centering/scaling choice as :func:`normalize_rewards`, but the
    mean (and std) are pooled across the whole batch instead of per group.
    ``exclude`` marks entries to leave out of the pooled statistics (used
    for truncated responses); excluded entries are still centered with the
    pooled mean so the output stays 1:1 with the input.  Empty input gives
    ``[]``; a one-sample basis centers to ~0 (epsilon-guarded, no floor
    needed).  If every entry is excluded, statistics fall back to the full
    vector so the call still returns aligned values.
    """
    if not all(math.isfinite(r) for r in rewards):
        raise ValueError("rewards must be finite")
    if not unbiased and (not math.isfinite(epsilon) or epsilon <= 0):
        raise ValueError("epsilon must be finite and positive for standardized advantages")
    n = len(rewards)
    if n == 0:
        return []
    if exclude is None:
        basis_idx = list(range(n))
    else:
        if len(exclude) != n:
            raise ValueError(f"exclude must pair 1:1 with rewards, got {len(exclude)} for {n}")
        basis_idx = [i for i, drop in enumerate(exclude) if not drop]
    basis = [rewards[i] for i in basis_idx] or list(rewards)
    try:
        mean_b = basis[0] if all(r == basis[0] for r in basis) else math.fsum(basis) / len(basis)
        centered = [r - mean_b for r in rewards]
        if unbiased:
            if all(math.isfinite(value) for value in centered):
                return centered
        else:
            var_b = sum((r - mean_b) ** 2 for r in basis) / len(basis)
            std_b = var_b**0.5 + epsilon
            if math.isfinite(std_b) and (var_b != 0 or all(r == mean_b for r in basis)):
                result = [c / std_b for c in centered]
                if all(math.isfinite(value) for value in result):
                    return result
    except OverflowError:
        pass
    return _normalize_extreme_rewards(rewards, basis, epsilon, unbiased)


def _nll_mask_for_group(
    rewards: Sequence[float],
    truncated: Sequence[bool],
    min_reward: float,
    mask_truncated: bool,
) -> List[bool]:
    """Per-response NLL-auxiliary flags: correct and (when masking) untruncated."""
    return [
        (reward >= min_reward) and not (mask_truncated and was_truncated)
        for reward, was_truncated in zip(rewards, truncated)
    ]


def _finite_reward(value: Any) -> float:
    """Reject invalid adapter scores before they reach updates or receipts."""
    try:
        reward = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("reward adapter must return a finite numeric score") from exc
    if not math.isfinite(reward):
        raise ValueError("reward adapter must return a finite numeric score")
    return reward


def heldout_reward_summary(test_rewards: Sequence[float]) -> Dict[str, float]:
    """Mean (+ bootstrap CI when n >= 2) for held-out eval logging.

    Routes through :func:`utils.stats.compute_bootstrap_ci` so a held-out
    eval run with n >= 2 scores logs uncertainty bounds alongside the mean.
    The import is lazy because
    ``utils.stats`` pulls heavy plotting deps; when it is unavailable the
    summary degrades to mean-only instead of failing the run.  A single
    held-out score likewise logs mean-only: one observation has no spread.
    """
    rewards = [_finite_reward(r) for r in test_rewards]
    if not rewards:
        return {}
    mean_r = _reward_mean(rewards)
    summary: Dict[str, float] = {
        "test/reward": mean_r,
        "test/reward_n": float(len(rewards)),
    }
    if len(rewards) < 2:
        return summary
    try:
        from utils.stats import compute_bootstrap_ci
    except ImportError:
        return summary
    mean_ci, ci_low, ci_high = compute_bootstrap_ci(rewards)
    summary["test/reward"] = float(mean_ci)
    summary["test/reward_ci_low"] = float(ci_low)
    summary["test/reward_ci_high"] = float(ci_high)
    return summary


def _response_logprobs(
    logprobs_list: Any, response_lens: Optional[Sequence[int]]
) -> List[torch.Tensor]:
    """Slice each datum's logprobs down to its response tokens.

    ``_build_datum`` targets ``prompt + response`` shifted by one, so Tinker
    returns ``P + R - 1`` logprobs per datum; the response tokens
    ``y_{i,1..R}`` are the last ``R``.  ``response_lens=None`` keeps the
    full tensors (callers that already pass response-only logprobs).
    """
    logprobs_list = list(logprobs_list)
    if response_lens is None:
        return logprobs_list
    if len(response_lens) != len(logprobs_list):
        raise ValueError(
            "response_lens must pair 1:1 with logprobs, got "
            f"{len(response_lens)} lengths for {len(logprobs_list)} responses"
        )
    sliced = []
    for i, (logprobs, n) in enumerate(zip(logprobs_list, response_lens)):
        total = logprobs.shape[0]
        if n < 0 or n > total:
            raise ValueError(f"response length {n} out of range for {total} logprobs on {i}")
        sliced.append(logprobs[total - n :])
    return sliced


def _loss_accumulator(logprobs: torch.Tensor) -> torch.Tensor:
    # Keep float64 or float32 untouched; cast before subtraction/reduction so
    # half precision does not erase clip-scale policy deltas or overflow sums.
    return logprobs.float() if logprobs.dtype in (torch.float16, torch.bfloat16) else logprobs


def _check_low_precision_gradient(value: float, logprobs: torch.Tensor) -> None:
    # Promoting the objective must not newly allow gradients that overflow when
    # autograd casts them back to the original half-precision input tensor.
    if logprobs.dtype in (torch.float16, torch.bfloat16) and (
        not math.isfinite(value) or abs(value) > torch.finfo(logprobs.dtype).max
    ):
        raise ValueError("policy gradient cannot be represented in the input dtype")


def make_grpo_loss_fn(
    advantages: Sequence[float],
    nll_mask: Optional[Sequence[bool]] = None,
    nll_coef: float = 0.0,
    response_lens: Optional[Sequence[int]] = None,
) -> Callable:
    """Return a Tinker-compatible loss closure bound to rollout inputs.

    Input masks and response lengths are snapshotted before the SDK can
    await its forward request; later caller mutation cannot rebind the loss.

    With ``nll_mask``/``nll_coef`` set, adds a VAPO-style NLL auxiliary term
    (VAPO §4.3, ``L = L_PPO + μ·L_NLL``) over masked (correct) responses:
    token-mean NLL averaged over the masked set, ``0`` when the set is
    empty.  Token-mean (not the GRPO term's token-sum) so responses of
    different lengths weigh equally.  ``nll_mask=None`` or ``nll_coef=0.0``
    reproduces the GRPO term exactly.

    ``response_lens`` restricts both terms to response tokens (ch05:
    ``sum_t log pi(y_{i,t} | x, y_{i,<t})``); prompt-token logprobs then get
    zero gradient.  The training loop always passes it (fixed 2026-10-03:
    prompt tokens were previously inside the sum).
    """

    advantages = tuple(advantages)
    nll_mask = None if nll_mask is None else tuple(nll_mask)
    response_lens = None if response_lens is None else tuple(response_lens)
    if not all(math.isfinite(value) for value in advantages):
        raise ValueError("advantages must be finite")
    if not math.isfinite(nll_coef):
        raise ValueError("nll_coef must be finite")

    def _loss_fn(data: Any, logprobs_list: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
        resp_logprobs = _response_logprobs(logprobs_list, response_lens)
        if len(advantages) != len(resp_logprobs):
            raise ValueError("advantages must pair 1:1 with logprobs")
        if nll_mask is not None and len(nll_mask) != len(resp_logprobs):
            raise ValueError(
                "nll_mask must pair 1:1 with logprobs, got "
                f"{len(nll_mask)} flags for {len(resp_logprobs)} responses"
            )
        # Share one cast node across policy and auxiliary terms. Separate
        # cast-back paths could each overflow before their gradients cancel.
        accumulators = [_loss_accumulator(lp) for lp in resp_logprobs]
        losses = []
        for i, logprobs in enumerate(resp_logprobs):
            if not torch.isfinite(logprobs).all():
                raise ValueError("response logprobs must be finite")
            losses.append(-advantages[i] * accumulators[i].sum())
        if not losses:
            return torch.tensor(0.0), {"grpo_loss": 0.0, "nll_loss": 0.0}
        loss = torch.stack(losses).mean()
        nll_loss = torch.tensor(0.0)
        if nll_mask is not None and nll_coef != 0.0:
            masked = [
                logprobs.mean()
                for keep, logprobs in zip(nll_mask, accumulators)
                if keep and logprobs.numel() > 0
            ]
            if masked:
                nll_loss = -torch.stack(masked).mean()
                loss = loss + nll_coef * nll_loss
        if not torch.isfinite(loss):
            raise ValueError("GRPO objective is non-finite; refusing an invalid update")
        masked_count = sum(
            bool(keep) and lp.numel() > 0
            for keep, lp in zip(() if nll_mask is None else nll_mask, resp_logprobs)
        )
        for i, logprobs in enumerate(resp_logprobs):
            if logprobs.numel() == 0:
                continue
            gradient = -advantages[i] / len(resp_logprobs)
            if masked_count and nll_mask is not None and nll_mask[i]:
                gradient -= nll_coef / masked_count / logprobs.numel()
            _check_low_precision_gradient(gradient, logprobs)
        # grpo_loss stays the optimized total (as before); nll_loss exposes
        # the auxiliary component for tracking.
        return loss, {"grpo_loss": loss.item(), "nll_loss": nll_loss.item()}

    return _loss_fn


def make_gspo_loss_fn(
    advantages: Sequence[float],
    old_logprobs: Optional[Sequence[Any]] = None,
    epsilon_low: float = 3e-4,
    epsilon_high: float = 4e-4,
    stash: Optional[List[Any]] = None,
    response_lens: Optional[Sequence[int]] = None,
) -> Callable:
    """Return a Tinker-compatible GSPO loss closure (Zheng et al. 2025, Eq. 5-7).

    Response lengths and behavior logprobs are snapshotted at creation;
    ``stash`` remains a caller-visible output list.

    Sequence-level clipped objective: ``s_i = exp(mean(new_lp - old_lp))``
    (length-normalized sequence ratio), loss ``-mean(min(s·A,
    clip(s, 1-ε_low, 1+ε_high)·A))``.  ``old_logprobs=None`` means first
    epoch (policy equals sampler, ratio exactly 1 with gradients flowing).
    When ``stash`` is given, the first call's detached logprobs are
    appended for use as the next epoch's ``old_logprobs``.  All inputs
    must pair 1:1, and each sequence's old and new token logprobs must
    have the same shape, or the closure raises instead of broadcasting
    a mis-assigned ratio.

    ``response_lens`` slices the trainer logprobs to response tokens before
    the ratio (see :func:`_response_logprobs`); ``old_logprobs`` and the
    stash are then response-only too.  The training loop passes the
    sampler's per-token logprobs as ``old_logprobs`` on every epoch, so
    the ratio measures pi_theta against the behaviour policy (fixed
    2026-10-03: epoch 0 used the trainer's own detached logprobs, so
    s_i == 1 and the clip never fired at the default 1 epoch).
    """

    advantages = tuple(advantages)
    response_lens = None if response_lens is None else tuple(response_lens)
    # The SDK may await its forward request before invoking this callback.
    # Bind rollout values now; detach/clone tensors without downcasting lists
    # or NumPy arrays before the trainer logprob dtype is known. Stash remains
    # intentionally shared output, not part of this immutable input snapshot.
    if old_logprobs is not None:
        old_logprobs = tuple(
            value.detach().clone() if isinstance(value, torch.Tensor) else deepcopy(value)
            for value in old_logprobs
        )
    if not all(math.isfinite(value) for value in advantages):
        raise ValueError("advantages must be finite")
    if any(not math.isfinite(value) or value < 0 for value in (epsilon_low, epsilon_high)):
        raise ValueError("GSPO clip epsilons must be finite and nonnegative")
    log_low = math.log1p(-epsilon_low) if epsilon_low < 1 else -math.inf
    log_high = math.log1p(epsilon_high)

    def _loss_fn(data: Any, logprobs_list: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
        logprobs_list = _response_logprobs(logprobs_list, response_lens)
        if len(advantages) != len(logprobs_list):
            raise ValueError(
                "advantages must pair 1:1 with logprobs, got "
                f"{len(advantages)} advantages for {len(logprobs_list)} responses"
            )
        if old_logprobs is not None and len(old_logprobs) != len(logprobs_list):
            raise ValueError(
                "old_logprobs must pair 1:1 with logprobs, got "
                f"{len(old_logprobs)} for {len(logprobs_list)} responses"
            )
        if not logprobs_list:
            return torch.tensor(0.0), {"gspo_loss": 0.0, "gspo_clip_frac": 0.0}
        terms = []
        clipped = 0
        for i, logprobs in enumerate(logprobs_list):
            if logprobs.numel() == 0:
                raise ValueError(f"sequence {i} has no token logprobs")
            if not torch.isfinite(logprobs).all():
                raise ValueError("response logprobs must be finite")
            accumulator = _loss_accumulator(logprobs)
            if old_logprobs is None:
                # First epoch: ratio is exactly 1, but keep it a function
                # of logprobs so gradients flow (length-normalized PG).
                log_ratio = (accumulator - accumulator.detach()).mean()
            else:
                old = torch.as_tensor(
                    old_logprobs[i], dtype=accumulator.dtype, device=logprobs.device
                ).detach()
                if logprobs.shape != old.shape:
                    raise ValueError(
                        "old_logprobs must match new logprobs token-for-token, "
                        f"got {tuple(old.shape)} vs {tuple(logprobs.shape)} on sequence {i}"
                    )
                if not torch.isfinite(old).all():
                    raise ValueError("behavior logprobs must be finite")
                log_ratio = (accumulator - old).mean()
            if not torch.isfinite(log_ratio):
                raise ValueError("GSPO log ratio is non-finite; refusing an invalid update")
            if log_ratio < log_low or log_ratio > log_high:
                clipped += 1
            # Select the min-objective branch *before* exponentiating.
            # exp(large) in an unused branch still causes 0 * inf = NaN
            # during backward, even when its clipped loss is finite.
            advantage = advantages[i]
            if advantage == 0:
                term = log_ratio * 0.0
            elif advantage > 0:
                term = torch.exp(torch.clamp(log_ratio, max=log_high)) * advantage
            else:
                term = torch.exp(torch.clamp(log_ratio, min=log_low)) * advantage
            active = (advantage > 0 and log_ratio <= log_high) or (
                advantage < 0 and log_ratio >= log_low
            )
            gradient = (
                float((term.detach() / len(logprobs_list) / logprobs.numel()).item())
                if active
                else 0.0
            )
            _check_low_precision_gradient(gradient, logprobs)
            terms.append(term)
        loss = -torch.stack(terms).mean()
        if not torch.isfinite(loss):
            raise ValueError("GSPO objective is non-finite; refusing an invalid update")
        if stash is not None and not stash:
            stash.extend(lp.detach().clone() for lp in logprobs_list)
        return loss, {
            "gspo_loss": loss.item(),
            "gspo_clip_frac": clipped / len(logprobs_list),
        }

    return _loss_fn


def _decode_response(tokenizer: Any, resp: Any) -> str:
    return tokenizer.decode(list(resp.tokens), skip_special_tokens=True)


def _sample_scored_group(
    tok: Any,
    sc: Any,
    T: Any,
    config: GRPOConfig,
    reward: RewardAdapter,
    example: TrainingExample,
    *,
    seed: Optional[int] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Tuple[List[int], List[List[int]], List[float], List[int]]:
    """Sample one group for an example and score it.

    Shared by the main loop and critic pretraining.  Returns
    ``(prompt_ids, response_ids_list, rewards, token_lengths)``.

    ``seed`` is forwarded to ``SamplingParams`` when the installed Tinker
    exposes a ``seed`` field.  When ``meta`` is a dict it receives
    ``sampler_logprobs`` (per-response token logprobs, or ``None`` when the
    sampler returned none or a mismatched length), ``truncated`` (from
    ``stop_reason == "length"`` when available, else ``length >=
    max_response_tokens``) and ``identical_group`` (a seeded group of 2+
    whose responses are all token-identical).
    """
    prompt_ids = tok.encode(example.prompt, add_special_tokens=False)
    if len(prompt_ids) > config.max_prompt_tokens:
        prompt_ids = prompt_ids[: config.max_prompt_tokens]

    sp_kwargs: Dict[str, Any] = {
        "max_tokens": config.max_response_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
    }
    if (
        config.seed_sampling
        and seed is not None
        and "seed" in (getattr(T.SamplingParams, "model_fields", None) or {})
    ):
        sp_kwargs["seed"] = seed
    sp = T.SamplingParams(**sp_kwargs)
    responses = sc.sample(
        T.ModelInput.from_ints(prompt_ids),
        num_samples=config.group_size,
        sampling_params=sp,
    ).result()

    sequences = list(responses.sequences)
    if len(sequences) != config.group_size:
        raise ValueError(
            f"sampler returned {len(sequences)} responses for requested group_size={config.group_size}"
        )
    resp_ids_list = [list(resp.tokens) for resp in sequences]
    rewards = [
        _finite_reward(reward.score(_decode_response(tok, resp), example)) for resp in sequences
    ]
    token_lengths = [len(ids) for ids in resp_ids_list]
    if meta is not None:
        sampler_logprobs: List[Optional[List[float]]] = []
        truncated: List[bool] = []
        for resp, ids in zip(sequences, resp_ids_list):
            lps = getattr(resp, "logprobs", None)
            sampler_logprobs.append(
                [float(v) for v in lps] if lps is not None and len(lps) == len(ids) else None
            )
            stop_reason = getattr(resp, "stop_reason", None)
            stop_reason = getattr(stop_reason, "value", stop_reason)
            if stop_reason in ("length", "stop"):
                truncated.append(stop_reason == "length")
            else:
                truncated.append(len(ids) >= config.max_response_tokens)
        meta["sampler_logprobs"] = sampler_logprobs
        meta["truncated"] = truncated
        meta["identical_group"] = (
            "seed" in sp_kwargs
            and len(resp_ids_list) > 1
            and all(ids == resp_ids_list[0] for ids in resp_ids_list)
        )
    return prompt_ids, resp_ids_list, rewards, token_lengths


def _sampling_seed(seed: int, step: int, index: int) -> int:
    """Deterministic per-request sampling seed from (run seed, step, index)."""
    return ((seed * 1_000_003 + step) * 10_007 + index) % (2**31 - 1)


def _datum_response_len(prompt_ids: List[int], response_ids: List[int]) -> int:
    """Number of response tokens that :func:`_build_datum` targets."""
    return len(response_ids) if prompt_ids else max(len(response_ids) - 1, 0)


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


def _training_result_loss(result: Any) -> float:
    """Require a trustworthy scalar loss before applying provider gradients."""
    error = "training result must contain a finite numeric loss; refusing optimizer update"
    if not isinstance(getattr(result, "metrics", None), Mapping):
        raise ValueError(error)
    value = _metric(result, ["grpo_loss", "gspo_loss", "loss"])
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(error)
    try:
        if not math.isfinite(value):
            raise ValueError(error)
        return float(value)
    except OverflowError as exc:
        raise ValueError(error) from exc


def _checkpoint_path(config: GRPOConfig, seed: int) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", config.name).strip("-")
    return Path(config.checkpoint_dir) / f"{safe_name}_seed{seed}.json"


def _critic_checkpoint_path(config: GRPOConfig, seed: int, digest: Optional[str] = None) -> Path:
    receipt = _checkpoint_path(config, seed)
    if digest is None:
        return receipt.with_suffix(".critic.pt")  # legacy, unbound sidecar
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise RuntimeError("invalid critic SHA-256 binding in checkpoint receipt")
    # No path from a receipt is ever trusted or joined to the filesystem.
    return receipt.parent / f"critic-{digest}.pt"


def _critic_config_hash(config: GRPOConfig, seed: int) -> str:
    encoded = json.dumps(_config_fingerprint(config, seed), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_regular_checkpoint_bytes(path: Path, *, kind: str) -> bytes:
    """Pin a nonsymlink regular-file descriptor before reading its snapshot."""
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise RuntimeError(f"secure {kind} loading requires O_NOFOLLOW and O_NONBLOCK support")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise RuntimeError(f"{kind} must be a readable regular file, not a symbolic link") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError(f"{kind} must be a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(descriptor)


def _read_regular_critic_bytes(path: Path) -> bytes:
    return _read_regular_checkpoint_bytes(path, kind="critic checkpoint")


def _save_critic(
    config: GRPOConfig,
    seed: int,
    critic: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    *,
    step: int,
) -> str:
    """Publish immutable critic bytes; return the digest for the commit receipt.

    A unique temporary is hard-linked only after its bytes are fully written.
    The JSON receipt is committed separately, last. An interrupted save leaves
    the previous receipt and its sidecar intact; unreferenced files are ignored.
    This provides process-interruption safety, not a cross-provider transaction.
    """
    if optimizer is None:
        raise ValueError("critic checkpoint requires optimizer state")
    payload: Dict[str, Any] = {
        "model": critic.state_dict(),
        "step": step,
        "config_sha256": _critic_config_hash(config, seed),
    }
    payload["optimizer"] = optimizer.state_dict()
    require_finite_critic_state(payload["model"], payload["optimizer"])
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    data = buffer.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    path = _critic_checkpoint_path(config, seed, digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".critic-", delete=False) as file:
            temporary = Path(file.name)
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            try:
                identical = _read_regular_critic_bytes(path) == data
            except (OSError, RuntimeError):
                identical = False
            if not identical:
                raise RuntimeError(
                    "existing content-addressed critic artifact is inconsistent"
                ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return digest


def _load_critic_checkpoint(
    config: GRPOConfig, seed: int, resume_step: int, receipt: Dict[str, Any]
) -> Dict[str, Any]:
    """Verify the committed bytes and state binding before restoring any model."""
    bound = "critic_state_sha256" in receipt
    digest = receipt.get("critic_state_sha256") if bound else None
    if bound and digest is None:
        raise RuntimeError("invalid critic SHA-256 binding in checkpoint receipt")
    critic_path = _critic_checkpoint_path(config, seed, digest)
    if not critic_path.exists():
        raise RuntimeError(
            f"[{config.name}] resume at step {resume_step} has no critic state at {critic_path}"
        )
    data = _read_regular_critic_bytes(critic_path)
    if bound and hashlib.sha256(data).hexdigest() != digest:
        raise RuntimeError("critic checkpoint SHA-256 mismatch; artifact is preserved")
    # Hash and deserialize the same snapshot, never reopen a changed file.
    payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or "model" not in payload
        or "optimizer" not in payload
        or type(payload.get("step")) is not int
        or payload["step"] != resume_step
    ):
        raise RuntimeError(
            f"[{config.name}] critic state must include model and optimizer "
            f"at actor checkpoint step {resume_step}; restore a matching "
            "checkpoint pair or start a fresh run"
        )
    if bound and payload.get("config_sha256") != _critic_config_hash(config, seed):
        raise RuntimeError("critic checkpoint configuration binding does not match")
    require_finite_critic_state(payload["model"], payload["optimizer"])
    return payload


def _config_fingerprint(config: GRPOConfig, seed: int) -> Dict[str, Any]:
    config._validate_boundary_inputs()
    values = asdict(config)
    values["seed"] = seed
    values.pop("resume", None)
    values.update(_campaign_metadata(config))
    values["held_out_suite_ids"] = list(values["heldout_suite_ids"])
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


def _preflight_checkpoint_storage(path: Path) -> None:
    """Probe an owned temporary before remote work; not a future-space guarantee."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".grpo-write-check-", delete=False
        ) as file:
            temporary = Path(file.name)
            file.write(b"{}\n")
            file.flush()
            os.fsync(file.fileno())
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    # Serialize before creating directories or files. Stage exclusively under
    # a unique name, never follow a predictable leftover temporary symlink.
    serialized = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary = Path(file.name)
            file.write(serialized)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _config_field_default(key: str, derived: Any) -> Any:
    """JSON-normalised dataclass default for a fingerprint key.

    A key missing from an old receipt was written by code that predates the
    field, i.e. under the field's default.  Backfilling with the *current*
    value (pre-2026-10-03) let any new setting, e.g. ``gspo_enabled=True``,
    resume as compatible.  Keys that are not ``GRPOConfig`` fields are
    derived from fields that are checked themselves, so ``derived`` stands.
    """
    spec = {f.name: f for f in fields(GRPOConfig)}.get(key)
    if spec is None:
        return derived
    if spec.default is not MISSING:
        value = spec.default
    elif spec.default_factory is not MISSING:
        value = spec.default_factory()
    else:
        return derived
    return json.loads(json.dumps(value))


def _load_checkpoint(config: GRPOConfig, seed: int) -> Dict[str, Any] | None:
    path = _checkpoint_path(config, seed)
    if not path.exists() and not path.is_symlink():
        return None
    raw = _read_regular_checkpoint_bytes(path, kind="JSON checkpoint")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("JSON checkpoint must be valid UTF-8") from None
    payload = _strict_json_object(text)
    if payload is None:
        raise ValueError(
            f"[{config.name}] checkpoint must be a JSON object with unique keys and finite numbers: {path}"
        )
    if not config.resume:
        # Fresh restart may change this experiment's settings, but it must
        # not erase another name that sanitizes to the same receipt path.
        stored = payload.get("config")
        if (
            not isinstance(stored, dict)
            or stored.get("name") != config.name
            or type(stored.get("seed")) is not int
            or stored.get("seed") != seed
        ):
            raise ValueError(
                "checkpoint identity does not match this fresh run; choose a new name or directory"
            )
        return None
    # Normalize through JSON so tuple-valued fields (e.g. wandb_tags) compare
    # equal to their stored list form; without this every resume fails as
    # "incompatible" (fixed 2026-10-03).
    expected = json.loads(json.dumps(_config_fingerprint(config, seed), sort_keys=True))
    stored = payload.get("config")
    if stored != expected:
        # Backfill keys added after the checkpoint was written so old
        # receipts resume under the current config instead of failing.
        # Genuine drift (changed values, removed keys) still fails closed.
        if isinstance(stored, dict):
            backfilled = dict(stored)
            for key, value in expected.items():
                if key not in backfilled:
                    backfilled[key] = _config_field_default(key, value)
            if backfilled == expected:
                payload["config"] = backfilled
                return payload
        raise ValueError(f"[{config.name}] incompatible checkpoint: {path}")
    return payload


def _resume_reward_history(
    prior: Optional[Dict[str, Any]], resume_step: int, steps: int
) -> Tuple[List[float], int, int]:
    """Validate committed history locally, then trim a legitimate rewind."""
    if prior is None:
        return [], 0, 0
    error = "checkpoint reward history is malformed or inconsistent with committed steps"
    result = prior.get("result", {})
    if result is None and prior.get("status") != "completed":
        result = {}
    if not isinstance(result, dict):
        raise RuntimeError(error)
    completed = prior.get("status") == "completed"
    source = result if completed else prior
    trace = source.get("reward_trace", result.get("reward_trace", []))
    if not isinstance(trace, list) or any(type(value) not in (int, float) for value in trace):
        raise RuntimeError(error)
    try:
        finite = all(math.isfinite(value) for value in trace)
    except OverflowError:
        finite = False
    if not finite or not resume_step <= len(trace) <= steps:
        raise RuntimeError(error)
    if completed and (resume_step != steps or len(trace) != steps):
        raise RuntimeError(error)
    zero_count = sum(value == 0 for value in trace)
    counters = []
    for key, default in (("zero_loss_steps", 0), ("zero_reward_steps", zero_count)):
        value = source.get(key, result.get(key, default))
        if type(value) is not int or not 0 <= value <= len(trace):
            raise RuntimeError(error)
        if key == "zero_reward_steps" and value != zero_count:
            raise RuntimeError(error)
        counters.append(value)
    kept = trace[:resume_step]
    if len(trace) > resume_step:
        # Loss is not stored per step, so discarded suffix totals cannot be
        # retained. Reward-zero counts can be reconstructed from the prefix.
        return kept, 0, sum(value == 0 for value in kept)
    return kept, counters[0], counters[1]


def _redact_error(exc: BaseException) -> str:
    """Return an error string with credential values removed."""
    message = str(exc)
    secrets = {
        os.environ.get(env_name)
        for env_name in ("TINKER_API_KEY", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "WANDB_API_KEY")
    }
    # Redact longer overlapping credentials first, so masking a short prefix
    # cannot leave the rest of another configured credential in a failure log.
    for secret in sorted((value for value in secrets if value), key=len, reverse=True):
        message = message.replace(secret, "[redacted]")
    message = re.sub(
        r"(?i)\b(?:hf|wandb|sk|ghp|github_pat)[_-][A-Za-z0-9][A-Za-z0-9._-]{7,}\b",
        "[redacted]",
        message,
    )
    message = re.sub(r"(?i)\bBearer\s+[A-Za-z0-9._-]+", "Bearer [redacted]", message)
    return message or exc.__class__.__name__


def _training_input_fingerprint(examples: Sequence[TrainingExample]) -> Optional[str]:
    """Hash ordered strict-JSON projections without coercing adapter objects."""
    encoder = json.JSONEncoder(
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    digest = hashlib.sha256(b"tinkerrl-training-inputs-v1\n")
    try:
        for example in examples:
            if type(example) is not TrainingExample:
                return None
            row = {"prompt": example.prompt, "target": example.target, "metadata": example.metadata}
            pending: List[Any] = [row]
            seen: set[int] = set()
            while pending:
                value = pending.pop()
                kind = type(value)
                if kind in (type(None), str, bool, int):
                    continue
                if kind is float:
                    if not math.isfinite(value):
                        return None
                elif kind in (dict, list):
                    if id(value) in seen:
                        continue  # Encoder still rejects cycles, unlike shared references.
                    seen.add(id(value))
                    if kind is dict:
                        if any(type(key) is not str for key in value):
                            return None
                        pending.extend(value.values())
                    else:
                        pending.extend(value)
                else:
                    return None
            for chunk in encoder.iterencode(row):
                digest.update(chunk.encode("utf-8"))
            digest.update(b"\n")
        digest.update(f"count={len(examples)}\n".encode())
        return digest.hexdigest()
    except (TypeError, ValueError, UnicodeError, RecursionError, OverflowError):
        return None


def _dataset_source_receipt(
    examples: Sequence[TrainingExample], resume_step: int, prior: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Bind new resume receipts to seed-start inputs, never infer legacy evidence."""
    error = "training input fingerprint is malformed, changed, or no longer verifiable"
    completed = bool(prior and prior.get("status") == "completed")
    container = (prior or {}).get("result", {}) if completed else (prior or {})
    stored = container.get("dataset_source", {})
    if not isinstance(stored, dict):
        raise RuntimeError(error)
    if stored:
        if (
            type(stored.get("schema_version")) is not int
            or stored.get("schema_version") != 1
            or stored.get("status") not in ("fingerprinted", "unrecorded")
            or type(stored.get("train_count")) is not int
            or stored["train_count"] < 0
        ):
            raise RuntimeError(error)
        if stored["status"] == "fingerprinted":
            origin = stored.get("fingerprinted_from_step")
            sha = stored.get("train_sha256")
            if (
                type(origin) is not int
                or not 0 <= origin <= resume_step
                or not isinstance(sha, str)
                or not re.fullmatch(r"[0-9a-f]{64}", sha)
            ):
                raise RuntimeError(error)
        elif (
            stored.get("train_sha256") is not None
            or stored.get("fingerprinted_from_step") is not None
        ):
            raise RuntimeError(error)
    if completed and (not stored or stored["status"] == "unrecorded"):
        return dict(stored)  # No new execution; do not invent a historical fingerprint.
    digest = _training_input_fingerprint(examples)
    origin = resume_step if digest is not None else None
    if stored.get("status") == "fingerprinted":
        if digest != stored["train_sha256"] or len(examples) != stored["train_count"]:
            raise RuntimeError(error)
        origin = stored["fingerprinted_from_step"]
    return {
        "schema_version": 1,
        "status": "fingerprinted" if digest is not None else "unrecorded",
        "train_count": len(examples),
        "train_sha256": digest,
        "fingerprinted_from_step": origin,
        "scope": "Ordered strict-JSON training inputs at seed start; not authenticated source or proof of immutable consumed data.",
    }


def _model_source_receipt(
    config: GRPOConfig,
    *,
    tokenizer_supplied: bool,
    resume_step: int,
    resume_state_path: Optional[str],
) -> Dict[str, Any]:
    """Record request scope, never infer managed weights from tokenizer metadata."""
    return {
        "schema_version": 1,
        "recorded_from_step": resume_step,
        "configured_model": config.model,
        "configured_model_revision": config.model_revision,
        "tokenizer": {
            "mode": "caller_supplied" if tokenizer_supplied else "pretrained_request",
            "requested_model": None if tokenizer_supplied else config.model,
            "requested_revision": None if tokenizer_supplied else config.model_revision,
        },
        "managed_base_weights": {"status": "unverified", "commit_sha": None},
        "initialization": {
            "mode": "optimizer_checkpoint" if resume_state_path else "managed_base_model_name",
            "checkpoint_path": resume_state_path,
        },
    }


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
        raise RuntimeError(
            f"Hugging Face authentication preflight failed: {_redact_error(exc)}"
        ) from exc

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
        # implicit owner override; ownership comes from the preflight-resolved
        # owner argument, which already applied configured overrides.
        _, base = base.split("/", 1)
    # Owner selection is resolved before provider work. Do not reread mutable
    # environment/config overrides and redirect a later checkpoint upload.
    if not isinstance(owner, str) or not owner.strip():
        raise ValueError("resolved checkpoint owner must be a non-empty string")
    repo_owner = owner.strip()
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
    for key, expected_url, label in (
        ("repo_url", expected_repo_url, "repo"),
        ("revision_url", f"{expected_repo_url}/tree/{value['revision']}", "revision"),
        ("commit_url", f"{expected_repo_url}/commit/{value['commit_sha']}", "commit"),
    ):
        if value[key] != expected_url:
            raise RuntimeError(
                f"Hugging Face checkpoint receipt has an invalid {label} URL for step {step}"
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
        raise RuntimeError(
            "Tinker CLI is unavailable; Hugging Face checkpoint export failed"
        ) from exc
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
        },
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
        "campaign/primary_evaluation_suite_ids": list(metadata["primary_evaluation_suite_ids"]),
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
    summary["primary_evaluation_suite_ids"] = list(metadata["primary_evaluation_suite_ids"])
    if metadata["heldout_suite_ids"]:
        summary["heldout_suite_ids"] = list(metadata["heldout_suite_ids"])
    summary["domain_tags"] = list(metadata["domain_tags"])
    summary["declared_domains"] = list(metadata["declared_domains"])
    summary["training_domain_union"] = list(metadata["training_domain_union"])
    summary["primary_evaluation_domain_union"] = list(metadata["primary_evaluation_domain_union"])
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


def _record_receipt(
    run: Any,
    checkpoint_receipts: List[Dict[str, Any]],
    checkpoint_urls: List[str],
    checkpoint_commit_shas: List[str],
    receipt: Dict[str, Any],
    *,
    step: int | str,
) -> Dict[str, Any]:
    """Validate one Hub receipt, append it to the run's lists, and log it to W&B."""
    receipt = _require_checkpoint_receipt(receipt, step=step)
    checkpoint_receipts.append(receipt)
    checkpoint_urls.append(receipt["revision_url"])
    checkpoint_commit_shas.append(receipt["commit_sha"])
    _log_wandb_checkpoint(run, receipt=receipt, receipts=checkpoint_receipts)
    return receipt


def _run_one_seed(
    config: GRPOConfig,
    dataset: DatasetAdapter,
    reward: RewardAdapter,
    tokenizer: Any,
    logger: Callable[[str], Any] = print,
) -> GRPORunResult:
    """Execute the GRPO loop for one seed.  Pure enough to unit-test with fakes."""
    config.validate_numerics()
    seed = config.seed
    config.validate_tracking()
    rng = random.Random(seed)
    torch.manual_seed(seed)

    train_examples = list(dataset.train_examples())
    if not train_examples:
        raise ValueError(f"[{config.name}] dataset returned 0 training examples")

    checkpoint_path = _checkpoint_path(config, seed)
    prior = _load_checkpoint(config, seed)
    resume_step = (prior or {}).get("step", 0)
    resume_state_path = (prior or {}).get("train_state_path")
    if type(resume_step) is not int or not 0 <= resume_step <= config.steps:
        raise RuntimeError(f"[{config.name}] checkpoint step must be an integer in [0, steps]")
    if prior and prior.get("status") != "completed" and resume_step > 0:
        if config.dynamic_sampling:
            raise RuntimeError(
                "resume cannot replay dynamic-sampling draws; start this seed "
                "from step 0 or turn dynamic sampling off"
            )
        if not isinstance(resume_state_path, str) or not resume_state_path.strip():
            raise RuntimeError(
                f"[{config.name}] checkpoint step {resume_step} has no actor optimizer state; "
                "restore a matching periodic receipt or start a fresh run"
            )
    step_rewards, zero_loss_steps, zero_reward_steps = _resume_reward_history(
        prior, resume_step, config.steps
    )
    dataset_source = _dataset_source_receipt(train_examples, resume_step, prior)
    if not (prior and prior.get("status") == "completed"):
        _preflight_checkpoint_storage(checkpoint_path)
    # Critic pretrain draws from this RNG before the step loop, and only
    # on a fresh run.  Replay those draws before the per-step draws so a
    # resume after pretraining keeps the same example order.
    if config.critic_enabled and config.critic_pretrain_batches > 0 and resume_step > 0:
        for _ in range(config.critic_pretrain_batches):
            rng.sample(train_examples, min(config.batch_size, len(train_examples)))
    for _ in range(resume_step):
        rng.sample(train_examples, min(config.batch_size, len(train_examples)))

    critic: Any = None
    critic_opt: Any = None
    # A completed receipt already holds the result. Loading the critic
    # first turns a missing or unreadable sidecar into a failed retry.
    if config.critic_enabled and not (prior and prior.get("status") == "completed"):
        critic = PromptValueCritic(
            hidden_dim=config.critic_hidden_dim,
            max_prompt_tokens=config.max_prompt_tokens,
        )
        critic_opt = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)
        if resume_step > 0:
            payload = _load_critic_checkpoint(config, seed, resume_step, prior or {})
            critic.load_state_dict(payload["model"])
            critic_opt.load_state_dict(payload["optimizer"])
            logger(f"[{config.name}] critic resumed from committed step {resume_step}")

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

        model_source = _model_source_receipt(
            config,
            tokenizer_supplied=tokenizer is not None,
            resume_step=resume_step,
            resume_state_path=resume_state_path,
        )
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
        checkpoint_urls: List[str] = [item["revision_url"] for item in checkpoint_receipts]
        checkpoint_commit_shas: List[str] = [item["commit_sha"] for item in checkpoint_receipts]
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
        initial_receipt = _record_receipt(
            wb,
            checkpoint_receipts,
            checkpoint_urls,
            checkpoint_commit_shas,
            initial_receipt,
            step=resume_step,
        )
        sc = tc.create_sampling_client(model_path=w0.path)

        save_every = config.effective_save_every()
        if config.critic_enabled and config.critic_pretrain_batches > 0 and resume_step == 0:
            logger(
                f"[{config.name}] critic pretraining: "
                f"{config.critic_pretrain_batches} batches (policy frozen)"
            )
            for pre_idx in range(config.critic_pretrain_batches):
                pre_batch = rng.sample(train_examples, min(config.batch_size, len(train_examples)))
                pre_prompts: List[List[int]] = []
                pre_targets: List[float] = []
                for ex_idx, example in enumerate(pre_batch):
                    pre_meta: Dict[str, Any] = {}
                    prompt_ids, _, rewards, _ = _sample_scored_group(
                        tok,
                        sc,
                        T,
                        config,
                        reward,
                        example,
                        seed=_sampling_seed(seed, -1 - pre_idx, ex_idx),
                        meta=pre_meta,
                    )
                    for is_trunc, r in zip(pre_meta["truncated"], rewards):
                        if config.mask_truncated_responses and is_trunc:
                            continue
                        pre_prompts.append(prompt_ids)
                        pre_targets.append(r)
                train_critic_step(
                    critic,
                    critic_opt,
                    pre_prompts,
                    pre_targets,
                    steps=config.critic_updates_per_step,
                )

        for step in range(resume_step, config.steps):
            batch = rng.sample(train_examples, min(config.batch_size, len(train_examples)))
            all_data: List[Any] = []
            all_advs: List[float] = []
            all_nll: List[bool] = []
            all_resp_lens: List[int] = []
            all_old: List[Optional[List[float]]] = []
            pending: List[Any] = []
            pending_extra: List[Any] = []
            batch_rewards: List[float] = []
            sampled_rewards: List[float] = []
            critic_prompts: List[List[int]] = []
            critic_targets: List[float] = []
            critic_ev = 0.0
            # The critic implies the batch-wide path: per-group centering
            # would cancel its per-prompt baseline.
            use_batch_path = config.global_advantage_normalization or config.critic_enabled

            # Dynamic sampling (DAPO §3.2) refills the batch pool with fresh
            # examples when a group is degenerate.  The refill loop is bounded
            # by dynamic_sampling_max_resamples, so degenerate batches
            # terminate instead of spinning.
            pool = list(batch)
            resamples_used = 0
            skipped_degenerate = 0
            identical_groups = 0
            pool_idx = 0
            while pool_idx < len(pool):
                example = pool[pool_idx]
                pool_idx += 1
                group_meta: Dict[str, Any] = {}
                prompt_ids, resp_ids_list, rewards, token_lengths = _sample_scored_group(
                    tok,
                    sc,
                    T,
                    config,
                    reward,
                    example,
                    seed=_sampling_seed(seed, step, pool_idx - 1),
                    meta=group_meta,
                )
                if group_meta["identical_group"] and config.temperature > 0:
                    identical_groups += 1
                # train/reward covers every sampled group, including the
                # degenerate ones dynamic sampling drops (fixed 2026-10-03).
                sampled_rewards.extend(rewards)
                if config.dynamic_sampling and is_degenerate_group(rewards):
                    skipped_degenerate += 1
                    if resamples_used < config.dynamic_sampling_max_resamples:
                        pool.append(rng.choice(train_examples))
                        resamples_used += 1
                    continue
                # stop_reason == "length" when the sampler reports it, so an
                # EOS exactly at the budget is not masked as truncated.
                truncated = group_meta["truncated"]
                sampler_lps = group_meta["sampler_logprobs"]
                nll_flags = _nll_mask_for_group(
                    rewards,
                    truncated,
                    config.nll_aux_min_reward,
                    config.mask_truncated_responses,
                )
                if use_batch_path:
                    # Stage raw rewards; one batch-wide pass runs after the pool.
                    pending.append(
                        (
                            prompt_ids,
                            resp_ids_list,
                            rewards,
                            token_lengths,
                            nll_flags,
                        )
                    )
                    pending_extra.append((truncated, sampler_lps))
                else:
                    advs = normalize_rewards(rewards, unbiased=config.debias_advantages)
                    if config.mask_truncated_responses:
                        advs = [0.0 if cut else adv for adv, cut in zip(advs, truncated)]
                    for resp_ids, adv, keep_nll, old_lps in zip(
                        resp_ids_list, advs, nll_flags, sampler_lps
                    ):
                        all_data.append(_build_datum(prompt_ids, resp_ids))
                        all_advs.append(adv)
                        all_nll.append(keep_nll)
                        all_resp_lens.append(_datum_response_len(prompt_ids, resp_ids))
                        all_old.append(old_lps)
                batch_rewards.extend(rewards)

            if use_batch_path:
                flat_rewards = [r for (_, _, rewards, _, _) in pending for r in rewards]
                flat_truncated = [cut for (cuts, _) in pending_extra for cut in cuts]
                flat_basis = flat_rewards
                if config.critic_enabled and pending:
                    with torch.no_grad():
                        pred = critic([item[0] for item in pending]).tolist()
                    flat_basis = [
                        r - v for (_, _, rewards, _, _), v in zip(pending, pred) for r in rewards
                    ]
                    # The number logged as critic EV is this baseline, before
                    # the fit.  V is subtracted from truncated rewards too.
                    baseline_pred = [
                        v for (_, _, rewards, _, _), v in zip(pending, pred) for _ in rewards
                    ]
                    critic_ev = explained_variance(baseline_pred, flat_rewards)
                exclude = (
                    flat_truncated
                    if config.mask_truncated_responses and config.global_norm_exclude_truncated
                    else None
                )
                flat_advs = normalize_advantages_global(
                    flat_basis,
                    unbiased=config.debias_advantages,
                    exclude=exclude,
                )
                if config.mask_truncated_responses:
                    flat_advs = [0.0 if cut else adv for adv, cut in zip(flat_advs, flat_truncated)]
                cursor = 0
                for (prompt_ids, resp_ids_list, rewards, _, nll_flags), (cuts, lps) in zip(
                    pending, pending_extra
                ):
                    if config.critic_enabled:
                        for cut, r in zip(cuts, rewards):
                            if config.mask_truncated_responses and cut:
                                continue
                            critic_prompts.append(prompt_ids)
                            critic_targets.append(r)
                    for resp_ids, keep_nll, old_lps in zip(resp_ids_list, nll_flags, lps):
                        all_data.append(_build_datum(prompt_ids, resp_ids))
                        all_advs.append(flat_advs[cursor])
                        all_nll.append(keep_nll)
                        all_resp_lens.append(_datum_response_len(prompt_ids, resp_ids))
                        all_old.append(old_lps)
                        cursor += 1

            # Logged reward covers every sampled group (dynamic sampling
            # included); batch_rewards is the subset that was trained on.
            avg = _reward_mean(sampled_rewards)
            if identical_groups:
                logger(
                    f"[{config.name}] WARNING: {identical_groups} seeded group(s) at "
                    f"temperature {config.temperature} returned identical responses"
                )
            if not all_data:
                # Every group was degenerate: no update, but keep the trace
                # aligned with steps (fixed 2026-10-03; skipped steps used to
                # vanish from reward_trace).
                step_rewards.append(avg)
                if avg == 0:
                    zero_reward_steps += 1
                logger(
                    f"[{config.name}] Step {step + 1:3d}/{config.steps}"
                    f" | skipped (all groups degenerate) | reward={avg:.3f}"
                )
                _wandb_log(
                    wb,
                    {
                        "train/reward": avg,
                        "train/step": step + 1,
                        "train/skipped_step": 1.0,
                        "train/skipped_degenerate_groups": float(skipped_degenerate),
                    },
                )
                continue

            adam_params = T.AdamParams(
                learning_rate=config.lr,
                beta1=config.beta1,
                beta2=config.beta2,
                eps=config.eps,
            )
            if config.gspo_enabled:
                # GSPO's ratio is pi_theta / pi_old with pi_old the policy
                # that generated the rollouts, so the sampler's own response
                # logprobs are old_logprobs on every epoch.  Fallback when
                # the sampler returned none: epoch 0 at ratio 1, stash for
                # later epochs (the pre-2026-10-03 behaviour), logged.
                gspo_sampler_old = all(lps is not None for lps in all_old)
                sampler_old = (
                    [lps[len(lps) - n :] for lps, n in zip(all_old, all_resp_lens)]
                    if gspo_sampler_old
                    else None
                )
                if not gspo_sampler_old:
                    logger(
                        f"[{config.name}] WARNING: sampler returned no logprobs for "
                        f"{sum(lps is None for lps in all_old)}/{len(all_old)} responses; "
                        "GSPO falls back to trainer logprobs (ratio 1 on epoch 0)"
                    )
                stashed_old: List[Any] = []
                for epoch in range(config.gspo_update_epochs):
                    loss_fn = make_gspo_loss_fn(
                        all_advs,
                        old_logprobs=(
                            sampler_old
                            if gspo_sampler_old
                            else (None if epoch == 0 else stashed_old)
                        ),
                        epsilon_low=config.gspo_epsilon_low,
                        epsilon_high=config.gspo_epsilon_high,
                        stash=stashed_old if epoch == 0 and not gspo_sampler_old else None,
                        response_lens=all_resp_lens,
                    )
                    train_result = tc.forward_backward_custom(
                        data=all_data,
                        loss_fn=loss_fn,
                        loss_type_input="logprobs",
                    ).result()
                    loss_val = _training_result_loss(train_result)
                    tc.optim_step(adam_params).result()
            else:
                if config.nll_aux_enabled:
                    loss_fn = make_grpo_loss_fn(
                        all_advs,
                        nll_mask=all_nll,
                        nll_coef=config.nll_aux_coef,
                        response_lens=all_resp_lens,
                    )
                else:
                    loss_fn = make_grpo_loss_fn(all_advs, response_lens=all_resp_lens)
                train_result = tc.forward_backward_custom(
                    data=all_data,
                    loss_fn=loss_fn,
                    loss_type_input="logprobs",
                ).result()
                loss_val = _training_result_loss(train_result)
                tc.optim_step(adam_params).result()

            critic_loss = 0.0
            if config.critic_enabled and critic_prompts:
                critic_loss = train_critic_step(
                    critic,
                    critic_opt,
                    critic_prompts,
                    critic_targets,
                    steps=config.critic_updates_per_step,
                )

            step_rewards.append(avg)
            if abs(loss_val) < 1e-6:
                zero_loss_steps += 1
            if avg == 0:
                zero_reward_steps += 1

            logger(
                f"[{config.name}] Step {step + 1:3d}/{config.steps}"
                f" | loss={loss_val:.4f} | reward={avg:.3f}"
            )
            step_log: Dict[str, float] = {
                "train/loss": loss_val,
                "train/reward": avg,
                "train/step": step + 1,
            }
            if config.dynamic_sampling:
                step_log["train/skipped_degenerate_groups"] = float(skipped_degenerate)
                step_log["train/reward_trained"] = _reward_mean(batch_rewards)
            if use_batch_path and all_advs:
                mean_a = sum(all_advs) / len(all_advs)
                var_a = sum((a - mean_a) ** 2 for a in all_advs) / len(all_advs)
                step_log["train/global_adv_std"] = var_a**0.5
            if config.nll_aux_enabled and all_nll:
                step_log["train/nll_loss"] = _metric(train_result, ["nll_loss"], default=0.0)
                step_log["train/nll_frac"] = sum(all_nll) / len(all_nll)
            if config.gspo_enabled:
                step_log["train/gspo_clip_frac"] = _metric(
                    train_result, ["gspo_clip_frac"], default=0.0
                )
                step_log["train/gspo_sampler_old"] = float(gspo_sampler_old)
            if config.critic_enabled:
                step_log["train/critic_loss"] = critic_loss
                step_log["train/critic_ev"] = critic_ev
            _wandb_log(wb, step_log)

            if (step + 1) % save_every == 0:
                # Prepare local critic bytes before creating remote exports.
                critic_binding = {}
                if config.critic_enabled:
                    critic_binding["critic_state_sha256"] = _save_critic(
                        config, seed, critic, critic_opt, step=step + 1
                    )
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
                checkpoint_receipt = _record_receipt(
                    wb,
                    checkpoint_receipts,
                    checkpoint_urls,
                    checkpoint_commit_shas,
                    checkpoint_receipt,
                    step=step + 1,
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
                        "model_source": model_source,
                        "dataset_source": dataset_source,
                        **critic_binding,
                    },
                )
                logger(f"[{config.name}]   -> Checkpoint step_{step + 1}")

        # Prepare local critic state before remote persistence/publication.
        # Later failures may leave unused sidecars; old receipts remain valid.
        critic_binding = {}
        if config.critic_enabled:
            critic_binding["critic_state_sha256"] = _save_critic(
                config, seed, critic, critic_opt, step=config.steps
            )
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
        final_receipt = _record_receipt(
            wb,
            checkpoint_receipts,
            checkpoint_urls,
            checkpoint_commit_shas,
            final_receipt,
            step="final",
        )
        sc = tc.create_sampling_client(model_path=final.path)

        last10 = step_rewards[-10:] if step_rewards else []
        first5 = step_rewards[:5] if step_rewards else []
        heldout_reward: Optional[float] = None
        heldout_evaluation: Dict[str, Any] = {
            "schema_version": 1,
            "status": "empty" if config.evaluate_heldout else "disabled",
            "total_n": 0,
            "failed_n": 0,
            "failure_policy": "score_zero_keep_denominator",
            "uncertainty_kind": "not_recorded",
            "summary": {},
        }

        if config.evaluate_heldout:
            test_examples = list(dataset.test_examples())
            if test_examples:
                test_rewards: List[float] = []
                heldout_failed = 0
                for ex_idx, ex in enumerate(test_examples):
                    pid = tok.encode(ex.prompt, add_special_tokens=False)
                    if len(pid) > config.max_prompt_tokens:
                        pid = pid[: config.max_prompt_tokens]
                    sp_kwargs: Dict[str, Any] = {
                        "max_tokens": config.max_response_tokens,
                        "temperature": 0.1,
                        "top_p": 0.95,
                    }
                    if config.seed_sampling and "seed" in (
                        getattr(T.SamplingParams, "model_fields", None) or {}
                    ):
                        sp_kwargs["seed"] = _sampling_seed(seed, config.steps, ex_idx)
                    sp = T.SamplingParams(**sp_kwargs)
                    try:
                        resp = sc.sample(
                            T.ModelInput.from_ints(pid), num_samples=1, sampling_params=sp
                        ).result()
                        sequences = list(resp.sequences)
                        if len(sequences) != 1:
                            raise ValueError("held-out sampler must return exactly one response")
                        text = _decode_response(tok, sequences[0])
                        test_rewards.append(_finite_reward(reward.score(text, ex)))
                    except Exception as exc:
                        # A failed sample scores 0 and stays in the
                        # denominator (fixed 2026-10-03: failures used to be
                        # dropped silently, shrinking the denominator).
                        heldout_failed += 1
                        test_rewards.append(0.0)
                        logger(
                            f"[{config.name}] held-out example {ex_idx} failed, "
                            f"scored 0: {_redact_error(exc)}"
                        )
                heldout_reward = _reward_mean(test_rewards)
                summary = heldout_reward_summary(test_rewards)
                summary["test/failed_n"] = float(heldout_failed)
                summary["test/total_n"] = float(len(test_examples))
                heldout_evaluation.update(
                    status=(
                        "all_failed"
                        if heldout_failed == len(test_examples)
                        else "partial_failure"
                        if heldout_failed
                        else "completed"
                    ),
                    total_n=len(test_examples),
                    failed_n=heldout_failed,
                    # Within-run example bootstrap, never across-seed uncertainty.
                    uncertainty_kind=(
                        "example_bootstrap" if "test/reward_ci_low" in summary else "not_recorded"
                    ),
                    summary=dict(summary),
                )
                _wandb_log(wb, summary)

        result = GRPORunResult(
            seed=seed,
            run_id=run_id,
            sampler_path=getattr(final, "path", None),
            reward_trace=step_rewards,
            avg_first5=(_reward_mean(first5)) if first5 else 0.0,
            avg_last10=(_reward_mean(last10)) if last10 else 0.0,
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
            model_source=model_source,
            heldout_evaluation=heldout_evaluation,
            dataset_source=dataset_source,
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
                **critic_binding,
            },
        )
        # The tinker run ID and checkpoint summary keys were already recorded
        # by _log_wandb_checkpoint for the final receipt; no re-write needed.
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
    # Keep the experiment fixed across seeds even if a caller callback edits
    # its original config during an earlier run. Metadata sequences are
    # normalized to immutable tuples by the copied config's constructor.
    config = replace(config)
    config.validate_numerics()
    results: List[GRPORunResult] = []
    for seed_idx in range(config.num_seeds):
        cfg = replace(config, seed=config.seed + seed_idx)
        result = _run_one_seed(cfg, dataset, reward, tokenizer, logger)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Built-in adapters
# ---------------------------------------------------------------------------


def _chat_prompt(system: str, user: str) -> str:
    """Format one system/user turn pair in the shared chat template."""
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


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
        return _chat_prompt(system_prompt, f"Available tools:\n{tool_schema}\n\nUser: {q}")

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
        return _chat_prompt(system_prompt, q)

    split = int(len(problems) * 0.8)
    train = [TrainingExample(prompt=_mkp(q), target=answer) for q, answer in problems[:split]]
    test = [TrainingExample(prompt=_mkp(q), target=answer) for q, answer in problems[split:]]
    return InMemoryDataset(train=train * 20, test=test)


def make_gsm8k_dataset(seed: int = 42, revision: Optional[str] = None) -> InMemoryDataset:
    """Load GSM8K through the dataset adapter seam."""
    from datasets import load_dataset

    system_prompt = (
        "You are a math assistant. Solve the problem step by step, then give "
        "your final numerical answer inside \\boxed{}."
    )

    def convert(split: str) -> List[TrainingExample]:
        rows: List[TrainingExample] = []
        revision_kwargs = {"revision": revision} if revision is not None else {}
        for row in load_dataset("openai/gsm8k", "main", split=split, **revision_kwargs):
            match = re.search(r"####\s*([\-\d,\.]+)", row["answer"])
            if not match:
                continue
            prompt = _chat_prompt(system_prompt, row["question"])
            rows.append(
                TrainingExample(
                    prompt=prompt,
                    target=match.group(1).replace(",", "").strip(),
                )
            )
        random.Random(seed).shuffle(rows)
        return rows

    return InMemoryDataset(train=convert("train"), test=convert("test"))


def make_xlam_dataset(seed: int = 42, revision: Optional[str] = None) -> InMemoryDataset:
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
            prompt = _chat_prompt(
                system_prompt,
                f"Available tools:\n{json.dumps(tools[:8])}\n\n"
                f"User: {row.get('query', row.get('instruction', ''))}",
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
    if (
        not isinstance(content, str)
        or "**Available Tools**" not in content
        or "[USER]" not in content
    ):
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
    tests: List[str] = []
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        names = row.get(key)
        if names is None:
            continue
        if isinstance(names, str):
            try:
                names = json.loads(names)
            except json.JSONDecodeError as exc:
                raise ValueError(f"SWE-Gym {key} must be a list of test names") from exc
        if not isinstance(names, list) or any(
            not isinstance(name, str) or not name.strip() for name in names
        ):
            raise ValueError(f"SWE-Gym {key} must be a list of nonempty test names")
        tests.extend(names)
    tests.sort()
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
    e4_path = (
        repo_root / "outputs/e4_banker_toolbench/official_repo_ff6db552/native-data/tasks.jsonl"
    )
    for required in (e1_path, e2_path, e4_path):
        if not required.exists():
            raise RuntimeError(f"required decontamination input is missing: {required}")

    e1 = load_dataset("parquet", data_files=str(e1_path), split="train")
    e1_ids = {str(value) for value in e1["instance_id"]}
    e1_pairs = {(str(repo), str(commit)) for repo, commit in zip(e1["repo"], e1["base_commit"])}
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

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"non-finite JSON number: {value}")
        return parsed

    try:
        parsed = json.loads(
            response.strip(),
            object_pairs_hook=object_pairs_hook,
            parse_constant=reject_nonstandard_constant,
            parse_float=finite_float,
        )
    except (json.JSONDecodeError, _DuplicateJSONKey, TypeError, ValueError, RecursionError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _tool_alias_value(payload: Dict[str, Any], primary: str, alias: str) -> Any:
    """Resolve an alias pair while rejecting conflicting values."""
    present = [payload[key] for key in (primary, alias) if key in payload]
    if not present:
        return _MISSING_TOOL_FIELD
    if len(present) == 2 and _canonical_tool_value(present[0]) != _canonical_tool_value(present[1]):
        return _CONFLICTING_TOOL_FIELD
    return present[0]


def _canonical_tool_value(value: Any) -> Any:
    """Preserve JSON boolean types, case, whitespace, and numeric equivalence."""
    if isinstance(value, dict):
        return {str(k): _canonical_tool_value(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_canonical_tool_value(v) for v in value]
    if isinstance(value, bool):
        # Python considers True == 1 and False == 0, unlike JSON tool schemas.
        # A tuple cannot collide with an input JSON array (represented as a list).
        return ("boolean", value)
    return value


class StrictToolCallReward:
    """Verifier-backed reward for one exact, unambiguous JSON tool call."""

    def score(self, response: str, example: TrainingExample) -> float:
        # Model output can parse successfully yet exceed the recursive
        # canonicalizer's depth. It must not abort the training/eval loop.
        try:
            return self._score(response, example)
        except RecursionError:
            return 0.0

    def _score(self, response: str, example: TrainingExample) -> float:
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
            _canonical_tool_value(predicted_args.get(key)) == _canonical_tool_value(arguments[key])
            for key in expected_keys
            if key in predicted_args
        )
        score += 0.3 * exact_values / len(expected_keys)
        return min(score, 1.0)


def _boxed_candidates(response: str) -> List[str]:
    """Return cleaned ``\\boxed{}`` answers from a math response."""
    return [
        item.strip().replace(",", "").replace(" ", "")
        for item in re.findall(r"\\boxed\{([^}]+)\}", response)
    ]


def _numeric_equal(candidate: str, answer: str, *, tolerance: float = 0.01) -> bool:
    """Return True when two answer strings match exactly or numerically."""
    context = Context(prec=28, rounding=ROUND_FLOOR, traps=[InvalidOperation])
    try:
        left = Decimal(candidate, context=context)
        right = Decimal(answer, context=context)
    except (InvalidOperation, ValueError, TypeError):
        return bool(candidate) and candidate == answer
    if not left.is_finite() or not right.is_finite():
        return False
    limit = Decimal(str(tolerance), context=context)
    if not limit.is_finite() or limit <= 0:
        return False
    context.prec = max(context.prec, len(limit.as_tuple().digits))
    # The limit is exactly representable. Rounding a nonnegative difference
    # DOWN preserves its strict comparison to that limit, even immediately
    # below the boundary; float conversion can merge distinct large integers.
    difference = context.subtract(max(left, right), min(left, right))
    return difference < limit


class MathReward:
    """Scores math completions: boxed answer > last number > partial credit."""

    def score(self, response: str, example: TrainingExample) -> float:
        answer = "" if example.target is None else str(example.target)
        response = response.strip()

        boxed = _boxed_candidates(response)
        for b_clean in boxed:
            if _numeric_equal(b_clean, answer):
                return 1.0
        if boxed:
            return 0.3

        nums = re.findall(r"\b" + re.escape(answer) + r"\b", response)
        if nums:
            return 0.5

        all_nums = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", response)
        if all_nums:
            last = all_nums[-1].replace(",", "")
            if _numeric_equal(last, answer):
                return 1.0

        if any(c in response for c in "+-*/="):
            return 0.1
        return 0.0


def _exact_math_box_spans(response: str) -> List[Tuple[int, int, Optional[str]]]:
    """Scan balanced boxes iteratively, retaining malformed trailing spans."""
    spans: List[Tuple[int, int, Optional[str]]] = []
    cursor = 0
    pattern = re.compile(r"\\boxed\s*\{")
    while match := pattern.search(response, cursor):
        start = match.start()
        content_start = match.end()
        position = content_start
        depth = 1
        while position < len(response) and depth:
            char = response[position]
            if char == "\\":
                position += 2
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            position += 1
        if depth:
            spans.append((start, len(response), None))
            break
        spans.append((start, position, response[content_start : position - 1]))
        cursor = position
    return spans


def _normalise_exact_math_candidate(value: str) -> str:
    value = value.strip().replace(" ", "")
    # Redundant balanced braces around a scalar are TeX grouping, not algebra.
    leading = len(value) - len(value.lstrip("{"))
    trailing = len(value) - len(value.rstrip("}"))
    if leading and leading == trailing:
        inner = value[leading:-trailing]
        if re.fullmatch(r"[-+]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", inner):
            value = inner
    # Remove thousands separators only for a valid grouped numeric token.
    if re.fullmatch(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d*)?(?:[eE][-+]?\d+)?", value):
        return value.replace(",", "")
    return value


def _unwrap_numeric_context(before: str, after: str) -> Tuple[str, str]:
    """Remove matched numeric wrappers using indices, without recursive slicing."""
    left_end, right_start = len(before), 0
    wrappers = ((r"\(", r"\)"), (r"\[", r"\]"), ("(", ")"))
    while True:
        left, right = left_end, right_start
        while left and before[left - 1].isspace():
            left -= 1
        while right < len(after) and after[right].isspace():
            right += 1
        for opening, closing in wrappers:
            if before.endswith(opening, 0, left) and after.startswith(closing, right):
                left_end, right_start = left - len(opening), right + len(closing)
                break
        else:
            return before[:left_end], after[right_start:]


def _exact_math_candidates(response: str) -> List[str]:
    spans = _exact_math_box_spans(response)
    candidates = [_normalise_exact_math_candidate(value) for _, _, value in spans if value]
    # Never rescue a wrong final box using an earlier number in the reasoning.
    suffix = response[spans[-1][1] :] if spans else response
    matches = list(re.finditer(r"[-+]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", suffix))
    if not matches:
        return candidates
    last = matches[-1]
    before, after = suffix[: last.start()], suffix[last.end() :]
    before, after = _unwrap_numeric_context(before, after)
    # A sentence period after a wrapper is punctuation, not a decimal suffix.
    if after.startswith(".") and (len(after) == 1 or after[1].isspace()):
        after = after[1:]
    if (before and (before[-1].isalnum() or before[-1] in "_.)]}")) or (
        after and (after[0].isalnum() or after[0] in "_.([{")
    ):
        return candidates
    # A component adjacent to expression/container syntax is not a bare answer.
    left, right = before.rstrip(), after.lstrip()
    if (left and left[-1] in "/^*+-,([{\\<>≤≥≠&|%×÷") or (
        right and right[0] in "/^*+-,)]}\\<>≤≥≠&|!×÷="
    ):
        return candidates
    if left.endswith(("<=", ">=", "!=", "==")):
        return candidates
    if left.endswith(":") and re.search(r"(?<![\w.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*$", left[:-1]):
        return candidates
    if last.group().startswith(("+", "-")) and left and (left[-1].isdigit() or left[-1] in ").]}"):
        return candidates
    candidate = _normalise_exact_math_candidate(last.group().removesuffix(","))
    if "," not in candidate:
        candidates.append(candidate)
    return candidates


class ExactMathReward:
    """Binary boxed-or-final-number reward used by held-out math benchmarks."""

    def score(self, response: str, example: TrainingExample) -> float:
        answer = "" if example.target is None else str(example.target)
        candidates = _exact_math_candidates(response.strip())
        for candidate in candidates:
            if _numeric_equal(candidate, answer):
                return 1.0
        return 0.0


class PatchReward:
    """Dense, format-aware reward for a SWE-Gym unified diff candidate."""

    @staticmethod
    def _normalise(value: str) -> str:
        # Context lines consisting of one space count toward unified-diff hunks.
        # Trailing content whitespace can also change the resulting source file.
        return value.replace("\r\n", "\n").strip("\n")

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
        reward_kind = example.metadata.get("reward_kind")

        def is_answer(text: str) -> bool:
            if reward_kind == "tool_call":
                return _strict_json_object(text) is not None
            return reward_kind == "patch" and text.startswith("diff --git")

        # Qwen's chat template exposes its reasoning channel in decoded token
        # text.  Score only the answer after a completed reasoning block, just
        # as the E11 verifier extracts HDL after ``</think>``.  An unclosed
        # block remains invalid and receives no special treatment.
        # A delimiter inside an already structured answer is literal data.
        # Select the first complete structured suffix, not the last delimiter
        # inside JSON string values or source-code diff lines.
        if "</think>" in response and not is_answer(response):
            for match in re.finditer("</think>", response):
                candidate = response[match.end() :].lstrip()
                if is_answer(candidate):
                    response = candidate
                    break
        if reward_kind == "tool_call":
            return self._tool.score(response, example)
        if reward_kind == "patch":
            return self._patch.score(response, example)
        return 0.0
