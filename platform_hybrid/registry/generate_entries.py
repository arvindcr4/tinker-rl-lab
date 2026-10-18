#!/usr/bin/env python3
"""Regenerate GRPO-Registry stack entries from the actual run sources.

Every `run_config` field is derived from a pinned run script (git blob at the
commit that was live when the run happened) or from a results/log JSON, and
records where it came from:

  status  logged    value read from a results/log artifact   source  <path>#<json key>
          code      constant in the pinned run script         source  <path>@<commit>:L<n>
          default   resolved default with runtime evidence    source  <path>@<commit>:L<n>
                    (unobserved CLI/library defaults remain unknown)
          inferred  derived by a stated rule                  source  rule text
          unknown   not derivable from any source in the repo source  why

The generator fails loudly if an anchor regex no longer matches its source,
so a moved/edited line cannot silently produce a stale provenance pointer.

Usage (from the repo root or anywhere):
  python3 platform_hybrid/registry/generate_entries.py           # rewrite entries + regen_changes.json
  python3 platform_hybrid/registry/generate_entries.py --check   # exit 1 if entries are stale
  python3 platform_hybrid/registry/generate_entries.py --dry-run # print the change summary only

Stdlib only; needs `git` for the pinned blobs.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import statistics
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ENTRIES = HERE / "entries"
CHANGES = HERE / "regen_changes.json"
GENERATOR = "platform_hybrid/registry/generate_entries.py"
GENERATOR_VERSION = "registry-generation-v3"
REWARD_FINGERPRINT_VERSION = "python-reward-assignment-and-module-v2"
sys.path.insert(0, str(HERE))
from query import (
    FIELDS,
    KNOWN_STATUSES,
    badge,
    manifest_view,
    strict_json_loads,
    COVERAGE_VERSION,
    ITEMS,
    validate_record,
)


# ---------------------------------------------------------------- sources
class Src:
    """One pinned source file: a git blob (commit given) or a working-tree file."""

    def __init__(self, path: str, commit: str | None = None):
        self.path, self.commit = path, commit
        if commit:
            data = subprocess.run(
                ["git", "show", f"{commit}:{path}"], cwd=ROOT, capture_output=True, check=True
            ).stdout
        else:
            data = (ROOT / path).read_bytes()
        self.sha256 = hashlib.sha256(data).hexdigest()
        self.text = data.decode()
        self.lines = self.text.splitlines()
        self._tree = ast.parse(self.text) if path.endswith(".py") else None
        self.json = strict_json_loads(self.text) if path.endswith(".json") else None

    @property
    def artifact(self) -> str:
        at = f"@{self.commit}" if self.commit else ""
        return f"{self.path}{at} sha256:{self.sha256}"

    def ref(self, line: int) -> str:
        at = f"@{self.commit}" if self.commit else ""
        return f"{self.path}{at}:L{line}"

    def find(self, pattern: str):
        rx = re.compile(pattern)
        for i, line in enumerate(self.lines, 1):
            m = rx.search(line)
            if m:
                return i, m
        raise SystemExit(f"anchor not found in {self.path}@{self.commit}: {pattern!r}")

    def cell(self, pattern, value=None, *, group=None, cast=None, status="code", note=None):
        """Field value anchored at the first line matching `pattern`."""
        line, m = self.find(pattern)
        if status == "default":
            return unknown(
                f"unresolved default at {self.ref(line)}; "
                + (note or "no effective launch/runtime configuration")
            )
        if value is None:
            value = m.group(group or 1)
            value = cast(value) if cast else value
        elif m.groups() and isinstance(value, str):
            raise ValueError(f"derive descriptions from captured values at {self.ref(line)}")
        elif m.groups() and (
            isinstance(value, (int, float, list, tuple)) and not isinstance(value, bool)
        ):
            # A matching anchor alone cannot certify a hard-coded numeric value.
            try:
                actual = [ast.literal_eval(g) for g in m.groups()]
            except (ValueError, SyntaxError) as exc:
                raise SystemExit(f"nonliteral numeric anchor at {self.ref(line)}") from exc
            expected = list(value) if isinstance(value, (list, tuple)) else [value]
            if actual != expected:
                raise SystemExit(
                    f"source assumption mismatch at {self.ref(line)}: "
                    f"expected {expected!r}, found {actual!r}"
                )
        elif status == "code":
            # A regex without captured values proves only source location.
            # Preserve semantic annotations as conditional interpretations.
            status = "inferred"
            note = "semantic source interpretation; value not mechanically extracted; " + (
                note or ""
            )
        return mk(value, status, self.ref(line), note)

    def keyword_cell(self, call_name, keyword):
        """Extract a literal keyword from an actual AST call, never a supplied value."""
        matches = []
        for node in ast.walk(self._tree):
            if (
                isinstance(node, ast.Call)
                and ast.get_source_segment(self.text, node.func) == call_name
            ):
                matches.extend(kw for kw in node.keywords if kw.arg == keyword)
        if len(matches) != 1:
            raise SystemExit(f"expected one {call_name} {keyword} in {self.path}")
        kw = matches[0]
        try:
            value = ast.literal_eval(kw.value)
        except (ValueError, SyntaxError) as exc:
            raise SystemExit(f"nonliteral {call_name} {keyword} at {self.ref(kw.lineno)}") from exc
        return mk(value, "code", self.ref(kw.lineno))

    def optimizer_betas(self):
        first, second = (
            self.keyword_cell("T.AdamParams", "beta1"),
            self.keyword_cell("T.AdamParams", "beta2"),
        )
        return mk(
            [first["value"], second["value"]], "code", first["source"] + "; " + second["source"]
        )

    def prompts_per_step(self):
        batch = self.keyword_cell("GRPOConfig", "generation_batch_size")
        group = self.keyword_cell("GRPOConfig", "num_generations")
        if (
            type(batch["value"]) is not int
            or type(group["value"]) is not int
            or group["value"] <= 0
            or batch["value"] % group["value"]
        ):
            raise SystemExit("generation batch size must be an integer multiple of group size")
        return mk(
            batch["value"] // group["value"],
            "code",
            batch["source"] + "; " + group["source"],
            "generation_batch_size / num_generations",
        )

    def assignment_src(self, name):
        nodes = [
            n
            for n in ast.walk(self._tree)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
        ]
        if not nodes:
            raise ValueError(f"missing assignment to {name}")
        nodes.sort(key=lambda node: node.lineno)
        return "\n".join(ast.get_source_segment(self.text, node) for node in nodes), nodes[0].lineno

    def reward_fingerprint(self):
        assignment, line = self.assignment_src("rewards")
        # Full module conservatively captures helper/constants and transformations,
        # including dependencies a narrow reward expression would omit.
        return mk(
            sha_of(REWARD_FINGERPRINT_VERSION, assignment, self.func_src("parse"), self.text),
            "code",
            self.ref(line),
            REWARD_FINGERPRINT_VERSION,
        )

    def _func(self, name: str):
        for node in ast.walk(self._tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        raise SystemExit(f"function {name} not in {self.path}@{self.commit}")

    def func_src(self, name: str) -> str:
        return ast.get_source_segment(self.text, self._func(name))

    def func_sha(self, name: str):
        """(sha256 of the function's source segment, its first line)."""
        return hashlib.sha256(self.func_src(name).encode()).hexdigest(), self._func(name).lineno

    def literal(self, name: str) -> str:
        for node in ast.walk(self._tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets
            ):
                return ast.literal_eval(node.value)
        raise SystemExit(f"literal {name} not in {self.path}@{self.commit}")

    def key(self, keypath: str, status="logged", note=None):
        """Value at a dotted/indexed JSON key path, e.g. frameworks[0].model."""
        cur = self.json
        for part in re.findall(r"[^.\[\]]+|\[\d+\]", keypath):
            cur = cur[int(part[1:-1])] if part.startswith("[") else cur[part]
        return mk(cur, status, f"{self.path}#{keypath}", note)


def mk(value, status, source, note=None):
    if status not in KNOWN_STATUSES + ("unknown",):
        raise ValueError(f"Unsupported evidence status: {status}")
    if status == "default":
        # Existing extractors have no resolved launch or pinned-library evidence.
        # Keep the source/default description, but never treat it as an observation.
        status, value = "unknown", None
        note = "unresolved default; " + (note or "effective runtime value not recorded")
    if status == "unknown":
        value = None
    out = {"value": value, "status": status, "source": source}
    if note:
        out["note"] = note
    return out


def unknown(why):
    return mk(None, "unknown", why)


def inferred(value, rule):
    return mk(value, "inferred", rule)


def sha_of(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


def outcome_evidence(src, keypath, computation_version="extract-json-number-v1"):
    """Lineage for extraction, not an invented claim about upstream computation."""
    return {
        "source": f"{src.path}#{keypath}",
        "input_sha256": src.sha256,
        "computation_version": computation_version,
        "note": "source-reported statistic; upstream computation not independently reproduced",
    }


def variant_rule(model):
    if not model:
        return unknown("checkpoint name is not reported")
    name = model.split("/")[-1].lower()
    if re.search(r"(?:^|[-_.])base(?:$|[-_.])", name):
        return inferred("base", f"checkpoint name '{model}' carries a -Base suffix")
    if re.search(r"(?:^|[-_.])instruct(?:$|[-_.])", name) or name.endswith(("-inst", "-it")):
        return inferred(
            "instruct", f"checkpoint name '{model}' explicitly identifies an instruct release"
        )
    return unknown(f"checkpoint name '{model}' does not establish base/instruct variant")


# ---------------------------------------------------- exhibit (a): Qwen3-8B
def fc_family():
    """framework_comparison.json rows: Tinker campaign_v2 w1, TRL on Modal, verl/OpenRLHF dry runs."""
    fc = Src("platform_hybrid/experiments/results/framework_comparison.json")
    rows = {r["framework"]: i for i, r in enumerate(fc.json["frameworks"])}
    # Scripts as committed with the results (2026-04-19, bf5b1db9f).
    c2 = Src("experiments/tinker-runs/campaign_v2.py", "bf5b1db9f")
    trl = Src("experiments/modal/modal_grpo_trl.py", "bf5b1db9f")
    out = {}

    # --- Tinker, campaign_v2 wave 1 (experiment_id campaign_v2_w1_qwen3-8b-base)
    i = rows["Tinker"]
    p = f"frameworks[{i}]"
    rsha, rline = c2.func_sha("reward_fn")
    sysp, suf = c2.literal("SYSTEM_PROMPT"), c2.literal("QUESTION_SUFFIX")
    model = fc.json["frameworks"][i]["model"]
    loss_ref = c2.find(r"losses = \[\(-_advs\[i\] \* lp\[i\]\.sum\(\)\)")[0]
    rc = {
        "execution.mode": fc.key(f"{p}.mode"),
        "checkpoint.id": fc.key(f"{p}.model"),
        "checkpoint.variant": variant_rule(model),
        "checkpoint.revision": unknown(
            f"unpinned: create_lora_training_client(base_model=model) has no revision "
            f"({c2.ref(c2.find(r'create_lora_training_client')[0])})"
        ),
        "reward.id": c2.cell(r"def reward_fn", "boxed-any|last-number-fallback|abs-tol-0.01"),
        "reward.fn_sha256": mk(
            rsha, "code", c2.ref(rline), "sha256 of the reward_fn source segment"
        ),
        "prompt.template_sha256": c2.cell(
            r"^SYSTEM_PROMPT = ",
            sha_of(sysp, suf),
            note="sha256(SYSTEM_PROMPT \\x1f QUESTION_SUFFIX)",
        ),
        "prompt.chat_template": c2.cell(
            r'prompt = \(f"<\|im_start\|>system',
            "manual-chatml",
            note="hand-built ChatML string applied to a Base "
            "checkpoint; no tokenizer chat template",
        ),
        "data.dataset": c2.cell(r'load_dataset\("openai/gsm8k", "main"', "openai/gsm8k:main"),
        "data.subset": c2.cell(
            r'split="train"',
            "train (all rows with a #### answer)",
            note="framework_comparison.json labels the task 'gsm8k-500'; "
            "the Tinker script samples the full train split",
        ),
        "data.selection": c2.cell(
            r"batch = random\.sample\(examples, min\(2",
            "seeded shuffle; random.sample per step (with replacement across steps)",
        ),
        "eval.metric": mk(TRAIN_LAST10, "logged", f"{fc.path}#metric", fc.json["metric"]),
        "eval.heldout": inferred("none", f"no evaluation loop in {c2.path}@{c2.commit}"),
        "trainer.name": c2.cell(r"create_lora_training_client", "tinker"),
        "trainer.version": unknown("tinker SDK version not logged in the result row or script"),
        "runtime.backend": c2.cell(r"sc = tc\.create_sampling_client", "tinker-managed sampler"),
        "runtime.precision": unknown("managed by Tinker; not exposed or logged"),
        "decoding.temperature": c2.cell(
            r"T\.SamplingParams\(max_tokens=512, temperature=([\d.]+)", cast=float
        ),
        "decoding.top_p": c2.cell(r"T\.SamplingParams\(.*top_p=([\d.]+)", cast=float),
        "decoding.max_tokens": c2.cell(r"T\.SamplingParams\(max_tokens=(\d+)", cast=int),
        "batch.prompts_per_step": c2.cell(
            r"batch = random\.sample\(examples, min\((\d+)", cast=int
        ),
        "batch.samples_per_prompt": fc.key(f"{p}.group_size"),
        "batch.group_schedule": c2.cell(r"num_samples=group_size", "fixed"),
        "train.steps": fc.key(f"{p}.steps"),
        "train.sampler_refresh": c2.cell(
            r"if \(step \+ 1\) % 10 == 0", "every 10 steps, no importance correction"
        ),
        "loss.objective": mk(
            "reinforce: -A_i * sum_t logp, mean over sequences", "code", c2.ref(loss_ref)
        ),
        "loss.importance_ratio": mk(
            "none", "code", c2.ref(loss_ref), "custom loss uses raw log-probs; no ratio"
        ),
        "loss.clip": mk("none", "code", c2.ref(loss_ref)),
        "loss.aggregation": mk("sequence-sum", "code", c2.ref(loss_ref)),
        "loss.advantage_norm": c2.cell(r"sr = \(sum\(\(r - mr\)\*\*2", "std"),
        "loss.token_mask": c2.cell(
            r"tid = fid\[1:\] \+ \[0\]",
            "none (prompt+completion targets; spurious token-0 final target)",
        ),
        "loss.length_normalization": mk(False, "code", c2.ref(loss_ref)),
        "loss.dynamic_sampling": mk(False, "code", c2.ref(c2.find(r"batch = random\.sample")[0])),
        "loss.reward_shaping": mk("none", "code", c2.ref(rline)),
        "loss.inner_epochs": c2.cell(r"fwdbwd = tc\.forward_backward_custom", 1),
        "kl.reference_policy": mk(False, "code", c2.ref(loss_ref), "custom loss has no KL term"),
        "kl.beta": mk(0.0, "code", c2.ref(loss_ref)),
        "kl.estimator": mk("none", "code", c2.ref(loss_ref)),
        "optimizer.name": c2.cell(r"T\.AdamParams\(", "adam (tinker AdamParams)"),
        "optimizer.lr": fc.key(f"{p}.learning_rate"),
        "optimizer.betas": c2.optimizer_betas(),
        "optimizer.eps": c2.cell(r"eps=([\de.-]+)\)", cast=float),
        "optimizer.weight_decay": unknown("not set; AdamParams default unresolved"),
        "lora.rank": c2.cell(
            r"create_lora_training_client\(base_model=model, rank=(\d+)", cast=int
        ),
        "lora.alpha": unknown("managed by Tinker; not set or logged"),
        "lora.dropout": unknown("managed by Tinker; not set or logged"),
        "lora.targets": unknown("managed by Tinker; not set or logged"),
        "seeds": mk([fc.json["frameworks"][i]["seed"]], "logged", f"{fc.path}#{p}.seed"),
    }
    out["tinker_grpo_qwen3-8b_gsm8k"] = dict(
        run_config=rc,
        sources=[fc, c2],
        label="grpo",
        framework={"name": "tinker", "version": None, "openness": "managed"},
        model=model,
        task="gsm8k",
        seeds=rc["seeds"]["value"],
        telemetry=(
            False,
            False,
            f"train/reward only ({c2.ref(c2.find(r'wb_run.log')[0])}); "
            "per-step ZVF logging was added 2026-06-14, after this run",
        ),
        outcomes={"mean_last10_train_reward": fc.json["frameworks"][i]["last10_avg"]},
        outcome_provenance={"mean_last10_train_reward": outcome_evidence(fc, f"{p}.last10_avg")},
        notes=(
            "Tinker campaign_v2 wave-1 run (experiment_id campaign_v2_w1_qwen3-8b-base). "
            "Custom REINFORCE loss via forward_backward_custom: no ratio, no clip, no KL; "
            "2 prompts x 8 samples, LoRA rank 32, temp 0.8, 512 tokens. "
            "The 44-field YAML dump previously cited describes neither this run nor the TRL one."
        ),
    )

    # --- TRL GRPOTrainer on Modal H100 (experiment_id modal_trl_trl_qwen3_8b)
    i = rows["TRL"]
    p = f"frameworks[{i}]"
    rsha, rline = trl.func_sha("reward_function")
    sysp = trl.literal("SYSTEM_PROMPT")
    _, msuf = trl.find(r'example\["question"\] \+ "(.*)"\}')
    model = fc.json["frameworks"][i]["model"]
    cfg = trl.find(r"training_args = GRPOConfig\(")[0]
    lib = (
        "not set in GRPOConfig; TRL library default applies but the resolved TRL "
        f"version is not logged (pip spec 'trl>=1.0', {trl.ref(trl.find(r'trl>=')[0])})"
    )
    rc = {
        "execution.mode": fc.key(f"{p}.mode"),
        "checkpoint.id": fc.key(f"{p}.model"),
        "checkpoint.variant": variant_rule(model),
        "checkpoint.revision": unknown(
            f"unpinned: GRPOTrainer(model=model_id) has no revision "
            f"({trl.ref(trl.find(r'model=model_id')[0])})"
        ),
        "reward.id": mk("boxed-any|no-fallback|abs-tol-0.01", "code", trl.ref(rline)),
        "reward.fn_sha256": mk(
            rsha, "code", trl.ref(rline), "sha256 of the reward_function source segment"
        ),
        "prompt.template_sha256": trl.cell(
            r'SYSTEM_PROMPT = "',
            sha_of(sysp, msuf.group(1).encode().decode("unicode_escape")),
            note="sha256(SYSTEM_PROMPT \\x1f user suffix)",
        ),
        "prompt.chat_template": trl.cell(
            r'"prompt": \[',
            "hf-apply_chat_template",
            note="TRL conversational prompt -> Qwen3 tokenizer chat "
            "template; enable_thinking not set",
        ),
        "data.dataset": trl.cell(r'load_dataset\("openai/gsm8k", "main"', "openai/gsm8k:main"),
        "data.subset": trl.cell(r'split="(train\[:500\])"'),
        "data.selection": mk("TRL default sampler", "default", trl.ref(cfg), lib),
        "eval.metric": mk(TRAIN_LAST10, "logged", f"{fc.path}#metric", fc.json["metric"]),
        "eval.heldout": inferred("none", f"no eval_dataset at {trl.path}@{trl.commit}"),
        "trainer.name": trl.cell(r"from trl import GRPOConfig, GRPOTrainer", "trl-GRPOTrainer"),
        "trainer.version": unknown(lib),
        "runtime.backend": unknown("generation backend (use_vllm) not set; " + lib),
        "runtime.precision": trl.cell(r"bf16=True", "bf16"),
        "decoding.temperature": unknown(lib),
        "decoding.top_p": unknown(lib),
        "decoding.max_tokens": trl.cell(r"max_completion_length=(\d+)", cast=int),
        "batch.prompts_per_step": trl.prompts_per_step(),
        "batch.samples_per_prompt": trl.cell(r"num_generations=(\d+)", cast=int),
        "batch.group_schedule": trl.cell(r"num_generations=", "fixed"),
        "train.steps": trl.cell(r"max_steps=(\d+)", cast=int),
        "train.sampler_refresh": mk(
            "every step (on-policy generation)", "default", trl.ref(cfg), lib
        ),
        "loss.objective": unknown("GRPOTrainer loss_type " + lib),
        "loss.importance_ratio": unknown(lib),
        "loss.clip": unknown("epsilon " + lib),
        "loss.aggregation": unknown("loss_type " + lib),
        "loss.advantage_norm": unknown("scale_rewards " + lib),
        "loss.token_mask": mk(
            "completion-only", "default", trl.ref(cfg), "GRPOTrainer masks prompt tokens"
        ),
        "loss.length_normalization": unknown("loss_type " + lib),
        "loss.dynamic_sampling": mk(False, "code", trl.ref(cfg), "no filtering configured"),
        "loss.reward_shaping": mk("none", "code", trl.ref(rline)),
        "loss.inner_epochs": unknown("num_iterations " + lib),
        "kl.reference_policy": unknown("beta " + lib),
        "kl.beta": unknown("beta " + lib),
        "kl.estimator": unknown(lib),
        "optimizer.name": mk(
            "adamw_torch", "default", trl.ref(cfg), "TrainingArguments default optim"
        ),
        "optimizer.lr": trl.cell(r"learning_rate=([\de.-]+)", cast=float),
        "optimizer.betas": mk(
            [0.9, 0.999], "default", trl.ref(cfg), "TrainingArguments adam_beta1/2 defaults"
        ),
        "optimizer.eps": mk(
            1e-8, "default", trl.ref(cfg), "TrainingArguments adam_epsilon default"
        ),
        "optimizer.weight_decay": mk(0.0, "default", trl.ref(cfg), "TrainingArguments default"),
        "lora.rank": trl.cell(r"r=(\d+), lora_alpha=", cast=int),
        "lora.alpha": trl.cell(r"lora_alpha=(\d+)", cast=int),
        "lora.dropout": trl.cell(r"lora_dropout=([\d.]+)", cast=float),
        "lora.targets": trl.keyword_cell("LoraConfig", "target_modules"),
        "seeds": mk(
            [fc.json["frameworks"][i]["seed"]],
            "logged",
            f"{fc.path}#{p}.seed",
            "script sets no seed; TrainingArguments default seed=42",
        ),
    }
    out["trl_grpo_qwen3-8b_gsm8k"] = dict(
        run_config=rc,
        sources=[fc, trl],
        label="grpo",
        framework={"name": "trl", "version": None, "openness": "open"},
        model=model,
        task="gsm8k",
        seeds=rc["seeds"]["value"],
        telemetry=(False, False, "TRL log_history reward only; no ZVF/GU logging in the script"),
        outcomes={"mean_last10_train_reward": fc.json["frameworks"][i]["last10_avg"]},
        outcome_provenance={"mean_last10_train_reward": outcome_evidence(fc, f"{p}.last10_avg")},
        notes=(
            "TRL GRPOTrainer run on Modal H100 (experiment_id modal_trl_trl_qwen3_8b). "
            "Post-trained Qwen3-8B with the tokenizer chat template, boxed-only reward, "
            "train[:500], 1 prompt x 8 samples, LoRA r32/a64/drop0.05. Loss/KL/decoding "
            "follow unpinned TRL defaults. 25 of 30 logged step rewards are 0.0."
        ),
    )

    # --- verl / OpenRLHF: dry-run fallbacks, no training executed
    for fw, eid in (
        ("verl", "verl_grpo_qwen3-8b_gsm8k"),
        ("OpenRLHF", "openrlhf_grpo_qwen3-8b_gsm8k"),
    ):
        i = rows[fw]
        p = f"frameworks[{i}]"
        model = fc.json["frameworks"][i]["model"]
        rc = {
            "execution.mode": fc.key(f"{p}.mode", note=fc.json["frameworks"][i].get("note")),
            "checkpoint.id": fc.key(f"{p}.model", note="intended model; never loaded"),
            "checkpoint.variant": variant_rule(model),
            "data.dataset": unknown("dry run: no data was loaded"),
            "eval.metric": mk(TRAIN_LAST10, "logged", f"{fc.path}#metric", fc.json["metric"]),
            "seeds": mk([fc.json["frameworks"][i]["seed"]], "logged", f"{fc.path}#{p}.seed"),
        }
        out[eid] = dict(
            run_config={"execution.mode": rc["execution.mode"]},
            intended_config=rc,
            sources=[fc],
            label="grpo",
            framework={"name": fw.lower(), "version": None, "openness": "open"},
            model=None,
            task=None,
            seeds=None,
            telemetry=(None, None, None),
            outcomes={},
            synthetic_diagnostics={"description": "seeded fallback only; no training executed"},
            notes="DRY RUN: synthetic diagnostics only; intended configuration is not execution evidence.",
        )
    return out


# ------------------------------------------------ exhibit (b): h2h2 Tinker cells
TRAIN_LAST10 = "mean sampled train reward over last 10 steps"
H2H_ARMS = ("grpo", "drgrpo", "dapo", "gspo")


def h2h_outcomes(runs):
    seeds, rewards, zvfs = [], [], []
    for src in runs:
        run = src.json
        seed = run.get("seed")
        if type(seed) is not int or seed in seeds or run.get("status") != "completed":
            raise ValueError("H2H requires distinct completed seed runs")
        seeds.append(seed)
        trace, log = run.get("reward_trace"), run.get("step_log")
        if (
            not isinstance(trace, list)
            or len(trace) < 10
            or not isinstance(log, list)
            or len(trace) != len(log)
            or len(log) != run.get("steps")
        ):
            raise ValueError("H2H run lacks complete per-step outcomes")
        for step, row in enumerate(log, 1):
            if row.get("step") != step or row.get("reward") != trace[step - 1]:
                raise ValueError("H2H step identity/reward mismatch")
        values = trace + [row.get("zvf") for row in log]
        if any(
            type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in values
        ):
            raise ValueError("invalid H2H outcome")
        rewards.append(statistics.mean(trace[-10:]))
        zvfs.append(statistics.mean(row["zvf"] for row in log))
    if not seeds:
        raise ValueError("H2H requires completed seed artifacts")
    outcomes = {
        "mean_last10_train_reward": statistics.mean(rewards),
        "mean_zvf": statistics.mean(zvfs),
    }
    evidence = {
        key: {
            "input_artifacts": [src.artifact for src in runs],
            "computation_version": "mean-of-seed-last10-reward-and-full-zvf-v1",
            "source": "per-run reward_trace and step_log",
            "seeds": seeds,
            "execution_binding": "unknown; historical logs lack execution receipts",
        }
        for key in outcomes
    }
    return outcomes, evidence


def h2h_family():
    probe = Src("experiments/tinker-runs/live_zvf_probe.py", "b0ac85bfb")
    cell = Src("experiments/tinker-runs/cell_runner.py", "b0ac85bfb")
    summ = Src("platform_hybrid/experiments/tinker-runs/results/week_h2h/SUMMARY.json")
    rsha, rline = probe.func_sha("reward_fn")
    sysp, suf = probe.literal("SYSTEM_PROMPT"), probe.literal("QUESTION_SUFFIX")
    base_loss = probe.find(r"losses = \[\(-_ADVS\[i\] \* logprobs_list\[i\]\.sum\(\)\)")[0]
    out = {}
    for arm in H2H_ARMS:
        runs = [
            Src(
                f"platform_hybrid/experiments/tinker-runs/results/week_h2h/"
                f"h2h2_{arm}_Qwen3.5-4B_s{s}_G8_Dmedium_R16_S15.json"
            )
            for s in (42, 123, 456)
        ]
        r0 = runs[0]
        for k in (
            "model",
            "rank",
            "group_size",
            "batch",
            "lr",
            "temperature",
            "steps",
            "difficulty",
        ):
            vals = {json.dumps(r.json[k]) for r in runs}
            if len(vals) != 1:
                raise ValueError(f"Inconsistent {arm} {k}: {vals}")
        if not all(r.json["status"] == "completed" for r in runs):
            raise ValueError(f"Incomplete source run for {arm}")
        model = r0.json["model"]
        dflt = "cell_runner argparse default; the sweep launcher command is not in the repo"
        rc = {
            "execution.mode": mk(
                "real", "logged", f"{r0.path}#status", "status=completed in all 3 seeds"
            ),
            "checkpoint.id": r0.key("model"),
            "checkpoint.variant": variant_rule(model),
            "checkpoint.revision": unknown(
                f"unpinned: create_lora_training_client(base_model=...) "
                f"({probe.ref(probe.find(r'create_lora_training_client')[0])})"
            ),
            "reward.id": mk(
                "boxed-any|last-number-fallback|abs-tol-0.01", "code", probe.ref(rline)
            ),
            "reward.fn_sha256": mk(
                rsha, "code", probe.ref(rline), "sha256 of the reward_fn source segment"
            ),
            "prompt.template_sha256": probe.cell(
                r"^SYSTEM_PROMPT = \(",
                sha_of(sysp, suf),
                note="sha256(SYSTEM_PROMPT \\x1f QUESTION_SUFFIX)",
            ),
            "prompt.chat_template": cell.cell(r'f"<\|im_start\|>system', "manual-chatml"),
            "data.dataset": cell.cell(r'load_dataset\("openai/gsm8k", "main"', "openai/gsm8k:main"),
            "data.subset": unknown(
                f"difficulty={r0.json['difficulty']!r} is logged at {r0.path}#difficulty; "
                f"example_limit is not logged and the launch command is absent"
            ),
            "data.selection": probe.cell(
                r"batch_examples = rng\.sample", "seeded rng.sample per step"
            ),
            "eval.metric": mk(
                TRAIN_LAST10, "logged", f"{summ.path}#by_arm.{arm}.mean_last10_reward"
            ),
            "eval.heldout": inferred("none", f"no evaluation loop in {probe.path}@{probe.commit}"),
            "trainer.name": probe.cell(r"create_lora_training_client", "tinker"),
            "trainer.version": unknown("tinker SDK version not logged"),
            "runtime.backend": probe.cell(
                r"sc = tc\.create_sampling_client", "tinker-managed sampler"
            ),
            "runtime.precision": unknown("managed by Tinker; not exposed or logged"),
            "decoding.temperature": r0.key("temperature"),
            "decoding.top_p": cell.cell(
                r'--top-p", dest="top_p", type=float, default=([\d.]+)',
                cast=float,
                status="default",
                note=dflt,
            ),
            "decoding.max_tokens": cell.cell(
                r'--max-tokens", type=int, default=(\d+)', cast=int, status="default", note=dflt
            ),
            "batch.prompts_per_step": r0.key("batch"),
            "batch.samples_per_prompt": r0.key("group_size"),
            "batch.group_schedule": probe.cell(r"num_samples=args\.group", "fixed"),
            "train.steps": r0.key("steps"),
            "train.sampler_refresh": cell.cell(
                r'--save-every", type=int, default=(\d+)', cast=int, status="default", note=dflt
            ),
            "loss.objective": mk(
                "reinforce: -A_i * sum_t logp, mean over sequences", "code", probe.ref(base_loss)
            ),
            "loss.importance_ratio": mk(
                "none", "code", probe.ref(base_loss), "custom loss uses raw log-probs; no ratio"
            ),
            "loss.clip": mk("none", "code", probe.ref(base_loss)),
            "loss.aggregation": mk("sequence-sum", "code", probe.ref(base_loss)),
            "loss.advantage_norm": probe.cell(r"else \(reward - mean_reward\) / std_reward", "std"),
            "loss.token_mask": probe.cell(
                r"target_ids = full_ids\[1:\] \+ \[0\]",
                "none (prompt+completion targets; spurious token-0 final target)",
            ),
            "loss.length_normalization": mk(False, "code", probe.ref(base_loss)),
            "loss.dynamic_sampling": mk(
                False, "code", probe.ref(probe.find(r"batch_examples = rng\.sample")[0])
            ),
            "loss.reward_shaping": mk("none", "code", probe.ref(rline)),
            "loss.inner_epochs": probe.cell(r"fwdbwd = tc\.forward_backward_custom", 1),
            "kl.reference_policy": mk(
                False, "code", probe.ref(base_loss), "custom loss has no KL term"
            ),
            "kl.beta": mk(0.0, "code", probe.ref(base_loss)),
            "kl.estimator": mk("none", "code", probe.ref(base_loss)),
            "optimizer.name": probe.cell(r"T\.AdamParams\(", "adam (tinker AdamParams)"),
            "optimizer.lr": r0.key("lr"),
            "optimizer.betas": probe.optimizer_betas(),
            "optimizer.eps": probe.cell(r"eps=([\de.-]+),", cast=float),
            "optimizer.weight_decay": unknown("not set; AdamParams default unresolved"),
            "lora.rank": r0.key("rank"),
            "lora.alpha": unknown("managed by Tinker; not set or logged"),
            "lora.dropout": unknown("managed by Tinker; not set or logged"),
            "lora.targets": unknown("managed by Tinker; not set or logged"),
            "seeds": mk(
                sorted(r.json["seed"] for r in runs),
                "logged",
                "platform_hybrid/experiments/tinker-runs/results/week_h2h/h2h2_"
                + arm
                + "_*.json#seed",
            ),
        }
        deltas = []
        if arm == "dapo":
            ln = cell.find(r"clip_lo, clip_hi = -2\.0, 3\.0")[0]
            rc["loss.clip"] = mk(
                "advantage[-2,3]",
                "code",
                cell.ref(ln),
                "clamps the standardized advantage, not the importance ratio",
            )
            deltas = [
                (
                    "delta_dapo",
                    "clip_higher",
                    "absent",
                    f"advantage clip [-2,3] ({cell.ref(ln)}); no ratio exists to clip",
                ),
                (
                    "delta_dapo",
                    "dynamic_sampling",
                    "absent",
                    "no zero-variance filter/resample in the loop",
                ),
                (
                    "delta_dapo",
                    "token_level_loss",
                    "absent",
                    f"sequence-sum loss ({probe.ref(base_loss)})",
                ),
                (
                    "delta_dapo",
                    "overlong_reward_shaping",
                    "absent",
                    f"binary reward ({probe.ref(rline)})",
                ),
                (
                    "delta_dapo",
                    "kl_removed",
                    "implemented",
                    f"custom loss has no KL term ({probe.ref(base_loss)})",
                ),
            ]
        elif arm == "gspo":
            ln = cell.find(r"seq = lp\.mean\(\) if lp\.numel\(\) > 0")[0]
            rc["loss.aggregation"] = mk("sequence-mean", "code", cell.ref(ln))
            rc["loss.length_normalization"] = mk(True, "code", cell.ref(ln))
            rc["loss.objective"] = mk(
                "reinforce: -A_i * mean_t logp, mean over sequences", "code", cell.ref(ln)
            )
            deltas = [
                (
                    "delta_gspo",
                    "sequence_level_ratio",
                    "absent",
                    f"no importance ratio; only mean-logp aggregation ({cell.ref(ln)})",
                ),
                ("delta_gspo", "sequence_level_clip", "absent", "no ratio, no clip"),
            ]
        elif arm == "drgrpo":
            ln = cell.find(r"probe_module\._ADV_NORMALIZE_STD = False")[0]
            rc["loss.advantage_norm"] = mk(
                "none",
                "code",
                cell.ref(ln),
                "honoured by live_zvf_probe@b0ac85bfb (no later getattr override)",
            )
            deltas = [
                (
                    "delta_drgrpo",
                    "advantage_std_normalization",
                    "implemented",
                    f"std division removed ({cell.ref(ln)})",
                ),
                (
                    "delta_drgrpo",
                    "length_normalization",
                    "implemented",
                    f"baseline loss is already an unnormalized sum ({probe.ref(base_loss)})",
                ),
            ]
        derived_outcomes, derived_evidence = h2h_outcomes(runs)
        out[f"tinker_{arm}_qwen3.5-4b_gsm8k"] = dict(
            run_config=rc,
            sources=[probe, cell, summ] + runs,
            label=arm,
            framework={"name": "tinker", "version": None, "openness": "managed"},
            model=model,
            task="gsm8k",
            seeds=rc["seeds"]["value"],
            deltas=deltas,
            telemetry=(True, True, f"step_log zvf/gu per step ({r0.path}#step_log)"),
            outcomes=derived_outcomes,
            outcome_provenance=derived_evidence,
            notes=(
                f"12-cell Tinker head-to-head, '{arm}' arm (3 seeds). cell_runner surrogate on the "
                "in-repo REINFORCE loop: no ratio, no KL, sequence-sum log-probs over prompt+completion. "
                + {
                    "dapo": "'dapo' = advantage clip [-2,3] only; none of DAPO's four components.",
                    "gspo": "'gspo' = mean instead of sum log-prob; no sequence ratio or clip.",
                    "drgrpo": "'drgrpo' = advantage without /std.",
                    "grpo": "Baseline arm.",
                }[arm]
            ),
        )
    return out


# ----------------------------------------------- exhibit (b): E3 open Colab trainer
E3_ARMS = {
    "grpo": "colab-open_grpo_e3",
    "drgrpo": "colab-open_drgrpo_e3",
    "dapo": "colab-open_dapo_e3",
    "grpo_adaptiveG": "colab-open_grpo-adaptiveg_e3",
}


def e3_family():
    e3 = Src("zvf-program/colab-experiments/e3_open_audit.py", "71e6f09ee")
    res = Src("zvf-program/colab-experiments/results/e3_open_audit.json")
    _, pline = e3.func_sha("parse")
    _, um = e3.find(r'"content": f"(Compute \{q\}\..*?)"\}')
    cline = e3.find(r"def clipped_loss")[0]
    ratio = e3.find(r"ratio = torch\.exp\(new_lp - old_lp\.detach\(\)\)")[0]
    seql = e3.find(r"return lp\[:, pids\.shape\[0\] - 1:\]\.sum\(\)")[0]
    model = res.json["model"]
    out = {}
    for arm, eid in E3_ARMS.items():
        rc = {
            "execution.mode": mk("real", "logged", f"{res.path}#_source", res.json["_source"]),
            "checkpoint.id": res.key("model"),
            "checkpoint.variant": variant_rule(model),
            "checkpoint.revision": unknown(
                f"unpinned from_pretrained ({e3.ref(e3.find(r'AutoModelForCausalLM.from_pretrained')[0])})"
            ),
            "reward.id": mk("after-####-first-integer|exact", "code", e3.ref(pline)),
            "reward.fn_sha256": e3.reward_fingerprint(),
            "prompt.template_sha256": mk(
                sha_of("", um.group(1)),
                "code",
                e3.ref(e3.find(r'"content": f"Compute')[0]),
                "sha256('' \\x1f user template); no system prompt",
            ),
            "prompt.chat_template": e3.cell(
                r"tok\.apply_chat_template\(", "hf-apply_chat_template"
            ),
            "data.dataset": e3.cell(r"def problem\(\)", "synthetic a+b, a,b in [11,60]"),
            "data.subset": e3.cell(r"a, b = random\.randint\(11, 60\)", "generated on the fly"),
            "data.selection": e3.cell(r"q, gold = problem\(\)", "fresh random problem per group"),
            "eval.metric": e3.cell(
                r'"heldout_delta": round\(post - pre', "greedy held-out accuracy delta (post - pre)"
            ),
            "eval.heldout": e3.cell(
                r"evalset = \[problem\(\) for _ in range\(HELDOUT_N\)\]",
                "20 problems from the training generator (not enforced disjoint)",
            ),
            "trainer.name": e3.cell(
                r"opt = torch\.optim\.AdamW", "e3-open-loop (hand-written torch)"
            ),
            "trainer.version": mk("71e6f09ee", "code", e3.artifact, "script commit"),
            "runtime.backend": e3.cell(
                r"out = model\.generate\(\*\*enc, do_sample=True",
                "hf-transformers generate (single process)",
            ),
            "runtime.precision": e3.cell(r"dtype=torch\.bfloat16", "bf16"),
            "decoding.temperature": e3.cell(r"do_sample=True, temperature=([\d.]+)", cast=float),
            "decoding.top_p": e3.cell(r"top_p=([\d.]+)", cast=float),
            "decoding.max_tokens": e3.cell(
                r"^BATCH, G0, GMAX, MAX_NEW, LR, STEPS, INNER = 3, 4, 10, (\d+)", cast=int
            ),
            "batch.prompts_per_step": e3.cell(
                r"^BATCH, G0, GMAX, MAX_NEW, LR, STEPS, INNER = (\d+)", cast=int
            ),
            "batch.samples_per_prompt": e3.cell(
                r"^BATCH, G0, GMAX, MAX_NEW, LR, STEPS, INNER = 3, (\d+)", cast=int
            ),
            "batch.group_schedule": e3.cell(r"^BATCH, G0", "fixed"),
            "train.steps": e3.cell(
                r"^BATCH, G0, GMAX, MAX_NEW, LR, STEPS, INNER = 3, 4, 10, 40, 2e-6, (\d+)", cast=int
            ),
            "train.sampler_refresh": e3.cell(
                r"old = \[seq_logprob\(model, pids, gens\[i\], grad=False\)",
                "every step (on-policy; old log-probs cached for inner epochs)",
            ),
            "loss.objective": mk(
                "ppo-clip on sequence log-prob ratio, summed over sequences", "code", e3.ref(cline)
            ),
            "loss.importance_ratio": mk(
                "sequence", "code", e3.ref(ratio), "exp(sum new logp - sum old logp)"
            ),
            "loss.clip": e3.cell(
                r"lo, hi = \(0\.2, 0\.28\) if arm == \"dapo\" else \(0\.2, 0\.2\)", "ratio[0.8,1.2]"
            ),
            "loss.aggregation": mk("sequence-sum", "code", e3.ref(seql)),
            "loss.advantage_norm": e3.cell(r"return \[\(r - m\) / \(s \+ 1e-6\)", "std"),
            "loss.token_mask": mk("completion-only", "code", e3.ref(seql)),
            "loss.length_normalization": mk(False, "code", e3.ref(seql)),
            "loss.dynamic_sampling": mk(
                False, "code", e3.ref(e3.find(r"if adv is not None or arm != \"dapo\"")[0])
            ),
            "loss.reward_shaping": mk(
                "none", "code", e3.ref(e3.find(r"rewards = \[1\.0 if parse")[0])
            ),
            "loss.inner_epochs": e3.cell(
                r"^BATCH, G0, GMAX, MAX_NEW, LR, STEPS, INNER = .*, (\d+)$", cast=int
            ),
            "kl.reference_policy": mk(
                False, "code", e3.ref(cline), "no KL term anywhere in the loop"
            ),
            "kl.beta": mk(0.0, "code", e3.ref(cline)),
            "kl.estimator": mk("none", "code", e3.ref(cline)),
            "optimizer.name": e3.cell(r"opt = torch\.optim\.AdamW", "adamw (torch)"),
            "optimizer.lr": e3.cell(
                r"^BATCH, G0, GMAX, MAX_NEW, LR, STEPS, INNER = 3, 4, 10, 40, ([\de.-]+)",
                cast=float,
            ),
            "optimizer.betas": e3.cell(
                r"opt = torch\.optim\.AdamW\(model\.parameters\(\), lr=LR\)",
                [0.9, 0.999],
                status="default",
                note="torch.optim.AdamW default",
            ),
            "optimizer.eps": e3.cell(
                r"opt = torch\.optim\.AdamW",
                1e-8,
                status="default",
                note="torch.optim.AdamW default",
            ),
            "optimizer.weight_decay": e3.cell(
                r"opt = torch\.optim\.AdamW",
                0.01,
                status="default",
                note="torch.optim.AdamW default",
            ),
            "lora.rank": e3.cell(
                r"AutoModelForCausalLM\.from_pretrained",
                "full-ft",
                note="no adapter; all parameters trained",
            ),
            "lora.alpha": e3.cell(r"AutoModelForCausalLM\.from_pretrained", "full-ft"),
            "lora.dropout": e3.cell(r"AutoModelForCausalLM\.from_pretrained", "full-ft"),
            "lora.targets": e3.cell(r"AutoModelForCausalLM\.from_pretrained", "full-ft"),
            "seeds": res.key("seeds"),
        }
        deltas = []
        if arm == "dapo":
            ds = e3.find(r"tries \+= 1")[0]
            rc["loss.clip"] = e3.cell(r"lo, hi = \(0\.2, 0\.28\)", "ratio[0.8,1.28]")
            rc["loss.dynamic_sampling"] = mk(
                "resample zero-variance groups up to 2 retries, then drop", "code", e3.ref(ds)
            )
            deltas = [
                (
                    "delta_dapo",
                    "clip_higher",
                    "implemented",
                    f"ratio clip [0.8, 1.28] ({e3.ref(cline + 2)})",
                ),
                (
                    "delta_dapo",
                    "dynamic_sampling",
                    "surrogate",
                    f"<=2 resamples per dead group, then the group is dropped; batch is not refilled ({e3.ref(ds)})",
                ),
                (
                    "delta_dapo",
                    "token_level_loss",
                    "absent",
                    f"sequence-level ratio on summed log-probs ({e3.ref(ratio)})",
                ),
                (
                    "delta_dapo",
                    "overlong_reward_shaping",
                    "absent",
                    "binary exact-match reward; 40-token cap",
                ),
                ("delta_dapo", "kl_removed", "implemented", f"no KL term ({e3.ref(cline)})"),
            ]
        elif arm == "drgrpo":
            ln = e3.find(r"return \[r - m for r in rewards\]")[0]
            rc["loss.advantage_norm"] = mk("none", "code", e3.ref(ln))
            deltas = [
                (
                    "delta_drgrpo",
                    "length_normalization",
                    "implemented",
                    f"log-probs summed, no 1/|o| ({e3.ref(seql)})",
                ),
                (
                    "delta_drgrpo",
                    "advantage_std_normalization",
                    "implemented",
                    f"std division removed ({e3.ref(ln)})",
                ),
            ]
        elif arm == "grpo_adaptiveG":
            ln = e3.find(r'if arm == "grpo_adaptiveG" and len\(recent_zvf\) >= 2')[0]
            rc["batch.group_schedule"] = mk(
                "adaptive: G += 2 (max 10) when mean ZVF of last 2 steps > 0.4; no de-escalation",
                "code",
                e3.ref(ln),
            )
            deltas = [
                (
                    "delta_adaptiveg",
                    "zvf_driven_group_size_ladder",
                    "implemented",
                    f"G 4 -> +2 per trigger, cap GMAX=10, trigger mean(last-2 ZVF) > 0.4, never decreases ({e3.ref(ln)})",
                )
            ]
        a = res.json["by_arm"][arm]
        out[eid] = dict(
            run_config=rc,
            sources=[e3, res],
            label=None if arm == "grpo_adaptiveG" else arm,
            framework={"name": "colab-open-trainer", "version": "71e6f09ee", "openness": "open"},
            model=model,
            task="synthetic-arithmetic (E3 audit)",
            seeds=rc["seeds"]["value"],
            deltas=deltas,
            telemetry=(
                True,
                False,
                f"per-step post-filter ZVF in traj ({e3.ref(e3.find(r'traj.append')[0])})",
            ),
            outcomes={
                "mean_zvf": a["mean_zvf"],
                "heldout_delta": a["mean_heldout_delta"],
                "rollouts": int(a["mean_rollouts"]),
            },
            outcome_provenance={
                "mean_zvf": outcome_evidence(res, f"by_arm.{arm}.mean_zvf"),
                "heldout_delta": outcome_evidence(res, f"by_arm.{arm}.mean_heldout_delta"),
                "rollouts": outcome_evidence(
                    res, f"by_arm.{arm}.mean_rollouts", "extract-json-number-int-cast-v1"
                ),
            },
            schedule=("adaptive" if arm == "grpo_adaptiveG" else "fixed"),
            notes=(
                "E3 open audit on Colab T4: Qwen2.5-0.5B-Instruct full fine-tune, synthetic a+b, "
                "3 prompts x G=4, 10 steps x 2 inner epochs, sequence-level PPO-clip ratio, no KL. "
                + (
                    "'dapo': clip-higher yes; dynamic sampling capped at 2 retries; ZVF counts groups "
                    "still dead after retries, so 0.00 is largely by construction."
                    if arm == "dapo"
                    else "Toy scale: directional evidence only."
                )
            ),
        )
    return out


# ------------------------------------------------------------- entry builder
def to_min_report(rc, old, spec):
    """Derive the MIN-REPORT block from run_config (unknown -> null)."""
    v = {k: (c["value"] if c["status"] in KNOWN_STATUSES else None) for k, c in rc.items()}
    clip_lo = clip_hi = None
    m = re.fullmatch(r"ratio\[([\d.]+),([\d.]+)\]", str(v.get("loss.clip")))
    if m:
        clip_lo, clip_hi = round(1 - float(m.group(1)), 6), round(float(m.group(2)) - 1, 6)
    ratio = v.get("loss.importance_ratio")
    dyn = v.get("loss.dynamic_sampling")
    per_zvf, per_gu, tsrc = spec.get("telemetry", (None, None, None))
    heldout = v.get("eval.heldout")
    return {
        "loss_form": {
            "importance_ratio_level": ratio,
            "clip_eps_low": clip_lo,
            "clip_eps_high": clip_hi,
            "clipping_applied": False if v.get("loss.clip") == "none" else True if m else None,
            "length_normalization": v.get("loss.length_normalization"),
            "advantage_normalization": v.get("loss.advantage_norm"),
            "token_mask": v.get("loss.token_mask"),
            "token_aggregation": v.get("loss.aggregation"),
            "reward_shaping_type": v.get("loss.reward_shaping"),
            "sampling_dynamic_filter": None if dyn is None else bool(dyn),
        },
        "reference_kl": {
            "reference_policy": v.get("kl.reference_policy"),
            "kl_beta": v.get("kl.beta"),
            "kl_estimator": v.get("kl.estimator"),
        },
        "sampler_backend": {
            "backend": v.get("runtime.backend"),
            "precision": v.get("runtime.precision"),
            "temperature": v.get("decoding.temperature"),
            "top_p": v.get("decoding.top_p"),
        },
        "telemetry": {"per_step_zvf": per_zvf, "per_step_gu": per_gu, "source": tsrc},
        "group_size_schedule": {
            "initial_g": v.get("batch.samples_per_prompt"),
            "schedule": spec.get(
                "schedule", "fixed" if v.get("batch.group_schedule") == "fixed" else None
            ),
            "adaptation_rule": (
                v.get("batch.adaptation_rule")
                if spec.get("schedule") == "adaptive"
                else "none"
                if v.get("batch.group_schedule") == "fixed"
                else None
            ),
        },
        "heldout_split": {
            "disjoint_from_reward_env": None,
            "n": None,
            "description": (
                "no held-out evaluation in the run script" if heldout == "none" else heldout
            ),
        },
        "decontamination": {
            key: (
                evidence["value"]
                if isinstance(evidence, dict)
                and evidence.get("status") in KNOWN_STATUSES
                and evidence.get("source")
                and type(evidence.get("value")) is bool
                else None
            )
            for key in ("performed", "parser_robustness_probe")
            for evidence in [spec.get("decontamination", {}).get(key)]
        },
    }


def build(eid, spec, old):
    # Only explicit human annotation namespaces survive; generated claims never
    # become their own regeneration inputs.
    new = {"record_type": "stack", "id": eid, "schema_version": "registry-v3"}
    if old.get("provenance", {}).get("manual_annotations"):
        new["provenance"] = {
            "manual_annotations": copy.deepcopy(old["provenance"]["manual_annotations"])
        }
    rc = copy.deepcopy(spec["run_config"])
    if "batch.adaptation_rule" not in rc:
        schedule = rc.get("batch.group_schedule", {})
        if schedule.get("value") == "fixed":
            rc["batch.adaptation_rule"] = mk("none", schedule["status"], schedule["source"])
        elif spec.get("schedule") == "adaptive":
            rc["batch.adaptation_rule"] = copy.deepcopy(schedule)
            rc["batch.group_schedule"] = mk("adaptive", schedule["status"], schedule["source"])

    if "effective_manifest" in spec:
        new["effective_manifest"] = copy.deepcopy(spec["effective_manifest"])
        rc.update(manifest_view(spec["effective_manifest"], "spec#effective_manifest"))
    for k in rc:
        if k not in FIELDS:
            raise ValueError(f"Unknown configuration field for {eid}: {k}")
    new["run_config"] = {k: rc.get(k, unknown("not recorded in source artifacts")) for k in FIELDS}
    if "intended_config" in spec:
        new["intended_config"] = copy.deepcopy(spec["intended_config"])
    if "synthetic_diagnostics" in spec:
        new["synthetic_diagnostics"] = copy.deepcopy(spec["synthetic_diagnostics"])
    srcs = []
    for s in spec["sources"]:
        if s.artifact not in srcs:
            srcs.append(s.artifact)
    new["provenance"] = {
        "source_artifacts": srcs,
        "wandb": spec.get("wandb"),
        "date_recorded": spec.get("date_recorded"),
        "generator_version": GENERATOR_VERSION,
        "execution_binding": "unknown: inspected source is not an execution receipt",
        "generated_by": GENERATOR,
        "outcome_provenance": copy.deepcopy(spec.get("outcome_provenance", {})),
        "manual_annotations": copy.deepcopy(
            old.get("provenance", {}).get("manual_annotations", {})
        ),
    }
    # Generated outcomes are a clean projection of current evidence. Historic
    # controller estimates, CIs and other annotations must not leak through.
    new["outcomes"] = copy.deepcopy(spec.get("outcomes", {}))
    new["notes"] = spec["notes"] + " Generated by generate_entries.py; do not hand-edit."
    new["label_claimed"] = spec.get("label") or "grpo"
    new["framework"] = spec["framework"]
    new["model"], new["task"], new["seeds"] = spec["model"], spec["task"], spec["seeds"]
    new["min_report"] = to_min_report(rc, {}, spec)
    new["variant_deltas_applied"] = [
        {"delta_id": d, "component": c, "status": st, "note": n}
        for d, c, st, n in spec.get("deltas", [])
    ]
    out = new["outcomes"]
    mr = new["min_report"]
    vd = new["variant_deltas_applied"]
    # Versioned names distinguish item presence, leaf reporting and actual
    # implementation. Explicit documented absence is complete reporting.
    cov = {
        "item_presence_coverage_v3": round(
            sum(score > 0 for score in badge(new)[1].values()) / len(ITEMS), 4
        ),
        "reported_leaf_coverage_v3": sum(badge(new)[1].values()) / len(ITEMS),
        "coverage_version": COVERAGE_VERSION,
        "declared_delta_reporting_completeness_v2": (
            round(
                sum(d["status"] in ("implemented", "surrogate", "absent") for d in vd) / len(vd), 4
            )
            if vd
            else None
        ),
        "declared_delta_implementation_fraction_v1": (
            round(sum(d["status"] == "implemented" for d in vd) / len(vd), 4) if vd else None
        ),
        "declared_delta_surrogate_fraction_v1": (
            round(sum(d["status"] == "surrogate" for d in vd) / len(vd), 4) if vd else None
        ),
        "audit_source": GENERATOR,
    }
    out["coverage"] = cov
    new["provenance"]["coverage_provenance"] = {
        "source": f"{GENERATOR}#build",
        "input_sha256": hashlib.sha256(
            json.dumps(
                {"min_report": mr, "variant_deltas_applied": vd},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "computation_version": "registry-coverage-v3",
    }
    return new


def leaf_diff(a, b, prefix=""):
    rows = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            rows += leaf_diff(a.get(k, "<absent>"), b.get(k, "<absent>"), f"{prefix}{k}.")
    elif a != b:
        rows.append({"path": prefix[:-1], "before": a, "after": b})
    return rows


def hand_written_baseline(path):
    rel = path.relative_to(ROOT).as_posix()
    r = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True)
    if r.returncode != 0:
        return None
    base = json.loads(r.stdout)
    return None if "run_config" in base else base


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--check", action="store_true", help="exit 1 if any generated entry differs from disk"
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="print the change summary, write nothing"
    )
    args = ap.parse_args(argv)
    specs = {**fc_family(), **h2h_family(), **e3_family()}
    changes, stale = {}, []
    for eid, spec in sorted(specs.items()):
        path = ENTRIES / f"{eid}.json"
        old = strict_json_loads(path.read_text())
        new = build(eid, spec, old)
        validate_record(new, strict_json_loads((HERE / "schema.json").read_text()))
        if new != old:
            stale.append(eid)
        if args.check:
            continue
        # Changes are recorded against the last hand-written version in git
        # (HEAD, while it has no run_config); once generated entries are
        # committed, the previously recorded rows are kept.
        base = hand_written_baseline(path)
        if base is not None:
            changes[eid] = [
                r for r in leaf_diff(base, new) if not r["path"].startswith("run_config")
            ]
        if not args.dry_run and new != old:
            path.write_text(json.dumps(new, indent=2, ensure_ascii=False) + "\n")
    if args.check:
        print(f"{len(specs)} generated entries; stale: {stale or 'none'}")
        return 1 if stale else 0
    for eid in sorted(changes):
        print(f"{eid}: {len(changes[eid])} min_report/top-level leaf changes vs hand-written HEAD")
    if not args.dry_run:
        prior = json.loads(CHANGES.read_text()) if CHANGES.exists() else {}
        merged = {eid: rows for eid, rows in prior.items() if eid != "_doc"}
        merged.update(changes)
        CHANGES.write_text(
            json.dumps(
                {
                    "_doc": "leaf-level before/after of each generated entry "
                    "versus its hand-written predecessor (run_config "
                    "additions omitted; see the entries)",
                    **{k: merged[k] for k in sorted(merged)},
                },
                indent=1,
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )
        print(f"wrote {CHANGES.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
