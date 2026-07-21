from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import validate_corpus_manifest, with_fingerprint
from .flops import PROFILED_STEPS, TorchPhaseProfiler
from .protocol import (
    PROTOCOL_PATH,
    REPO_ROOT,
    AuthorizationError,
    PilotUnit,
    build_screening_plan,
    PilotProtocol,
    execution_blockers,
    load_protocol,
    sha256_file,
)
from .remote_core import (
    RemoteContractError,
    gsm8k_reward,
    math500_reward,
    prompt_messages,
    require_a100,
    seed_everything,
    source_manifest,
    verified_train_order,
    verify_runtime_versions,
)
from .replay import (
    ReplayCandidate,
    ReplayContractError,
    ReplayLedger,
    balanced_equal_length_group,
    filtered_variable_length_pool,
)
from .training import ReplayBatch, completion_logps, run_replay_step


CORPUS_RESULT_PREFIX = "FPILOT_CORPUS_RESULT "


def _require_credentials() -> None:
    missing = [key for key in ("HF_TOKEN", "WANDB_API_KEY") if not os.environ.get(key)]
    if missing:
        raise RemoteContractError(f"required credentials are missing: {', '.join(missing)}")


def _dataset(protocol: PilotProtocol, regime: str) -> tuple[Any, Sequence[Mapping[str, Any]]]:
    from datasets import load_dataset

    contract = protocol.payload["regimes"][regime]
    if regime == "balanced_equal_length":
        dataset = load_dataset(
            contract["dataset"],
            "main",
            revision=contract["dataset_revision"],
            trust_remote_code=False,
        )
        return dataset, dataset["train"]
    dataset = load_dataset(
        contract["dataset"],
        revision=contract["dataset_revision"],
        trust_remote_code=False,
    )
    return dataset, dataset["test"]


def _row_fields(regime: str, row: Mapping[str, Any]) -> tuple[str, str]:
    if regime == "balanced_equal_length":
        return str(row["question"]), str(row["answer"])
    return str(row["problem"]), str(row["answer"])


def _prompt_tokens(tokenizer: Any, *, regime: str, question: str, max_length: int) -> Any:
    import torch

    encoded = tokenizer.apply_chat_template(
        prompt_messages(regime, question),
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=True,
    )
    if hasattr(encoded, "input_ids"):
        # Transformers v5 returns a BatchEncoding here instead of a tensor.
        encoded = encoded["input_ids"]
    if not torch.is_tensor(encoded):
        encoded = torch.as_tensor(encoded, dtype=torch.long)
    if encoded.ndim == 1:
        encoded = encoded[None, :]
    if encoded.shape[0] != 1:
        raise RemoteContractError("chat template did not produce one prompt row")
    return encoded[:, -max_length:]


def _completion_tokens(sequence: Any, *, prompt_width: int, eos_token_id: int) -> tuple[int, ...]:
    tokens = [int(token) for token in sequence[prompt_width:].tolist()]
    if eos_token_id in tokens:
        tokens = tokens[: tokens.index(eos_token_id) + 1]
    if not tokens:
        tokens = [eos_token_id]
    return tuple(tokens)


def _generate_candidates(
    *,
    model: Any,
    tokenizer: Any,
    prompt_ids: Any,
    regime: str,
    answer: str,
    group_index: int,
    seed: int,
    count: int,
    max_new_tokens: int,
) -> list[ReplayCandidate]:
    import torch

    device = next(model.parameters()).device
    prompt_ids = prompt_ids.to(device)
    attention_mask = torch.ones_like(prompt_ids)
    # Transformers v5 removed the `generator` generate kwarg; reproducible
    # sampling now requires scoped global seeding. fork_rng restores the
    # outer RNG state so corpus groups stay independent of call order.
    # The math SDPA backend is pinned for the same reason as in
    # training.completion_logps: fused CUDA kernels on torch 2.7.1 can reject
    # this attention configuration under deterministic algorithms.
    fork_devices = [] if device.type == "cpu" else [device]
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with torch.random.fork_rng(devices=fork_devices), sdpa_kernel([SDPBackend.MATH]):
        torch.manual_seed(seed * 1_000_003 + group_index)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                num_return_sequences=count,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
    candidates: list[ReplayCandidate] = []
    for candidate_index, sequence in enumerate(generated):
        tokens = _completion_tokens(
            sequence,
            prompt_width=prompt_ids.shape[1],
            eos_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        reward = (
            gsm8k_reward(text, answer)
            if regime == "balanced_equal_length"
            else math500_reward(text, answer)
        )
        candidates.append(
            ReplayCandidate.from_tokens(
                candidate_id=f"g{group_index:03d}-c{candidate_index:02d}",
                token_ids=tokens,
                reward=reward,
            )
        )
    return candidates


def _group_batch(
    *,
    model: Any,
    prompt_ids: Any,
    group: Any,
) -> tuple[ReplayBatch, Any]:
    import torch

    prompt = prompt_ids.cpu().repeat(8, 1)
    prompt_mask = torch.ones_like(prompt)
    completion_ids = torch.tensor(group.padded_token_ids, dtype=torch.long)
    completion_mask = torch.tensor(group.optimization_masks, dtype=torch.float32)
    active_rows = torch.zeros(8, dtype=torch.bool)
    active_rows[list(group.active_indices)] = True
    rewards = torch.tensor(
        [candidate.reward for candidate in group.candidates], dtype=torch.float32
    )
    provisional = ReplayBatch(
        group_fingerprint=group.fingerprint,
        prompt_ids=prompt,
        prompt_mask=prompt_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        rewards=rewards,
        active_rows=active_rows,
        old_logps=torch.zeros_like(completion_mask),
    )
    with torch.inference_mode():
        old_logps = completion_logps(model, provisional).cpu()
    batch = ReplayBatch(
        group_fingerprint=provisional.group_fingerprint,
        prompt_ids=provisional.prompt_ids,
        prompt_mask=provisional.prompt_mask,
        completion_ids=provisional.completion_ids,
        completion_mask=provisional.completion_mask,
        rewards=provisional.rewards,
        active_rows=provisional.active_rows,
        old_logps=old_logps,
    )
    return batch, old_logps


def _save_group(path: Path, *, batch: ReplayBatch, group: Any) -> None:
    import torch

    payload = {
        "schema_version": "flagship-pilot-replay-group-v1",
        "group": group.as_record(),
        "prompt_ids": batch.prompt_ids,
        "prompt_mask": batch.prompt_mask,
        "completion_ids": batch.completion_ids,
        "completion_mask": batch.completion_mask,
        "rewards": batch.rewards,
        "active_rows": batch.active_rows,
        "old_logps": batch.old_logps,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def build_corpus(
    *,
    protocol: PilotProtocol,
    regime: str,
    seed: int,
    hf_repo: str,
    wandb_project: str,
    wandb_entity: str | None,
) -> dict[str, Any]:
    protocol.require_gpu_authorization()
    plan = build_screening_plan(
        protocol,
        PilotUnit(condition="intended_full", regime=regime, seed=seed),
    )
    if hf_repo != plan["identity"]["corpus_hf_repo"]:
        raise RemoteContractError(
            f"corpus repository does not match frozen identity: expected "
            f"{plan['identity']['corpus_hf_repo']}, got {hf_repo}"
        )
    _require_credentials()

    import torch
    import wandb
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer

    versions = verify_runtime_versions(protocol)
    accelerator = require_a100(torch)
    seed_everything(seed, torch)
    runtime = protocol.payload["runtime"]
    contract = runtime["execution_contract"]
    regime_contract = protocol.payload["regimes"][regime]
    if seed not in runtime["screening_seeds"]:
        raise RemoteContractError(f"seed is outside screening: {seed}")

    dataset, train_rows = _dataset(protocol, regime)
    order = verified_train_order(train_rows, protocol=protocol, regime=regime, seed=seed)
    tokenizer = AutoTokenizer.from_pretrained(
        runtime["model"]["id"], revision=runtime["model"]["revision"]
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        runtime["model"]["id"],
        revision=runtime["model"]["revision"],
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
    )
    model.eval()

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=hf_repo, repo_type="dataset", private=True, exist_ok=True)
    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        group=contract["tracking"]["wandb_corpus_group"],
        name=plan["identity"]["corpus_wandb_run"],
        config={
            "protocol_sha256": protocol.sha256,
            "regime": regime,
            "seed": seed,
            "model": runtime["model"],
            "dataset_revision": regime_contract["dataset_revision"],
        },
    )
    started_at = time.time()
    groups = []
    profiled_flops = 0.0
    profiled_tokens = 0
    group_records: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="flagship-pilot-corpus-") as temporary:
        root = Path(temporary)
        for group_index, row_index in enumerate(order):
            question, answer = _row_fields(regime, train_rows[row_index])
            prompt_ids = _prompt_tokens(
                tokenizer,
                regime=regime,
                question=question,
                max_length=int(contract["max_prompt_length"]),
            )
            candidate_count = int(contract["generated_candidates_per_group"][regime])
            profiler = TorchPhaseProfiler(
                torch,
                enabled=(group_index + 1) in PROFILED_STEPS,
            )
            with profiler("replay_generation"):
                candidates = _generate_candidates(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_ids=prompt_ids,
                    regime=regime,
                    answer=answer,
                    group_index=group_index,
                    seed=seed,
                    count=candidate_count,
                    max_new_tokens=int(contract["max_completion_length"]),
                )
            if regime == "balanced_equal_length":
                group = balanced_equal_length_group(
                    candidates,
                    pad_token_id=tokenizer.eos_token_id,
                )
            else:
                group = filtered_variable_length_pool(
                    candidates,
                    pad_token_id=tokenizer.eos_token_id,
                )
            batch, _ = _group_batch(model=model, prompt_ids=prompt_ids, group=group)
            artifact_path = root / "groups" / f"group-{group_index:03d}.pt"
            _save_group(artifact_path, batch=batch, group=group)
            groups.append(group)
            if profiler.phase_flops:
                profiled_flops += profiler.phase_flops["replay_generation"]
                profiled_tokens += group.charged_generated_tokens
            record = {
                "index": group_index,
                "source_row_index": int(row_index),
                "fingerprint": group.fingerprint,
                "active_rows": len(group.active_indices),
                "selected_length_cv": group.selected_length_cv,
                "charged_generated_tokens": group.charged_generated_tokens,
                "artifact_path": str(artifact_path.relative_to(root)),
            }
            group_records.append(record)
            run.log(
                {
                    "corpus/group": group_index + 1,
                    "corpus/charged_generated_tokens": sum(
                        item.charged_generated_tokens for item in groups
                    ),
                    "corpus/selected_length_cv": group.selected_length_cv,
                },
                step=group_index + 1,
            )

        ledger = ReplayLedger.build(groups)
        ceiling = int(contract["charged_generated_token_ceiling"])
        if ledger.charged_generated_tokens > ceiling:
            raise RemoteContractError(
                f"corpus charged {ledger.charged_generated_tokens} tokens above ceiling {ceiling}"
            )
        if profiled_flops <= 0 or profiled_tokens <= 0:
            raise RemoteContractError("replay generation profiler coverage is missing")
        replay_generation_flops = profiled_flops * (
            ledger.charged_generated_tokens / profiled_tokens
        )
        if not math.isfinite(replay_generation_flops) or replay_generation_flops <= 0:
            raise RemoteContractError("replay generation FLOP extrapolation is invalid")

        sources = source_manifest(REPO_ROOT)
        source_path = root / "source_manifest.json"
        source_path.write_text(json.dumps(sources, indent=2, sort_keys=True) + "\n")
        artifact_files = {
            str(path.relative_to(root)): sha256_file(path)
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }
        manifest = with_fingerprint(
            {
                "schema_version": "flagship-pilot-corpus-v1",
                "status": "complete",
                "protocol_sha256": protocol.sha256,
                "regime": regime,
                "seed": seed,
                "model": runtime["model"],
                "dataset": regime_contract["dataset"],
                "dataset_revision": regime_contract["dataset_revision"],
                "train_order_hash": contract["train_order_hash"][regime][str(seed)],
                "groups": group_records,
                "rejected_generated_tokens": 0,
                "charged_generated_tokens": ledger.charged_generated_tokens,
                "flop_ledger": {
                    "profiled_steps": list(PROFILED_STEPS),
                    "profiled_generated_tokens": profiled_tokens,
                    "profiled_generation_flops": profiled_flops,
                    "replay_generation_flops": replay_generation_flops,
                },
                "runtime_versions": versions,
                "accelerator": accelerator,
                "source_manifest": sources,
                "artifact_files": artifact_files,
                "wall_clock_seconds": time.time() - started_at,
                "wandb": {
                    "run_id": run.id,
                    "run_url": run.url,
                    "entity": run.entity,
                    "project": run.project,
                },
            }
        )
        validate_corpus_manifest(
            manifest,
            protocol=protocol,
            regime=regime,
            seed=seed,
        )
        manifest_path = root / "corpus_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        commit = api.upload_folder(
            repo_id=hf_repo,
            repo_type="dataset",
            folder_path=root,
            path_in_repo=".",
            commit_message=f"Complete flagship pilot corpus {regime} seed {seed}",
        )

    result = {
        "status": "completed",
        "protocol_sha256": protocol.sha256,
        "regime": regime,
        "seed": seed,
        "corpus_fingerprint": manifest["fingerprint"],
        "charged_generated_tokens": manifest["charged_generated_tokens"],
        "replay_generation_flops": manifest["flop_ledger"]["replay_generation_flops"],
        "hf_repo": hf_repo,
        "hf_commit": commit.oid,
        "wandb_run_id": run.id,
        "wandb_run_url": run.url,
    }
    run.summary.update(result)
    wandb.finish(exit_code=0)
    return result


def describe(protocol: PilotProtocol) -> dict[str, Any]:
    return {
        "status": protocol.status,
        "gpu_authorized": protocol.gpu_authorized,
        "protocol_sha256": protocol.sha256,
        "execution_blockers": list(execution_blockers(protocol)),
    }


def smoke(protocol: PilotProtocol) -> dict[str, Any]:
    protocol.require_gpu_authorization()
    import torch

    from .remote_training import _model_stack

    versions = verify_runtime_versions(protocol)
    accelerator = require_a100(torch)
    model, tokenizer, optimizer, scheduler = _model_stack(protocol, seed=11)
    _, rows = _dataset(protocol, "balanced_equal_length")
    question, answer = _row_fields("balanced_equal_length", rows[0])
    contract = protocol.payload["runtime"]["execution_contract"]
    prompt_ids = _prompt_tokens(
        tokenizer,
        regime="balanced_equal_length",
        question=question,
        max_length=int(contract["max_prompt_length"]),
    )
    candidates = _generate_candidates(
        model=model,
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        regime="balanced_equal_length",
        answer=answer,
        group_index=0,
        seed=11,
        count=8,
        max_new_tokens=int(contract["max_completion_length"]),
    )
    group = balanced_equal_length_group(candidates, pad_token_id=tokenizer.eos_token_id)
    batch, _ = _group_batch(model=model, prompt_ids=prompt_ids, group=group)
    profiler = TorchPhaseProfiler(torch, enabled=True)
    receipt = run_replay_step(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        batch=batch,
        condition="intended_full",
        step=1,
        max_grad_norm=float(contract["max_grad_norm"]),
        phase_context=profiler,
    )
    profiler.require_training_coverage()
    return {
        "status": "smoke_pass",
        "runtime_versions": versions,
        "accelerator": accelerator,
        "group_fingerprint": group.fingerprint,
        "charged_generated_tokens": group.charged_generated_tokens,
        "receipt": receipt.as_record(),
        "phase_flops": profiler.phase_flops,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flagship pilot remote executor")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("describe")
    subparsers.add_parser("verify-environment")
    subparsers.add_parser("smoke")
    corpus = subparsers.add_parser("build-corpus")
    corpus.add_argument("--regime", choices=("balanced_equal_length", "filtered_variable_length"), required=True)
    corpus.add_argument("--seed", type=int, required=True)
    corpus.add_argument("--hf-repo", required=True)
    corpus.add_argument("--wandb-project", default="tinker-rl-lab")
    corpus.add_argument("--wandb-entity")
    training = subparsers.add_parser("train-unit")
    training.add_argument(
        "--condition",
        choices=("intended_full", "native_trl", "epsilon_only", "reduction_only"),
        required=True,
    )
    training.add_argument(
        "--regime",
        choices=("balanced_equal_length", "filtered_variable_length"),
        required=True,
    )
    training.add_argument("--seed", type=int, required=True)
    training.add_argument("--wandb-project", default="tinker-rl-lab")
    training.add_argument("--wandb-entity")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    protocol = load_protocol(args.protocol)
    if args.command == "describe":
        print(json.dumps(describe(protocol), indent=2, sort_keys=True))
        return 0
    if args.command == "verify-environment":
        protocol.require_gpu_authorization()
        import torch

        print(
            json.dumps(
                {
                    "runtime_versions": verify_runtime_versions(protocol),
                    "accelerator": require_a100(torch),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "smoke":
        try:
            result = smoke(protocol)
        except (AuthorizationError, RemoteContractError, ReplayContractError) as exc:
            raise SystemExit(str(exc)) from exc
        print("FPILOT_SMOKE_RESULT " + json.dumps(result, sort_keys=True))
        return 0
    try:
        if args.command == "build-corpus":
            result = build_corpus(
                protocol=protocol,
                regime=args.regime,
                seed=args.seed,
                hf_repo=args.hf_repo,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
            )
        else:
            from .remote_training import train_unit

            result = train_unit(
                protocol=protocol,
                condition=args.condition,
                regime=args.regime,
                seed=args.seed,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
            )
    except (AuthorizationError, RemoteContractError, ReplayContractError) as exc:
        raise SystemExit(str(exc)) from exc
    prefix = CORPUS_RESULT_PREFIX if args.command == "build-corpus" else "FPILOT_UNIT_RESULT "
    print(prefix + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
