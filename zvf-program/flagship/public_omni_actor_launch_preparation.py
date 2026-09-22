#!/usr/bin/env python3
"""Read-only, one-shot Omni-MATH actor export preparation; never exports or calls APIs.

Prints a prospective immutable contract and exact full-batch hash to stdout.
The caller must separately authorize materialization/export, reserve a bounded
session, and reconcile every started task before selecting any unstarted suffix.
Existing actor state is rejected; this utility is deliberately not a resume tool.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "outputs/public_portfolio_2026-09-05"
RUNTIME_PINS = {
    "modal_public_portfolio_group_runtime.py": "609a37d36634915ebb46be7b863911b2322fca61f7c231c76c302126c4508000",
    "public_colab_runtime_group.py": "6755e4e41d9ebb06ad7febdc93eb71f092e98e910d81048bca34cbffffdc5118",
    "public_runtime_journal_group_commit.py": "3493b27ff29fa80f3b931e224458e727d546305833e2b22d35441aae6dd716a4",
    "modal_public_portfolio_fast_runtime.py": "1f1c1b6b78c74a25a6e3927094540767ec5511f3a627e9456fc6e58a073a3232",
    "public_colab_runtime_fast.py": "968462cee08c2089b7aec016ba52836505ad898dcf6b7a553b5e0cb1970818dd",
}
NATIVE_PINS = {
    "public_omni_math_native.py": "c0409e355efbbb2283658c7b790271d51a659e0704e24ff786edf0958b6c5329",
    "public_portfolio_native.py": "8d0712e467502078103b33dc87e82731c3975b87ab4637e537c44adffaabdd7c",
}
CANARY_SHA256 = "232651ded61389a3f823a816f9823b5730d84e2bac042d6f2f87578c5554cd5a"
MANIFEST_SHA256 = "3ebc60af5c3725e2367a2020480b27e526b3d8d1c54e1570940f1c9b0f6e5865"
BASE = {"repo_id": "Qwen/Qwen3.6-35B-A3B", "revision": "995ad96eacd98c81ed38be0c5b274b04031597b0"}
ADAPTER = {"repo_id": "arvindcr4/pavlov-portfolio-qwen36-seed809-stepfinal-tinker-cf0ad8c1-1f1b-5ff-9f777c4018b6",
           "revision": "64444133c55d88c3f1bf0df8a2f5d7ac646125c8"}
SERVED_MODEL = "pavlov-public-portfolio-bf16"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def load_verified(name, path):
    # Import only verified, lightweight local modules. No Modal launcher import.
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.dont_write_bytecode = previous


def source_modules():
    paths = {}
    for name, expected in {**RUNTIME_PINS, **NATIVE_PINS}.items():
        directory = ROOT / (".codex-run" if name.startswith("public_colab_") or name == "public_runtime_journal_group_commit.py" else "zvf-program/flagship")
        path = directory / name
        require(sha(path.read_bytes()) == expected, f"Frozen source changed: {name}")
        paths[name] = path
    load_verified("public_portfolio_native", paths["public_portfolio_native.py"])
    native = load_verified("public_omni_math_native", paths["public_omni_math_native.py"])
    fast = load_verified("omni_plan_frozen_fast", paths["public_colab_runtime_fast.py"])
    return native, fast, paths


def check_canary(value, reconciliation, result_sha, pointer, fast):
    require(result_sha == CANARY_SHA256, "Use the frozen successful group-canary01 receipt")
    require(reconciliation.get("result_sha256") == result_sha, "Provider reconciliation belongs to another result")
    require(reconciliation.get("status") == "TERMINAL_SMOKES_ONLY" and reconciliation.get("native_calls") == 0,
            "Canary reconciliation must be synthetic-only and terminal")
    app = reconciliation.get("app", {})
    require(app.get("app_id") == "ap-gZF6Cxw7OBiv7Sbr3otqmj" and app.get("state") == "stopped" and str(app.get("tasks")) == "0" and app.get("stopped_at"),
            "Frozen canary provider allocation is not confirmed stopped with zero tasks")
    rpc, receipt = value["rpc_receipt"], value["receipt"]
    require(rpc.get("status") == "RPC_RETURNED" and rpc.get("reservation_id") == "public-20260905-group-canary01", "Wrong canary invocation")
    require(rpc.get("source_sha256") == RUNTIME_PINS and value.get("client_source_dependencies_sha256_before_dispatch") == RUNTIME_PINS,
            "Remote/client runtime source identity mismatch")
    require(receipt.get("status") == "RUNTIME_SMOKE_COMPLETE" and receipt.get("requests_completed") == 34 and receipt.get("server_process_stopped") is True,
            "All 34 runtime smokes and server teardown must complete")
    require(receipt.get("score") is None and not receipt.get("full_suite_scores"), "Canary cannot be benchmark evidence")
    group = receipt["group_commit"]
    subset = {name: value for name, value in RUNTIME_PINS.items() if not name.startswith("modal_")}
    require(group.get("source_dependencies_sha256") == subset, "Loaded runtime dependency mismatch")
    require(group.get("pending_bytes") == 0 and group.get("unacknowledged_tickets") == 0 and group.get("poisoned") is False and group.get("writer_alive") is False,
            "Canary journals are not acknowledged and cleanly drained")
    merge = receipt["merge"]
    exact = {"base_model": BASE["repo_id"], "base_commit": BASE["revision"], "adapter_repo": ADAPTER["repo_id"], "adapter_commit": ADAPTER["revision"],
             "all_adapter_tensors_consumed": True, "merge_method": "streaming_lora_delta_merge_v1", "weight_file_count": 26,
             "adapter_module_count": 431, "adapter_tensor_count": 862}
    require(all(merge.get(k) == v and pointer.get(k) == v for k, v in exact.items()), "Wrong or incomplete saved checkpoint merge")
    require(len(merge.get("weight_shard_sha256", {})) == 26 and merge["weight_shard_sha256"] == pointer.get("weight_shard_sha256"), "Full 26-shard identity mismatch")
    require(receipt["huggingface"].get("base_commit") == BASE["revision"] and receipt["huggingface"].get("adapter_commit") == ADAPTER["revision"], "HF source revisions differ")
    require(receipt["packages"].get("vllm") == "0.28.0+cu129" and receipt.get("arm_id") == fast.ARM, "Runtime version/arm mismatch")
    condition = receipt["serving_condition"]
    require(condition.get("compilation_config") == fast.GRAPH_CONFIG and condition.get("max_num_seqs") == 32 and condition.get("max_num_batched_tokens") == 8192,
            "Frozen graph/concurrency settings differ")
    command = receipt["server_command"]
    for key, expected in {"--dtype": "bfloat16", "--seed": "809", "--max-model-len": "32768", "--served-model-name": SERVED_MODEL}.items():
        require(key in command and command[command.index(key) + 1] == expected, f"Runtime argument mismatch: {key}")
    starts, responses = value["started_requests"], value["responses"]
    require(len(starts) == len(responses) == 34, "Expected exactly 34 intents and raw-backed responses")
    require({r["request_index"] for r in starts} == set(range(34)) and {r["request_index"] for r in responses} == set(range(34)), "Duplicate or missing canary indices")
    starts = {r["request_index"]: r for r in starts}
    responses = {r["request_index"]: r for r in responses}
    synthetic = fast.performance_smoke_requests(SERVED_MODEL)
    for index, response in responses.items():
        start = starts[index]
        require(start.get("status") == "GENERATION_STARTED" and response.get("status") == "HTTP_RESPONSE_RECORDED" and response.get("http_status") == 200,
                "Canary contains failed or unresolved generation")
        require(all(r.get("is_runtime_smoke") is True and r.get("score") is None and not r.get("task_id") for r in [start, response]), "A canary row is not synthetic")
        require(all(start.get(k) == response.get(k) for k in ["custom_id", "payload_sha256", "max_completion_tokens", "prompt_token_count"]), "Intent/result binding mismatch")
        raw = json.loads(base64.b64decode(response["raw_body_base64"], validate=True))
        require(raw == response["response"] and raw.get("model") == SERVED_MODEL, "HTTP bytes disagree with parsed response/model")
        require(len(raw.get("choices", [])) == 1 and raw["choices"][0].get("index") == 0, "Canary has unexpected response choices")
        if index < 2:
            expected = "ready" if index == 0 else "red"
            require(raw["choices"][0]["message"].get("content", "").strip().lower() == expected, "Basic text/image capability smoke did not pass")
        else:
            row = synthetic[index - 2]
            require(response["custom_id"] == row["custom_id"] and response["payload_sha256"] == fast.stable_hash(row["payload"]), "Synthetic throughput payload changed")
            require(response["max_completion_tokens"] == 1024 and raw["usage"]["completion_tokens"] == 1024, "Synthetic output cap was not fully exercised")
    performance = receipt["performance_smoke"]
    require(performance.get("count") == 32 and performance.get("completion_tokens") == 32768 and performance.get("max_tokens_each") == 1024 and performance.get("synthetic_only") is True and performance.get("score") is None,
            "Performance receipt denominator changed")
    elapsed = performance["elapsed_seconds"]
    require(elapsed > 0 and math.isclose(performance["aggregate_completion_tokens_per_second"], 32768 / elapsed, rel_tol=1e-12), "Throughput does not match measured token/time receipt")
    return receipt


def check_journals(directory, value):
    verification = read(directory / "verification.json")
    require(verification.get("status") == "REMOTE_VISIBLE_PREFIXES_VERIFIED" and verification.get("verifier_sha256") == RUNTIME_PINS["public_runtime_journal_group_commit.py"], "Missing frozen journal verification")
    data = {}
    for name in ["started_requests.jsonl", "http_responses.jsonl", "responses.jsonl", "journal_epochs.jsonl"]:
        data[name] = (directory / name).read_bytes()
        require(sha(data[name]) == verification["files"][name], f"Recovered journal hash mismatch: {name}")
    rows = lambda name: [json.loads(line) for line in data[name].split(b"\n") if line.strip()]
    require(rows("started_requests.jsonl") == value["started_requests"] and rows("responses.jsonl") == value["responses"], "RPC result differs from recovered intent/response journals")
    fences = rows("journal_epochs.jsonl")
    require(fences == value["journal_epoch_fences"], "RPC journal fences differ from recovered file")
    phase_files = {"STARTED": "started_requests.jsonl", "RAW_HTTP": "http_responses.jsonl", "PARSED": "responses.jsonl"}
    counts = {phase: 0 for phase in phase_files}
    for fence in fences:
        for phase, prefix in fence["prefixes"].items():
            require(sha(data[phase_files[phase]][:prefix["bytes"]]) == prefix["sha256"], "Recovered journal prefix mismatch")
        for record in fence["records"]:
            phase = record["phase"]
            begin, end = record["byte_range"]
            require(sha(data[phase_files[phase]][begin:end]) == record["record_sha256"], "Recovered journal record mismatch")
            counts[phase] += 1
    require(counts == {phase: 34 for phase in phase_files}, "Expected 34 durable records in each phase")
    return verification["files"]


def check_fresh_paths(run_dir, output):
    require(not run_dir.is_symlink() and not output.is_symlink(), "Symlink run/export paths are not accepted")
    require(not run_dir.exists() or (run_dir.is_dir() and not any(run_dir.iterdir())), "Existing Omni run state: reconcile it; do not export or resample again")
    require(not output.exists() and not output.with_suffix(output.suffix + ".receipt.json").exists(), "Batch or export receipt already exists")
    require(output.resolve().is_relative_to(run_dir.resolve()), "Prospective export must be inside its run directory")


def estimates(receipt):
    perf = receipt["performance_smoke"]
    tps = perf["completion_tokens"] / perf["elapsed_seconds"]
    non_performance = receipt["elapsed_seconds"] - perf["elapsed_seconds"]
    # These are frozen launcher/actor limits, not inferred GPU performance.
    actor_wall_ceiling, new_wave_guard = 3600 - 60 - 35, 2 * 180 + 35
    useful = actor_wall_ceiling - non_performance - new_wave_guard
    require(useful > 0, "No useful modeled session allowance")
    scenarios = []
    for average in [512, 1024, 2048]:
        seconds = 4428 * average / tps
        sessions = math.ceil(seconds / useful)
        scenarios.append({"assumed_mean_completion_tokens": average, "actor_output_tokens": 4428 * average,
                          "synthetic_rate_extrapolated_generation_seconds": seconds,
                          "conservative_session_count_under_this_scenario": sessions,
                          "nominal_sum_of_8_usd_session_holds_not_actual_cost": sessions * 8})
    return {"measurement_is_synthetic_not_native": True, "measured_aggregate_output_tokens_per_second": tps,
            "measured_non_performance_runtime_seconds": non_performance, "actor_wall_seconds_ceiling_before_container_bootstrap": actor_wall_ceiling,
            "required_time_before_starting_next_wave_seconds": new_wave_guard,
            "scenario_useful_generation_seconds_per_session": useful, "scenarios": scenarios,
            "native_input_token_count": None, "native_throughput": None, "actual_future_cost_usd": None,
            "limits": ["512 and 1024 are hypothetical output lengths; 2048 is the real per-request ceiling.",
                       "Synthetic 32-request throughput includes its preparation and persistence; native context, reasoning, prefill, output lengths, startup and image-free workload may differ.",
                       "The conservative session scenario subtracts all observed non-performance time and the full next-wave guard; it is not a measured capacity or completion guarantee.",
                       "Each fullbatch needs a separate 8 USD reservation; refunds may be reconciled between sessions. Holds are not invoices or actual spend.",
                       "Actor work and separately served native Omni-Judge work require separate validated terminal receipts and cost accounting."]}


def build_plan(args):
    native, fast, paths = source_modules()
    result_bytes = args.canary.read_bytes()
    result = json.loads(result_bytes)
    reconciliation = read(args.reconciliation)
    pointer = read(args.checkpoint)
    receipt = check_canary(result, reconciliation, sha(result_bytes), pointer, fast)
    journal_hashes = check_journals(args.journals, result)
    manifest, references = native.verified_prepared(args.setup, args.prepared)
    require(native.fingerprint(manifest) == MANIFEST_SHA256, "Prepared full split differs from frozen 4428-row manifest")
    check_fresh_paths(args.run_dir, args.output)
    extra = {"seed": 809, "chat_template_kwargs": {"enable_thinking": False}} if args.actor_condition == "no-thinking-seed809" else {}
    provenance = {"base_model": BASE, "adapter": ADAPTER, "model_loaded": True, "adapter_loaded": True,
                  "served_model": SERVED_MODEL, "deployment_receipt_sha256": sha(result_bytes),
                  "deployment_verified_at": receipt["finished_at"], "arm_id": receipt["arm_id"],
                  "evidence_role": "Prior successful canary; every future execution must independently match the checkpoint, shard and runtime identities."}
    wandb = read(args.wandb)
    contract = native.create_contract(manifest, "actor", provenance, wandb, extra)
    contract_hash = native.fingerprint(contract)
    batch_id = native.fingerprint({"contract": contract_hash, "output": str(args.output.resolve())})
    exported_hash = hashlib.sha256()
    payload_bindings = []
    for task in manifest["tasks"]:
        payload = native.actor_payload(task, contract)
        require(payload["max_tokens"] == 2048 and payload["temperature"] == 0 and payload["top_p"] == payload["n"] == 1 and payload["stream"] is False, "Native F.2 sampling changed")
        record = {"task_id": task["task_id"], "kind": "actor", "contract_sha256": contract_hash, "payload": payload, "dependency": None}
        row = {"task_id": task["task_id"], "kind": "actor", "batch_id": batch_id, "contract_sha256": contract_hash,
               "request_sha256": native.fingerprint(record), "api_path": "/v1/chat/completions", "payload": payload}
        exported_hash.update(native.canonical(row))
        payload_bindings.append([task["task_id"], row["request_sha256"], fast.stable_hash(payload)])
    problem_count = len({row["problem"] for row in references["rows"].values()})
    full_row_count = len({native.fingerprint(row) for row in references["rows"].values()})
    require(len(payload_bindings) == 4428 and problem_count == 4406 and full_row_count == 4424, "Duplicate-preserving denominator changed")
    return {"schema": "omni-actor-read-only-launch-preparation-v1", "status": "READY_FOR_ROOT_EXPORT_REVIEW",
            "score": None, "authorization_required": True, "actual_export_performed": False,
            "actual_model_calls": 0, "budget_reserved": False, "created_at": native.now(),
            "helper_sha256": sha(Path(__file__).read_bytes()), "native_source_sha256": NATIVE_PINS, "runtime_source_sha256": RUNTIME_PINS,
            "source_evidence": {str(p): sha(p.read_bytes()) for p in [args.canary, args.reconciliation, args.checkpoint, args.wandb,
                                args.prepared / "manifest.json", args.prepared / "reference_rows.json", args.journals / "verification.json"]},
            "canary_provider_terminal": reconciliation["app"], "canary_journal_sha256": journal_hashes,
            "full_scope": {"rows": 4428, "unique_problem_strings": problem_count, "unique_full_rows": full_row_count,
                           "duplicates_retained": True, "manifest_sha256": MANIFEST_SHA256, "maximum_actor_output_tokens": 4428 * 2048},
            "proposed_actor_condition": args.actor_condition, "prospective_provenance": provenance,
            "prospective_extra_body": extra, "prospective_contract": contract, "prospective_contract_sha256": contract_hash,
            "prospective_batch": {"run_dir": str(args.run_dir.resolve()), "output": str(args.output.resolve()), "count": 4428,
                                  "batch_id": batch_id, "predicted_export_sha256": exported_hash.hexdigest(),
                                  "ordered_task_request_payload_bindings_sha256": native.fingerprint(payload_bindings)},
            "export_arguments_after_explicit_authorization": [str(paths["public_omni_math_native.py"]), "--setup", str(args.setup.resolve()),
                 "export-requests", "--kind", "actor", "--prepared", str(args.prepared.resolve()), "--run-dir", str(args.run_dir.resolve()),
                 "--provenance", "<materialized-prospective-provenance.json>", "--wandb", str(args.wandb.resolve()),
                 "--extra-body", "<materialized-prospective-extra-body.json>", "--output", str(args.output.resolve())],
            "runtime_after_export": {"launcher": str(paths["modal_public_portfolio_group_runtime.py"]),
                 "environment": {"PUBLIC_GROUP_RUNTIME_PROFILE": "fullbatch"}, "required_unset_environment": ["PUBLIC_RUNTIME_INCLUDE_AGENTDOJO"],
                 "fullbatch_max_requests_including_two_smokes": 4430, "performance_smoke": False,
                 "per_request_seconds": 180, "context_tokens": 32768, "concurrency": 32,
                 "separate_root_reservation_required": True, "original_full_export_must_remain_immutable": True},
            "feasibility_scenarios": estimates(receipt),
            "remaining_requirements": ["Root authorizes the condition and materializes the prospective provenance/extra-body before the first full export; omit max-new so all 4428 rows are claimed together.",
                 "Verify exported bytes equal predicted_export_sha256; preserve every repeated source row and the original contract across all sessions.",
                 "Root reconciles the live budget and supplies each separate bounded reservation. This planner neither reads nor changes spend authorization.",
                 "Actual native prompts must pass runtime token counting within context32768 and per-call180s; this dry run does not tokenize or contact a model.",
                 "Every future terminal actor receipt must prove exact checkpoint/shards, source hashes, immutable input hash, raw response binding and provider release.",
                 "Recover all started/raw/parsed journals; resume only demonstrably unstarted original export rows. Never resample completed or ambiguous intents.",
                 "Run all separate pinned Omni-Judge prompts and the unchanged scorer. No full-suite score before all4428 actor/judge records validate."]}


def self_test(args):
    native, fast, unused = source_modules()
    original = read(args.canary)
    reconciliation, pointer = read(args.reconciliation), read(args.checkpoint)
    checked = 0
    mutations = [lambda x: x["receipt"].update(status="RUNTIME_BATCH_PARTIAL_CLEAN_STOP"),
                 lambda x: x["receipt"]["group_commit"].update(unacknowledged_tickets=1),
                 lambda x: x["rpc_receipt"]["source_sha256"].update({"public_colab_runtime_group.py": "0" * 64}),
                 lambda x: x["responses"].pop(),
                 lambda x: x["responses"][2].update(payload_sha256="0" * 64),
                 lambda x: x["receipt"]["merge"].update(adapter_commit="0" * 40),
                 lambda x: x["responses"][2].update(raw_body_base64=base64.b64encode(b"{}").decode())]
    for mutate in mutations:
        broken = copy.deepcopy(original)
        mutate(broken)
        try:
            check_canary(broken, reconciliation, CANARY_SHA256, pointer, fast)
        except (ValueError, KeyError):
            checked += 1
        else:
            raise AssertionError("Invalid canary accepted")
    broken = copy.deepcopy(reconciliation)
    broken["app"]["state"] = "running"
    try:
        check_canary(original, broken, CANARY_SHA256, pointer, fast)
    except ValueError:
        checked += 1
    else:
        raise AssertionError("Running provider accepted")
    plan = build_plan(args)
    require(plan["full_scope"]["maximum_actor_output_tokens"] == 9068544 and plan["prospective_batch"]["count"] == 4428, "Full-scope planning regression")
    return {"status": "OFFLINE_READ_ONLY_CHECKS_PASSED", "negative_checks": checked, "full_native_rows_bound": 4428,
            "score": None, "model_calls": 0, "exports_created": 0, "prospective_contract_sha256": plan["prospective_contract_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", type=Path, default=EVIDENCE / "omni_setup")
    parser.add_argument("--prepared", type=Path, default=EVIDENCE / "omni_setup/prepared")
    parser.add_argument("--run-dir", type=Path, default=EVIDENCE / "omni_run")
    parser.add_argument("--output", type=Path, default=EVIDENCE / "omni_run/actor-batch-001.jsonl", help="Prospective export path only; never written")
    parser.add_argument("--canary", type=Path, default=EVIDENCE / "group_canary01_result.json")
    parser.add_argument("--reconciliation", type=Path, default=EVIDENCE / "group_canary01_reconciliation.json")
    parser.add_argument("--journals", type=Path, default=EVIDENCE / "group_canary01_journals")
    parser.add_argument("--checkpoint", type=Path, default=EVIDENCE / "merged_checkpoint_pointer.json")
    parser.add_argument("--wandb", type=Path, default=EVIDENCE / "wandb_runtime_receipt.json")
    parser.add_argument("--actor-condition", choices=["no-thinking-seed809", "native-fields-only"], default="no-thinking-seed809",
                        help="Proposed condition; freeze its exact extra-body before first export")
    parser.add_argument("--self-test", action="store_true", help="Offline mutation checks and real read-only full-split binding")
    args = parser.parse_args()
    try:
        print(json.dumps(self_test(args) if args.self_test else build_plan(args), indent=2, sort_keys=True))
    except (ValueError, OSError, KeyError) as exc:
        print(json.dumps({"status": "PREPARATION_BLOCKED", "reason": str(exc), "score": None,
                          "actual_export_performed": False, "actual_model_calls": 0}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
