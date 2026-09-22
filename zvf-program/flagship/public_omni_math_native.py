#!/usr/bin/env python3
"""Omni-MATH public split: immutable actor/judge batches and native Omni-Judge score.

This utility NEVER calls models. Root executes exported actor and separate judge
batches. The official judge tokenizer/prompt/report scorer are used unchanged.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import types

import public_portfolio_native as durable
from public_portfolio_native import ContractError, canonical, digest, fingerprint, now, read_json, require, write_once


SCHEMA = "omni-math-public-native-v1"
EXPECTED_TOTAL = 4428
DATASET_REVISION = "40ba231d8f16e29ecd40e6407e2c8640145a8f62"
CODE_REVISION = "23be225c8e268df51990f6c5c1448f34d3b56911"
JUDGE_REPO = "KbsdJames/Omni-Judge"
JUDGE_REVISION = "de5bdca15ff3c366b90718c4b4be555d25c655b0"
SYSTEM_PROMPT = "You are an experienced educator in the field of MATHEMATICS."
DEFAULT_SETUP = Path(__file__).resolve().parents[2] / "outputs/public_portfolio_2026-09-05/omni_setup"
FILES = {
    "test.jsonl": "7c87be8ee41ac7c7a597ef5a5500e84bd2b639a85a06db3da7f69bf9a32ef168",
    "dataset_README.md": "2688bd0d51f329de3647c540dc1df4fa699681d029068ae575bb2eb1eb8515f0",
    "tokenizer_config.json": "e49fba1c797c4ee525dfcd657704f5173379b45c41c5830ef2fff83be6749198",
    "tokenizer.json": "79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4",
    "tokenization_omnijudge.py": "3c05f5ec35482b1b61e75de488adea9ac474904994a3ca8d7a9dfdf797a92853",
    "special_tokens_map.json": "1b1835caa5b4d70acaa210fa222b0036f1882f9525c4660fd4810fb3e1e40ff8",
    "native_source/README.md": "fefae872e14c9545a10afd1a60a60e069878258d3ded4683547f485e128ee4b5",
    "native_source/Omni-Judge_eval/omni_judge_vllm.py": "8f832467443444e3c8e9f50546d7b15ebe0ceef18b0ed6122417462b1b64930b",
    "native_source/Omni-Judge_eval/get_result.py": "dd7928ac30dd3aabef26f4d8be9eac52462224817af70f27cd44f174e1fd3a90",
}


def sources(setup):
    for name, expected in FILES.items():
        require(digest((setup / name).read_bytes()) == expected, f"Omni source drift: {name}")
    return {"dataset_repo": "KbsdJames/Omni-MATH", "dataset_revision": DATASET_REVISION,
            "dataset_url": f"https://huggingface.co/datasets/KbsdJames/Omni-MATH/tree/{DATASET_REVISION}",
            "code_repo": "KbsdJames/Omni-MATH", "code_revision": CODE_REVISION,
            "code_url": f"https://github.com/KbsdJames/Omni-MATH/tree/{CODE_REVISION}",
            "judge_repo": JUDGE_REPO, "judge_revision": JUDGE_REVISION,
            "judge_url": f"https://huggingface.co/{JUDGE_REPO}/tree/{JUDGE_REVISION}",
            "files_sha256": FILES, "dataset_license": "Apache-2.0 per pinned dataset card",
            "judge_license": "Apache-2.0 declared in HF model card",
            "github_scorer_license": "not separately declared in inspected root; do not assert Apache-2.0",
            "actor_prompt_source": "https://arxiv.org/html/2410.07985v1#A6.SS2"}


def dataset_rows(setup):
    require(digest((setup / "test.jsonl").read_bytes()) == FILES["test.jsonl"], "dataset hash drift")
    rows = [json.loads(line) for line in (setup / "test.jsonl").read_text().splitlines() if line.strip()]
    require(len(rows) == EXPECTED_TOTAL, "full Omni-MATH test split must have 4428 rows")
    for row in rows:
        require(set(row) == {"domain", "difficulty", "source", "problem", "solution", "answer"}, "unexpected native row schema")
        require(isinstance(row["problem"], str) and row["problem"] and isinstance(row["answer"], str), "invalid problem/reference answer")
    return rows


def task_id(index, row):
    # The official split has repeated problems/rows; preserve their original indices.
    return f"omni-{index:05d}-{fingerprint(row)[:16]}"


def prepare(setup, output):
    source = sources(setup)
    rows = dataset_rows(setup)
    tasks, gold = [], {}
    for index, row in enumerate(rows):
        ident = task_id(index, row)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": row["problem"]}]
        tasks.append({"task_id": ident, "row_index": index, "source_row_sha256": fingerprint(row),
                      "messages": messages, "prompt_sha256": fingerprint(messages)})
        gold[ident] = row
    manifest = {"schema": SCHEMA, "suite_id": "omni_math_eval", "expected_total": EXPECTED_TOTAL,
                "source": source, "split": "entire pinned HF test.jsonl in original row order; duplicates retained",
                "protocol": {"actor_system_prompt": SYSTEM_PROMPT, "actor_prompt_source": source["actor_prompt_source"],
                             "actor_temperature": 0, "actor_top_p": 1, "actor_max_tokens": 2048,
                             "judge": "native Omni-Judge, separate model; not GPT-4o leaderboard judge",
                             "judge_temperature": 0, "judge_max_tokens": 300, "n": 1,
                             "malformed_judge_policy": "block full score if native parser would skip any row"},
                "tasks": tasks}
    validate_manifest(manifest)
    keys = {"manifest_sha256": fingerprint(manifest), "rows": gold}
    write_once(output / "manifest.json", manifest)
    write_once(output / "reference_rows.json", keys)
    write_once(output / "source_receipt.json", source)
    return {"status": "PREPARED_NOT_EVALUATED", "expected_total": EXPECTED_TOTAL, "score": None,
            "manifest_sha256": fingerprint(manifest), "unique_problems": len({r['problem'] for r in rows}),
            "unique_full_rows": len({fingerprint(r) for r in rows})}


def validate_manifest(manifest):
    require(manifest.get("schema") == SCHEMA and manifest.get("expected_total") == EXPECTED_TOTAL, "Omni denominator must be 4428")
    tasks = manifest["tasks"]
    require(len(tasks) == EXPECTED_TOTAL and len({t["task_id"] for t in tasks}) == EXPECTED_TOTAL, "incomplete or duplicate task IDs")
    for index, task in enumerate(tasks):
        require(task["row_index"] == index and re.fullmatch(rf"omni-{index:05d}-[a-f0-9]{{16}}", task["task_id"]), "row order or task ID drift")
        require(set(task) == {"task_id", "row_index", "source_row_sha256", "messages", "prompt_sha256"}, "gold or unexpected fields in actor task")
        require(fingerprint(task["messages"]) == task["prompt_sha256"], "actor prompt drift")
        require(task["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}, "native actor system prompt drift")


def verified_prepared(setup, prepared):
    manifest = read_json(prepared / "manifest.json")
    keys = read_json(prepared / "reference_rows.json")
    validate_manifest(manifest)
    require(manifest["source"] == sources(setup), "manifest/source mismatch")
    require(keys["manifest_sha256"] == fingerprint(manifest), "reference manifest mismatch")
    rows = dataset_rows(setup)
    require(set(keys["rows"]) == {t["task_id"] for t in manifest["tasks"]}, "reference rows incomplete")
    for index, (task, row) in enumerate(zip(manifest["tasks"], rows)):
        require(task["task_id"] == task_id(index, row) and keys["rows"][task["task_id"]] == row, "reference row drift")
        require(task["messages"] == [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": row["problem"]}], "answer leakage or problem prompt drift")
    return manifest, keys


def load_tokenizer(setup):
    require(importlib.metadata.version("transformers") == "4.44.2", "use pinned Omni tokenizer environment transformers==4.44.2")
    for name in ("tokenization_omnijudge.py", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        require(digest((setup / name).read_bytes()) == FILES[name], f"native tokenizer drift: {name}")
    spec = importlib.util.spec_from_file_location("portfolio_native_omni_tokenizer", setup / "tokenization_omnijudge.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OmniJudgeTokenizer.from_pretrained(setup, local_files_only=True)


def load_scorer(setup):
    path = setup / "native_source/Omni-Judge_eval/get_result.py"
    require(digest(path.read_bytes()) == FILES["native_source/Omni-Judge_eval/get_result.py"], "native scorer drift")
    spec = importlib.util.spec_from_file_location("portfolio_native_omni_score", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_provenance(provenance, kind):
    require(provenance.get("served_model"), "served model required")
    require(provenance.get("model_loaded") is True, "actual loaded model receipt required")
    require(re.fullmatch(r"[a-f0-9]{64}", provenance.get("deployment_receipt_sha256", "")), "deployment receipt SHA256 required")
    require(provenance.get("deployment_verified_at"), "deployment verification timestamp required")
    if kind == "actor":
        require(provenance.get("adapter_loaded") is True, "base-only actor is forbidden")
        for key in ("base_model", "adapter"):
            model = provenance.get(key, {})
            require(isinstance(model.get("repo_id"), str) and "/" in model["repo_id"], f"HF {key} repo required")
            require(re.fullmatch(r"[a-f0-9]{40}", model.get("revision", "")), f"pinned HF {key} revision required")
    else:
        require(provenance.get("model") == {"repo_id": JUDGE_REPO, "revision": JUDGE_REVISION}, "only pinned native Omni-Judge may judge")


def create_contract(manifest, kind, provenance, wandb, extra_body=None):
    validate_manifest(manifest)
    require(kind in {"actor", "judge"}, "invalid stage")
    validate_provenance(provenance, kind)
    durable.validate_wandb(wandb)
    extra_body = extra_body or {}
    require(isinstance(extra_body, dict) and not set(extra_body) & {"model", "messages", "prompt", "n", "temperature", "top_p", "max_tokens", "stop", "stream", "echo"}, "extra body may not override native parameters")
    require(kind == "actor" or not extra_body, "native judge accepts no chat-template or sampling overrides")
    sampling = {"temperature": 0, "top_p": 1, "max_tokens": 2048 if kind == "actor" else 300, "n": 1}
    return {"schema": SCHEMA, "kind": kind, "manifest_sha256": fingerprint(manifest), "provenance": provenance,
            "wandb": wandb, "sampling": sampling, "extra_body": extra_body,
            "runner_sha256": digest(Path(__file__).read_bytes()), "durability_helper_sha256": digest(Path(durable.__file__).read_bytes()),
            "transport": "OpenAI chat completions" if kind == "actor" else "raw OpenAI completions; no second chat template"}


def actor_payload(task, contract):
    return {"model": contract["provenance"]["served_model"], "messages": task["messages"],
            **contract["sampling"], "stream": False, **contract["extra_body"]}


def judge_payload(row, actor_text, contract, tokenizer):
    # This calls the exact, hashed custom HF tokenizer.get_context implementation.
    prompt = tokenizer.get_context(row["problem"], row["answer"], actor_text)
    return {"model": contract["provenance"]["served_model"], "prompt": prompt, **contract["sampling"],
            "stop": [tokenizer.eos_token, "<|eot_id|>"], "stream": False, "echo": False}


def checked_response(run_dir, kind, task_id, contract):
    directory = run_dir / kind / "tasks" / task_id
    request, response = read_json(directory / "request.json"), read_json(directory / "response.json")
    require(request["contract_sha256"] == fingerprint(contract) and response["contract_sha256"] == fingerprint(contract), "response contract drift")
    require(response["request_sha256"] == fingerprint(request), "response/request mismatch")
    raw = base64.b64decode(response["raw_body_base64"], validate=True)
    require(digest(raw) == response["raw_body_sha256"], "raw response drift")
    require(response["http_status"] == 200, "non-successful model response")
    try:
        envelope = json.loads(raw)
        require(envelope["model"] == contract["provenance"]["served_model"], "served model mismatch")
        require(len(envelope["choices"]) == 1, "exactly one sample required")
        choice = envelope["choices"][0]
        require(choice.get("index", 0) == 0, "unexpected sample index")
        text = choice["message"]["content"] if kind == "actor" else choice["text"]
        require(isinstance(text, str), "non-text model sample")
    except (ValueError, KeyError, TypeError) as exc:
        raise ContractError("malformed model envelope; do not resample") from exc
    return text, request, response, envelope


def export_requests(setup, prepared, run_dir, kind, provenance, wandb, output, max_new=None, extra_body=None):
    manifest, keys = verified_prepared(setup, prepared)
    contract = create_contract(manifest, kind, provenance, wandb, extra_body)
    write_once(run_dir / kind / "contract.json", contract)
    require(not output.exists(), "batch already exported; execute/recover the original batch once")
    batch_id = fingerprint({"contract": fingerprint(contract), "output": str(output.resolve())})
    write_once(run_dir / kind / "batches" / f"{batch_id}.json", {"batch_id": batch_id, "output": str(output.resolve()),
               "created_at": now(), "state": "RESERVED_ONCE", "contract_sha256": fingerprint(contract)})
    tokenizer = load_tokenizer(setup) if kind == "judge" else None
    actor_contract = read_json(run_dir / "actor/contract.json") if kind == "judge" else None
    if kind == "judge":
        require(provenance["served_model"] != actor_contract["provenance"]["served_model"], "actor cannot masquerade as native judge")
    requests = []
    for task in manifest["tasks"]:
        if max_new is not None and len(requests) >= max_new:
            break
        directory = run_dir / kind / "tasks" / task["task_id"]
        if (directory / "request.json").exists() or (directory / "attempt.claim").exists():
            continue
        dependency = None
        if kind == "judge":
            if not (run_dir / "actor/tasks" / task["task_id"] / "response.json").exists():
                continue
            actor_text, actor_request, actor_response, _ = checked_response(run_dir, "actor", task["task_id"], actor_contract)
            require(actor_request["payload"] == actor_payload(task, actor_contract), "actor request/prompt drift")
            payload = judge_payload(keys["rows"][task["task_id"]], actor_text, contract, tokenizer)
            dependency = {"actor_contract_sha256": fingerprint(actor_contract), "actor_response_sha256": fingerprint(actor_response),
                          "reference_row_sha256": task["source_row_sha256"], "native_prompt_sha256": digest(payload["prompt"].encode())}
        else:
            payload = actor_payload(task, contract)
        record = {"task_id": task["task_id"], "kind": kind, "contract_sha256": fingerprint(contract),
                  "payload": payload, "dependency": dependency}
        directory.mkdir(parents=True, exist_ok=True)
        try:
            with (directory / "attempt.claim").open("x") as claim:
                claim.write(batch_id)
                claim.flush()
                os.fsync(claim.fileno())
        except FileExistsError as exc:
            raise ContractError(f"concurrent/uncertain task reservation: {task['task_id']}") from exc
        write_once(directory / "request.json", record)
        write_once(directory / "batch_assignment.json", {"batch_id": batch_id})
        requests.append({"task_id": task["task_id"], "kind": kind, "batch_id": batch_id,
                         "contract_sha256": fingerprint(contract), "request_sha256": fingerprint(record),
                         "api_path": "/v1/chat/completions" if kind == "actor" else "/v1/completions", "payload": payload})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as f:
        for row in requests:
            f.write(canonical(row))
        f.flush()
        os.fsync(f.fileno())
    receipt = {"kind": kind, "count": len(requests), "batch_id": batch_id, "requests_sha256": digest(output.read_bytes()),
               "status": "EXPORTED_NOT_EVALUATED", "score": None}
    write_once(output.with_suffix(output.suffix + ".receipt.json"), receipt)
    return receipt


def ingest_responses(setup, prepared, run_dir, kind, input_path):
    manifest, _ = verified_prepared(setup, prepared)
    expected = {t["task_id"] for t in manifest["tasks"]}
    contract = read_json(run_dir / kind / "contract.json")
    require(contract == create_contract(manifest, kind, contract["provenance"], contract["wandb"], contract["extra_body"]), "stage contract drift")
    seen = set()
    for line in input_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        require(row["task_id"] in expected and row["task_id"] not in seen, "unknown/duplicate response task")
        require(row["kind"] == kind, "actor/judge stage confusion")
        seen.add(row["task_id"])
        directory = run_dir / kind / "tasks" / row["task_id"]
        record = read_json(directory / "request.json")
        require(row["contract_sha256"] == fingerprint(contract) and row["request_sha256"] == fingerprint(record), "response/prompt/model mismatch")
        require(row["batch_id"] == read_json(directory / "batch_assignment.json")["batch_id"], "batch assignment mismatch")
        require(row.get("started_at") and row.get("received_at"), "actual model call timestamps required")
        raw = base64.b64decode(row["raw_body_base64"], validate=True)
        response = {k: row[k] for k in ("task_id", "kind", "batch_id", "contract_sha256", "request_sha256", "http_status", "raw_body_base64", "started_at", "received_at")}
        response["raw_body_sha256"] = digest(raw)
        write_once(directory / "response.json", response)
    return {"status": "INGESTED_NOT_SCORED", "kind": kind, "count": len(seen), "score": None}


def validate_native_reports(rows, scorer):
    """Native parser skips malformed reports; block instead of shrinking 4428."""
    require(len(rows) == EXPECTED_TOTAL, "full native scoring requires all 4428 reports")
    for row in rows:
        parsed = scorer.parse_report(row["omni_judge"])
        require(parsed and "Equivalence Judgement" in parsed, "native parser would skip a malformed judge report")


def native_score(rows, scorer, artifact_dir):
    validate_native_reports(rows, scorer)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    outfile = artifact_dir / "native_judged_rows.jsonl"
    data = b"".join(canonical(row) for row in rows)
    if outfile.exists():
        require(outfile.read_bytes() == data, "native judged-row collision")
    else:
        with outfile.open("xb") as f:
            f.write(data)
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        scorer.main(types.SimpleNamespace(in_file=str(outfile), out_file=None))
    match = re.fullmatch(r"Total Accuracy:([0-9.]+)\s*", stdout.getvalue())
    require(match is not None, "unexpected native scorer output")
    return float(match.group(1)), stdout.getvalue(), digest(data)


def score(setup, prepared, run_dir, output):
    manifest, keys = verified_prepared(setup, prepared)
    contracts = {}
    for kind in ("actor", "judge"):
        path = run_dir / kind / "contract.json"
        if not path.exists():
            receipt = {"schema": SCHEMA, "status": "INCOMPLETE_PUBLIC_SPLIT", "expected_total": EXPECTED_TOTAL,
                       "score": None, "error": f"{kind} stage has not started", "manifest_sha256": fingerprint(manifest)}
            write_once(output, receipt)
            return receipt
        contract = read_json(path)
        require(contract == create_contract(manifest, kind, contract["provenance"], contract["wandb"], contract["extra_body"]), "source/model/sampling contract drift")
        contracts[kind] = contract
    scorer, tokenizer = load_scorer(setup), load_tokenizer(setup)
    rows, blocked, actor_count = [], [], 0
    for task in manifest["tasks"]:
        ident = task["task_id"]
        if not (run_dir / "actor/tasks" / ident / "response.json").exists():
            continue
        try:
            actor_text, actor_request, actor_response, actor_envelope = checked_response(run_dir, "actor", ident, contracts["actor"])
            require(actor_request["payload"] == actor_payload(task, contracts["actor"]), "actor prompt/source mismatch")
            actor_count += 1
            if not (run_dir / "judge/tasks" / ident / "response.json").exists():
                continue
            judge_text, judge_request, judge_response, judge_envelope = checked_response(run_dir, "judge", ident, contracts["judge"])
            require(judge_request["dependency"]["actor_response_sha256"] == fingerprint(actor_response), "judge used another actor sample")
            native_payload = judge_payload(keys["rows"][ident], actor_text, contracts["judge"], tokenizer)
            require(judge_request["payload"] == native_payload, "native judge prompt/reference/actor mismatch")
            report = "## Student Final Answer\n" + judge_text.strip()  # exact native vLLM postprocessing
            parsed = scorer.parse_report(report)
            require(parsed and "Equivalence Judgement" in parsed, "native scorer would skip malformed report; full score blocked")
            source = keys["rows"][ident]
            row = {k: source[k] for k in ("domain", "difficulty", "source", "problem", "answer")}
            row.update({"model_generation": actor_text, "omni_judge": report})
            grade = {"task_id": ident, "row_index": task["row_index"], "native_parsed_report": parsed,
                     "correct": parsed["Equivalence Judgement"] == "TRUE", "actor_response_sha256": fingerprint(actor_response),
                     "judge_response_sha256": fingerprint(judge_response), "actor_usage": actor_envelope.get("usage"),
                     "judge_usage": judge_envelope.get("usage"), "native_row_sha256": fingerprint(row)}
            write_once(run_dir / "judge/tasks" / ident / "native_grade.json", grade)
            rows.append(row)
        except (ContractError, OSError, ValueError) as exc:
            blocked.append({"task_id": ident, "reason": str(exc)})
    complete = len(rows) == EXPECTED_TOTAL and not blocked
    result, stdout, rows_hash = (None, None, None)
    if complete:
        result, stdout, rows_hash = native_score(rows, scorer, run_dir / "native_score")
    receipt = {"schema": SCHEMA, "suite_id": "omni_math_eval", "experiment": "E14 replacement",
               "status": "COMPLETE_PUBLIC_SPLIT" if complete else "INCOMPLETE_PUBLIC_SPLIT", "expected_total": EXPECTED_TOTAL,
               "actor_samples": actor_count, "native_judged_samples": len(rows), "score": result,
               "metric": "native Omni-Judge Total Accuracy; all 4428 rows",
               "blocked_tasks": blocked, "manifest_sha256": fingerprint(manifest),
               "actor_contract_sha256": fingerprint(contracts["actor"]), "judge_contract_sha256": fingerprint(contracts["judge"]),
               "native_scorer_stdout": stdout, "native_judged_rows_sha256": rows_hash,
               "native_scorer_sha256": FILES["native_source/Omni-Judge_eval/get_result.py"],
               "actor_provenance": contracts["actor"]["provenance"], "judge_provenance": contracts["judge"]["provenance"],
               "claim_boundary": "Omni-MATH public replacement with Omni-Judge; not GPT-4o leaderboard score, FrontierMath, or original E14."}
    write_once(output, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", type=Path, default=DEFAULT_SETUP)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--output", type=Path, required=True)
    export = sub.add_parser("export-requests")
    for name in ("prepared", "run-dir", "provenance", "wandb", "output"):
        export.add_argument(f"--{name}", type=Path, required=True)
    export.add_argument("--kind", choices=["actor", "judge"], required=True)
    export.add_argument("--max-new", type=int)
    export.add_argument("--extra-body", type=Path)
    ingest = sub.add_parser("ingest-responses")
    for name in ("prepared", "run-dir", "input"):
        ingest.add_argument(f"--{name}", type=Path, required=True)
    ingest.add_argument("--kind", choices=["actor", "judge"], required=True)
    score_parser = sub.add_parser("score")
    for name in ("prepared", "run-dir", "output"):
        score_parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare(args.setup, args.output)
        elif args.command == "export-requests":
            require(args.max_new is None or args.max_new >= 0, "max-new cannot be negative")
            result = export_requests(args.setup, args.prepared, args.run_dir, args.kind, read_json(args.provenance), read_json(args.wandb),
                                     args.output, args.max_new, read_json(args.extra_body) if args.extra_body else None)
        elif args.command == "ingest-responses":
            result = ingest_responses(args.setup, args.prepared, args.run_dir, args.kind, args.input)
        else:
            result = score(args.setup, args.prepared, args.run_dir, args.output)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ContractError, OSError) as exc:
        print(json.dumps({"status": "BLOCKED", "score": None, "error": str(exc)}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
