#!/usr/bin/env python3
"""Grade with the native CORE-Bench grader (run_corebench.py grade) and write result.json."""
import json, math, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
E2 = HERE.parent
sys.path.insert(0, str(HERE))
import run_corebench as rc  # noqa: E402

g = subprocess.run([sys.executable, str(HERE / "run_corebench.py"), "grade"], capture_output=True, text=True)
(E2 / "raw/grader_stdout.log").write_text(g.stdout + g.stderr)
print(g.stdout[-1200:], g.stderr[-600:])

native = json.load(open(E2 / "native_results_codeocean_hard.json"))["summary"]
per = json.load(open(E2 / "per_task.json"))
n, k = len(per), sum(p["correct"] for p in per)


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0, c - h), 4), round(min(1, c + h), 4)]


status = {}
for p in per:
    ms = sorted((E2 / "raw" / p["capsule_id"]).glob("meta_*.json"))
    last = json.load(open(ms[-1])) if ms else {}
    status[p["capsule_id"]] = {"status": last.get("status", "missing"), "end": (last.get("agent") or {}).get("end"),
                               "has_report": p["has_report"], "correct": p["correct"]}
ends = {}
for s in status.values():
    key = s["status"] if s["status"] != "completed" else f"completed/{s['end']}"
    ends[key] = ends.get(key, 0) + 1

res = {
    "benchmark": "CORE-Bench (codeocean_hard)",
    "scope": "replacement scope: all 45 CORE-Bench test capsules at the codeocean_hard difficulty, trained actor "
             "pavlov-public-portfolio-bf16 (seed809 merged), single attempt per task (one STANDARD rerun only on spot preemption)",
    "n_attempted": n,
    "n_graded": n,
    "n_correct": k,
    "metric": "task accuracy: all written+vision questions correct (native benchmark.evaluations.eval_result_json / score_results)",
    "score": round(k / n, 4),
    "ci95_wilson": wilson(k, n),
    "native_summary": native,
    "outcome_breakdown": ends,
    "spend_usd": {"gcp_vms_estimate_list_price": round(rc.cost_estimate(), 2),
                  "note": "estimate from vm_ledger*.jsonl create/delete timestamps x us-central1 list prices (+disk, +Ubuntu Pro allowance), "
                          "includes aborted attempts; actor inference is billed to the shared_trained_actor_endpoint cap, not E2"},
    "deviations": [
        "Per-task GCP VMs (e2-highmem-2 / n1-highmem-4+T4, Ubuntu Pro 20.04) instead of the native Azure shapes; spot first, one STANDARD rerun on preemption",
        "Agent is a thin OpenAI-tools loop (bash/query_image/finish) driving the trained actor, not the native AutoGPT/CORE-Agent scaffold; native 8100 s budget, 150-turn cap, 26k-token context window with oldest-turn truncation",
        "query_image answered by the same trained actor (no separate vision model)",
        "Harness infrastructure restarts: utf8-decoding bug, driver bug, a local DNS outage (30 tasks never reached a VM) and a driver-process death (10 in-flight tasks) were all infra aborts; those attempts are kept under raw/aborted_* and the affected tasks were rerun from scratch. No task was rerun because of its score.",
    ],
    "caveats": [
        "n=45; wide CI",
        "Errors/timeouts/no-report count as failures in the denominator",
        "Replacement-scope number; never pool with original-contract numbers",
        "The dataset (dataset/core_test.json) contains the answer keys and is git-ignored",
        "12 of the 27 correct tasks left no files under the capsule results/ dir (outputs written elsewhere or printed); native result_paths_success is recorded but, as in the native score_results, is not part of task accuracy",
        "Agent commands were scanned for corebench/princeton/huggingface/codeocean URLs (answer-leak routes): none found",
        "Vision questions were answered by the same text+vision actor; some are categorical and could be guessed",
    ],
    "per_task": status,
}
json.dump(res, open(E2 / "result.json", "w"), indent=1)
print(json.dumps({k_: res[k_] for k_ in ("n_attempted", "n_correct", "score", "ci95_wilson", "outcome_breakdown", "spend_usd")}, indent=1))
