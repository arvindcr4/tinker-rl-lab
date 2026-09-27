"""Build E1 result.json from score_breakdown.json (run score.py first). Headline = this lane only,
resolved / attempted. Prior 2026-09-12 waves reported as a separate, unpooled block."""
import glob
import json
import time
from pathlib import Path

E1 = Path(__file__).parent.parent
sb = json.load(open(E1 / "score_breakdown.json"))
lane = sb["this_lane_total"]

# context-overflow ids (prompt alone >= 32768 served context)
overflow = sorted({Path(p).parent.name for p in glob.glob(str(E1 / "remaining/*/attempts/*/context_overflow.json"))})
overflow_log = sorted({l.split()[1] for l in open(E1 / "logs/remaining.log")
                       if "maximum context length" in l})
overflow = sorted(set(overflow) | set(overflow_log))

# Modal sandbox spend estimate: 4 CPU + 16 GiB, $0.047/core-h + $0.008/GiB-h ~= $0.32/sandbox-hour
hours = 0.0
for sp in glob.glob(str(E1 / "remaining/*/*_status.json")):
    st = json.load(open(sp))
    hours += ((st.get("finished_at") or time.time()) - st["started_at"]) / 3600
hours += 1.0  # wave10 sandbox + probe/validation sandboxes (upper-bound allowance)
spend = round(hours * 0.32 * 1.25, 2)  # 25% margin for image-pull egress / rounding

result = {
    "lane": "E1",
    "benchmark": "SWE-bench Multilingual (300 tasks, test split)",
    "scope": "this lane: wave10 (16 sealed v6 recovery tasks) + the 174 tasks never attempted in the "
             "2026-09-12 waves 01-09 = 190 tasks",
    "actor": "pavlov-public-portfolio-bf16 (Qwen3.6-35B-A3B + seed809 LoRA merged) on shared Modal app "
             "pavlov-trained-actor-e1e14; temp 0.0, top_p 0.95, seed 809, non-thinking, max_tokens<=8192 "
             "capped to 32768 - prompt_tokens",
    "grader": "native swebench 5.0.2 CLI (`swebench eval`) in Modal dockerd sandboxes, pinned instance images",
    "metric": "resolved / attempted (errors, patch-apply failures, empty patches, provider errors and "
              "context overflows count as failures)",
    "n_attempted": lane["attempted"],
    "n_graded": lane["graded"],
    "n_resolved": lane["resolved"],
    "score": lane["resolved_over_attempted"],
    "ci95_wilson": lane["resolved_over_attempted_ci95"],
    "secondary_resolved_over_graded": {"score": lane["resolved_over_graded"],
                                       "ci95_wilson": lane["resolved_over_graded_ci95"]},
    "resolved_ids": lane["resolved_ids"],
    "breakdown": {"wave10": sb["this_lane_wave10"], "remaining174": sb["this_lane_remaining"]},
    "outcome_counts": {k: lane[k] for k in ("errors_patch_apply_or_eval", "empty_patch")},
    "not_attempted_of_300": sb["not_attempted"],
    "prior_waves_separate_not_pooled": {
        **sb["prior_waves01_09_2026_09_12"],
        "source": "outputs/PES_Phase2_Review_2026-09-12/finish/e1_completion",
        "pooling_verdict": "NOT POOLED. Verified same task universe (inventory sha256 765d75f9...), same "
                           "actor weights (base 995ad96e + adapter 64444133) and same native grader family, "
                           "but the contract is not identical: 2026-09-12 actor served max-model-len 65536 "
                           "vs 32768 here (different max_tokens caps and overflow behaviour), and this lane's "
                           "source-context collector is a re-implementation (original metadata_wave08.py "
                           "lost), validated byte-identical on only 3 recorded tasks. combined_300 in "
                           "score_breakdown.json is informational only and not a headline.",
    },
    "spend_usd": spend,
    "spend_basis": f"Modal CPU sandboxes ~{hours:.1f} sandbox-hours x $0.32/h x 1.25 margin (estimate); "
                   "actor GPU time is billed to shared_trained_actor_endpoint cap, not E1",
    "deviations": [
        "Actor served at max-model-len 32768 (2026-09-12: 65536); max_tokens capped per task to fit",
        "Source-context collector re-implemented (e1_runtime.collect_sources) from Pro flagship helpers",
        "Native eval in Modal dockerd sandboxes (same shape as 2026-09-12 runtime0X), thin runner "
        "run_remaining.py bypassing fail-closed paperwork gates per AUTHORIZATION.json",
        "Batches r06/r07: driver killed during native eval; eval completed in the orphaned sandboxes and "
        "reports/logs were harvested (code/harvest_orphans.py); exec stdout/returncode lost",
        "Driver relaunched twice (agent kills); completed generations reused, never resampled",
        "Context-overflow short-circuit added on 2nd relaunch (no retry when prompt alone fills context)",
    ],
    "caveats": [
        f"Context-overflow task ids (prompt >= 32768 tokens; scored as failures): {overflow}",
        "Non-generated (provider error) ids scored as failures: "
        + json.dumps(sb.get("this_lane_non_generated", {})),
        "High patch-apply/eval error rate: most failures are malformed or non-applying patches, not test failures",
        "Single sample per task at temperature 0; no pass@k",
        "Task order is manifest order, not random; wave10 is a sealed recovery selection (gson/terraform/immutable-js)",
    ],
    "wandb": json.load(open(E1 / "wandb_run.json"))["url"],
    "authorization": "outputs/finish_pending_2026-09-27/AUTHORIZATION.json (E1 cap $30)",
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
json.dump(result, open(E1 / "result.json", "w"), indent=2)
print(json.dumps({k: result[k] for k in ("n_attempted", "n_graded", "n_resolved", "score", "ci95_wilson",
                                          "spend_usd")}), overflow)
