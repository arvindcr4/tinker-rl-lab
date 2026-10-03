#!/usr/bin/env python3
"""Check selected current thesis arithmetic offline, not provenance or native grading.

Inputs are the M1b summary/raw correctness arrays, six replacement-lane result
receipts, and fourteen small-scale paired receipts. No model or provider runs.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Any, Callable

M1B = "platform_hybrid/experiments/results/samestack_gsm8k_cot.json"
M1B_FULL = "platform_hybrid/experiments/results/samestack_gsm8k_cot_full.json"
FINISH = "outputs/finish_pending_2026-09-27"
PAIRED = "outputs/e1_e14_small_scale_2026-09-26"
REPLACEMENT_COUNTS = {"E1": 190, "E2": 45, "E5": 97, "E6": 812, "E9": 34, "E13": 255}
PAIRED_COUNTS = {
    "E1": 10,
    "E2": 30,
    "E3": 8,
    "E4": 6,
    "E5": 10,
    "E6": 36,
    "E7": 6,
    "E8": 80,
    "E9": 5,
    "E10": 60,
    "E11": 50,
    "E12": 151,
    "E13": 26,
    "E14": 100,
}
# Required relative paths are also the submission packager's input allowlist.
EVIDENCE_FILES = (
    M1B,
    M1B_FULL,
    *(f"{FINISH}/{lane}/result.json" for lane in REPLACEMENT_COUNTS),
    *(f"{PAIRED}/{lane}/paired.json" for lane in PAIRED_COUNTS),
)
SCOPE = "Selected arithmetic consistency only; not source provenance or native re-evaluation."
NOT_CHECKED = [
    "Source authenticity, model identity, task selection, training execution, or native grades.",
    "Confidence intervals, bootstrap, t-test/TOST p-values, M16/M18, or manuscript agreement.",
    "Original-contract campaign scores and other historical evidence.",
    "Replacement E1/E5/E6/E13 raw outcomes: only receipt subdivisions are checked.",
    "Paired secondary metrics except E10 scores and E13 full-success McNemar.",
    "Inferential validity: E12 item-level McNemar is clustered and not an app-level test.",
    "M1b PPO step-log ZVF is preserved as NaN (undefined at G=1); it is not a checked score.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"expected number, got {value!r}",
    )
    require(math.isfinite(value), f"non-finite number: {value!r}")
    return value


def count(value):
    require(type(value) is int and value >= 0, f"expected nonnegative integer, got {value!r}")
    return value


def same(actual, expected, label, digits=None):
    """Compare exact arithmetic, or the documented decimal output precision."""
    actual = number(actual)
    expected = number(expected)
    tolerance = 1e-10 if digits is None else 0.5 * 10**-digits + 1e-12
    require(
        abs(actual - expected) <= tolerance,
        f"{label}: recorded {actual!r}, recomputed {expected!r}",
    )


def values(rows, key, binary=False):
    out = []
    for row in rows:
        value = row[key]
        if binary and isinstance(value, bool):
            value = int(value)
        value = number(value)
        require(
            value in (0, 1) if binary else 0 <= value <= 1,
            f"{key}: outcome outside {'{0,1}' if binary else '[0,1]'}: {value!r}",
        )
        out.append(value)
    require(bool(out), "empty outcome collection")
    return out


def unique(items, label):
    require(len(items) == len(set(items)), f"{label}: duplicate identifiers")


def mcnemar(trained, base):
    b = sum(t == 1 and a == 0 for t, a in zip(trained, base))
    c = sum(t == 0 and a == 1 for t, a in zip(trained, base))
    n = b + c
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n)
    return b, c, p


def paired_metric(record, trained, base, difference_key, binary=True, staged_rounding=False):
    trained_mean, base_mean = statistics.mean(trained), statistics.mean(base)
    same(record["trained_value"], trained_mean, "trained mean", digits=4)
    same(record["base_value"], base_mean, "base mean", digits=4)
    raw_delta = statistics.mean(t - b for t, b in zip(trained, base))
    if staged_rounding:
        # E3/code/paired_finalize.py also builds E7 and E12: each arm's k/n
        # is rounded to four places before subtracting and rounding again.
        staged_delta = round(round(trained_mean, 4) - round(base_mean, 4), 4)
        same(record[difference_key], staged_delta, "staged-rounding paired difference")
    else:
        same(record[difference_key], raw_delta, "paired difference", digits=4)
    if binary:
        b, c, p = mcnemar(trained, base)
        prefix = "discordant_" if "discordant_b_trained_only" in record else ""
        same(count(record[prefix + "b_trained_only"]), b, "trained-only discordances")
        same(count(record[prefix + "c_base_only"]), c, "base-only discordances")
        same(record["mcnemar_exact_p"], p, "McNemar exact p", digits=4)


def check_m1b(data):
    short, full = data[M1B], data[M1B_FULL]
    for section in ("config", "summary", "contrasts"):
        require(short[section] == full[section], f"M1b summary/full {section} mismatch")
    seeds = [42, 123, 456, 789, 1024]
    arms = {"grpo_g8": 8, "grpo_g2": 2, "ppo": 1}
    require(short["config"]["seeds"] == seeds, "M1b seed set changed")
    require(short["config"]["group"] == arms, "M1b arm set changed")
    for cfg_key, expected in (("n_eval", 200), ("n_steps", 30), ("n_gen", 64)):
        same(count(short["config"][cfg_key]), expected, f"M1b config {cfg_key}")
    require(len(short["runs"]) == len(full["runs"]) == 15, "M1b requires 15 runs")
    runs: dict[tuple[str, int], dict[str, Any]] = {}
    for small, run in zip(short["runs"], full["runs"]):
        projected = {
            k: v for k, v in run.items() if k not in ("pre_correct", "post_correct", "step_log")
        }
        require(small == projected, "M1b per-seed summary/full mismatch")
        key = (run["arm"], run["seed"])
        require(key not in runs, f"M1b duplicate run {key}")
        runs[key] = run
        require(key[0] in arms and key[1] in seeds, f"unexpected M1b run {key}")
        same(run["group"], arms[key[0]], f"{key} group")
        same(run["prompts_per_step"] * run["group"], 64, f"{key} rollout budget")
        for field in ("n_eval", "n_steps", "n_gen", "model", "k_epochs", "lr", "max_new"):
            require(run[field] == short["config"][field], f"{key} config mismatch: {field}")
        pre = values([{"x": x} for x in run["pre_correct"]], "x", binary=True)
        post = values([{"x": x} for x in run["post_correct"]], "x", binary=True)
        require(len(pre) == len(post) == 200, f"{key} requires 200 evaluation items")
        same(run["heldout_pre_acc"], statistics.mean(pre), f"{key} pre accuracy")
        same(run["heldout_post_acc"], statistics.mean(post), f"{key} post accuracy")
        b, c, _ = mcnemar(post, pre)
        same(count(run["wrong_to_right"]), b, f"{key} wrong-to-right")
        same(count(run["right_to_wrong"]), c, f"{key} right-to-wrong")
        require(
            [r["step"] for r in run["step_log"]] == list(range(30)),
            f"{key} training steps missing/duplicated",
        )
        rewards = values(run["step_log"], "mean_reward")
        same(run["last10_avg"], statistics.mean(rewards[-10:]), f"{key} last10 reward")
    require(set(short["summary"]) == set(arms), "M1b summary arms changed")
    for arm in arms:
        rows = [runs[arm, seed] for seed in seeds]
        summary = short["summary"][arm]
        same(count(summary["n_seeds"]), 5, f"{arm} seed count")
        for metric, field in (
            ("heldout_pre_mean", "heldout_pre_acc"),
            ("heldout_post_mean", "heldout_post_acc"),
            ("last10_mean", "last10_avg"),
        ):
            same(summary[metric], statistics.mean(r[field] for r in rows), f"{arm} {metric}")
        same(
            summary["delta_mean"],
            statistics.mean(r["heldout_post_acc"] - r["heldout_pre_acc"] for r in rows),
            f"{arm} delta",
        )
    require(
        set(short["contrasts"]) == {"grpo_g8_minus_ppo", "grpo_g8_minus_grpo_g2"},
        "M1b contrast set changed",
    )
    for arm in ("ppo", "grpo_g2"):
        contrast = short["contrasts"][f"grpo_g8_minus_{arm}"]
        require(contrast["seeds"] == seeds, f"M1b contrast seed order: {arm}")
        same(count(contrast["n_seeds"]), 5, f"{arm} contrast count")
        diffs = [
            runs["grpo_g8", s]["heldout_post_acc"] - runs[arm, s]["heldout_post_acc"] for s in seeds
        ]
        require(len(contrast["diffs"]) == 5, f"{arm} missing contrast differences")
        for recorded, computed in zip(contrast["diffs"], diffs):
            same(recorded, computed, f"{arm} paired seed difference")
        same(contrast["mean_diff"], statistics.mean(diffs), f"{arm} mean difference")
        same(contrast["sd"], statistics.stdev(diffs), f"{arm} SD")
    return "15 runs: raw pre/post correctness, transitions, last-10 means, arm means and paired deltas/SD"


def check_replacement(lane, d):
    n = count(d["n_attempted"])
    same(n, REPLACEMENT_COUNTS[lane], f"{lane} attempted scope")
    graded = count(d["n_graded"])
    require(graded <= n, f"{lane} graded exceeds attempted")
    if lane == "E13":
        envs = d["per_env"]
        require(
            set(envs) == {"babyai", "babaisai", "textworld", "crafter", "nle", "minihack"},
            "E13 environment set changed",
        )
        for key in ("n_planned", "n_graded", "n_missing_scored_0"):
            same(count(d[key]), sum(count(e[key]) for e in envs.values()), f"E13 {key}")
        same(d["n_planned"], n, "E13 planned/attempted")
        same(graded + count(d["n_missing_scored_0"]), n, "E13 graded+missing")
        same(
            count(d["n_attempt_records"]),
            n + count(d["n_killed_for_restart_records"]),
            "E13 restart accounting",
        )
        for env, record in envs.items():
            same(
                record["n_planned"],
                count(record["n_graded"]) + count(record["n_missing_scored_0"]),
                f"E13 {env} coverage",
            )
            require(0 <= number(record["progression_pct"]) <= 100, f"E13 {env} progression")
        same(
            d["score"],
            statistics.mean(e["progression_pct"] for e in envs.values()),
            "E13 equal-environment mean (not pooled episodes)",
            digits=2,
        )
        return "255 episodes: coverage/restart counts, equal mean of six recorded environment means"
    success_key = {"E1": "n_resolved", "E2": "n_correct"}.get(lane, "n_success")
    success = count(d[success_key])
    require(success <= graded, f"{lane} successes exceed graded")
    same(d["score"], success / n, f"{lane} success/attempted", digits=4)
    if lane == "E1":
        parts = list(d["breakdown"].values())
        require(set(d["breakdown"]) == {"wave10", "remaining174"}, "E1 partition changed")
        for field, expected in (("attempted", n), ("graded", graded), ("resolved", success)):
            same(sum(count(p[field]) for p in parts), expected, f"E1 partition {field}")
        for part in [*parts, d["prior_waves_separate_not_pooled"]]:
            same(
                count(part["attempted"]),
                count(part["graded"])
                + count(part["errors_patch_apply_or_eval"])
                + count(part["empty_patch"]),
                "E1 outcome partition",
            )
            same(count(part["resolved"]), len(part["resolved_ids"]), "E1 resolved IDs")
            unique(part["resolved_ids"], "E1 resolved IDs")
            for denominator in ("graded", "attempted"):
                same(
                    part[f"resolved_over_{denominator}"],
                    part["resolved"] / part[denominator],
                    f"E1 resolved/{denominator}",
                    digits=4,
                )
        for field in ("errors_patch_apply_or_eval", "empty_patch"):
            same(count(d["outcome_counts"][field]), sum(p[field] for p in parts), f"E1 {field}")
        same(success, len(d["resolved_ids"]), "E1 headline resolved IDs")
        require(
            sorted(d["resolved_ids"]) == sorted(i for p in parts for i in p["resolved_ids"]),
            "E1 resolved ID partition mismatch",
        )
        same(
            d["secondary_resolved_over_graded"]["score"],
            success / graded,
            "E1 graded rate",
            digits=4,
        )
        same(
            count(d["prior_waves_separate_not_pooled"]["attempted"]) + n,
            300,
            "E1 task coverage only",
        )
        require(d["not_attempted_of_300"] == [], "E1 unexpected unattempted tasks")
    elif lane == "E2":
        rows = list(d["per_task"].values())
        same(len(rows), n, "E2 per-task count")
        same(sum(values(rows, "correct", binary=True)), success, "E2 task successes")
        same(sum(r["status"] == "completed" for r in rows), graded, "E2 graded count")
        require(
            dict(Counter(f"{r['status']}/{r['end']}" for r in rows)) == d["outcome_breakdown"],
            "E2 completion partition mismatch",
        )
        same(count(d["native_summary"]["correct_tasks"]), success, "E2 native successes")
        same(count(d["native_summary"]["total_tasks"]), n, "E2 native total")
    elif lane == "E5":
        infra = count(d["n_infrastructure_error"])
        same(graded + infra, n, "E5 graded+infrastructure errors")
        same(
            sum(count(v) for v in d["infrastructure_error_breakdown"].values()),
            infra,
            "E5 infra partition",
        )
        same(len(d["infrastructure_error_tasks"]), infra, "E5 infra task count")
        unique(d["infrastructure_error_tasks"], "E5 infrastructure tasks")
        same(d["native_pass1_excluding_infra"], success / graded, "E5 native pass1")
        parts = list(d["subsets"].values())
        require(len(parts) == 2, "E5 subset partition changed")
        same(sum(count(p["n"]) for p in parts), n, "E5 subset counts")
        same(sum(count(p["passes"]) for p in parts), success, "E5 subset successes")
        for part in parts:
            same(part["pass1"], part["passes"] / part["n"], "E5 subset pass1", digits=4)
    elif lane == "E6":
        pending = count(d["n_ungraded_judge_unavailable"])
        same(graded + pending, n, "E6 graded+ungraded")
        require(count(d["n_errors"]) <= graded, "E6 error count exceeds graded")
        secondary = d["secondary"]
        parts = list(secondary["per_split"].values())
        require(len(parts) == 5, "E6 split partition changed")
        for field, expected in (("n", n), ("success", success), ("judge_pending", pending)):
            same(sum(count(p[field]) for p in parts), expected, f"E6 split {field}")
        same(secondary["score_over_graded_only"], success / graded, "E6 graded score", digits=4)
        same(
            secondary["score_upper_bound_if_all_ungraded_passed"],
            (success + pending) / n,
            "E6 upper bound",
            digits=4,
        )
    elif lane == "E9":
        rows = d["per_task"]
        same(len(rows), n, "E9 per-task count")
        unique([r["task"] for r in rows], "E9 tasks")
        same(sum(values(rows, "success", binary=True)), success, "E9 task successes")
        errors = sum(r["rc"] != 0 for r in rows)
        same(count(d["n_errors_or_timeouts"]), errors, "E9 errors")
        same(graded, n - errors, "E9 graded")
    return f"{success}/{n}: headline rate and receipt outcome/coverage subdivisions" + (
        "; prior 4/110 score checked separately, not pooled" if lane == "E1" else ""
    )


def check_paired(lane, d):
    require(d["lane"] == lane, f"{lane} wrong lane identifier")
    if lane == "E10":
        require(set(d["n_items"]) == {"harmful", "benign"}, "E10 scope partition changed")
        require(len(d["per_item"]) == 60, "E10 requires 60 outcomes")
        for scope in ("harmful", "benign"):
            rows = [r for k, r in d["per_item"].items() if k.startswith(scope + "/")]
            same(count(d["n_items"][scope]), 30, f"E10 {scope} count")
            same(len(rows), 30, f"E10 {scope} outcomes")
            scores = {a: values([r[a] for r in rows], "score") for a in ("trained", "base")}
            metric = "harm_score" if scope == "harmful" else "benign_score"
            paired_metric(
                d[metric], scores["trained"], scores["base"], "paired_mean_difference", binary=False
            )
            if scope == "harmful":
                refusals = {a: values([r[a] for r in rows], "refusal", binary=True) for a in scores}
                paired_metric(
                    d["refusal_rate"], refusals["trained"], refusals["base"], "difference"
                )
        return "60 outcomes: harm/benign means and deltas, harmful-refusal McNemar"
    same(count(d["n_items"]), PAIRED_COUNTS[lane], f"{lane} paired scope")
    collection = d["items"] if lane in ("E1", "E2", "E13") else d["per_item"]
    if isinstance(collection, dict):
        rows = list(collection.values())
        ids = list(collection)
    else:
        require(isinstance(collection, list), f"{lane} outcomes must be list or mapping")
        rows = collection
        id_key = {"E1": "instance_id", "E2": "problem_idx", "E13": "id"}.get(lane, "item")
        ids = [r[id_key] for r in rows]
    unique(ids, f"{lane} paired items")
    same(len(rows), d["n_items"], f"{lane} outcome count")
    if lane in ("E4", "E5"):
        require(d["item_ids"] == ids, f"{lane} item ID mismatch")
    binary = lane not in ("E4", "E13")
    arm_values = {}
    for arm in ("trained", "base"):
        key = {"E1": f"{arm}_resolved", "E2": f"{arm}_passed"}.get(lane, arm)
        arm_rows = [{"x": r[arm]["any_medal"]} for r in rows] if lane == "E9" else rows
        arm_values[arm] = values(arm_rows, "x" if lane == "E9" else key, binary=binary)
    trained, base = arm_values["trained"], arm_values["base"]
    if lane == "E13":
        envs = {i.split("/")[0] for i in ids}
        require(
            envs == set(d["per_env"]) == {"babyai", "textworld", "babaisai", "minihack", "crafter"},
            "E13 paired environment set changed",
        )
        means: dict[str, list[float]] = {arm: [] for arm in arm_values}
        for env in sorted(envs):
            indices = [i for i, ident in enumerate(ids) if ident.split("/")[0] == env]
            same(count(d["per_env"][env]["n"]), len(indices), f"E13 {env} episode count")
            for arm in arm_values:
                value = statistics.mean(arm_values[arm][i] for i in indices) * 100
                means[arm].append(value)
                same(d["per_env"][env][arm], value, f"E13 {env} {arm}", digits=2)
        for arm in means:
            same(
                d[f"{arm}_value"],
                statistics.mean(means[arm]),
                f"E13 {arm} equal-env mean",
                digits=2,
            )
        same(
            d["difference"],
            statistics.mean(means["trained"]) - statistics.mean(means["base"]),
            "E13 equal-env paired difference",
            digits=3,
        )
        same(
            d["mean_item_difference"],
            statistics.mean(t - b for t, b in zip(trained, base)),
            "E13 mean item difference",
            digits=4,
        )
        same(
            count(d["n_items_identical_progression"]),
            sum(t == b for t, b in zip(trained, base)),
            "E13 identical episodes",
        )
        b, c, p = mcnemar([int(t == 1) for t in trained], [int(b == 1) for b in base])
        secondary = d["secondary_full_success_mcnemar"]
        same(count(secondary["trained_only_b"]), b, "E13 full-success trained discordances")
        same(count(secondary["base_only_c"]), c, "E13 full-success base discordances")
        same(secondary["exact_p"], p, "E13 full-success McNemar", digits=4)
    else:
        diff_key = "difference_trained_minus_base" if lane in ("E3", "E7", "E12") else "difference"
        paired_metric(
            d, trained, base, diff_key, binary=binary, staged_rounding=lane in ("E3", "E7", "E12")
        )
        if lane == "E4":
            same(
                d["paired_mean_difference"],
                statistics.mean(t - b for t, b in zip(trained, base)),
                "E4 paired mean difference",
                digits=4,
            )
    if lane == "E13":
        return "26 outcomes: equal-env means/delta, item mean delta and full-success McNemar; no bootstrap"
    if lane in ("E3", "E7", "E12"):
        return {
            "description": f"{len(rows)} outcomes: means, exact staged-rounding delta, McNemar",
            "raw_paired_delta": statistics.mean(t - b for t, b in zip(trained, base)),
            "recorded_delta": d["difference_trained_minus_base"],
            "rounding_rule": "round(round(mean(trained), 4) - round(mean(base), 4), 4)",
            "generator": f"{PAIRED}/E3/code/paired_finalize.py",
        }
    return f"{len(rows)} paired outcomes: means, deltas" + (
        ", exact McNemar" if binary else "; no bootstrap check"
    )


def load_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def validate_finite(value, relative, document, path=()):
    # The preserved runner writes NaN for PPO ZVF (G=1). It is not a score.
    if isinstance(value, float) and not math.isfinite(value):
        allowed = (
            relative == M1B_FULL
            and len(path) == 5
            and path[0] == "runs"
            and path[2] == "step_log"
            and path[4] == "zvf"
            and document["runs"][path[1]].get("arm") == "ppo"
            and math.isnan(value)
        )
        require(allowed, f"non-finite value at {path}")
    elif isinstance(value, dict):
        for key, item in value.items():
            validate_finite(item, relative, document, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            validate_finite(item, relative, document, (*path, index))


def check_evidence(root: Path):
    data, checks, errors = {}, [], []
    for relative in EVIDENCE_FILES:
        try:
            data[relative] = json.loads(
                (root / relative).read_text(encoding="utf-8"), object_pairs_hook=load_object
            )
            require(isinstance(data[relative], dict), "top-level JSON must be an object")
            validate_finite(data[relative], relative, data[relative])
        except (OSError, ValueError) as exc:
            errors.append(f"{relative}: {exc}")
    if not errors:
        # Data lookups stay inside the job so a missing file is reported as a FAIL row.
        def replacement_job(lane: str) -> Any:
            return check_replacement(lane, data[f"{FINISH}/{lane}/result.json"])

        def paired_job(lane: str) -> Any:
            return check_paired(lane, data[f"{PAIRED}/{lane}/paired.json"])

        jobs: list[tuple[str, Callable[[], Any]]] = [("M1b", partial(check_m1b, data))]
        jobs += [
            (f"replacement/{lane}", partial(replacement_job, lane)) for lane in REPLACEMENT_COUNTS
        ]
        jobs += [(f"paired/{lane}", partial(paired_job, lane)) for lane in PAIRED_COUNTS]
        for name, check in jobs:
            try:
                checks.append({"check": name, "detail": check(), "status": "PASS"})
            except (
                ValueError,
                KeyError,
                TypeError,
                AttributeError,
                IndexError,
                ZeroDivisionError,
            ) as exc:
                errors.append(f"{name}: {exc}")
                checks.append({"check": name, "detail": str(exc), "status": "FAIL"})
    return {
        "schema_version": "thesis-selected-arithmetic-v1",
        "status": "FAIL" if errors else "PASS",
        "scope": SCOPE,
        "required_files": list(EVIDENCE_FILES),
        "checks": checks,
        "errors": errors,
        "not_checked": NOT_CHECKED,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", type=Path, metavar="PATH", help="also write the report as JSON")
    args = parser.parse_args(argv)
    report = check_evidence(args.root)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json:
        args.json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
