# Historical review contract only; not interchangeable with canonical v3 evidence.
"""Strict same-stack evidence gate. Historical bundles require revalidation.

Merging requires a pinned item manifest, independent completion regrading, and an
explicit hardware pooling policy. Reports alone never establish equivalence.
"""

import argparse
import gzip
import hashlib
import json
import math
import pathlib
import re
import shutil
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
STAGE = ROOT / "platform_hybrid/experiments/results/samestack_gsm8k_v2s_colab"
CANON = ROOT / "platform_hybrid/experiments/results/samestack_gsm8k_v2s"
OUT = ROOT / "reports/project_finish_2026-10-11/validation/samestack_v2/colab_runs_check.json"
CONFIG_KEYS = ("model", "n_steps", "k_epochs", "lr", "max_new", "n_gen", "n_eval")
GROUPS = {"grpo_g8": 8, "grpo_g2": 2, "ppo": 1, "ppo_8x8": 8}


def load(path):
    return json.loads(
        path.read_text(),
        parse_constant=lambda v: (_ for _ in ()).throw(ValueError("Non-finite JSON")),
    )


def validate_result(name, result, texts, reference, items=None):
    match = re.fullmatch(r"(grpo_g8|grpo_g2|ppo|ppo_8x8)_s([0-9]+)", name)
    arm = match[1] if match else None
    n = reference.get("n_eval")
    pre, post = result.get("pre_correct", []), result.get("post_correct", [])
    binary = lambda v: type(v) is int and v in (0, 1)
    grades_valid = (
        isinstance(pre, list)
        and isinstance(post, list)
        and type(n) is int
        and n > 0
        and len(pre) == len(post) == n
        and all(binary(v) for v in pre + post)
    )
    text_valid = isinstance(texts, dict) and all(
        isinstance(texts.get(k), list)
        and len(texts[k]) == n
        and all(isinstance(t, str) and t.strip() for t in texts[k])
        for k in ("pre", "post")
    )
    env = result.get("execution_environment") or result.get("env") or {}
    logs = result.get("step_log", [])
    checks = {
        "filename_identity": bool(
            match and result.get("arm") == arm and result.get("seed") == int(match[2])
        ),
        "not_smoke": result.get("smoke") is False,
        "config_matches": all(result.get(k) == reference.get(k) for k in CONFIG_KEYS),
        "group_matches": bool(arm and result.get("group") == GROUPS[arm]),
        "critic_matches": bool(
            arm
            and (
                (isinstance(result.get("critic"), dict) and bool(result["critic"]))
                if arm and arm.startswith("ppo")
                else bool(arm) and result.get("critic") is None
            )
        ),
        "actual_steps": isinstance(logs, list)
        and len(logs) == reference.get("n_steps")
        and [row.get("step") for row in logs if isinstance(row, dict)]
        == list(range(reference.get("n_steps", 0))),
        "binary_grade_cardinality": grades_valid,
        "completion_cardinality": text_valid,
        "substantive_environment": isinstance(env, dict)
        and all(isinstance(env.get(k), str) and env[k].strip() for k in ("gpu", "python", "torch")),
        "accuracy_recomputed": grades_valid
        and all(
            type(result.get(k)) in (int, float)
            and math.isfinite(result[k])
            and abs(result[k] - sum(v) / n) < 1e-12
            for k, v in (("heldout_pre_acc", pre), ("heldout_post_acc", post))
        ),
        "pinned_item_identity": bool(
            items and result.get("item_ids") == [item["id"] for item in items] and len(items) == n
        ),
    }
    checks["independent_regrade"] = False
    if checks["pinned_item_identity"] and text_valid:
        from platform_local.trl_integrations.scientific import numeric_answer

        def grade(text, gold):
            boxes = re.findall(r"\\boxed\{([^{}]+)\}", text)
            return int(
                bool(boxes)
                and numeric_answer(boxes[-1]) is not None
                and numeric_answer(boxes[-1]) == numeric_answer(gold)
            )

        checks["independent_regrade"] = all(
            [grade(t, item["answer"]) for t, item in zip(texts[key], items)] == vector
            for key, vector in (("pre", pre), ("post", post))
        )
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--item-manifest", type=pathlib.Path)
    parser.add_argument("--item-manifest-sha256")
    parser.add_argument("--pooling-policy", type=pathlib.Path)
    args = parser.parse_args(argv)
    items = None
    if args.item_manifest:
        if hashlib.sha256(args.item_manifest.read_bytes()).hexdigest() != args.item_manifest_sha256:
            raise ValueError("Item manifest must match an independently reviewed digest")
        items = load(args.item_manifest)["items"]
    reference = load(CANON / "grpo_g8_s42.json")
    report = {
        "schema": "colab-verification-v2",
        "runs": {},
        "pooling": "stratified_by_recorded_hardware",
    }
    accepted = []
    # Completion markers are transfer receipts, never experimental results.
    # Exclude only the exact known marker grammar: malformed result names must
    # still enter validation and fail rather than disappear silently.
    paths = sorted(
        path
        for path in STAGE.glob("*_s*.json")
        if not re.fullmatch(r"(grpo_g8|grpo_g2|ppo|ppo_8x8)_s[0-9]+\.complete\.json", path.name)
    )
    for path in paths:
        result = load(path)
        text_path = path.with_name(path.stem + "_texts.json.gz")
        try:
            with gzip.open(text_path, "rt") as handle:
                texts = json.load(handle)
        except (OSError, ValueError):
            texts = None
        checks = validate_result(path.stem, result, texts, reference, items)
        report["runs"][path.stem] = {"checks": checks, "ok": all(checks.values())}
        if all(checks.values()):
            accepted.append((path, text_path))
    ok = bool(paths) and len(accepted) == len(paths)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    if args.merge:
        if not ok or not args.pooling_policy:
            raise ValueError(
                "Merging requires complete regraded bundles and an explicit hardware pooling policy"
            )
        policy = load(args.pooling_policy)
        if not policy.get("calibration") or policy.get("mode") != "stratified_by_recorded_hardware":
            raise ValueError("Pooling policy must specify calibration and hardware stratification")
        # Publish one complete directory atomically, never partially overwrite canonical files.
        parent = CANON / "verified_bundles"
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent, prefix=".staging-") as temp:
            bundle = pathlib.Path(temp) / "bundle"
            bundle.mkdir()
            for pair in accepted:
                for path in pair:
                    shutil.copyfile(path, bundle / path.name)
            (bundle / "verification.json").write_text(json.dumps(report, sort_keys=True))
            (bundle / "pooling_policy.json").write_text(json.dumps(policy, sort_keys=True))
            digest = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
            target = parent / digest
            if target.exists():
                raise FileExistsError("Verified bundle already exists; it was not overwritten")
            bundle.rename(target)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
