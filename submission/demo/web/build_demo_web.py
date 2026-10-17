"""Export repository evidence for the TinkerRL defense web demo (React app).

Reads stored results only (no training, no network) and writes src/data.json.
Then build the offline single-file page with `npm run build` (see README.md).
"""

import argparse
import gzip
import math
import hashlib
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
V2 = ROOT / "platform_hybrid/experiments/results/samestack_gsm8k_v2s"
V1 = ROOT / "reports/project_finish_2026-10-11/validation/samestack_v2/v1_recompute.json"
GOLD = ROOT / "reports/project_finish_2026-10-11/validation/samestack_v2/gsm8k_test_gold.json"
AUDIT = ROOT / "reports/public_revision_2026-10-11/PUBLIC_CHECKOUT_NUMBER_AUDIT.json"
ANALYSIS_DIR = ROOT / "reports/project_finish_2026-10-11/validation/samestack_v2"
sys.path.insert(0, str(ANALYSIS_DIR))
from analysis_plan import analyze_runs, default_plan, shared_stack_fingerprint
from platform_hybrid.experiments.samestack_contract import validate_completion

PLATFORM = ANALYSIS_DIR / "platform_calibration.json"
ARMS = ["grpo_g8", "grpo_g2", "ppo", "ppo_8x8"]

sys.path.insert(0, str(ROOT / "platform_hybrid/registry"))
import query as registry


def source_name(path):
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def samestack(plan=None, results_dir=None):
    """Export immutable records and the canonical selector's output, without guessed identity."""
    runs, records, sources, bundle_failures, validated_ids = [], [], [], [], []
    results_dir = results_dir or V2
    for path in sorted(results_dir.glob("*_s*.json")):
        if not re.fullmatch(r"(?:grpo_g8|grpo_g2|ppo_8x8|ppo)_s\d+\.json", path.name):
            continue
        r = json.loads(path.read_text())
        if r.get("arm") not in ARMS or r.get("smoke"):
            continue
        source_path = source_name(path)
        completion_validated = False
        failure = None
        if r.get("manifest") is not None:
            try:
                r = validate_completion(path)
                completion_validated = True
                validated_ids.append(r["manifest"]["run_id"])
            except (OSError, ValueError, KeyError, TypeError) as error:
                failure = str(error)
        if failure is None:
            records.append({**r, "source_path": source_path})
        else:
            manifest = r.get("manifest") if isinstance(r.get("manifest"), dict) else {}
            bundle_failures.append({"run_id": manifest.get("run_id"), "arm": r.get("arm"), "seed": r.get("seed"),
                                    "source_path": source_path, "stage": "artifact_completion", "reasons": [failure]})
        runs.append({
            **{k: (r.get("manifest") or {}).get(k) for k in ("run_id", "environment", "environment_hash", "protocol_hash", "ordered_test_item_hash")},
            "shared_stack_hash": shared_stack_fingerprint(r),
            "manifest": r.get("manifest"), "arm": r.get("arm"), "seed": r.get("seed"),
            "source_path": source_path, "completion_bundle_validated": completion_validated,
            "completion_failure": failure,
            "pre": r.get("heldout_pre_acc"), "post": r.get("heldout_post_acc"),
            "post_trunc": r.get("heldout_post_trunc"), "n_eval": r.get("n_eval"),
            "w2r": r.get("wrong_to_right"), "r2w": r.get("right_to_wrong"),
            "steps": [{"r": t["mean_reward"], "zvf": t["zvf"]} for t in r.get("step_log", [])
                      if isinstance(t.get("zvf"), (int, float)) and math.isfinite(t["zvf"])
                      and isinstance(t.get("mean_reward"), (int, float)) and math.isfinite(t["mean_reward"])],
        })
        sources.append({"path": source_path, "sha256": sha(path), "role": "immutable run record"})
    first = records[0] if records else {}
    setup = {k: first.get(k) for k in ("model", "n_steps", "max_new", "lr", "k_epochs", "n_gen", "n_eval")}
    analysis = analyze_runs(records, plan=plan, completion_validated_run_ids=validated_ids,
                            input_artifact_exclusions=bundle_failures)
    return {"runs": runs, "setup": setup, "sources": sources,
            "seeds_planned": len(analysis["plan"]["planned_seeds"])}, analysis


def stackdiff():
    recs = registry.load("stack")
    entries = [{"id": rid, "label": r["label_claimed"],
                "framework": f"{r['framework']['name']}/{r['framework']['openness']}"}
               for rid, r in sorted(recs.items())]
    pairs = {}
    for (a, b), r in registry.all_pairs(recs).items():
        pairs[f"{a}|{b}"] = {
            "level": r["level"], "coverage": round(r["coverage"], 4),
            "underspecified": r["underspecified"], "same_label": r["same_label"],
            "differs": [[f, s, x, y] for f, s, x, y in sorted(r["differs"], key=lambda d: -d[1])],
            "unknown_n": len(r["unknown"]),
            "unknown_r5": sorted({u[0] for u in r["unknown"] if u[1] == 5}),
        }
    return {"entries": entries, "pairs": pairs, "risk": registry.RISK}


def examples(seed=42, per_pattern=4, results_dir=None):
    """Stored greedy completions for one seed: base, GRPO G8 and PPO on the same test items.

    Items are grouped by outcome pattern (base, GRPO, PPO correct?) over all test items;
    the first `per_pattern` items by test index are taken from each pattern, so the sample
    is fixed by rule rather than chosen by hand.
    """
    results_dir = results_dir or V2
    required = [GOLD] + [results_dir / f"{a}_s{seed}{suffix}" for a in ("grpo_g8", "ppo")
                         for suffix in (".json", "_texts.json.gz")]
    if not all(p.exists() for p in required):
        return {"seed": seed, "n": 0, "pattern_counts": {}, "per_pattern": per_pattern, "items": [],
                "status": "unavailable", "missing": [source_name(p) for p in required if not p.exists()]}
    gold = json.loads(GOLD.read_text())["test"]
    runs = {a: json.loads((results_dir / f"{a}_s{seed}.json").read_text()) for a in ("grpo_g8", "ppo")}
    texts = {a: json.loads(gzip.open(results_dir / f"{a}_s{seed}_texts.json.gz", "rt").read()) for a in runs}
    def unavailable(reason):
        return {"seed": seed, "n": len(gold), "pattern_counts": {}, "per_pattern": per_pattern,
                "items": [], "status": "unavailable", "missing": [reason]}
    for arm in runs:
        for key in ("pre_correct", "post_correct", "pre_trunc", "post_trunc"):
            if not isinstance(runs[arm].get(key), list) or len(runs[arm][key]) != len(gold):
                return unavailable(f"{arm} {key}: index-aligned example coverage does not match gold")
        for phase in ("pre", "post"):
            if not isinstance(texts[arm].get(phase), list) or len(texts[arm][phase]) != len(gold):
                return unavailable(f"{arm} {phase}: completion coverage does not match gold")
    if texts["grpo_g8"]["pre"] != texts["ppo"]["pre"]:
        return unavailable("Per-arm base completions differ; shared-base examples are unavailable")
    g, p = runs["grpo_g8"], runs["ppo"]
    patterns = {}
    for i in range(len(gold)):
        patterns.setdefault(f"{g['pre_correct'][i]}{g['post_correct'][i]}{p['post_correct'][i]}", []).append(i)
    items = []
    for key, idx in sorted(patterns.items(), key=lambda kv: -len(kv[1])):
        for i in idx[:per_pattern]:
            items.append({
                "i": i, "pattern": key, "question": gold[i]["question"],
                "gold": gold[i]["answer"].split("####")[-1].strip(),
                "base": {"text": texts["grpo_g8"]["pre"][i], "stored": g["pre_correct"][i], "trunc": g["pre_trunc"][i]},
                "grpo_g8": {"text": texts["grpo_g8"]["post"][i], "stored": g["post_correct"][i], "trunc": g["post_trunc"][i]},
                "ppo": {"text": texts["ppo"]["post"][i], "stored": p["post_correct"][i], "trunc": p["post_trunc"][i]},
            })
    return {"status": "historical_unverified_index_alignment", "seed": seed, "n": len(gold), "pattern_counts": {k: len(v) for k, v in patterns.items()},
            "per_pattern": per_pattern, "items": items}


def validate_grader_report(report):
    """A publication audit must prove complete coverage and zero disagreements."""
    if not isinstance(report, dict) or report.get("grading_spec_id") != "boxed-exact-rational-v2":
        raise ValueError("grader audit: missing or unsupported grading specification")
    counts = ("runs", "runs_checked", "expected_items_per_phase", "completions", "disagreements",
              "stored_correct_now_incorrect", "stored_incorrect_now_correct")
    if any(type(report.get(k)) is not int or report[k] < 0 for k in counts):
        raise ValueError("grader audit: invalid count schema")
    if report.get("historical_scores_mutated") is not False:
        raise ValueError("grader audit: immutable-score contract missing")
    if report.get("coverage_complete") is not True or any(
        report.get(k) != [] for k in ("missing_sidecars", "invalid_inputs", "length_mismatches", "invalid_scores")
    ):
        raise ValueError("grader audit: incomplete or invalid coverage")
    if (report["runs"] == 0 or report["expected_items_per_phase"] == 0
            or report["runs_checked"] != report["runs"]
            or report["completions"] != 2 * report["runs"] * report["expected_items_per_phase"]):
        raise ValueError("grader audit: inconsistent coverage denominators")
    if any(report[k] != 0 for k in counts[-3:]):
        raise ValueError("grader audit: stored-score disagreements detected")
    return report


def grader_check(results_dir=None):
    """Fail closed before publishing evidence; diagnostic report-only audits stay separate."""
    try:
        out = subprocess.run(["node", str(HERE / "tests/grade_all.mjs"), "--json", "--results-dir", str(results_dir or V2), "--gold-path", str(GOLD)], capture_output=True, text=True)
    except OSError as error:
        raise RuntimeError(f"grader audit could not run: {error}") from error
    if out.returncode != 0:
        raise RuntimeError(f"grader audit failed (exit {out.returncode}): {out.stderr or out.stdout}")
    try:
        report = validate_grader_report(json.loads(out.stdout))
    except (ValueError, TypeError) as error:
        raise RuntimeError(f"grader audit rejected: {error}") from error
    return {**report, "audit_exit_code": out.returncode}


def samestack_v1():
    """v1 (200-token cap, cold critic, 200 items): superseded, shown for the correction story."""
    if not V1.exists():
        return None
    d = json.loads(V1.read_text())
    arms = {a: {"pre": v["pre_mean"], "post_by_seed": v["post_by_seed"]} for a, v in d["arms"].items()}
    c = d["contrasts"]["g8-ppo"]
    return {"arms": arms, "g8_ppo": {k: c[k] for k in ("mean", "t_ci", "p_t", "p_signflip", "holm_sf")},
            "path": str(V1.relative_to(ROOT)), "sha256": sha(V1)}


def audit():
    d = json.loads(AUDIT.read_text())
    return {"status": d["status"], "summary": d["summary"], "token_counts": d["token_counts"],
            "scope": d.get("scope"), "path": str(AUDIT.relative_to(ROOT)), "sha256": sha(AUDIT)}


def provenance(data, plan_path=None):
    """Hash the export's complete known input/code chain; disclose unknown historical links."""
    sources = {}
    missing = []
    def add(path, role):
        key = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        if path.is_file():
            entry = sources.setdefault(key, {"path": key, "role": role, "roles": [], "sha256": sha(path)})
            if role not in entry["roles"]:
                entry["roles"].append(role)
        else:
            missing.append(f"{role}: {key}")
    for r in data["samestack"]["sources"]:
        add(ROOT / r["path"], "immutable run record")
        add((ROOT / r["path"]).with_name(pathlib.Path(r["path"]).stem + "_texts.json.gz"), "completion sidecar")
    for run in data["samestack"]["runs"]:
        if run.get("manifest") is not None:
            path = ROOT / run["source_path"]
            marker = path.with_suffix(".complete.json")
            add(marker, "completion marker")
            if run["completion_bundle_validated"]:
                inventory = json.loads(marker.read_text())["files"]
                for relative_path in inventory:
                    add(path.parent / relative_path, "verified completion-bundle artifact")
            else:
                missing.append(f"Completion bundle excluded: {run['source_path']}: {run['completion_failure']}")
    for path, role in [(GOLD, "ordered gold questions and answers"), (V1, "superseded historical analysis"),
                       (AUDIT, "audit summary"), (PLATFORM, "historical cross-platform calibration"), (HERE / "build_demo_web.py", "exporter"),
                       (HERE / "package-lock.json", "frontend dependency lock"),
                       (HERE / "package.json", "build commands"),
                       (ROOT / "platform_hybrid/registry/query.py", "registry comparison implementation"),
                       (ROOT / "platform_hybrid/registry/schema.json", "registry schema")]:
        add(path, role)
    audit_input = json.loads(AUDIT.read_text()) if AUDIT.is_file() else {}
    for group in audit_input.get("groups", []):
        chapter = group.get("chapter")
        if chapter and audit_input.get("thesis_directory"):
            chapter_path = (ROOT / audit_input["thesis_directory"] / chapter).resolve()
            if chapter_path.is_relative_to(ROOT):
                add(chapter_path, "audited thesis chapter")
        for artifact in group.get("artifacts", []):
            path = (ROOT / artifact.get("path", "")).resolve()
            if not path.is_relative_to(ROOT):
                missing.append(f"Audit reference outside repository not inspected: {artifact.get('path')}")
                continue
            add(path, "audit evidence reference")
            if path.is_file() and artifact.get("sha256"):
                entry = sources[source_name(path)]
                entry.setdefault("audit_recorded_sha256", [])
                if artifact["sha256"] not in entry["audit_recorded_sha256"]:
                    entry["audit_recorded_sha256"].append(artifact["sha256"])
                if entry["sha256"] != artifact["sha256"]:
                    missing.append(f"Audit evidence changed since recorded audit: {source_name(path)}")
    for path in sorted((ROOT / "platform_hybrid/registry/entries").glob("*.json")):
        add(path, "registry input entry")
    for path in sorted(ANALYSIS_DIR.glob("*.py")):
        add(path, "offline analysis implementation")
    for path in sorted((HERE / "src").rglob("*")):
        if path.suffix in {".js", ".jsx", ".css"}:
            add(path, "interactive presentation implementation")
    for path in sorted((ROOT / "platform_hybrid/experiments").glob("samestack*.py")):
        add(path, "protocol, validation and training implementation")
    add(HERE / "tests/grade_all.mjs", "completion re-grading implementation")
    if plan_path:
        add(plan_path, "explicit analysis plan")
    else:
        missing.append("No explicit bound analysis plan supplied; default retrospective plan is not a preregistration")
    for r in data["samestack"]["runs"]:
        absent = [k for k in ("run_id", "environment", "environment_hash", "shared_stack_hash", "protocol_hash", "ordered_test_item_hash") if r.get(k) is None]
        if absent:
            missing.append(f"{r['source_path']}: unknown {', '.join(absent)}")
    missing.extend(data["examples"].get("missing", []))
    check = data["grader_check"]
    missing.extend(str(x) for x in check.get("missing", []))
    for category in ("missing_sidecars", "invalid_inputs", "length_mismatches", "invalid_scores"):
        missing.extend(f"completion grading {category}: {json.dumps(x)}" for x in check.get(category, []))
    if check.get("coverage_complete") is False:
        missing.append("Completion re-grading coverage is incomplete")
    return {"schema_version": 1, "sources": sorted(sources.values(), key=lambda s: s["path"]),
            "analysis_sha256": hashlib.sha256(json.dumps(data["analysis"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
            "missing": sorted(set(missing)),
            "coverage_note": "Hashes identify current export inputs, not unrecorded historical runtime provenance. Source run records are never overwritten."}


def export_data(plan_path=None, results_dir=None):
    plan = json.loads(plan_path.read_text()) if plan_path else default_plan()
    ss, analysis = samestack(plan, results_dir)
    data = {"samestack": ss, "analysis": analysis, "samestack_v1": samestack_v1(),
            "examples": examples(results_dir=results_dir), "grader_check": grader_check(results_dir), "stackdiff": stackdiff(), "audit": audit(), "platform": platform()}
    data["manifest"] = provenance(data, plan_path)
    return data


def platform():
    """Cross-platform calibration (Modal A10G vs Colab A100), if platform_calibration.py has been run."""
    if not PLATFORM.exists():
        return None
    d = json.loads(PLATFORM.read_text(), parse_constant=lambda _: None)
    return {**d, "path": str(PLATFORM.relative_to(ROOT)), "sha256": sha(PLATFORM)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=pathlib.Path, help="Explicit versioned analysis-plan JSON; missing historical identity is never filled in")
    parser.add_argument("--results-dir", type=pathlib.Path, help="Explicit environment/protocol leaf directory containing arm_sSEED.json records; defaults to immutable historical v2 records")
    parser.add_argument("--output", type=pathlib.Path, default=HERE / "src/data.json")
    args = parser.parse_args()
    data = export_data(args.plan.resolve() if args.plan else None, args.results_dir.resolve() if args.results_dir else None)
    args.output.write_text(json.dumps(data, separators=(",", ":"), default=str, allow_nan=False) + "\n")
    print(f"wrote {args.output} ({args.output.stat().st_size // 1024} KB); analysis={data['analysis']['status']}")


if __name__ == "__main__":
    main()
