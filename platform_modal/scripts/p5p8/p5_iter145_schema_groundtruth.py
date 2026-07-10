#!/usr/bin/env python3
"""
Iter 145 P5 schema-ground-truth audit.

For each of the n=98 mega-campaign manifests, derive the 5-component
stack (model, task, G, temperature, seed) from the cell_id filename and
compare against the declared fields (heldout_split, group_size_schedule,
decontamination_notes, per_step_zvf_path). Also test whether the
per_step_zvf_path points to an existing tensor file and whether the
loaded tensor's `cell.group_size` matches the declared G.

Outputs:
  - platform_hybrid/experiments/results/p5p8/p5_iter145_schema_groundtruth.tsv
  - platform_hybrid/experiments/results/p5p8/p5_iter145_summary.json

Falsifiable headline: zero manifest drift on a stack-consistency audit
(n=X / 98 perfect on cell_id parsability + hash uniqueness + tensor
existence + G match) OR measurable drift on a per-check axis.

This iter is the schema-ground-truth audit that complements
iter-105 (live field coverage), iter-117 (structural ambiguity),
iter-121 (value correctness), iter-137 (cross-corpus portability),
iter-141 (algorithm-axis eta^2). None of those audited the
cross-reference between the manifest *declarations* and the *derived
fields* in the cell_id + the *linked tensor file* on disk.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import re
import sys
from pathlib import Path

REPO = Path("/home/claude/tinker-rl-lab-minimax")
MAN_DIR = REPO / "platform_hybrid/experiments/results/mega_20260704/manifests"
TENS_DIR = REPO / "platform_hybrid/experiments/results/mega_20260704/group_tensors"
OUT_TSV = REPO / "platform_hybrid/experiments/results/p5p8/p5_iter145_schema_groundtruth.tsv"
OUT_JSON = REPO / "platform_hybrid/experiments/results/p5p8/p5_iter145_summary.json"

# Regex for cell_id: {model}_{task}_G{N}_t{T}_s{S}_{10hex}
CELL_ID_RE = re.compile(
    r"^(?P<model>.+?)_(?P<task>[^_]+(?:_[^_]+)*?)_G(?P<G>\d+)_t(?P<T>[\d.]+)_s(?P<S>\d+)_(?P<hash>[0-9a-f]{10})$"
)

# Regex for fixed-G=N in group_size_schedule
GSS_RE = re.compile(r"^fixed-G=(\d+)$")

TASK_TO_DECONTAM = {
    "gsm8k_easy": "gsm8k-train-slice",
    "gsm8k_hard": "gsm8k-train-slice",
    "humaneval_subset": "humaneval-openai-subset",
}


def parse_cell_id(cell_id: str):
    """Return (parsed_ok, dict_or_None, reason)."""
    m = CELL_ID_RE.match(cell_id)
    if not m:
        return False, None, f"cell_id does not match pattern {CELL_ID_RE.pattern}"
    d = m.groupdict()
    try:
        d["G"] = int(d["G"])
        d["T"] = float(d["T"])
        d["S"] = int(d["S"])
    except ValueError as e:
        return False, None, f"int/float parse fail: {e}"
    return True, d, "OK"


def parse_gss(gss: str):
    m = GSS_RE.match(gss)
    if not m:
        return False, None, f"group_size_schedule {gss!r} does not match {GSS_RE.pattern}"
    return True, int(m.group(1)), "OK"


def audit_one(manifest_path: Path):
    """Audit a manifest, returning per-check pass/fail. The `m` dict is
    left attached for the perturbation-test path to use."""
    rec = {"manifest": manifest_path.name, "_manifest_obj": None}
    try:
        m = json.loads(manifest_path.read_text())
        rec["_manifest_obj"] = m
    except Exception as e:
        rec["status"] = "FAIL_LOAD"
        rec["reason"] = str(e)
        return rec
    cid = m.get("cell_id", "")
    rec["cell_id"] = cid

    # CHECK 1: cell_id parses to (model, task, G, T, S, hash)
    ok, parsed, why = parse_cell_id(cid)
    rec["check1_cellid_parses"] = "PASS" if ok else f"FAIL:{why}"
    if not ok:
        # cannot do further checks
        rec["check2_hash_unique"] = "SKIP"
        rec["check3_tensor_exists"] = "SKIP"
        rec["check4_tensor_G_match"] = "SKIP"
        rec["check5_tensor_model_match"] = "SKIP"
        rec["check6_heldout_matches_task"] = "SKIP"
        rec["check7_decontam_matches_task"] = "SKIP"
        rec["check8_gss_matches_G"] = "SKIP"
        return rec

    # CHECK 2: 10-hex hash uniqueness: does the same hash appear in the
    # linked tensor filename? Manifest filename already uses cell_id
    # (with hash). Tensor filename also uses cell_id. So the check is
    # whether the manifest's cell_id matches the basename of the
    # tensor file under TENS_DIR.
    expected_tensor_basename = cid + ".json"
    actual_tensor_basename = Path(m.get("per_step_zvf_path", "")).name
    rec["check2_hash_unique"] = (
        "PASS" if expected_tensor_basename == actual_tensor_basename
        else f"FAIL:tensor basename={actual_tensor_basename!r} vs manifest={expected_tensor_basename!r}"
    )

    # CHECK 3: per_step_zvf_path points to an existing file
    p = m.get("per_step_zvf_path", "")
    rec["check3_tensor_exists"] = (
        "PASS" if p and os.path.exists(p) else f"FAIL:path={p!r}"
    )

    # CHECK 4 & 5: load tensor if exists
    tensor_cell = None
    if rec["check3_tensor_exists"] == "PASS":
        try:
            t = json.loads(Path(p).read_text())
            tensor_cell = t.get("cell", {})
        except Exception as e:
            rec["check3_tensor_exists"] = f"FAIL:load_error={e}"
    if tensor_cell:
        tg = tensor_cell.get("group_size")
        tm = tensor_cell.get("model")
        rec["check4_tensor_G_match"] = (
            "PASS" if tg == parsed["G"]
            else f"FAIL:tensor G={tg} vs cell_id G={parsed['G']}"
        )
        # Naming-convention check: cell_id encodes model as e.g.
        # "meta-llama-Llama-3-2-3B" (slashes replaced with dashes,
        # dots replaced with dashes). Tensor cell.model is the
        # canonical HuggingFace handle e.g. "meta-llama/Llama-3.2-3B".
        # We measure BOTH (a) strict string equality (will detect the
        # systematic drift), and (b) canonicalized equality (the
        # schema-ground-truth verdict).
        def _canon(s: str) -> str:
            return s.lower().replace("/", "-").replace(".", "-")
        if not tm:
            rec["check5_tensor_model_match_strict"] = "FAIL:tensor cell.model missing"
            rec["check5_tensor_model_match_canon"] = "FAIL:tensor cell.model missing"
            rec["check5_tensor_model_match"] = "FAIL:tensor cell.model missing"
        else:
            strict_ok = (parsed["model"] == tm)
            canon_ok = (_canon(parsed["model"]) == _canon(tm))
            rec["check5_tensor_model_match_strict"] = (
                "PASS" if strict_ok
                else f"FAIL:naming_drift cell_id={parsed['model']!r} vs tensor={tm!r}"
            )
            rec["check5_tensor_model_match_canon"] = (
                "PASS" if canon_ok
                else f"FAIL:canon cell_id={_canon(parsed['model'])!r} vs tensor={_canon(tm)!r}"
            )
            rec["check5_tensor_model_match"] = (
                rec["check5_tensor_model_match_canon"]  # canonical is the authoritative verdict
            )
    else:
        rec["check4_tensor_G_match"] = "SKIP"
        rec["check5_tensor_model_match"] = "SKIP"
        rec["check5_tensor_model_match_strict"] = "SKIP"
        rec["check5_tensor_model_match_canon"] = "SKIP"

    # CHECK 6: heldout_split matches task encoded in cell_id
    hs = m.get("heldout_split", "")
    rec["check6_heldout_matches_task"] = (
        "PASS" if hs == parsed["task"] else f"FAIL:heldout={hs!r} vs task={parsed['task']!r}"
    )

    # CHECK 7: decontamination_notes matches task family
    dn = m.get("decontamination_notes", "")
    expected_dn = TASK_TO_DECONTAM.get(parsed["task"])
    rec["check7_decontam_matches_task"] = (
        "PASS" if expected_dn is None or dn == expected_dn
        else f"FAIL:decontam={dn!r} vs expected={expected_dn!r} for task={parsed['task']!r}"
    )

    # CHECK 8: group_size_schedule matches declared G (from cell_id)
    gss = m.get("group_size_schedule", "")
    ok_gss, gss_g, _ = parse_gss(gss)
    if ok_gss:
        rec["check8_gss_matches_G"] = (
            "PASS" if gss_g == parsed["G"]
            else f"FAIL:gss_G={gss_g} vs cell_id_G={parsed['G']}"
        )
    else:
        rec["check8_gss_matches_G"] = f"FAIL:{gss_g}"

    return rec


def main():
    files = sorted(MAN_DIR.glob("*.json"))
    n = len(files)
    print(f"Loaded {n} manifest files from {MAN_DIR}")
    rows = [audit_one(p) for p in files]
    # write TSV
    check_keys = [f"check{i}_{name}" for i, name in enumerate(
        ["cellid_parses", "hash_unique", "tensor_exists",
         "tensor_G_match", "tensor_model_match", "heldout_matches_task",
         "decontam_matches_task", "gss_matches_G"], start=1)]
    cols = ["manifest", "cell_id"] + check_keys + [
        "check5_tensor_model_match_strict",
        "check5_tensor_model_match_canon",
    ]
    with OUT_TSV.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"Wrote {OUT_TSV}")

    # aggregate
    summary = {"n_manifests": n, "checks": {}}
    pass_counts = collections.Counter()
    fail_counts = collections.Counter()
    skip_counts = collections.Counter()
    fail_examples = collections.defaultdict(list)
    for r in rows:
        for c in check_keys:
            v = str(r.get(c, ""))
            if v == "PASS":
                pass_counts[c] += 1
            elif v.startswith("FAIL"):
                fail_counts[c] += 1
                if len(fail_examples[c]) < 5:
                    fail_examples[c].append({"manifest": r["manifest"], "reason": v})
            elif v == "SKIP":
                skip_counts[c] += 1
            else:
                fail_counts[c] += 1
                if len(fail_examples[c]) < 5:
                    fail_examples[c].append({"manifest": r["manifest"], "reason": v})
    for c in check_keys:
        summary["checks"][c] = {
            "pass": pass_counts[c],
            "fail": fail_counts[c],
            "skip": skip_counts[c],
            "fail_examples": fail_examples[c],
        }

    # overall pass rate across non-skipped checks
    non_skip_total = pass_counts.total() + fail_counts.total()
    summary["overall_pass_rate"] = (
        pass_counts.total() / non_skip_total if non_skip_total else 0.0
    )
    summary["overall_pass_count"] = pass_counts.total()
    summary["overall_fail_count"] = fail_counts.total()
    summary["overall_skip_count"] = skip_counts.total()
    summary["total_checks"] = n * len(check_keys)

    # universal-pass manifests (PASS on every applicable check)
    fully_pass = 0
    for r in rows:
        if all(str(r.get(c, "")) == "PASS" for c in check_keys):
            fully_pass += 1
    summary["fully_consistent_manifests"] = fully_pass
    summary["fully_consistent_rate"] = fully_pass / n if n else 0.0

    # PERTURBATION TEST (non-vacuity check) — modify 5 random manifests
    # by swapping G, temperature, task_slice, or heldout_split to a
    # DIFFERENT valid value, then re-audit. Detect-rate should be
    # 100% (every perturbation produces at least one FAIL on the
    # non-canonical checks). This proves the audit is not a vacuous
    # ceiling.
    import random as _r
    _r.seed(20260705)
    perturb_rows = []
    perturb_detect = 0
    perturb_total = 0
    valid_tasks = ["gsm8k_easy", "gsm8k_hard", "humaneval_subset"]
    valid_G = [2, 4, 8, 16, 32]
    valid_T = [0.6, 1.0]
    perturb_specs = []  # (manifest_idx, field, new_value)
    rng = _r.Random(20260705)
    sample_indices = rng.sample(range(n), min(20, n))
    perturbation_kinds = [
        "heldout_swap", "gss_swap", "decontam_swap", "path_swap",
    ]
    for idx in sample_indices:
        rec = rows[idx]
        m = rec.get("_manifest_obj")
        if not m:
            continue
        kind = rng.choice(perturbation_kinds)
        cid = m.get("cell_id", "")
        ok, parsed, _ = parse_cell_id(cid)
        if not ok:
            continue
        if kind == "heldout_swap":
            others = [t for t in valid_tasks if t != parsed["task"]]
            new = rng.choice(others)
            perturb_specs.append((idx, "heldout_split", new))
        elif kind == "gss_swap":
            others = [g for g in valid_G if g != parsed["G"]]
            new_g = rng.choice(others)
            perturb_specs.append((idx, "group_size_schedule", f"fixed-G={new_g}"))
        elif kind == "decontam_swap":
            # pick a decontam value that's WRONG for the current task
            other_decontams = [
                d for t, d in TASK_TO_DECONTAM.items() if t != parsed["task"]
                for d in [TASK_TO_DECONTAM[t]]
            ]
            other_decontams = list(set(other_decontams))
            new = rng.choice(other_decontams)
            perturb_specs.append((idx, "decontamination_notes", new))
        elif kind == "path_swap":
            # point to a non-existent path
            perturb_specs.append((idx, "per_step_zvf_path", "/tmp/does-not-exist.json"))

    for idx, field, new in perturb_specs:
        # build perturbed manifest in-memory and re-audit
        m = dict(rows[idx]["_manifest_obj"])
        m[field] = new
        # write to a tmp file, audit, delete
        tmp = REPO / f"platform_hybrid/experiments/results/p5p8/.tmp_perturb_{idx}.json"
        tmp.write_text(json.dumps(m))
        audited = audit_one(tmp)
        tmp.unlink()
        audited.pop("_manifest_obj", None)
        audited["perturb_field"] = field
        audited["perturb_value"] = new
        audited["original_manifest"] = rows[idx]["manifest"]
        perturb_rows.append(audited)
        perturb_total += 1
        # detect if ANY of the canonical checks is FAIL (string starts with "FAIL:")
        canon_fail = any(
            isinstance(audited.get(c), str) and audited[c].startswith("FAIL")
            for c in check_keys
        )
        if canon_fail:
            perturb_detect += 1

    summary["perturbation_test"] = {
        "n_perturbations": perturb_total,
        "n_detected": perturb_detect,
        "detect_rate": perturb_detect / perturb_total if perturb_total else 0.0,
        "examples": perturb_rows[:10],
    }

    # NAMING-DRIFT FINDING — measure how many would FAIL without
    # canonicalization (i.e. on naive string equality).
    strict_fail_count = sum(
        1 for r in rows
        if str(r.get("check5_tensor_model_match_strict", "")).startswith("FAIL")
    )
    summary["naming_drift_without_canonicalization"] = {
        "n_strict_fail": strict_fail_count,
        "n_strict_pass": n - strict_fail_count,
        "strict_fail_rate": strict_fail_count / n if n else 0.0,
        "canonicalization_rule": "lower-case + replace / with - + replace . with -",
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {OUT_JSON}")
    print(f"Pass: {pass_counts.total()}, Fail: {fail_counts.total()}, Skip: {skip_counts.total()}")
    print(f"Fully consistent manifests: {fully_pass} / {n}  "
          f"({summary['fully_consistent_rate']*100:.1f}%)")
    for c in check_keys:
        pc = pass_counts[c]; fc = fail_counts[c]; sc = skip_counts[c]
        print(f"  {c}: PASS={pc}  FAIL={fc}  SKIP={sc}")
    print(f"\nNaming-drift (without canonicalization) check5_strict: "
          f"FAIL={strict_fail_count}/{n}  "
          f"({summary['naming_drift_without_canonicalization']['strict_fail_rate']*100:.1f}%)")
    print(f"Perturbation detection: {perturb_detect}/{perturb_total}  "
          f"({summary['perturbation_test']['detect_rate']*100:.1f}%)")


if __name__ == "__main__":
    sys.exit(main())