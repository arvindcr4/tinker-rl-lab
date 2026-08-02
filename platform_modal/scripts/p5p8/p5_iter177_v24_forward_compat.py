#!/usr/bin/env python3
r"""P5 MIN-REPORT v2.4 -> v2.5 forward-compatibility stress test (iter 177).

Fresh vein, not in 173 prior rows. Closes brief vein (a) at the
**schema-evolution** layer: prior P5 audits (iter-105, iter-117,
iter-121, iter-137, iter-145, iter-153) audited the *current* corpus
against the *current* spec. iter-177 audits what happens when we
**propose** the next schema version (v2.5): for each candidate v2.5
mutation that should make the manifest INCOMPATIBLE with v2.5, does
the v2.4 audit detect it (it shouldn't, because v2.4 doesn't enforce
the new rule), and does the proposed v2.5 audit correctly detect it?

Mutations (each produces a manifest that should FAIL v2.5 audit):
  M1 REMOVE_FIELD   drop sampler_backend_precision (v2.5 still requires it)
  M2 TYPE_VIOLATION heldout_split = 1 (v2.5 requires str)
  M3 VOCAB_VIOLATION heldout_split = "TRAIN-INTERNAL" (not in v2.5 vocab)
  M4 REGEX_VIOLATION group_size_schedule = "fixed-G=8-extra" (invalid)
  M5 NA_SENTINEL    replace "n/a" -> "missing" (v2.5 enforces canonical)

v2.4 audits (existing; iter-145 + iter-153 + iter-105):
  - v24_identifier_stamp: 8 keys + id-bearing field
  - schema_ground_truth: cell_id regex + ^fixed-G=\d+$ on group_size
  - field_coverage_rate: 7-item presence rate

v2.5 audits (proposed in iter-177):
  - v25_required_keys: V24_REQUIRED_KEYS subset (8 keys present)
  - v25_type_strict: heldout_split is str
  - v25_vocab_strict: heldout_split in {gsm8k_easy, gsm8k_hard,
    gsm8k_train, humaneval_subset, math_hard, MATH-Hard}
  - v25_regex_strict: union of v2.4 + v2.5 schedule patterns
  - v25_na_sentinel_strict: n/a sentinels in canonical set

5 falsifiable hypotheses
------------------------
H1 v2.5 audits catch >= 4/5 mutations (each by >=1 v2.5 audit)
H2 v2.4 audits miss >= 3 of {M2, M3, M4, M5} (v2.4 can't see new rules)
H3 v2.5 detection rate strictly > v2.4 detection rate
H4 the BEST v2.5 audit (max single-audit mutations caught) catches >= 4
H5 union of v2.5 audits catches exactly 5/5 mutations (full coverage)

Outputs
-------
- platform_hybrid/experiments/results/p5p8/p5_iter177_mutation_panel.tsv
- platform_hybrid/experiments/results/p5p8/p5_iter177_detection_rates.tsv
- platform_hybrid/experiments/results/p5p8/p5_iter177_v25_spec.tsv
- platform_hybrid/experiments/results/p5p8/p5_iter177_summary.json
"""
from __future__ import annotations
import json
import math
import random
import re
from pathlib import Path

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
RES = ROOT / "experiments" / "results" / "p5p8"
MEGA = ROOT / "experiments" / "results" / "mega_20260704" / "manifests"
RES.mkdir(parents=True, exist_ok=True)

N_SAMPLE = 20
RNG = random.Random(20260705)

V24_REQUIRED_KEYS = {
    "cell_id", "loss_form", "ref_policy_kl", "sampler_backend_precision",
    "per_step_zvf_path", "group_size_schedule", "heldout_split",
    "decontamination_notes",
}
V25_HELDOUT_VOCAB = {
    "gsm8k_easy", "gsm8k_hard", "gsm8k_train",
    "humaneval_subset", "math_hard", "MATH-Hard",
}
V25_NA_CANONICAL = {"n/a", "n/a-sampling", "n/a-parser", "n/a-trainer"}


def wilson(k: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    z = 1.959963984540054
    denom = 1 + z * z / n
    c = (p + z * z / (2 * n)) / denom
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, c - h), min(1.0, c + h)


def load_sample_manifests(n: int = N_SAMPLE) -> tuple[list[dict], list[str]]:
    paths = sorted(MEGA.glob("*.json"))
    chosen = RNG.sample(paths, n)
    out = []
    for p in chosen:
        with open(p) as f:
            out.append(json.load(f))
    return out, [p.name for p in chosen]


# ----- v2.4 audits (existing) -----

def audit_v24_identifier_stamp(m, _src=""):
    """8 required keys present + cell_id + per_step_zvf_path non-empty."""
    if not all(k in m for k in V24_REQUIRED_KEYS):
        return False
    return bool(m.get("cell_id")) and bool(m.get("per_step_zvf_path"))


def audit_schema_ground_truth(m, _src=""):
    cid = m.get("cell_id", "")
    if not re.match(r"^[A-Za-z0-9._\-]+_G\d+_t[\d.]+_s\d+_[0-9a-f]+$", cid):
        return False
    gs = m.get("group_size_schedule", "")
    return bool(re.match(r"^fixed-G=\d+$", gs))


def audit_field_coverage_rate(m, _src=""):
    items = ["loss_form", "ref_policy_kl", "sampler_backend_precision",
             "per_step_zvf_path", "group_size_schedule", "heldout_split",
             "decontamination_notes"]
    n_present = sum(1 for it in items if m.get(it))
    return n_present / len(items)


# ----- v2.5 audits (proposed) -----

def audit_v25_required_keys(m, _src=""):
    """v2.5 keeps the 8 v2.4 keys as required."""
    return all(k in m for k in V24_REQUIRED_KEYS)


def audit_v25_type_strict(m, _src=""):
    """v2.5 enforces types: heldout_split must be str."""
    return isinstance(m.get("heldout_split"), str)


def audit_v25_vocab_strict(m, _src=""):
    """v2.5 enforces heldout_split is in a fixed vocab."""
    return m.get("heldout_split") in V25_HELDOUT_VOCAB


def audit_v25_regex_strict(m, _src=""):
    """v2.5 regex: union of v2.4 + adaptive schedule.
    ^fixed-G=\\d+$  OR  ^schedule-(fixed|adaptive)-G=\\d+(-\\d+)?$"""
    gs = m.get("group_size_schedule", "")
    return bool(re.match(r"^(fixed-G=\d+|schedule-(fixed|adaptive)-G=\d+(-\d+)?)$", gs))


def audit_v25_na_sentinel_strict(m, _src=""):
    """v2.5 enforces canonical n/a sentinel set."""
    for k, v in m.items():
        if isinstance(v, str):
            lv = v.lower().strip()
            # any "na"-ish marker
            if lv in {"missing", "not_applicable", "na", "n.a.",
                      "not-applicable", "n / a", "n.a"}:
                return False
            # "n/a" embedded as substring but not canonical
            if "n/a" in lv and v not in V25_NA_CANONICAL:
                return False
    return True


# ----- mutations (v2.5 incompatibilities) -----

def mut_remove_field(m):
    out = dict(m); out.pop("sampler_backend_precision", None); return out


def mut_type_violation(m):
    out = dict(m); out["heldout_split"] = 1; return out


def mut_vocab_violation(m):
    out = dict(m); out["heldout_split"] = "TRAIN-INTERNAL"; return out


def mut_regex_violation(m):
    out = dict(m); out["group_size_schedule"] = "fixed-G=8-extra"; return out


def mut_na_sentinel(m):
    out = dict(m)
    for k, v in list(out.items()):
        if v == "n/a":
            out[k] = "missing"
    return out


MUTATIONS = [
    ("M1_remove_field", mut_remove_field),
    ("M2_type_violation", mut_type_violation),
    ("M3_vocab_violation", mut_vocab_violation),
    ("M4_regex_violation", mut_regex_violation),
    ("M5_na_sentinel", mut_na_sentinel),
]

V24_AUDITS = [
    ("v24_identifier_stamp", audit_v24_identifier_stamp, "v2.4"),
    ("schema_ground_truth", audit_schema_ground_truth, "v2.4"),
    ("field_coverage_rate", audit_field_coverage_rate, "v2.4"),
]
V25_AUDITS = [
    ("v25_required_keys", audit_v25_required_keys, "v2.5"),
    ("v25_type_strict", audit_v25_type_strict, "v2.5"),
    ("v25_vocab_strict", audit_v25_vocab_strict, "v2.5"),
    ("v25_regex_strict", audit_v25_regex_strict, "v2.5"),
    ("v25_na_sentinel_strict", audit_v25_na_sentinel_strict, "v2.5"),
]
ALL_AUDITS = V24_AUDITS + V25_AUDITS


def run_audit(fn, m) -> float:
    r = fn(m)
    return r if isinstance(r, float) else (1.0 if r else 0.0)


def main():
    manifests, names = load_sample_manifests(N_SAMPLE)
    print(f"Sampled {len(manifests)} manifests (seed=20260705)")

    # Baseline
    baseline = {an: [run_audit(fn, m) for m in manifests]
                for an, fn, _ in ALL_AUDITS}

    # Apply mutations, re-audit
    detection_rows = []
    panel_rows = []
    for mut_name, mut_fn in MUTATIONS:
        mutated = [mut_fn(m) for m in manifests]
        for an, fn, ver in ALL_AUDITS:
            base_vals = baseline[an]
            mut_vals = [run_audit(fn, m) for m in mutated]
            n_base_pass = sum(1 for v in base_vals if v == 1.0)
            n_det = sum(1 for b, mv in zip(base_vals, mut_vals)
                        if b == 1.0 and mv == 0.0)
            p, lo, hi = wilson(n_det, n_base_pass) if n_base_pass else (0.0, 0.0, 0.0)
            detection_rows.append({
                "mutation": mut_name, "audit": an, "version": ver,
                "n_baseline_pass": n_base_pass,
                "baseline_rate": f"{sum(base_vals)/len(base_vals):.3f}",
                "mutated_rate": f"{sum(mut_vals)/len(mut_vals):.3f}",
                "n_detected": n_det,
                "detection_rate": f"{p:.3f}",
                "ci_lo": f"{lo:.3f}", "ci_hi": f"{hi:.3f}",
                "verdict": "DETECTED" if n_det > 0 else "MISSED",
            })
            for i in range(len(manifests)):
                bv = base_vals[i]
                mv = mut_vals[i]
                detected = int(bv == 1.0 and mv == 0.0)
                panel_rows.append({
                    "cell_idx": i, "cell_id": names[i],
                    "mutation": mut_name, "audit": an, "version": ver,
                    "baseline": bv, "mutated": mv, "detected": detected,
                })

    # Write per-cell panel
    panel_tsv = RES / "p5_iter177_mutation_panel.tsv"
    cols = ["cell_idx", "cell_id", "mutation", "audit", "version",
            "baseline", "mutated", "detected"]
    with open(panel_tsv, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in panel_rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")

    # Write detection rates
    detect_tsv = RES / "p5_iter177_detection_rates.tsv"
    cols2 = ["mutation", "audit", "version", "n_baseline_pass",
             "baseline_rate", "mutated_rate", "n_detected",
             "detection_rate", "ci_lo", "ci_hi", "verdict"]
    with open(detect_tsv, "w") as f:
        f.write("\t".join(cols2) + "\n")
        for r in detection_rows:
            f.write("\t".join(str(r[c]) for c in cols2) + "\n")

    # Write v2.5 spec TSV
    spec_tsv = RES / "p5_iter177_v25_spec.tsv"
    with open(spec_tsv, "w") as f:
        f.write("audit\trule\tmutation_caught\n")
        f.write("v25_required_keys\t8 v2.4 keys all present\tM1_remove_field\n")
        f.write("v25_type_strict\theldout_split is str\tM2_type_violation\n")
        f.write("v25_vocab_strict\theldout_split in fixed vocab\tM3_vocab_violation\n")
        f.write("v25_regex_strict\tfixed-G=\\d+ OR schedule-(fixed|adaptive)-G=\\d+(-\\d+)?\tM4_regex_violation\n")
        f.write("v25_na_sentinel_strict\tna sentinels in canonical set\tM5_na_sentinel\n")

    # Summary
    v24_dets = [r for r in detection_rows if r["version"] == "v2.4"]
    v25_dets = [r for r in detection_rows if r["version"] == "v2.5"]
    v24_n_det = sum(1 for r in v24_dets if r["verdict"] == "DETECTED")
    v25_n_det = sum(1 for r in v25_dets if r["verdict"] == "DETECTED")
    v24_rate = v24_n_det / max(len(v24_dets), 1)
    v25_rate = v25_n_det / max(len(v25_dets), 1)

    mut_v24 = {m: 0 for m, _ in MUTATIONS}
    mut_v25 = {m: 0 for m, _ in MUTATIONS}
    for r in detection_rows:
        if r["verdict"] == "DETECTED":
            (mut_v25 if r["version"] == "v2.5" else mut_v24)[r["mutation"]] += 1

    h1_count = sum(1 for m, _ in MUTATIONS if mut_v25[m] >= 1)
    h2_count = sum(1 for m, _ in MUTATIONS
                   if m != "M1_remove_field" and mut_v24[m] == 0)
    h3_pass = v25_rate > v24_rate
    audit_v25_max = {}
    for r in v25_dets:
        if r["verdict"] == "DETECTED":
            audit_v25_max.setdefault(r["audit"], 0)
            audit_v25_max[r["audit"]] += 1
    h4_max = max(audit_v25_max.values()) if audit_v25_max else 0
    h5_pass = all(mut_v25[m] >= 1 for m, _ in MUTATIONS)

    summary = {
        "iter": 177, "pillar": "P5",
        "vein": "MIN-REPORT v2.4 -> v2.5 forward-compatibility stress test",
        "n_sample": N_SAMPLE, "n_mutations": len(MUTATIONS),
        "n_audits": len(ALL_AUDITS), "n_panel_rows": len(panel_rows),
        "v24_detected": v24_n_det, "v24_total": len(v24_dets),
        "v24_rate": v24_rate,
        "v25_detected": v25_n_det, "v25_total": len(v25_dets),
        "v25_rate": v25_rate,
        "per_mutation": {m: {"v24": mut_v24[m], "v25": mut_v25[m]}
                         for m, _ in MUTATIONS},
        "best_v25_audit": max(audit_v25_max, key=audit_v25_max.get) if audit_v25_max else "(none)",
        "best_v25_audit_count": h4_max,
        "hypotheses": {
            "H1_v25_catches_ge_4_mut": {"bar": 4, "actual": h1_count,
                                         "verdict": "PASS" if h1_count >= 4 else "FAIL"},
            "H2_v24_misses_ge_3_new": {"bar": 3, "actual": h2_count,
                                        "verdict": "PASS" if h2_count >= 3 else "FAIL"},
            "H3_v25_strictly_gt_v24": {"v24_rate": v24_rate, "v25_rate": v25_rate,
                                        "verdict": "PASS" if h3_pass else "FAIL"},
            "H4_best_v25_catches_ge_4": {"bar": 4, "actual": h4_max,
                                          "verdict": "PASS" if h4_max >= 4 else "FAIL"},
            "H5_v25_union_full_coverage": {"bar": "5/5", "actual": f"{sum(1 for m,_ in MUTATIONS if mut_v25[m]>=1)}/5",
                                            "verdict": "PASS" if h5_pass else "FAIL"},
        },
    }
    summary_path = RES / "p5_iter177_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print
    print("\n=== Per (mutation, audit) detection ===")
    print(f"{'mutation':<22} {'audit':<28} {'version':<6} {'verdict':<10}")
    for r in detection_rows:
        print(f"{r['mutation']:<22} {r['audit']:<28} {r['version']:<6} {r['verdict']:<10}")

    print(f"\nv2.4 audits: {v24_n_det}/{len(v24_dets)} ({v24_rate:.1%})")
    print(f"v2.5 audits: {v25_n_det}/{len(v25_dets)} ({v25_rate:.1%})")

    print("\n=== Per-mutation caught-by (v2.4 / v2.5) ===")
    for m, _ in MUTATIONS:
        print(f"  {m:<22} v2.4={mut_v24[m]}  v2.5={mut_v25[m]}")

    print("\n=== Hypotheses ===")
    for hk, hv in summary["hypotheses"].items():
        print(f"  {hk}: {hv['verdict']}  bar={hv.get('bar')}  actual={hv.get('actual', hv)}")

    print(f"\n  panel:    {panel_tsv}")
    print(f"  detect:   {detect_tsv}")
    print(f"  spec:     {spec_tsv}")
    print(f"  summary:  {summary_path}")


if __name__ == "__main__":
    main()