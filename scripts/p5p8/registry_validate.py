#!/usr/bin/env python3
"""P6 Registry Validation: schema check + measured variant deltas from N2 tensors.

Three deliverables:
  1. experiments/results/p5p8/registry_schema_check.tsv
     - one row per registry/entries/*.json
     - PASS / FAIL on schema validation + a leaf-coverage table per MIN-REPORT item
  2. experiments/results/p5p8/registry_measured_deltas.tsv
     - one row per (method_pair) on N2 reward tensors (same stack, G=8, seed=0)
     - measured delta_reward_mean, delta_zvf, delta_loss, paired bootstrap CI
  3. experiments/results/p5p8/registry_measured_deltas.json
     - machine-readable dump (also consumed by paper_P6 §measured-evidence patch)

This is the P6 T3 item: validate entries against what the N2 four-method run
actually logged. Per the P5P8 brief, prototype on real data.
"""
import argparse
import csv
import json
import math
import pathlib
import statistics
import sys

import jsonschema

WORKTREE = pathlib.Path("/home/claude/tinker-rl-lab-minimax")
REGISTRY = WORKTREE / "registry"
RESULTS = WORKTREE / "experiments" / "results" / "p5p8"
N2 = WORKTREE / "experiments" / "results" / "n2_reward_tensor_resume"
MIN_REPORT_ITEMS = ["loss_form", "reference_kl", "sampler_backend", "telemetry",
                    "group_size_schedule", "heldout_split", "decontamination"]


# ---------------------------------------------------------------------------
# 1. Schema validation + per-leaf coverage table
# ---------------------------------------------------------------------------
def leaf_values(d):
    """Yield (path, value) for every leaf in a (possibly nested) dict."""
    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in leaf_values(v):
                yield f"{k}.{k2}", v2
        else:
            yield k, v


def coverage(d):
    """Fraction of leaf fields in a MIN-REPORT item that are non-null."""
    leaves = list(leaf_values(d))
    if not leaves:
        return 0.0, 0, 0
    nonnull = sum(1 for _, v in leaves if v is not None)
    return nonnull / len(leaves), nonnull, len(leaves)


def schema_check():
    schema = json.loads((REGISTRY / "schema.json").read_text())
    out_rows = []
    for p in sorted((REGISTRY / "entries").glob("*.json")):
        rec = json.loads(p.read_text())
        rid = rec.get("id", p.stem)
        rec_type = rec.get("record_type", "?")
        # try schema
        try:
            jsonschema.validate(rec, schema)
            ok = "PASS"
            err = ""
        except jsonschema.ValidationError as e:
            ok = "FAIL"
            err = str(e.message)[:80]
        # coverage (stack records only)
        cov_per_item = {}
        if rec_type == "stack" and "min_report" in rec:
            for it in MIN_REPORT_ITEMS:
                c, n, t = coverage(rec["min_report"].get(it, {}))
                cov_per_item[it] = (c, n, t)
            total = sum(c for c, _, _ in cov_per_item.values()) / len(MIN_REPORT_ITEMS)
            badge = round(100 * total)
        else:
            cov_per_item = {it: (1.0, 0, 0) for it in MIN_REPORT_ITEMS}
            badge = 100 if rec_type == "variant_delta" else 0
        out_rows.append({
            "id": rid,
            "record_type": rec_type,
            "schema": ok,
            "badge": badge,
            "error": err,
            **{f"cov_{it}": f"{cov_per_item[it][1]}/{cov_per_item[it][2]}" for it in MIN_REPORT_ITEMS},
        })
    return out_rows


# ---------------------------------------------------------------------------
# 2. Measured variant deltas from N2 tensors
# ---------------------------------------------------------------------------
def load_n2_metrics():
    """Parse the n2_metrics.tsv file: per-(method,seed,step) scalar rollouts."""
    metrics = {}
    with (N2 / "n2_metrics.tsv").open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            key = (row["method"], int(row["seed"]), int(row["step"]))
            metrics[key] = {
                "zvf": float(row["zvf"]),
                "reward_mean": float(row["reward_mean"]),
                "loss": float(row["loss"]),
                "mean_len": float(row["mean_len"]),
                "cv_len": float(row["cv_len"]),
                "frac_all_zero": float(row["frac_all_zero"]),
                "frac_all_one": float(row["frac_all_one"]),
                "pcd": float(row["pcd"]),
                "larq": float(row["larq"]),
            }
    return metrics


def load_n2_tensor_prompts():
    """For each method/seed/step, re-derive per-step *per-prompt* mean reward from
    the JSONL tensors (this is what each method actually saw as a training signal
    summary, NOT the scalar reward_mean column). Returns a dict keyed by
    (method, seed, step) -> list[float] of per-prompt mean rewards.

    Why this is useful: the same-stack delta is more honestly measured at the
    per-prompt reward level than at the per-step scalar level, because the
    scalar already pools over groups with different contrast structures.
    """
    out = {}
    for method in ("grpo", "aero", "gift", "areal"):
        path = N2 / f"{method}_s0_tensors.jsonl"
        with path.open() as f:
            for line in f:
                rec = json.loads(line)
                key = (rec["method"], rec["seed"], rec["step"])
                # per-prompt mean over G rollouts
                out[key] = [sum(g) / len(g) for g in rec["rewards"]]
    return out


def bootstrap_paired_diff(a, b, n_boot=1000, seed=0):
    """Paired bootstrap on the difference a-b over aligned indices. Returns
    (mean, ci_lo, ci_hi, n). a, b must be aligned."""
    n = min(len(a), len(b))
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    diffs = [a[i] - b[i] for i in range(n)]
    mean = sum(diffs) / n
    # deterministic bootstrap using LCG
    state = seed or 1
    rng = []
    for _ in range(n_boot):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        idx = state % n
        rng.append(diffs[idx])
    rng.sort()
    lo = rng[int(0.025 * n_boot)]
    hi = rng[int(0.975 * n_boot) - 1]
    return mean, lo, hi, n


def measured_deltas(metrics):
    """Compute same-stack GRPO vs (AERO/GIFT/AREAL) deltas on last-10 steps."""
    # align per-step across methods by step number (seed 0 only, all methods use s0)
    methods = ["grpo", "aero", "gift", "areal"]
    steps = sorted({k[2] for k in metrics if k[0] == "grpo" and k[1] == 0})
    # last 10 steps
    last10 = steps[-10:]
    out = []
    for field in ("reward_mean", "zvf", "loss", "mean_len", "cv_len",
                  "frac_all_zero", "frac_all_one", "pcd", "larq"):
        for other in ("aero", "gift", "areal"):
            a = [metrics[("grpo", 0, s)][field] for s in last10]
            b = [metrics[(other, 0, s)][field] for s in last10]
            mean, lo, hi, n = bootstrap_paired_diff(a, b, n_boot=2000)
            out.append({
                "metric": field,
                "baseline": "grpo",
                "variant": other,
                "n_steps": n,
                "grpo_mean": round(statistics.mean(a), 4),
                "variant_mean": round(statistics.mean(b), 4),
                "paired_delta": round(mean, 4),
                "ci_lo": round(lo, 4),
                "ci_hi": round(hi, 4),
                "ci_excludes_0": "yes" if (lo > 0 or hi < 0) else "no",
            })
    return out


def measured_per_prompt(prompts):
    """Per-prompt reward mean over all steps (each prompt gets many steps; we
    pool by computing each method's mean per prompt across steps it's seen)."""
    by_method_prompt = {m: {} for m in ("grpo", "aero", "gift", "areal")}
    for (method, seed, step), per_prompt in prompts.items():
        if seed != 0:
            continue
        for pi, r in enumerate(per_prompt):
            by_method_prompt[method].setdefault(pi, []).append(r)
    out = []
    for m in ("aero", "gift", "areal"):
        a_means = [sum(by_method_prompt["grpo"][p]) / len(by_method_prompt["grpo"][p])
                   for p in sorted(by_method_prompt["grpo"])
                   if p in by_method_prompt["grpo"]]
        b_means = [sum(by_method_prompt[m][p]) / len(by_method_prompt[m][p])
                   for p in sorted(by_method_prompt["grpo"])
                   if p in by_method_prompt[m]]
        n = min(len(a_means), len(b_means))
        if n == 0:
            continue
        mean, lo, hi, _ = bootstrap_paired_diff(a_means[:n], b_means[:n], n_boot=2000)
        out.append({
            "scope": "per_prompt_pooled",
            "baseline": "grpo",
            "variant": m,
            "n_prompts": n,
            "grpo_mean": round(statistics.mean(a_means), 4),
            "variant_mean": round(statistics.mean(b_means), 4),
            "paired_delta": round(mean, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
            "ci_excludes_0": "yes" if (lo > 0 or hi < 0) else "no",
        })
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help="Write outputs under experiments/results/p5p8/")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)

    print("# 1. Schema validation + MIN-REPORT coverage")
    rows = schema_check()
    cols = ["id", "record_type", "schema", "badge"] + \
           [f"cov_{it}" for it in MIN_REPORT_ITEMS] + ["error"]
    if args.write:
        with (RESULTS / "registry_schema_check.tsv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
            w.writeheader()
            for r in rows:
                w.writerow(r)
    cov_keys = [f"cov_{it}" for it in MIN_REPORT_ITEMS]
    for r in rows:
        cov_str = "  ".join(f"{it}={r[k]}" for it, k in zip(MIN_REPORT_ITEMS, cov_keys))
        print(f"  {r['id']:40s} {r['schema']:4s} badge={r['badge']:3d}  {cov_str}")
    print(f"  ({len(rows)} entries; "
          f"{sum(r['schema']=='PASS' for r in rows)}/{len(rows)} pass)")

    print()
    print("# 2. Measured variant deltas from N2 tensors (seed 0, last 10 steps)")
    metrics = load_n2_metrics()
    deltas = measured_deltas(metrics)
    if args.write:
        with (RESULTS / "registry_measured_deltas.tsv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(deltas[0].keys()), delimiter="\t")
            w.writeheader()
            for d in deltas:
                w.writerow(d)
    for d in deltas:
        print(f"  {d['metric']:18s} grpo vs {d['variant']:5s}  "
              f"Δ={d['paired_delta']:+.4f}  CI=[{d['ci_lo']:+.4f},{d['ci_hi']:+.4f}]  "
              f"sig={d['ci_excludes_0']}")

    print()
    print("# 3. Per-prompt pooled reward-mean deltas")
    prompts = load_n2_tensor_prompts()
    pp = measured_per_prompt(prompts)
    for d in pp:
        print(f"  {d['baseline']:5s} vs {d['variant']:5s} (n={d['n_prompts']:4d} prompts)  "
              f"Δ={d['paired_delta']:+.4f}  CI=[{d['ci_lo']:+.4f},{d['ci_hi']:+.4f}]  "
              f"sig={d['ci_excludes_0']}")

    if args.write:
        out_json = {
            "schema_check_summary": {
                "n_total": len(rows),
                "n_pass": sum(r["schema"] == "PASS" for r in rows),
                "n_stack": sum(r["record_type"] == "stack" for r in rows),
                "n_variant_delta": sum(r["record_type"] == "variant_delta" for r in rows),
                "by_record": rows,
            },
            "measured_deltas_stepwise": deltas,
            "measured_deltas_per_prompt": pp,
            "source": "experiments/results/n2_reward_tensor_resume/{grpo,aero,gift,areal}_s0_tensors.jsonl",
            "note": "All four methods share the same stack (Tinker-managed sampler, G=8, seed 0); "
                    "deltas isolate the variant label, not the runtime.",
        }
        (RESULTS / "registry_measured_deltas.json").write_text(
            json.dumps(out_json, indent=2))
        print(f"\nwrote {RESULTS}/registry_schema_check.tsv")
        print(f"wrote {RESULTS}/registry_measured_deltas.{{tsv,json}}")


if __name__ == "__main__":
    sys.exit(main())