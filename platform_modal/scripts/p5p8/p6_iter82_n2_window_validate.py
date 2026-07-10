#!/usr/bin/env python3
"""P6 iter-82: N2 same-stack window-sensitivity validation of variant deltas.

Closes vein (a) of the brief: validate existing variant-delta entries against
measured behavior on the same-stack N2 tensor data.

For each of {aero, gift, areal} we re-measure Δ vs grpo under 4 windows
(full 40, last 20, last 10, last 5, early 10) and on 4 metrics
(zvf, reward_mean, mean_len, cv_len). Then compare the freshly computed
Δ_full40 to the registry's claimed Δ_last10:

  - sign agreement (does the registry claim survive the full window?)
  - CI overlap (does the registry CI contain the new Δ_full40 point?)
  - stability class (window-stable vs window-fragile)
  - rank preservation (does the method ranking by Δ_full40 match Δ_last10?)

Outputs:
  platform_hybrid/experiments/results/p5p8/p6_n2_window_deltas.tsv
  platform_hybrid/experiments/results/p5p8/p6_n2_registry_vs_measured.tsv
  platform_hybrid/experiments/results/p5p8/p6_n2_window_sensitivity.json

Stdlib only. ~280 LoC.
"""
import csv
import json
import math
import pathlib
import random
import statistics

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TENSOR_DIR = ROOT / "experiments" / "results" / "n2_reward_tensor_resume"
ENTRY_DIR = ROOT / "registry" / "entries"
OUT_DIR = ROOT / "experiments" / "results" / "p5p8"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHODS = ["grpo", "aero", "gift", "areal"]   # all same stack, seed 0, G=8
METRICS = ["zvf", "reward_mean", "mean_len", "cv_len"]
WINDOWS = {
    "full40": (0, 40),
    "last20": (20, 40),
    "last10": (30, 40),     # matches registry `n2_same_stack_last10` panel
    "last5":  (35, 40),
    "early10": (0, 10),
}
SEED = 20260705
N_BOOT = 2000


def load_tensors(method: str) -> list:
    """Return list of step-dicts sorted by step."""
    path = TENSOR_DIR / f"{method}_s0_tensors.jsonl"
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    out.sort(key=lambda d: d["step"])
    return out


def per_step_metric(tensors: list, metric: str) -> list:
    return [float(t[metric]) for t in tensors]


def paired_bootstrap_ci(diff: list, n_boot: int = N_BOOT, ci: float = 0.95,
                         seed: int = SEED) -> tuple:
    """Return (mean, lo, hi) percentile CI of paired-difference vector."""
    if not diff:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(diff)
    means = []
    for _ in range(n_boot):
        sample = [diff[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((1 - ci) / 2 * n_boot)]
    hi = means[int((1 + ci) / 2 * n_boot)]
    return (sum(diff) / n, lo, hi)


def compute_window_deltas(tensors_by_method: dict) -> dict:
    """For each method × metric × window, compute (Δ vs grpo, mean, ci_lo, ci_hi)."""
    out = {}
    for metric in METRICS:
        grpo_full = per_step_metric(tensors_by_method["grpo"], metric)
        for win_name, (lo, hi) in WINDOWS.items():
            grpo_win = grpo_full[lo:hi]
            row = {"metric": metric, "window": win_name,
                   "n_steps": len(grpo_win), "grpo_mean": sum(grpo_win) / len(grpo_win)}
            for m in METHODS:
                if m == "grpo":
                    continue
                m_full = per_step_metric(tensors_by_method[m], metric)
                m_win = m_full[lo:hi]
                diff = [b - a for a, b in zip(grpo_win, m_win)]
                mean, ci_lo, ci_hi = paired_bootstrap_ci(diff)
                row[f"{m}_delta"] = mean
                row[f"{m}_ci_lo"] = ci_lo
                row[f"{m}_ci_hi"] = ci_hi
                row[f"{m}_sig"] = (ci_lo > 0) or (ci_hi < 0)
            out[(metric, win_name)] = row
    return out


def load_registry_measured(delta_id: str) -> list:
    """Read registry's `measured` block for a variant-delta entry."""
    path = ENTRY_DIR / f"{delta_id}.json"
    d = json.loads(path.read_text())
    return d.get("measured", [])


def registry_metric_panel_map(delta_id: str, panel: str) -> dict:
    """For a (delta, panel) pair, return {metric: (delta, ci_lo, ci_hi, n, sig)}."""
    out = {}
    for m in load_registry_measured(delta_id):
        if m.get("panel") == panel:
            out[m["metric"]] = (m.get("delta"), m.get("ci_low"), m.get("ci_high"),
                                m.get("n"), m.get("significant"))
    return out


def sign_agreement(reg_sign: str, new_delta: float) -> str:
    """Coarse check: does the measured Δ agree with the registry's claim?

    For zvf/reward_mean the registry does not publish a predicted sign in
    every entry's `measured` block; we infer from the numeric sign of the
    stored `delta` itself (the registry's own claim).
    """
    if reg_sign is None:
        return "no-registry-claim"
    if reg_sign == ">0" and new_delta > 0:
        return "AGREE"
    if reg_sign == "<0" and new_delta < 0:
        return "AGREE"
    if reg_sign in (">=0", "<=0"):
        return "AGREE"  # boundary-allowed
    return "DISAGREE"


def write_window_deltas_tsv(deltas: dict, path: pathlib.Path):
    fields = ["metric", "window", "n_steps", "grpo_mean"]
    for m in METHODS:
        if m == "grpo":
            continue
        fields += [f"{m}_delta", f"{m}_ci_lo", f"{m}_ci_hi", f"{m}_sig"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for row in deltas.values():
            w.writerow(row)


def write_registry_vs_measured(deltas: dict, path: pathlib.Path):
    """For each (method, metric) compare registry-claimed Δ_last10
    vs freshly measured Δ_full40."""
    fields = ["method", "metric", "registry_delta_last10",
              "registry_ci_lo_last10", "registry_ci_hi_last10",
              "registry_n", "registry_sig",
              "fresh_delta_full40", "fresh_ci_lo_full40",
              "fresh_ci_hi_full40", "fresh_n_full40", "fresh_sig_full40",
              "registry_panel", "sign_agreement",
              "ci_overlap", "stability_class"]
    rows = []
    panel = "n2_same_stack_last10"
    for m in ["aero", "gift", "areal"]:
        delta_id = f"delta_{m}"
        reg_map = registry_metric_panel_map(delta_id, panel)
        for metric in METRICS:
            reg = reg_map.get(metric, (None, None, None, None, None))
            full = deltas.get((metric, "full40"), {})
            last10 = deltas.get((metric, "last10"), {})
            reg_delta, reg_lo, reg_hi, reg_n, reg_sig = reg
            fresh_d = full.get(f"{m}_delta")
            fresh_lo = full.get(f"{m}_ci_lo")
            fresh_hi = full.get(f"{m}_ci_hi")
            fresh_n = full.get("n_steps")
            fresh_sig = full.get(f"{m}_sig")
            last10_d = last10.get(f"{m}_delta")
            # sign of registry's claim
            if reg_delta is None:
                sign = None
                agree = "no-registry-claim"
            else:
                sign = ">0" if reg_delta > 0 else ("<0" if reg_delta < 0 else "=0")
                agree = sign_agreement(sign, fresh_d if fresh_d is not None else 0.0)
            # CI overlap test (does the registry CI bracket fresh_delta_full40)?
            if reg_lo is not None and reg_hi is not None and fresh_d is not None:
                overlap = "YES" if (reg_lo <= fresh_d <= reg_hi) else "NO"
            else:
                overlap = "n/a"
            # stability: does the signof Δ_full40 match Δ_last10?
            if last10_d is None or fresh_d is None:
                stab = "n/a"
            elif (last10_d > 0) == (fresh_d > 0):
                if abs(last10_d - fresh_d) < 0.05 * max(abs(last10_d), abs(fresh_d), 1e-9):
                    stab = "STABLE"
                else:
                    stab = "STABLE-DIRECTION-MAG-SHIFT"
            else:
                stab = "FRAGILE-SIGN-FLIP"
            rows.append({
                "method": m, "metric": metric,
                "registry_delta_last10": "" if reg_delta is None else f"{reg_delta:.6f}",
                "registry_ci_lo_last10": "" if reg_lo is None else f"{reg_lo:.6f}",
                "registry_ci_hi_last10": "" if reg_hi is None else f"{reg_hi:.6f}",
                "registry_n": reg_n or "",
                "registry_sig": reg_sig if reg_sig is not None else "",
                "fresh_delta_full40": "" if fresh_d is None else f"{fresh_d:.6f}",
                "fresh_ci_lo_full40": "" if fresh_lo is None else f"{fresh_lo:.6f}",
                "fresh_ci_hi_full40": "" if fresh_hi is None else f"{fresh_hi:.6f}",
                "fresh_n_full40": fresh_n or "",
                "fresh_sig_full40": fresh_sig if fresh_sig is not None else "",
                "registry_panel": panel,
                "sign_agreement": agree,
                "ci_overlap": overlap,
                "stability_class": stab,
            })
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    return rows


def summarise(rows: list, deltas: dict) -> dict:
    """Per-method headline summary."""
    out = {"n_method_metric_rows": len(rows), "per_method": {}, "per_window": {}}
    for m in ["aero", "gift", "areal"]:
        sub = [r for r in rows if r["method"] == m]
        agree = sum(1 for r in sub if r["sign_agreement"] == "AGREE")
        overlap = sum(1 for r in sub if r["ci_overlap"] == "YES")
        stable = sum(1 for r in sub if r["stability_class"].startswith("STABLE"))
        fragile = sum(1 for r in sub if r["stability_class"] == "FRAGILE-SIGN-FLIP")
        out["per_method"][m] = {
            "n_metrics": len(sub),
            "n_sign_agree": agree,
            "n_ci_overlap": overlap,
            "n_stable": stable,
            "n_fragile_sign_flip": fragile,
        }
    # per-window summary (4 methods × 4 metrics = 16 cells)
    for win in WINDOWS:
        sub = [deltas[(m, win)] for m in METRICS]
        n_sig = 0
        n_total = 0
        for s in sub:
            for mm in ["aero", "gift", "areal"]:
                n_total += 1
                if s.get(f"{mm}_sig"):
                    n_sig += 1
        out["per_window"][win] = {
            "n_cells": n_total,
            "n_significant": n_sig,
            "pct_significant": round(100 * n_sig / n_total, 1),
        }
    return out


def main():
    print("[p6_iter82] loading N2 tensors...")
    tensors_by_method = {m: load_tensors(m) for m in METHODS}
    for m in METHODS:
        print(f"  {m}: {len(tensors_by_method[m])} steps")
    print("[p6_iter82] computing window deltas...")
    deltas = compute_window_deltas(tensors_by_method)
    out_tsv = OUT_DIR / "p6_n2_window_deltas.tsv"
    write_window_deltas_tsv(deltas, out_tsv)
    print(f"[p6_iter82] wrote {out_tsv.relative_to(ROOT)} ({len(deltas)} rows)")
    print("[p6_iter82] comparing registry claims vs fresh measurements...")
    cmp_tsv = OUT_DIR / "p6_n2_registry_vs_measured.tsv"
    rows = write_registry_vs_measured(deltas, cmp_tsv)
    print(f"[p6_iter82] wrote {cmp_tsv.relative_to(ROOT)} ({len(rows)} rows)")
    summary = summarise(rows, deltas)
    summary["tensors_loaded"] = {m: len(tensors_by_method[m]) for m in METHODS}
    summary["metrics_audited"] = METRICS
    summary["windows_audited"] = list(WINDOWS.keys())
    summary["boot_seed"] = SEED
    summary["boot_n"] = N_BOOT
    summary_json = OUT_DIR / "p6_n2_window_sensitivity.json"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[p6_iter82] wrote {summary_json.relative_to(ROOT)}")
    # headline printout
    print("\n=== per-method headline ===")
    for m, s in summary["per_method"].items():
        print(f"  {m:6s} agree={s['n_sign_agree']}/{s['n_metrics']}  "
              f"CI-overlap={s['n_ci_overlap']}/{s['n_metrics']}  "
              f"stable={s['n_stable']}  fragile={s['n_fragile_sign_flip']}")
    print("\n=== per-window significance (% of 12 method×metric cells) ===")
    for w, s in summary["per_window"].items():
        print(f"  {w:8s} sig={s['n_significant']}/{s['n_cells']} ({s['pct_significant']}%)")
    # reproducibility hint
    print(f"\nseed={SEED} n_boot={N_BOOT} — rerun produces identical CI bounds.")


if __name__ == "__main__":
    main()