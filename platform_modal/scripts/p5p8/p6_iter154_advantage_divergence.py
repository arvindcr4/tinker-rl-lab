#!/usr/bin/env python3
"""P6 iter-154: per-step advantage-distribution divergence between methods on N2.

Brief vein (a) at the **distribution level**: the registry currently reports
*scalar* measured[] deltas (zvf, reward_mean, pcd, mean_len, zvf_risk_mean).
But the registry also records *code-level* component deltas (advantage_guided
shaping for AERO, gamma-baseline for GIFT, decoupled clipping for AREAL).
The strongest falsifiable test is: do these code-level differences actually
manifest in the **per-step advantage distribution** on the N2 same-stack
tensors? If the methods are algorithmically inert (per P5 same-stack
finding), the per-step advantages should be identical up to finite-precision
noise, and distribution divergences (KL, Wasserstein, JS) should be ≈ 0.

For every step t in N2 (40 steps, G=8, seed 0, 4 methods):
  - Pool the 16x8 = 128 advantages into a 1-D vector per method.
  - Compute the histogram-based divergences:
      * KL(p_var || p_grpo)  with ε-smoothing (epsilon=1e-6)
      * JS  (symmetrised KL on the midpoint)
      * Wasserstein-1  on sorted samples (empirical cdf distance)
  - Compare to scalar measured deltas in platform_hybrid/registry/entries/delta_*.json:
      * `delta.zvf`            (registry measured[zvf, panel=n2_same_stack_last10])
      * `delta.reward_mean`    (registry measured[reward_mean, panel=...] )
  - Aggregate per-method summary statistics across steps (mean, sd, max,
    fraction-of-steps-with-d_KL > 0.01).

Outputs (platform_hybrid/experiments/results/p5p8/):
  - p6_iter154_adv_div_per_step.tsv   (one row per (method, step))
  - p6_iter154_adv_div_summary.json   (per-method aggregates + scalars)
  - p6_iter154_adv_div_vs_scalar.tsv  (cross-checks adv-div vs registry deltas)

Stdlib + numpy only.
"""
import json
import pathlib
import statistics
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
TENSORS = ROOT / "platform_hybrid/experiments/results/n2_reward_tensor_resume"
REG_ENTRIES = ROOT / "platform_hybrid/registry/entries"
OUT = ROOT / "platform_hybrid/experiments/results/p5p8"
OUT.mkdir(parents=True, exist_ok=True)

METHODS = ["grpo", "aero", "gift", "areal"]
EPS = 1e-6
N_BINS = 25  # quantile-binned histogram (avoids empty bins)
RNG = np.random.default_rng(20260705)


# -------------------------------------------------------------------------
# 1. Load tensors
# -------------------------------------------------------------------------
def load_tensors(method: str) -> list[dict]:
    p = TENSORS / f"{method}_s0_tensors.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines()]


def to_advantage_array(records: list[dict]) -> np.ndarray:
    """Stack all per-step advantages into a (n_steps, n_prompts*G) array."""
    out = []
    for r in records:
        a = np.array(r["advantages"], dtype=np.float64)
        out.append(a.reshape(-1))
    return np.stack(out, axis=0)


# -------------------------------------------------------------------------
# 2. Divergence functions (1-D)
# -------------------------------------------------------------------------
def kl_div(p: np.ndarray, q: np.ndarray, eps: float = EPS) -> float:
    """KL(p || q) on normalised histograms. Sym uses log2."""
    p = (p + eps) / (p + eps).sum()
    q = (q + eps) / (q + eps).sum()
    return float(np.sum(p * (np.log(p) - np.log(q))) / np.log(2.0))


def js_div(p: np.ndarray, q: np.ndarray, eps: float = EPS) -> float:
    m = 0.5 * (p + q)
    return 0.5 * (kl_div(p, m, eps) + kl_div(q, m, eps))


def wasserstein_1(samples_p: np.ndarray, samples_q: np.ndarray) -> float:
    """Empirical 1-Wasserstein distance on sorted samples (sliced 1-D)."""
    n = max(len(samples_p), len(samples_q))
    a = np.sort(samples_p)
    b = np.sort(samples_q)
    # Interpolate to same length
    a_x = np.linspace(0, 1, len(a))
    b_x = np.linspace(0, 1, len(b))
    grid = np.linspace(0, 1, n)
    a_i = np.interp(grid, a_x, a)
    b_i = np.interp(grid, b_x, b)
    return float(np.mean(np.abs(a_i - b_i)))


def hist_binned(samples: np.ndarray, edges: np.ndarray) -> np.ndarray:
    h, _ = np.histogram(samples, bins=edges)
    return h.astype(np.float64)


# -------------------------------------------------------------------------
# 3. Per-step divergence
# -------------------------------------------------------------------------
def compute_per_step_div():
    rows = []
    arrays = {m: to_advantage_array(load_tensors(m)) for m in METHODS}
    n_steps = arrays["grpo"].shape[0]
    assert all(arrays[m].shape[0] == n_steps for m in METHODS), \
        "all methods must have same step count"

    # Use a shared bin grid spanning the union of all advantage ranges
    all_advs = np.concatenate([arrays[m].reshape(-1) for m in METHODS], axis=0)
    edges = np.quantile(all_advs, np.linspace(0, 1, N_BINS + 1))
    edges[0] -= 1e-6
    edges[-1] += 1e-6

    for s in range(n_steps):
        a_grpo = arrays["grpo"][s]
        for m in METHODS:
            if m == "grpo":
                rows.append({
                    "method": m, "step": s,
                    "kl_to_grpo_bits": 0.0,
                    "js_to_grpo_bits": 0.0,
                    "wass1_to_grpo": 0.0,
                    "adv_mean_diff": 0.0,
                    "adv_var_ratio": 1.0,
                    "n_advs_grpo": len(a_grpo),
                    "n_advs_var": len(a_grpo),
                    "adv_mean_grpo": float(a_grpo.mean()),
                    "adv_mean_var": float(a_grpo.mean()),
                })
                continue
            a_var = arrays[m][s]
            p_h = hist_binned(a_grpo, edges)
            q_h = hist_binned(a_var, edges)
            kl = kl_div(p_h, q_h)
            js = js_div(p_h, q_h)
            w1 = wasserstein_1(a_grpo, a_var)
            rows.append({
                "method": m, "step": s,
                "kl_to_grpo_bits": kl,
                "js_to_grpo_bits": js,
                "wass1_to_grpo": w1,
                "adv_mean_diff": float(a_var.mean() - a_grpo.mean()),
                "adv_var_ratio": float(a_var.var() / max(a_grpo.var(), 1e-12)),
                "n_advs_grpo": len(a_grpo),
                "n_advs_var": len(a_var),
                "adv_mean_grpo": float(a_grpo.mean()),
                "adv_mean_var": float(a_var.mean()),
            })
    return rows


# -------------------------------------------------------------------------
# 4. Per-method summary
# -------------------------------------------------------------------------
def summarise(rows: list[dict]) -> dict:
    summary = {}
    for m in METHODS:
        if m == "grpo":
            summary[m] = {
                "n_steps": 0,
                "mean_kl_bits": 0.0,
                "sd_kl_bits": 0.0,
                "max_kl_bits": 0.0,
                "frac_steps_kl_gt_0.01": 0.0,
                "mean_js_bits": 0.0,
                "mean_wass1": 0.0,
                "max_wass1": 0.0,
                "mean_adv_var_ratio": 1.0,
                "sd_adv_var_ratio": 0.0,
                "mean_adv_mean_diff": 0.0,
                "max_abs_adv_mean_diff": 0.0,
                "baseline": True,
            }
            continue
        mr = [r for r in rows if r["method"] == m]
        kls = np.array([r["kl_to_grpo_bits"] for r in mr])
        jss = np.array([r["js_to_grpo_bits"] for r in mr])
        w1s = np.array([r["wass1_to_grpo"] for r in mr])
        ratios = np.array([r["adv_var_ratio"] for r in mr])
        diffs = np.array([r["adv_mean_diff"] for r in mr])
        summary[m] = {
            "n_steps": len(mr),
            "mean_kl_bits": float(kls.mean()),
            "sd_kl_bits": float(kls.std(ddof=1)) if len(kls) > 1 else 0.0,
            "max_kl_bits": float(kls.max()),
            "frac_steps_kl_gt_0.01": float((kls > 0.01).mean()),
            "frac_steps_kl_gt_0.001": float((kls > 0.001).mean()),
            "mean_js_bits": float(jss.mean()),
            "mean_wass1": float(w1s.mean()),
            "max_wass1": float(w1s.max()),
            "mean_adv_var_ratio": float(ratios.mean()),
            "sd_adv_var_ratio": float(ratios.std(ddof=1)) if len(ratios) > 1 else 0.0,
            "mean_adv_mean_diff": float(diffs.mean()),
            "max_abs_adv_mean_diff": float(np.abs(diffs).max()),
            "baseline": False,
        }
    return summary


# -------------------------------------------------------------------------
# 5. Cross-check against registry's scalar measured deltas
# -------------------------------------------------------------------------
def load_registry_scalar_deltas() -> dict:
    """Pull the per-method scalar deltas in the registry's measured[].

    Returns: {method: {"zvf_n2_last10": float|None, "reward_mean_n2_last10": float|None,
                       "zvf_risk_mean_5seed": float|None, "panel_list": list[str]}}
    """
    out = {}
    for m in ["aero", "gift", "areal"]:
        p = REG_ENTRIES / f"delta_{m}.json"
        rec = json.loads(p.read_text())
        d = {"measured_rows": []}
        for row in rec.get("measured", []):
            d["measured_rows"].append({
                "metric": row["metric"],
                "panel": row["panel"],
                "delta": row.get("delta"),
                "significant": row.get("significant"),
                "ci_low": row.get("ci_low"),
                "ci_high": row.get("ci_high"),
            })
        out[m] = d
    return out


def cross_check(per_step: list[dict], summary: dict, registry: dict) -> list[dict]:
    rows = []
    for m in ["aero", "gift", "areal"]:
        # Aggregate adv-div across all 40 steps (full panel)
        mr = [r for r in per_step if r["method"] == m]
        full_kl = float(np.mean([r["kl_to_grpo_bits"] for r in mr]))
        # Last 10 steps (matches registry's n2_same_stack_last10 panel)
        last10 = mr[-10:]
        last10_kl = float(np.mean([r["kl_to_grpo_bits"] for r in last10]))
        last10_w1 = float(np.mean([r["wass1_to_grpo"] for r in last10]))
        last10_var_ratio = float(np.mean([r["adv_var_ratio"] for r in last10]))

        # Pull registry's scalar deltas
        scalar_zvf = next((r for r in registry[m]["measured_rows"]
                           if r["metric"] == "zvf" and r["panel"] == "n2_same_stack_last10"),
                          {})
        scalar_rew = next((r for r in registry[m]["measured_rows"]
                           if r["metric"] == "reward_mean" and r["panel"] == "n2_same_stack_last10"),
                          {})

        rows.append({
            "method": m,
            "adv_kl_full40_mean": full_kl,
            "adv_kl_last10_mean": last10_kl,
            "adv_w1_last10_mean": last10_w1,
            "adv_var_ratio_last10_mean": last10_var_ratio,
            "registry_zvf_delta": scalar_zvf.get("delta"),
            "registry_zvf_sig": scalar_zvf.get("significant"),
            "registry_reward_mean_delta": scalar_rew.get("delta"),
            "registry_reward_mean_sig": scalar_rew.get("significant"),
            # Interpretation: adv divergence at this magnitude indicates
            # whether the variant has any *distributional* effect on the same
            # step (anything > 1e-3 bits is meaningful vs the bin noise floor)
            "adv_div_inert_flag": last10_kl < 1e-3,
        })
    return rows


# -------------------------------------------------------------------------
# 6. Write outputs
# -------------------------------------------------------------------------
def write_per_step_tsv(rows: list[dict]) -> pathlib.Path:
    cols = ["method", "step",
            "kl_to_grpo_bits", "js_to_grpo_bits", "wass1_to_grpo",
            "adv_mean_diff", "adv_var_ratio",
            "n_advs_grpo", "n_advs_var",
            "adv_mean_grpo", "adv_mean_var"]
    p = OUT / "p6_iter154_adv_div_per_step.tsv"
    with p.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    return p


def write_vs_scalar_tsv(rows: list[dict]) -> pathlib.Path:
    cols = ["method",
            "adv_kl_full40_mean", "adv_kl_last10_mean", "adv_w1_last10_mean",
            "adv_var_ratio_last10_mean",
            "registry_zvf_delta", "registry_zvf_sig",
            "registry_reward_mean_delta", "registry_reward_mean_sig",
            "adv_div_inert_flag"]
    p = OUT / "p6_iter154_adv_div_vs_scalar.tsv"
    with p.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    return p


def write_summary_json(per_step: list[dict], summary: dict,
                       cross: list[dict]) -> pathlib.Path:
    p = OUT / "p6_iter154_adv_div_summary.json"
    obj = {
        "iter": 154,
        "pillar": "P6",
        "vein": "(a) distribution-level",
        "n_steps": 40,
        "n_methods_compared": 3,
        "n_bins": N_BINS,
        "eps": EPS,
        "per_method_summary": summary,
        "registry_vs_adv_div_cross": cross,
        "headline": {
            "h1_no_adv_distribution_effect_on_same_stack":
                all(c["adv_div_inert_flag"] for c in cross),
            "mean_kl_bits_last10": {
                m: round(summary[m]["mean_kl_bits"] / 10.0, 6) if False
                else round(summary[m]["mean_kl_bits"], 6) if False
                else float(round(
                    statistics.mean(
                        r["kl_to_grpo_bits"]
                        for r in per_step
                        if r["method"] == m
                    ), 6))
                for m in ["aero", "gift", "areal"]
            },
            "mean_kl_bits_full40": {
                m: float(round(
                    statistics.mean(
                        r["kl_to_grpo_bits"]
                        for r in per_step
                        if r["method"] == m
                    ), 6))
                for m in ["aero", "gift", "areal"]
            },
        },
    }
    p.write_text(json.dumps(obj, indent=2))
    return p


def main():
    print(f"[iter154] loading N2 tensors ...", file=sys.stderr)
    per_step = compute_per_step_div()
    summary = summarise(per_step)
    print(f"[iter154] loaded {len(per_step)} per-step rows", file=sys.stderr)

    registry = load_registry_scalar_deltas()
    cross = cross_check(per_step, summary, registry)

    p1 = write_per_step_tsv(per_step)
    p2 = write_vs_scalar_tsv(cross)
    p3 = write_summary_json(per_step, summary, cross)

    # Headline printout
    print("\n=== iter-154 advantage-distribution divergence (last 10 steps) ===",
          file=sys.stderr)
    for c in cross:
        print(f"  {c['method']}: "
              f"KL={c['adv_kl_last10_mean']:.5f} bits  "
              f"W1={c['adv_w1_last10_mean']:.5f}  "
              f"var-ratio={c['adv_var_ratio_last10_mean']:.4f}  "
              f"inert={c['adv_div_inert_flag']}",
              file=sys.stderr)
    print("\nregistry scalar deltas vs grpo (n2_same_stack_last10):",
          file=sys.stderr)
    for c in cross:
        print(f"  {c['method']}: "
              f"zvf Δ={c['registry_zvf_delta']} (sig={c['registry_zvf_sig']}); "
              f"reward_mean Δ={c['registry_reward_mean_delta']} "
              f"(sig={c['registry_reward_mean_sig']})",
              file=sys.stderr)

    print(f"\n[iter154] wrote{p1}", file=sys.stderr)
    print(f"[iter154] wrote {p2}", file=sys.stderr)
    print(f"[iter154] wrote {p3}", file=sys.stderr)


if __name__ == "__main__":
    main()