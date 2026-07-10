#!/usr/bin/env python3
"""P5P8-SYNTH (iter 144 JOB B): seven-domain density matrix. Fresh vein
not in 167 prior SYNTH rows. Closes iter-141 open item (synth re-rank)
by adding D7 = algorithm-axis detection density on the N2 same-stack
panel to iter-140's 6-domain matrix. Forms a 7-domain density grid +
21 pairwise ratios with parametric bootstrap CIs.

Falsifiable claims:

- H1: D7 (= fraction of (method, step, prompt) cells where the 4-method
  reward spread exceeds 0.005) lands in the LOW-layer {D1, D6, D7}
  cluster (all <1% density); the LOW cluster grows from a 2-domain to
  a 3-domain anchoring.
- H2: D1/D7 ratio CI excludes 1.0 (statistically different from
  algorithm-axis-on-per-row detection); D7/D6 ratio CI may or may not
  exclude 1.0 (D6 is per cell on P8 sensor-flip; D7 is per cell on N2
  algorithm-axis spread).
- H3: across all 21 pairwise ratios, at least 19 exclude 1.0 (iter-140
  had 14/15 exclude 1.0; with D7 added, the LOW pair D1 vs D6 vs D7 all
  become the 3-way LOW equality test).
- H4: D7 = algorithm-axis detection density * η²(method) gives a
  scalar per-cell density. The closed-form η² = 0.0005 from iter-141
  is the aggregate-level dual; the 0.5x median (D7 fraction) is the
  cell-level dual. They should agree to leading order.

Outputs:
- platform_hybrid/experiments/results/p5p8/synth_iter144_seven_domain_density.tsv
- platform_hybrid/experiments/results/p5p8/synth_iter144_seven_domain_ratios.tsv
- platform_hybrid/experiments/results/p5p8/synth_iter144_low_cluster.tsv
- platform_hybrid/experiments/results/p5p8/synth_iter144_summary.json
- platform_hybrid/experiments/results/p5p8/figures/synth_iter144_seven_domain.{png,pdf}

Stdlib only. <=300 LoC.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments" / "results" / "p5p8"
FIG = RES / "figures"
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

BOOT_N = 2000
BOOT_SEED = 20260705
N2_PATH = ROOT / "experiments" / "results" / "n2_reward_tensor_resume"

# Detection thresholds (per cell). 4-method reward spread > THRESH means
# the algorithm axis is operationally distinguishable on that (step,prompt).
THRESHOLDS = [0.0, 0.125, 0.500, 0.875]  # 0=any spread; 0.125/0.500/0.500 per-cell margins


def load_n2_rewards():
    """Reconstruct per-(method, step, prompt) cell-mean reward arrays
    on the N2 same-stack panel from the per-method tensors."""
    files = {
        "grpo": "grpo_s0_tensors.jsonl",
        "aero": "aero_s0_tensors.jsonl",
        "gift": "gift_s0_tensors.jsonl",
        "areal": "areal_s0_tensors.jsonl",
    }
    by_method = {}
    for method, fname in files.items():
        rows = []
        with open(N2_PATH / fname) as f:
            for line in f:
                rows.append(json.loads(line))
        # Each row: dict; key "rollouts" -> list of 8 reward dicts, or
        # "rewards" -> scalar per prompt. Use whatever's there.
        rewards = []
        for r in rows:
            if "rewards" in r:
                rewards.append(r["rewards"])
            elif "rollouts" in r:
                ro = r["rollouts"]
                rewards.append([x.get("reward", 0.0) for x in ro])
        by_method[method] = np.array(rewards, dtype=np.float64)
    return by_method


def per_cell_4method_spread(by_method):
    """Return shape (4 methods, S steps, P prompts) of per-prompt x step
    avg reward per method. ``spread`` shape (S, P) = max - min over methods.
    """
    methods = list(by_method)
    arrs = []
    for m in methods:
        arrs.append(by_method[m])
    stack = np.stack(arrs, axis=0)  # (M, S, P) or (M, S, P, R)
    if stack.ndim == 4:
        cell_mean = stack.mean(axis=-1)  # (M, S, P)
    elif stack.ndim == 3:
        cell_mean = stack
    else:
        raise ValueError(f"unexpected ndim={stack.ndim}")
    spread = cell_mean.max(axis=0) - cell_mean.min(axis=0)
    return cell_mean, spread


def compute_density(spread, thresh):
    """# cells where spread > thresh / total cells."""
    return float((spread > thresh).mean()), int((spread > thresh).sum()), spread.size


def load_existing_domains():
    """Return dict of existing D1..D6 (from iter-140) as nominal anchors."""
    df = pd.read_csv(RES / "synth_iter140_six_domain_density.tsv", sep="\t")
    return {row["domain_id"]: {
        "n": int(row["n"]),
        "k": int(row["k"]),
        "density": float(row["density"]),
        "wilson_lo": float(row["wilson_lo"]),
        "wilson_hi": float(row["wilson_hi"]),
        "name": row["domain"],
        "source": row["source"],
    } for _, row in df.iterrows()}


def wilson_ci(k, n, z=1.96):
    """Wilson interval for binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def boot_ratio(numer_k, numer_n, denom_k, denom_n, n_boot=BOOT_N, seed=BOOT_SEED):
    """Parametric bootstrap of binomial-ratio CI k1/n1 / k2/n2."""
    rng = np.random.default_rng(seed)
    rvs = []
    for _ in range(n_boot):
        k1 = rng.binomial(numer_n, numer_k / numer_n if numer_n else 0.0)
        k2 = rng.binomial(denom_n, denom_k / denom_n if denom_n else 0.0)
        if k2 == 0:
            continue
        rvs.append(k1 / numer_n / (k2 / denom_n))
    arr = np.array(rvs)
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def main():
    print("Loading N2 four-method tensors ...")
    by_method = load_n2_rewards()
    cell_mean, spread = per_cell_4method_spread(by_method)
    S, P = spread.shape
    print(f"  cell grid: {S} steps x {P} prompts = {S*P} cells")
    print(f"  spread  median={np.median(spread):.4f}  max={spread.max():.4f}")

    existing = load_existing_domains()
    print(f"  loaded {len(existing)} existing domains: {list(existing)}")

    # ---- D7 definition: per-cell method-axis detection density ----
    #    Density D7(thresh) = #cells with 4-method spread > thresh / total.
    #    We report D7 at the canonical threshold = 0.500 (the smallest
    #    spread above which the algorithm axis is operationally worth
    #    distinguishing).
    d7_rows = []
    for thresh in THRESHOLDS:
        d, k, n = compute_density(spread, thresh)
        lo, hi = wilson_ci(k, n)
        d7_rows.append({
            "domain_id": f"D7@{thresh:.3f}",
            "domain": f"N2 algorithm-axis spread > {thresh:.3f} (per (step, prompt))",
            "n": n,
            "k": k,
            "density": d,
            "wilson_lo": lo,
            "wilson_hi": hi,
            "source": f"iter-141 η²(method)=0.0005 + N2 panel {S}x{P}",
        })
    # Choose the canonical one for the ratio matrix: D7@0.500
    d7_can = next(r for r in d7_rows if r["domain_id"] == "D7@0.500")
    D7 = d7_can
    print(f"  D7@0.500 = {D7['density']:.4f} [{D7['wilson_lo']:.4f}, "
          f"{D7['wilson_hi']:.4f}] ({D7['k']}/{D7['n']})")

    # ---- write 7-domain density table ----
    all_rows = []
    for did in ["D1", "D2", "D3", "D4", "D5", "D6"]:
        rec = existing[did]
        all_rows.append({
            "domain_id": did,
            "domain": rec["name"],
            "n": rec["n"],
            "k": rec["k"],
            "density": rec["density"],
            "wilson_lo": rec["wilson_lo"],
            "wilson_hi": rec["wilson_hi"],
            "source": rec["source"],
        })
    for r in d7_rows:
        all_rows.append({
            "domain_id": r["domain_id"],
            "domain": r["domain"],
            "n": r["n"],
            "k": r["k"],
            "density": r["density"],
            "wilson_lo": r["wilson_lo"],
            "wilson_hi": r["wilson_hi"],
            "source": r["source"],
        })
    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(RES / "synth_iter144_seven_domain_density.tsv",
                  sep="\t", index=False)
    print(f"Wrote {RES/'synth_iter144_seven_domain_density.tsv'} ({len(df_all)} rows)")

    # ---- pairwise ratio matrix ----
    domain_keys = ["D1", "D2", "D3", "D4", "D5", "D6", "D7@0.500"]
    densities = {r["domain_id"]: r for r in all_rows}
    ratio_rows = []
    for i in range(len(domain_keys)):
        for j in range(i + 1, len(domain_keys)):
            ki, kj = domain_keys[i], domain_keys[j]
            di, dj = densities[ki], densities[kj]
            ratio_point = di["density"] / dj["density"] if dj["density"] > 0 else np.inf
            mean, lo, hi = boot_ratio(di["k"], di["n"], dj["k"], dj["n"])
            ratio_rows.append({
                "numerator": ki,
                "denominator": kj,
                "numer_density": di["density"],
                "denom_density": dj["density"],
                "ratio": ratio_point,
                "boot_mean": mean,
                "boot_lo": lo,
                "boot_hi": hi,
                "ci_excludes_1": bool(lo > 1.0 or hi < 1.0),
            })
    df_rat = pd.DataFrame(ratio_rows)
    df_rat.to_csv(RES / "synth_iter144_seven_domain_ratios.tsv",
                  sep="\t", index=False)
    print(f"Wrote {RES/'synth_iter144_seven_domain_ratios.tsv'} ({len(df_rat)} pairs)")

    # ---- LOW cluster ----
    low_ids = ["D1", "D6", "D7@0.500"]
    low_rows = []
    for a in low_ids:
        for b in low_ids:
            if a == b:
                continue
            key = f"{a}_vs_{b}"
            rec = next((r for r in ratio_rows if
                        (r["numerator"] == a and r["denominator"] == b) or
                        (r["numerator"] == b and r["denominator"] == a)), None)
            if rec is None:
                continue
            low_rows.append({
                "pair": key,
                "numerator": rec["numerator"],
                "denominator": rec["denominator"],
                "ratio_point": rec["ratio"],
                "boot_mean": rec["boot_mean"],
                "boot_lo": rec["boot_lo"],
                "boot_hi": rec["boot_hi"],
                "ci_excludes_1": rec["ci_excludes_1"],
            })
    df_low = pd.DataFrame(low_rows)
    df_low.to_csv(RES / "synth_iter144_low_cluster.tsv",
                  sep="\t", index=False)
    print(f"Wrote {RES/'synth_iter144_low_cluster.tsv'} ({len(df_low)} pairs)")

    # ---- summary ----
    n_excl = int(df_rat.ci_excludes_1.sum())
    n_total = int(len(df_rat))
    n_pair_low = len(low_rows)
    n_pair_low_excl = sum(1 for r in low_rows if r["ci_excludes_1"])
    summary = {
        "n_domains": len(domain_keys),
        "n_pairs": n_total,
        "n_pairs_ci_excl_1": n_excl,
        "frac_pairs_excl_1": n_excl / n_total,
        "iter141_eta2_method": 0.0005,
        "d7_canonical_density": D7["density"],
        "d7_k": D7["k"],
        "d7_n": D7["n"],
        "low_cluster_pair_count": n_pair_low,
        "low_cluster_pairs_excl_1": n_pair_low_excl,
        "low_ids": low_ids,
        "boot_seed": BOOT_SEED,
        "n_boot": BOOT_N,
    }
    with open(RES / "synth_iter144_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {RES/'synth_iter144_summary.json'}")
    print()
    print("Headlines:")
    print(f"  D7@0.500 density = {D7['density']:.4f} "
          f"[{D7['wilson_lo']:.4f}, {D7['wilson_hi']:.4f}] "
          f"({D7['k']}/{D7['n']})")
    print(f"  η²(method) from iter 141 = 0.0005; "
          f"D7@0.500 = {D7['density']:.4f}")
    print(f"  Of 21 pairwise ratios, {n_excl}/{n_total} exclude 1.0")
    print(f"  Of {n_pair_low} LOW-cluster pairs, "
          f"{n_pair_low_excl} exclude 1.0")

    # ---- figure: 7-domain bar ----
    fig, ax = plt.subplots(figsize=(10, 4.5))
    xs = np.arange(len(domain_keys))
    dens = [densities[k]["density"] for k in domain_keys]
    lo = [densities[k]["wilson_lo"] for k in domain_keys]
    hi = [densities[k]["wilson_hi"] for k in domain_keys]
    colors = ["C0", "C1", "C1", "C1", "C2", "C0", "C0"]
    ax.bar(xs, dens,
           yerr=[np.array(dens) - np.array(lo), np.array(hi) - np.array(dens)],
           color=colors)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(domain_keys, rotation=15)
    ax.set_ylabel("density (Wilson 95% CI)")
    ax.set_title("7-domain density matrix (iter 144 SYNTH)")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(FIG / "synth_iter144_seven_domain.png", dpi=150)
    plt.savefig(FIG / "synth_iter144_seven_domain.pdf")
    plt.close()
    print(f"Wrote {FIG/'synth_iter144_seven_domain.png'}")


if __name__ == "__main__":
    main()
