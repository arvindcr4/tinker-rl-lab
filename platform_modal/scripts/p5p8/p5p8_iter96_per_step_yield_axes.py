#!/usr/bin/env python3
"""P5+P7 SYNTH (iter 96 JOB B): per-step analog of iter-81 row 96 yield-residual Items 14-17.

Closes the iter-83 row 98 mint recommendation (mint vein #2 from iter-83):
"extend iter-81 row 96 Items 14-17 to per-step granularity on N2 - does the
multi-axis discrimination hold at per-step level, or only at per-cell level?"

The iter-81 Items (P5 axis) at per-cell granularity:
- Item 14  K_variance_residual = Var(K)_obs - G*p*(1-p)
- Item 15  K_unique_count = |{k : k ∈ K_obs}|
- Item 16  max_K_share = max frequency of K as fraction of n_groups [REJECTED]
- Item 17  prompt_p_hat_var = Var(K_x/G)

This iter computes the per-STEP analog:
- Item 14-step  = Var(k_p)_obs - G*p_mean*(1-p_mean)   over 16 prompts in the step
- Item 15-step  = |{k_p : k_p ∈ K_obs}|                 unique k values per step
- Item 16-step  = max_k #prompts(k_p==k) / 16           max share per step [REJECTED re-test]
- Item 17-step  = Var(k_p/8)_obs                         = Var(p_hat)_obs per step

Predicted discrimination (per iter-83 mint): items 14, 15, 17 carry signal
on the per-step axis; item 16 should be placebo-null.

Data: experiments/results/n2_reward_tensor/{grpo,aero,gift,areal}_s0_tensors.jsonl
(n2_reward_tensor_resume/{...}_s0_tensors.jsonl is iter-91 used; n2_reward_tensor
is the canonical N2 panel).

Outputs:
- experiments/results/p5p8/p5p8_iter96_per_step_yield_axes.tsv  (4 methods x 40 steps = 160 rows)
- experiments/results/p5p8/p5p8_iter96_per_step_summary.json
- experiments/results/p5p8/p5p8_iter96_per_step_rho.tsv  (Spearman rho of Item_k vs |zvf_drop|)
- docs/p5p8_improvements/113_p5p8_per_step_yield_axes.md
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
RES = ROOT / "experiments" / "results" / "p5p8"
TENSORS_DIR = ROOT / "experiments" / "results" / "n2_reward_tensor"
TENSORS_DIR_RESUME = ROOT / "experiments" / "results" / "n2_reward_tensor_resume"
RES.mkdir(parents=True, exist_ok=True)

SEED = 20260705
G = 8  # observed group size on N2


def load_step_records() -> list[dict]:
    """Load all (method, step, rewards[16][8]) from the N2 four-method panels."""
    out = []
    for d in (TENSORS_DIR, TENSORS_DIR_RESUME):
        for f in sorted(d.glob("*_s0_tensors.jsonl")):
            if f.name.startswith("smoke_"):
                continue
            for line in f.open():
                r = json.loads(line)
                if "rewards" in r:
                    rewards = r["rewards"]
                    k_per_prompt = [sum(int(x) for x in rollout) for rollout in rewards]
                    out.append(dict(
                        method=r.get("method", f.stem.split("_s")[0]),
                        seed=int(r.get("seed", 0)),
                        step=int(r["step"]),
                        k_per_prompt=k_per_prompt,
                        zvf_obs=float(r.get("zvf", -1.0)),
                    ))
    return out


def compute_per_step_items(record: dict) -> dict:
    """Compute Item 14-step, 15-step, 16-step, 17-step for one record."""
    k = record["k_per_prompt"]
    n = len(k)
    p_hat = sum(k) / (n * G)
    # Item 13-step : zvf_yield_residual analog = zvf_obs - binom(zvf)
    import math
    def binom_zvf(p, g):
        return p ** g + (1 - p) ** g
    binom_zvf_g = binom_zvf(p_hat, G)
    item_13 = (record["zvf_obs"] - binom_zvf_g) / max(1 - binom_zvf_g, 1e-12)

    # Item 14-step: Var(k) - G*p*(1-p)
    if n > 1:
        mean_k = sum(k) / n
        var_k = sum((x - mean_k) ** 2 for x in k) / (n - 1)
    else:
        var_k = 0.0
    var_iid = G * p_hat * (1 - p_hat)
    item_14 = (var_k - var_iid) / max(var_iid, 1e-12)

    # Item 15-step: unique k values
    item_15 = len(set(k))

    # Item 16-step: max share
    from collections import Counter
    cnt = Counter(k)
    item_16 = max(cnt.values()) / n

    # Item 17-step: Var(p_hat_p) = Var(k/G)
    if n > 1:
        mean_p = sum(x / G for x in k) / n
        item_17 = sum(((x / G) - mean_p) ** 2 for x in k) / (n - 1)
    else:
        item_17 = 0.0

    return dict(
        item_13_step=item_13,
        item_14_step=item_14,
        item_15_step=item_15,
        item_16_step=item_16,
        item_17_step=item_17,
        p_hat=p_hat, n_unique=item_15,
    )


def spearman_rho(xs: list[float], ys: list[float]) -> float:
    """Rank-correlation coefficient. Equal to 0 if n < 2."""
    n = len(xs)
    if n < 2:
        return 0.0
    rx = _rank(xs); ry = _rank(ys)
    mean_rx = sum(rx) / n; mean_ry = sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    dx = sum((rx[i] - mean_rx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ry[i] - mean_ry) ** 2 for i in range(n)) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def _rank(xs: list[float]) -> list[float]:
    sorted_pairs = sorted(enumerate(xs), key=lambda t: t[1])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j < len(xs) and sorted_pairs[j][1] == sorted_pairs[i][1]:
            j += 1
        avg_rank = (i + j - 1) / 2 + 1
        for k in range(i, j):
            ranks[sorted_pairs[k][0]] = avg_rank
        i = j
    return ranks


def binomial_null_test(item_13_orig: list[float], item_X_orig: list[float],
                       items_per_step: list[dict], n_sim: int = 500) -> dict:
    """Shuffle k_per_prompt within each step to break per-step Item axes but
    preserve (p_hat, n_prompts, n_steps). Compute the per-step Items on the
    shuffled data and report the empirical H_bits uplift mean & CI for item_X."""
    rng = random.Random(SEED)
    items_X_shuffled = []
    for r in records:
        pass  # need records; we'll recompute


def main() -> None:
    print("[iter96-SYNTH] loading N2 four-method tensors ...")
    records = load_step_records()
    print(f"  n_records = {len(records)}; unique methods = {set(r['method'] for r in records)}")
    # Compute per-step Items
    enriched = []
    for r in records:
        items = compute_per_step_items(r)
        enriched.append({**r, **items})

    out_rows = []
    for r in enriched:
        out_rows.append(dict(
            method=r["method"], seed=r["seed"], step=r["step"],
            n_prompts=len(r["k_per_prompt"]),
            k_mean=sum(r["k_per_prompt"]) / len(r["k_per_prompt"]),
            item_13_step=r["item_13_step"],
            item_14_step=r["item_14_step"],
            item_15_step=r["item_15_step"],
            item_16_step=r["item_16_step"],
            item_17_step=r["item_17_step"],
            p_hat=r["p_hat"],
            zvf_obs=r["zvf_obs"],
        ))
    with (RES / "p5p8_iter96_per_step_yield_axes.tsv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()), delimiter="\t")
        w.writeheader(); [w.writerow(r) for r in out_rows]

    # ---- Spearman rho per (item, method) ----
    # Predict each Item_k_step vs |zvf_obs - mean(p_hat)^G - (1-mean(p_hat))^G|
    import math
    def b_zvf(p, g=G): return p ** g + (1 - p) ** g
    target = []
    for r in enriched:
        target.append(abs(r["zvf_obs"] - b_zvf(r["p_hat"])))

    methods = sorted(set(r["method"] for r in enriched))
    rho_rows = []
    for m in methods:
        sub = [r for r in enriched if r["method"] == m]
        sub_t = [target[enriched.index(r)] for r in sub]
        item_cols = ["item_13_step", "item_14_step", "item_15_step", "item_16_step", "item_17_step"]
        for c in item_cols:
            xs = [r[c] for r in sub]
            rho = spearman_rho(xs, sub_t)
            rho_rows.append(dict(method=m, item=c, spearman_rho=rho))

    # Cross-method pooled rho
    pool_t = target
    for c in ["item_13_step", "item_14_step", "item_15_step", "item_16_step", "item_17_step"]:
        xs = [r[c] for r in enriched]
        rho = spearman_rho(xs, pool_t)
        rho_rows.append(dict(method="POOLED", item=c, spearman_rho=rho))

    with (RES / "p5p8_iter96_per_step_rho.tsv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=["method", "item", "spearman_rho"], delimiter="\t")
        w.writeheader(); [w.writerow(r) for r in rho_rows]

    # ---- Binomial(G, p) null control on Items 14-17 ----
    # For each step, the null is Binomial(G, p_hat): the expected Item values
    # under independent binomial rewards. Compare empirical to null.
    # Approx: simulate n=500 k-prompt vectors per step under Binomial(G, p_hat).
    rng = random.Random(SEED)
    null_results = {m: {c: [] for c in ["item_14_step", "item_15_step", "item_16_step", "item_17_step"]}
                    for m in methods + ["POOLED"]}
    for r in enriched:
        p_hat = r["p_hat"]
        n = len(r["k_per_prompt"])
        if n < 2:
            continue
        for _ in range(500):
            k_sim = [sum(rng.random() < p_hat for _ in range(G)) for _ in range(n)]
            mean_k = sum(k_sim) / n
            var_k = sum((x - mean_k) ** 2 for x in k_sim) / max(n - 1, 1)
            var_iid = G * p_hat * (1 - p_hat)
            item_14_sim = (var_k - var_iid) / max(var_iid, 1e-12)
            uniq = len(set(k_sim))
            from collections import Counter
            cnt = Counter(k_sim)
            item_16_sim = max(cnt.values()) / n
            mean_p = sum(x / G for x in k_sim) / n
            item_17_sim = sum(((x / G) - mean_p) ** 2 for x in k_sim) / max(n - 1, 1)
            for c, v in zip(["item_14_step", "item_15_step", "item_16_step", "item_17_step"],
                             [item_14_sim, uniq, item_16_sim, item_17_sim]):
                null_results[r["method"]][c].append(v)

    # Empirical vs null: difference of means
    excess = {m: {} for m in methods + ["POOLED"]}
    seen_methods = set()
    for r in enriched:
        seen_methods.add(r["method"])
    methods = sorted(seen_methods)
    for m in methods + ["POOLED"]:
        for c in ["item_14_step", "item_15_step", "item_16_step", "item_17_step"]:
            if m == "POOLED":
                emp_vals = [r[c] for r in enriched]
            else:
                emp_vals = [r[c] for r in enriched if r["method"] == m]
            null_vals = null_results[m][c] if m in null_results else []
            if not emp_vals or not null_vals:
                continue
            emp_mean = sum(emp_vals) / len(emp_vals)
            null_mean = sum(null_vals) / len(null_vals)
            null_std = (sum((x - null_mean) ** 2 for x in null_vals) / max(len(null_vals) - 1, 1)) ** 0.5
            excess_z = (emp_mean - null_mean) / max(null_std, 1e-12)
            excess[m][c] = dict(emp_mean=emp_mean, null_mean=null_mean,
                                diff=emp_mean - null_mean, z=excess_z)

    summary = {
        "n_records": len(records),
        "methods": methods,
        "rho_per_method_per_item": rho_rows,
        "excess_signal_vs_binom_null": {
            m: {c: excess[m][c] for c in ["item_14_step", "item_15_step", "item_16_step", "item_17_step"]
                if c in excess[m]}
            for m in methods + ["POOLED"] if m in excess and excess[m]
        },
    }
    (RES / "p5p8_iter96_per_step_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
