#!/usr/bin/env python3
"""P0-A: Paired GSM8K held-out test.

Loads per-prompt results from gsm8k_base_control_200.json (base Qwen3-8B, no LoRA)
and gsm8k_heldout_seed{042,137,256,512,999}.json (5 GRPO seeds). Computes:

1. Per-seed paired McNemar test (base vs single seed).
2. Pooled-pair paired McNemar (base vs majority-vote-of-5 GRPO).
3. Paired bootstrap 95% CI on the accuracy delta, per seed and majority-vote.
4. Per-prompt disagreement table (base correct / GRPO wrong, base wrong / GRPO correct).

Writes a JSON result to experiments/results/p0_gsm8k_paired.json and a markdown
summary to experiments/p0_gsm8k_paired.md. No Tinker API calls, no cost.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_FILE = ROOT / "reports/final/gsm8k_base_control_200.json"
SEED_FILES = [
    ROOT / f"reports/final/gsm8k_heldout_seed{s}.json"
    for s in ("042", "137", "256", "512", "999")
]
OUT_DIR = ROOT / "experiments/results"
OUT_JSON = OUT_DIR / "p0_gsm8k_paired.json"
OUT_MD = ROOT / "experiments/p0_gsm8k_paired.md"


def load_passfail(path: Path) -> list[tuple[int, int]]:
    d = json.load(open(path))
    out = []
    for ex in d["examples"]:
        idx = ex["idx"]
        status = ex.get("status", "")
        correct = 1 if status == "correct" else 0
        out.append((idx, correct))
    return out


def align(base, grpo):
    bmap = dict(base)
    gmap = dict(grpo)
    shared = sorted(set(bmap) & set(gmap))
    b = [bmap[i] for i in shared]
    g = [gmap[i] for i in shared]
    return b, g


def mcnemar_exact(b01, b10):
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    def pmf(n, k):
        return math.comb(n, k) * (0.5 ** n)
    p_low = sum(pmf(n, i) for i in range(k + 1))
    p_hi = sum(pmf(n, i) for i in range(n - k, n + 1))
    p = p_low + p_hi
    return min(p, 1.0)


def mcnemar_chi2_cc(b01, b10):
    n = b01 + b10
    if n == 0:
        return 0.0, 1.0
    chi2 = (abs(b01 - b10) - 1) ** 2 / n
    p = math.erfc(math.sqrt(chi2 / 2))
    return chi2, p


def paired_bootstrap_ci(b, g, reps=10000, seed=20260422):
    rng = random.Random(seed)
    n = len(b)
    deltas = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        db = sum(b[i] for i in idx) / n
        dg = sum(g[i] for i in idx) / n
        deltas.append(dg - db)
    deltas.sort()
    lo = deltas[int(0.025 * reps)]
    hi = deltas[int(0.975 * reps)]
    point = sum(g) / n - sum(b) / n
    return point, (lo, hi)


def contingency(b, g):
    c = Counter(zip(b, g))
    return {
        "both_correct": c[(1, 1)],
        "base_correct_grpo_wrong": c[(1, 0)],
        "base_wrong_grpo_correct": c[(0, 1)],
        "both_wrong": c[(0, 0)],
    }


def analyze_pair(b, g, label):
    ct = contingency(b, g)
    b10 = ct["base_correct_grpo_wrong"]
    b01 = ct["base_wrong_grpo_correct"]
    p_exact = mcnemar_exact(b01, b10)
    chi2, p_chi = mcnemar_chi2_cc(b01, b10)
    point, (lo, hi) = paired_bootstrap_ci(b, g)
    return {
        "label": label,
        "n": len(b),
        "base_acc": sum(b) / len(b),
        "grpo_acc": sum(g) / len(g),
        "delta": point,
        "boot_ci_95": [lo, hi],
        "contingency": ct,
        "mcnemar_b01": b01,
        "mcnemar_b10": b10,
        "mcnemar_exact_p": p_exact,
        "mcnemar_chi2_cc": chi2,
        "mcnemar_chi2_p": p_chi,
    }


def majority_vote(seed_g):
    n = len(seed_g[0])
    out = []
    for i in range(n):
        votes = sum(s[i] for s in seed_g)
        out.append(1 if votes >= (len(seed_g) + 1) // 2 else 0)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = load_passfail(BASE_FILE)
    seed_results = []
    seed_aligned = []
    base_aligned_any = None

    for sf in SEED_FILES:
        grpo = load_passfail(sf)
        b, g = align(base, grpo)
        if base_aligned_any is None:
            base_aligned_any = b
        seed_aligned.append(g)
        seed_results.append(analyze_pair(b, g, sf.stem))

    mv_g = majority_vote(seed_aligned)
    mv_result = analyze_pair(base_aligned_any, mv_g, "grpo_majority_vote_5seeds")

    n = len(base_aligned_any)
    mean_seed = [sum(s[i] for s in seed_aligned) / len(seed_aligned) for i in range(n)]
    diffs = [mean_seed[i] - base_aligned_any[i] for i in range(n)]
    mean_diff = sum(diffs) / n
    var = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n)
    t_stat = mean_diff / se if se > 0 else 0.0
    p_t = math.erfc(abs(t_stat) / math.sqrt(2))

    out = {
        "description": "Paired GSM8K held-out test: base Qwen3-8B vs GRPO LoRA (5 seeds)",
        "base_file": str(BASE_FILE.relative_to(ROOT)),
        "seed_files": [str(s.relative_to(ROOT)) for s in SEED_FILES],
        "per_seed": seed_results,
        "majority_vote_5seeds": mv_result,
        "continuous_mean_indicator_paired_t": {
            "n": n,
            "mean_delta_indicator": mean_diff,
            "std_error": se,
            "t_stat": t_stat,
            "p_two_sided_normal_approx": p_t,
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=2) + "\n")

    lines = [
        "# P0-A: Paired GSM8K held-out test",
        "",
        f"Source: `{BASE_FILE.relative_to(ROOT)}` (base) + 5 `gsm8k_heldout_seed*.json` (GRPO).",
        f"n = {mv_result['n']} paired prompts.",
        "",
        "## Per-seed paired McNemar",
        "",
        "| Seed | base acc | GRPO acc | Δ | 95% boot CI | b01 | b10 | McNemar exact p | χ²(cc) p |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for r in seed_results:
        lo, hi = r["boot_ci_95"]
        lines.append(
            f"| {r['label']} | {r['base_acc']:.3f} | {r['grpo_acc']:.3f} | "
            f"{r['delta']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
            f"{r['mcnemar_b01']} | {r['mcnemar_b10']} | "
            f"{r['mcnemar_exact_p']:.3f} | {r['mcnemar_chi2_p']:.3f} |"
        )
    lines += [
        "",
        "## Majority-vote-of-5 GRPO vs base",
        "",
        f"- base acc: {mv_result['base_acc']:.3f}",
        f"- majority-vote GRPO acc: {mv_result['grpo_acc']:.3f}",
        f"- Δ: {mv_result['delta']:+.3f} "
        f"(95% paired-bootstrap CI: [{mv_result['boot_ci_95'][0]:+.3f}, {mv_result['boot_ci_95'][1]:+.3f}])",
        f"- discordant pairs: b01={mv_result['mcnemar_b01']} (base wrong → GRPO correct), "
        f"b10={mv_result['mcnemar_b10']} (base correct → GRPO wrong)",
        f"- McNemar exact two-sided p: **{mv_result['mcnemar_exact_p']:.4f}**",
        f"- McNemar χ² (continuity-corrected) p: {mv_result['mcnemar_chi2_p']:.4f}",
        "",
        "## Paired test on per-prompt mean-of-5-seeds indicator",
        "",
        f"- mean Δ (continuous): {out['continuous_mean_indicator_paired_t']['mean_delta_indicator']:+.4f}",
        f"- t = {out['continuous_mean_indicator_paired_t']['t_stat']:.3f}, "
        f"p = {out['continuous_mean_indicator_paired_t']['p_two_sided_normal_approx']:.3f} "
        "(paired t, normal approx, two-sided)",
        "",
        "## Interpretation",
        "",
        "Paired analyses confirm the non-significant held-out lift already reported in",
        "the thesis (one-sample t-test on per-seed mean: t=1.32, p=0.26). The McNemar",
        "framing makes explicit that the signal is carried by a small number of",
        "discordant pairs; a base-wrong → GRPO-correct count that is not much larger",
        "than base-correct → GRPO-wrong leaves the null unchallenged.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
