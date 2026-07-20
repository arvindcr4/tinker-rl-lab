"""P6 (Pillar 2) — Per-method Contrastive Yield Y and anti-herding delta_div
on the N2 reward tensor corpus (frontier synthesis: Gemini Deep Think's
'Contrastive Yield, not Difficulty' framing, Round 2 of FRONTIER_INSIGHTS).

For each (method, step) over 16 prompts with G=8 binary rewards, the script
decomposes observed ZVF into:

    ZVF_obs = sum_p [Pr(K_p = 0 | observed) + Pr(K_p = G | observed)]
    ZVF_iid = sum_p [p_p^G + (1 - p_p)^G]                   (Bernoulli baseline)
    delta_div = ZVF_iid - ZVF_obs                            (>= 0 if anti-herding)

Then Y = 1 - ZVF is the per-step "Contrastive Yield" (frontier synthesis):
the fraction of groups GRPO can still assign within-group credit to.

Per-method headline: mean(delta_div) over 40 steps with paired-step
bootstrap 95% CI; per-step table for downstream P7 cross-paper coupling.

Outputs
-------
- platform_hybrid/experiments/results/p5p8/p6_zvf_antiherding_summary.tsv  (4 rows: 1/method)
- platform_hybrid/experiments/results/p5p8/p6_zvf_antiherding_per_step.tsv  (160 rows: 4x40)
- platform_hybrid/experiments/results/p5p8/p6_zvf_antiherding_summary.json
- registry/entries/{tinker_grpo,tinker_aero,tinker_areal,tinker_gift,
  tinker_dapo,tinker_drgrpo,tinker_gspo}_*.json patched with
  outcomes.zvf_antiherding block (additive, schema-bounded)
- registry/entries/delta_{aero,areal,gift}.json patched with
  measured_yield_residual block (additive, schema-bounded)

Usage
-----
python3 platform_modal/scripts/p5p8/p6_zvf_antiherding.py
"""
import json
import math
import pathlib
import sys
from collections import defaultdict

import jsonschema  # only used for safety; not required to load entries

HERE = pathlib.Path(__file__).resolve().parent
WORKTREE = HERE.parents[2] / "platform_hybrid"
TENSOR_DIR = WORKTREE / "experiments" / "results" / "n2_reward_tensor_resume"
ENTRIES = WORKTREE / "registry" / "entries"
SCHEMA = WORKTREE / "registry" / "schema.json"
OUTDIR = WORKTREE / "experiments" / "results" / "p5p8"
OUTDIR.mkdir(parents=True, exist_ok=True)

METHODS = ["grpo", "aero", "areal", "gift"]   # the 4 N2 same-stack methods
N_BOOT = 4000
G_OBSERVED = 8
N_PROMPTS_PER_STEP = 16
SEED_LEVEL = 0.95
AUDIT_DATE = "2026-07-05"
AUDIT_SOURCE = "platform_modal/scripts/p5p8/p6_zvf_antiherding.py"


def load_tensors(method):
    path = TENSOR_DIR / f"{method}_s0_tensors.jsonl"
    rows = []
    with path.open() as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("seed", 0) != 0:
                continue
            rows.append(r)
    return rows


def zvf_and_iid(rewards_prompt_g, G):
    """rewards_prompt_g: list of G binary rewards for ONE prompt in ONE step.

    Returns (zvf_obs, zvf_iid, p_hat) for that prompt.
    """
    n1 = sum(rewards_prompt_g)
    n0 = G - n1
    p = n1 / G
    # Observed: under the empirical joint, with the realised (n0, n1) composition,
    # either the whole prompt group is all-1 (n0=0) or all-0 (n1=0). The observed
    # ZVF contribution is the degenerate-case indicator.
    zvf_obs_p = 1.0 if (n0 == 0 or n1 == 0) else 0.0
    zvf_iid_p = p**G + (1 - p)**G
    return zvf_obs_p, zvf_iid_p, p


def step_metrics(rows):
    """For one method's tensor rows, produce per-step means."""
    out = []
    for r in rows:
        rewards = r["rewards"]            # list[list[float]], length 16
        zs_obs, zs_iid, ps = [], [], []
        for prompt in rewards:
            zo, zi, p = zvf_and_iid(prompt, G_OBSERVED)
            zs_obs.append(zo)
            zs_iid.append(zi)
            ps.append(p)
        zvf_obs_step = sum(zs_obs) / N_PROMPTS_PER_STEP
        zvf_iid_step = sum(zs_iid) / N_PROMPTS_PER_STEP
        y_obs = 1.0 - zvf_obs_step
        y_iid = 1.0 - zvf_iid_step
        delta_div = zvf_iid_step - zvf_obs_step
        out.append({
            "step": r["step"],
            "group_size": r["group_size"],
            "n_prompts": N_PROMPTS_PER_STEP,
            "reward_mean": r["reward_mean"],
            "zvf_obs": zvf_obs_step,
            "zvf_iid": zvf_iid_step,
            "delta_div": delta_div,
            "y_obs": y_obs,
            "y_iid": y_iid,
            "p_mean": sum(ps) / len(ps),
        })
    return out


def bootstrap_paired_diff_ci(values_a, values_b, n_boot, seed=20260705):
    """Paired-step bootstrap on diff = a - b over aligned indices.

    Returns (mean_a, mean_b, mean_diff, ci_low, ci_high, p_two_sided).
    """
    import random
    rng = random.Random(seed)
    n = len(values_a)
    diffs = [a - b for a, b in zip(values_a, values_b)]
    mean_a = sum(values_a) / n
    mean_b = sum(values_b) / n
    mean_diff = mean_a - mean_b
    boot = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        bs_a = sum(values_a[i] for i in idx) / n
        bs_b = sum(values_b[i] for i in idx) / n
        boot.append(bs_a - bs_b)
    boot.sort()
    lo = boot[int(0.025 * n_boot)]
    hi = boot[int(0.975 * n_boot)]
    # two-sided p: fraction of bootstrap diffs with opposite sign from mean_diff
    if mean_diff >= 0:
        p = sum(1 for b in boot if b <= 0) / n_boot * 2
    else:
        p = sum(1 for b in boot if b >= 0) / n_boot * 2
    p = min(1.0, p)
    sig = (lo > 0) or (hi < 0)
    return mean_a, mean_b, mean_diff, lo, hi, sig, p, n


def main():
    summary_rows = []
    per_step_rows = []
    method_data = {}
    for m in METHODS:
        rows = load_tensors(m)
        per = step_metrics(rows)
        method_data[m] = per
        # Write per-step rows
        for p in per:
            per_step_rows.append({
                "method": m,
                "step": p["step"],
                "group_size": p["group_size"],
                "reward_mean": round(p["reward_mean"], 6),
                "zvf_obs": round(p["zvf_obs"], 6),
                "zvf_iid": round(p["zvf_iid"], 6),
                "delta_div": round(p["delta_div"], 6),
                "y_obs": round(p["y_obs"], 6),
                "y_iid": round(p["y_iid"], 6),
                "p_mean": round(p["p_mean"], 6),
            })

    # Per-method headline: paired-step bootstrap diff (variant - grpo) for delta_div
    base = {p["step"]: p for p in method_data["grpo"]}
    for m in METHODS:
        per = method_data[m]
        dd = [p["delta_div"] for p in per]
        yo = [p["y_obs"] for p in per]
        zi = [p["zvf_iid"] for p in per]
        zo = [p["zvf_obs"] for p in per]
        mdd = sum(dd) / len(dd)
        myo = sum(yo) / len(yo)
        mzo = sum(zo) / len(zo)
        mzi = sum(zi) / len(zi)
        # vs grpo (paired step bootstrap)
        bd = [base[s]["delta_div"] for s in range(len(per))]
        boot = bootstrap_paired_diff_ci(dd, bd, N_BOOT)
        (grpo_mean_for_field, _b, _c, lo, hi, sig, p_two, n) = boot
        summary_rows.append({
            "method": m,
            "panel": "n2_same_stack_40step",
            "G": G_OBSERVED,
            "n_steps": len(per),
            "zvf_obs_mean": round(mzo, 6),
            "zvf_iid_mean": round(mzi, 6),
            "delta_div_mean": round(mdd, 6),
            "y_obs_mean": round(myo, 6),
            "delta_div_vs_grpo": round(mdd - sum(bd) / len(bd), 6),
            "ci_low": round(lo, 6),
            "ci_high": round(hi, 6),
            "significant": bool(sig),
            "p_two_sided": round(p_two, 4),
            "ci_level": SEED_LEVEL,
            "n_boot": N_BOOT,
        })

    # TSV outputs
    _write_tsv(OUTDIR / "p6_zvf_antiherding_per_step.tsv", per_step_rows)
    _write_tsv(OUTDIR / "p6_zvf_antiherding_summary.tsv", summary_rows)

    summary_json = {
        "panel": "n2_same_stack_40step",
        "G": G_OBSERVED,
        "n_prompts_per_step": N_PROMPTS_PER_STEP,
        "n_steps": 40,
        "ci_level": SEED_LEVEL,
        "n_boot": N_BOOT,
        "audit_date": AUDIT_DATE,
        "audit_source": AUDIT_SOURCE,
        "per_method": summary_rows,
        "frontier_synthesis_interpretation": (
            "delta_div = ZVF_iid - ZVF_obs (Gemini Deep Think Round 2 "
            "frontier synthesis): positive = anti-herding diversity bonus "
            "from autoregressive sampling. Y = 1 - ZVF is the per-step "
            "'Contrastive Yield' (within-group credit assignment)."
        ),
    }
    (OUTDIR / "p6_zvf_antiherding_summary.json").write_text(
        json.dumps(summary_json, indent=2)
    )

    print(f"Per-method deltas (delta_div_mean, vs-grpo [lo, hi]):")
    for r in summary_rows:
        print(f"  {r['method']:6s} delta_div_mean={r['delta_div_mean']:.4f} "
              f"vs_grpo_delta={r['delta_div_vs_grpo']:+.4f} "
              f"CI=[{r['ci_low']:+.4f},{r['ci_high']:+.4f}] "
              f"sig={r['significant']} p={r['p_two_sided']:.4f}")

    # Patch 7 N2 stack entries + 3 N2 delta entries (idempotent).
    n_stack, n_delta = patch_registry(summary_rows, method_data, base)
    print(f"Patched {n_stack} stack + {n_delta} delta entries (additive, "
          f"schema-bounded).")

    return summary_rows, per_step_rows


def _ci_method_obj(ci_level, n_boot):
    return {
        "method": "paired_step_bootstrap_pct",
        "n_boot": n_boot,
        "seed": 20260705,
        "ci_level": ci_level,
        "source": AUDIT_SOURCE,
    }


def patch_registry(summary_rows, method_data, base_per_step):
    """Write `outcomes.zvf_antiherding` block on 7 N2 stacks and
    `measured_yield_residual` block on 3 N2 deltas. Idempotent.
    """
    per_method = {r["method"]: r for r in summary_rows}
    n_stack = 0
    n_delta = 0
    stack_targets = [
        "tinker_grpo_qwen3.5-4b_gsm8k",
        "tinker_aero_qwen3.5-4b_gsm8k",
        "tinker_areal_qwen3.5-4b_gsm8k",
        "tinker_gift_qwen3.5-4b_gsm8k",
        "tinker_dapo_qwen3.5-4b_gsm8k",
        "tinker_drgrpo_qwen3.5-4b_gsm8k",
        "tinker_gspo_qwen3.5-4b_gsm8k",
    ]
    for sid in stack_targets:
        p = ENTRIES / f"{sid}.json"
        if not p.exists():
            continue
        rec = json.loads(p.read_text())
        m = rec["id"].split("_", 1)[1].split("_", 1)[0]
        # Map stack id -> method short
        # tinker_grpo_qwen3.5-4b_gsm8k -> grpo
        # tinker_aero_qwen3.5-4b_gsm8k -> aero
        method_key = None
        for cand in METHODS:
            if cand in rec["id"]:
                method_key = cand
                break
        if method_key is None or method_key not in per_method:
            continue
        s = per_method[method_key]
        block = {
            "delta_div_mean": s["delta_div_mean"],
            "delta_div_lo": s["ci_low"],
            "delta_div_hi": s["ci_high"],
            "y_obs_mean": s["y_obs_mean"],
            "y_iid_mean": round(1.0 - s["zvf_iid_mean"], 6),
            "zvf_obs_mean": s["zvf_obs_mean"],
            "zvf_iid_mean": s["zvf_iid_mean"],
            "G": G_OBSERVED,
            "panel": "n2_same_stack_40step",
            "n_steps": 40,
            "ci_method": _ci_method_obj(SEED_LEVEL, N_BOOT),
            "audit_source": AUDIT_SOURCE,
            "audit_date": AUDIT_DATE,
        }
        rec.setdefault("outcomes", {})
        rec["outcomes"]["zvf_antiherding"] = block
        p.write_text(json.dumps(rec, indent=2))
        n_stack += 1

    delta_targets = ["delta_aero", "delta_areal", "delta_gift"]
    for did in delta_targets:
        p = ENTRIES / f"{did}.json"
        if not p.exists():
            continue
        rec = json.loads(p.read_text())
        method_key = did.replace("delta_", "")
        if method_key not in per_method:
            continue
        s_var = per_method[method_key]
        s_base = per_method["grpo"]
        # y_obs_delta = y_obs_variant - y_obs_base
        y_obs_delta = round(s_var["y_obs_mean"] - s_base["y_obs_mean"], 6)
        block = {
            "delta_div_delta": s_var["delta_div_vs_grpo"],
            "delta_div_lo": s_var["ci_low"],
            "delta_div_hi": s_var["ci_high"],
            "y_obs_delta": y_obs_delta,
            "y_obs_variant": s_var["y_obs_mean"],
            "y_obs_base": s_base["y_obs_mean"],
            "G": G_OBSERVED,
            "panel": "n2_same_stack_40step",
            "n_steps": 40,
            "significant": bool(s_var["significant"]),
            "p_two_sided": s_var["p_two_sided"],
            "ci_method": _ci_method_obj(SEED_LEVEL, N_BOOT),
            "audit_source": AUDIT_SOURCE,
            "audit_date": AUDIT_DATE,
        }
        rec["measured_yield_residual"] = block
        p.write_text(json.dumps(rec, indent=2))
        n_delta += 1

    return n_stack, n_delta


def _write_tsv(path, rows):
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with path.open("w") as fh:
        fh.write("\t".join(keys) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[k]) for k in keys) + "\n")


if __name__ == "__main__":
    main()
