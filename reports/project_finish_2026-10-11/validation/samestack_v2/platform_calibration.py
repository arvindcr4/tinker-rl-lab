"""Cross-platform calibration for the same-stack v2 runs (Modal A10G reference vs Colab A100).

Greedy bf16 decoding is not bit-identical across GPU generations, so arms trained on different
platforms are not directly comparable. This script:
  1. recomputes the pre-specified primary contrasts with one pair per seed, both arms from the same
     platform (Modal where both arms exist there, else the Colab platform), stratified by platform;
  2. measures the platform effect (Colab - Modal) for arms run on both platforms at the same seeds,
     tested for equivalence at the pre-specified +/-0.02 margin (gain over own base is primary here,
     because the base model itself scores differently per platform);
  3. tests the platform x arm interaction for g8-ppo, DiD_s = (g8-ppo)_Colab,s - (g8-ppo)_Modal,s,
     which is what could bias contrasts pooled across platforms.
Primary statistic: post accuracy, paired t with Holm over the primary family; sign-flip, gain and
Modal-base-state-matched results are sensitivity analyses. Modal produced two base states (0.6535 and
0.6550); each run's state is recorded and a sensitivity analysis keeps only state-matched Modal pairs.

Usage: python3 -I platform_calibration.py <scripts_dir> <results_dir> [<results_dir> ...] > platform_calibration.json
Runs without an execution_environment field are the Modal A10G runs.
"""

import glob, hashlib, json, os, re, sys
import numpy as np

sys.path.insert(0, sys.argv[1])
from stats_lib import paired, holm
from scipy import stats

FAMILY = {
    "g8-ppo": ("grpo_g8", "ppo"),
    "g8-ppo_8x8": ("grpo_g8", "ppo_8x8"),
    "g8-g2": ("grpo_g8", "grpo_g2"),
}
MARGIN = 0.02
REF = "A10G"


def family(r):
    gpu = (r.get("execution_environment") or {}).get("gpu") or "NVIDIA A10G (Modal)"
    m = re.search(r"A10G|A100|H100|L4|T4", gpu)
    if not m:
        sys.exit(f"unknown GPU {gpu!r}")
    return m.group(0)


runs = {}  # (platform, arm, seed) -> run
for d in sys.argv[2:]:
    for f in sorted(glob.glob(os.path.join(d, "*_s*.json"))):
        r = json.loads(open(f).read().replace("NaN", "null"))
        r["_file"] = f
        r["_base_state"] = hashlib.sha1(bytes(r["pre_correct"]), usedforsecurity=False).hexdigest()[
            :8
        ]
        key = (family(r), r["arm"], r["seed"])
        if key in runs:
            sys.exit(f"duplicate run {key}: {runs[key]['_file']} and {f}")
        runs[key] = r
plats = sorted({p for p, _, _ in runs}, key=lambda p: p != REF)  # reference platform first
if not plats or len(plats) > 2 or plats[0] != REF:
    sys.exit(f"expected Modal {REF} plus one Colab platform, got {plats}")
OTHER = plats[1] if len(plats) == 2 else None


def post(r):
    return float(np.mean(r["post_correct"]))


def gain(r):
    return post(r) - float(np.mean(r["pre_correct"]))


def tost(c):
    n, m, sd = c["n"], c["mean"], c["sd"]
    if n < 2 or not sd > 0:
        return None
    se = sd / np.sqrt(n)
    return float(
        max(1 - stats.t.cdf((m + MARGIN) / se, n - 1), stats.t.cdf((m - MARGIN) / se, n - 1))
    )


def one_sample(d):
    """Report insufficient pairs without relaxing paired()'s validation."""
    if len(d) < 2:
        seeds = sorted(d)
        values = [d[s] for s in seeds]
        if not np.isfinite(values).all():
            raise ValueError("finite paired differences are required")
        return {
            "status": "unavailable",
            "reason": "insufficient_pairs",
            "n": len(d),
            "seeds": seeds,
            "diffs": values,
            "mean": float(np.mean(values)) if values else None,
            "sd": None,
            "p_t": None,
            "p_signflip": None,
            "tost_p": None,
            "t_ci": None,
            "tost_ci90": None,
            "boot_ci": None,
            "difference_ci_95": None,
            "equivalence_ci_90": None,
            "within_margin_unadjusted": None,
        }
    c = paired(d, {k: 0.0 for k in d})
    c["tost_p"] = tost(c)
    return c


R = {
    "platforms": plats,
    "reference": REF,
    "margin": MARGIN,
    "n_runs": {p: sum(k[0] == p for k in runs) for p in plats},
    "environments": sorted(
        {json.dumps(r.get("execution_environment"), sort_keys=True) for r in runs.values()}
    ),
}

# Base model: deterministic greedy evaluation; Modal has two states.
R["base"] = {}
for p in plats:
    states = {}
    for k, r in runs.items():
        if k[0] == p:
            states.setdefault(
                r["_base_state"], {"acc": float(np.mean(r["pre_correct"])), "runs": 0}
            )["runs"] += 1
    R["base"][p] = states
if OTHER:
    ref_pre = {k[1:]: np.array(r["pre_correct"]) for k, r in runs.items() if k[0] == REF}
    oth_pre = next(np.array(r["pre_correct"]) for k, r in runs.items() if k[0] == OTHER)
    R["base"]["verdicts_agree_with_each_modal_state"] = sorted(
        {int((v == oth_pre).sum()) for v in ref_pre.values()}
    )

# 1. Primary contrasts: one same-platform pair per seed, Modal preferred.
for metric, f in (("post", post), ("gain", gain)):
    C = {}
    for name, (x, y) in FAMILY.items():
        d, plat_of, state_ok = {}, {}, {}
        for p in plats:
            for (pp, a, s), rx in runs.items():
                ry = runs.get((p, y, s))
                if pp == p and a == x and ry is not None and s not in d:
                    d[s], plat_of[s] = f(rx) - f(ry), p
                    state_ok[s] = rx["_base_state"] == ry["_base_state"]
        c = one_sample(d)
        c["platform_of_seed"] = {str(s): plat_of[s] for s in sorted(d)}
        c["strata"] = {}
        for p in plats:
            ds = [d[s] for s in d if plat_of[s] == p]
            if ds:
                c["strata"][p] = {"n": len(ds), "mean": float(np.mean(ds))}
        a_, b_ = ([d[s] for s in d if plat_of[s] == p] for p in plats) if OTHER else ([], [])
        c["stratum_difference_welch_p"] = (
            float(stats.ttest_ind(b_, a_, equal_var=False).pvalue)
            if len(a_) > 1 and len(b_) > 1
            else None
        )
        ok = {s: v for s, v in d.items() if plat_of[s] != REF or state_ok[s]}
        c["modal_state_matched"] = {
            "n": len(ok),
            **({"mean": float(np.mean(list(ok.values())))} if ok else {}),
        }
        C[name] = c
    for stat in ("p_t", "p_signflip", "tost_p"):
        # Reserve the entire pre-specified family, including unavailable tests.
        # A placeholder p=1 protects the family size, but is not a test result.
        ps = {k: c[stat] if c[stat] is not None else 1.0 for k, c in C.items()}
        for k, v in holm(ps).items():
            C[k][f"holm_{stat}"] = v if C[k][stat] is not None else None
    R[f"contrasts_{metric}"] = C
R["primary"] = (
    "contrasts_post: mean, paired t with holm_p_t; holm_p_signflip and holm_tost_p alongside"
)

# 2. Platform effect (Colab - Modal) for arms run on both platforms at the same seeds.
R["platform_effect"] = {}
if OTHER:
    for arm in sorted({k[1] for k in runs}):
        seeds = sorted(s for (p, a, s) in runs if p == OTHER and a == arm and (REF, arm, s) in runs)
        if not seeds:
            continue
        e = {"direction": f"{OTHER} - {REF}", "seeds": seeds}
        for metric, f in (("gain", gain), ("post", post)):
            d = {s: f(runs[OTHER, arm, s]) - f(runs[REF, arm, s]) for s in seeds}
            e[metric] = one_sample(d) if len(seeds) > 1 else {"n": 1, "diff": d[seeds[0]]}
        R["platform_effect"][arm] = e

# 3. Platform x arm interaction for g8-ppo.
if OTHER:
    seeds = sorted(
        s
        for s in {k[2] for k in runs}
        if all((p, a, s) in runs for p in plats for a in ("grpo_g8", "ppo"))
    )
    R["interaction_g8_ppo"] = {"direction": f"(g8-ppo)_{OTHER} - (g8-ppo)_{REF}", "seeds": seeds}
    for metric, f in (("post", post), ("gain", gain)):
        d = {
            s: (f(runs[OTHER, "grpo_g8", s]) - f(runs[OTHER, "ppo", s]))
            - (f(runs[REF, "grpo_g8", s]) - f(runs[REF, "ppo", s]))
            for s in seeds
        }
        R["interaction_g8_ppo"][metric] = one_sample(d) if len(d) > 1 else {"n": len(d), "diffs": d}

print(json.dumps(R, indent=1, default=float))
