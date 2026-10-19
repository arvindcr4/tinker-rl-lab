#!/usr/bin/env python3
"""Recompute P11 (E1) historical statistics from the checked-in seed records.

Seed is the unit (n=8 paired seeds, df=7). Uses the corrected aggregator's own
functions (bootstrap: 10,000 resamples, RNG seed 20260714; exact noncentral
paired-t MDE80; paired-t p; Benjamini-Hochberg) so numbers match
zvf-program/audit/STATISTICAL_REANALYSIS.md. These records do NOT pass the
current strengthened admission contract (results/audit.json: 0/40 validated);
outputs are a historical recomputation, not a certified result.

Outputs (under platform_hybrid/experiments/results/review_2026-10-18/P11_reproducibility_audit/):
  e1_recompute.json   all statistics, incl. 90% CIs, paired-t CIs, margin
                      sensitivity (0.005/0.01/0.02), secondary table (ii), and
                      the pre-replay GRPO seed-11 sensitivity (0.646 vs 0.650)
  e1_historical.tex   \\Hist* macros consumed by the manuscript
"""

import importlib
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "zvf-program" / "audit"
sys.path.insert(0, str(AUDIT))
aa = importlib.import_module("aggregate_audit")

FULL = AUDIT / "results" / "full"
LEGACY = FULL / "legacy-unhashed-2026-07-19"
OUT = ROOT / "platform_hybrid/experiments/results/review_2026-10-18/P11_reproducibility_audit"
ARMS = ["dapo", "gspo", "drgrpo", "aero"]
ALL = ["grpo"] + ARMS
MARGINS = [0.005, 0.01, 0.02]


def load(directory):
    out = {}
    for p in sorted(directory.glob("*-seed-*.json")):
        r = json.loads(p.read_text())
        out[(r["arm"], r["seed"])] = r
    return out


def t_ci(d, conf):
    n = len(d)
    se = statistics.stdev(d) / math.sqrt(n)
    tc = aa._student_t_critical(n - 1, 1 - conf)  # two-sided critical value
    m = statistics.fmean(d)
    return [m - tc * se, m + tc * se]


def analyse(recs, seeds):
    res, pvals = {}, {}
    for arm in ARMS:
        d = [recs[(arm, s)]["heldout_score"] - recs[("grpo", s)]["heldout_score"] for s in seeds]
        r = {
            "paired_differences": d,
            "delta": statistics.fmean(d),
            "sd": statistics.stdev(d),
            "se": statistics.stdev(d) / math.sqrt(len(d)),
            "df": len(d) - 1,
            "boot_ci95": list(aa.paired_bootstrap_ci(d, 0.95)),
            "boot_ci90": list(aa.paired_bootstrap_ci(d, 0.90)),
            "t_ci95": t_ci(d, 0.95),
            "t_ci90": t_ci(d, 0.90),
            "mde80": aa.achieved_mde_80(d),
            "p": aa.paired_t_p_value(d),
        }
        pvals[arm] = r["p"]
        res[arm] = r
    bh = aa.benjamini_hochberg(pvals, 0.05)
    for arm, r in res.items():
        r["bh_reject"] = bh[arm]
        # lowercase on purpose: secondary-margin labels, not primary verdicts (stale-verdict gate)
        r["verdict_by_margin"] = {
            str(m): aa.verdict(
                tuple(r["boot_ci95"]),
                tuple(r["boot_ci90"]),
                r["mde80"],
                m,
                None,
                difference_rejected=bh[arm],
            ).lower()
            for m in MARGINS
        }
    return res


def secondary(recs, seeds):
    tab = {}
    for arm in ALL:
        rows = [recs[(arm, s)] for s in seeds]
        tab[arm] = {
            "last10_reward": statistics.fmean(r["last10_reward"] for r in rows),
            "mean_zvf": statistics.fmean(r["mean_zvf"] for r in rows),
            "mean_gu": statistics.fmean(r["mean_gu"] for r in rows),
            "collapse_rate": statistics.fmean(float(r["collapse"]) for r in rows),
            "rollouts": statistics.fmean(r["rollouts"] for r in rows),
            "wall_clock_h": statistics.fmean(r["wall_clock_seconds"] for r in rows) / 3600,
        }
    return tab


def main():
    prereg = json.loads((AUDIT / "preregistration.json").read_text())
    seeds = prereg["core_stratum"]["seeds"]
    recs = load(FULL)
    legacy = load(LEGACY)
    main_res = analyse(recs, seeds)
    sec = secondary(recs, seeds)
    replay = {
        f"{a}-seed-{s}": {
            "legacy": legacy[(a, s)]["heldout_score"],
            "current": recs[(a, s)]["heldout_score"],
        }
        for (a, s) in legacy
    }
    pre = dict(recs)
    pre[("grpo", 11)] = legacy[("grpo", 11)]
    pre_res = analyse(pre, seeds)
    out = {
        "source": "zvf-program/audit/results/full/*-seed-*.json (+ legacy-unhashed-2026-07-19 for replay)",
        "admission_status": json.loads((AUDIT / "results" / "audit.json").read_text())["status"],
        "seeds": seeds,
        "grpo_mean": statistics.fmean(recs[("grpo", s)]["heldout_score"] for s in seeds),
        "main": main_res,
        "secondary_table_ii": sec,
        "replay_scores": replay,
        "pre_replay_grpo_s11_sensitivity": pre_res,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "e1_recompute.json").write_text(json.dumps(out, indent=1))

    name = {"dapo": "DAPO", "gspo": "GSPO", "drgrpo": "DrGRPO", "aero": "AERO", "grpo": "GRPO"}
    f5 = lambda x: f"{x:+.5f}"
    lines = [
        "% Generated by tools/review_ledger/P11_reproducibility_audit_recompute.py",
        "% Historical recomputation from uncertified seed records; NOT an admission-gate output.",
    ]
    for arm in ARMS:
        r, n = main_res[arm], name[arm]
        lines += [
            f"\\newcommand{{\\Hist{n}Delta}}{{{f5(r['delta'])}}}",
            f"\\newcommand{{\\Hist{n}CIlowNinetyFive}}{{{f5(r['boot_ci95'][0])}}}",
            f"\\newcommand{{\\Hist{n}CIhighNinetyFive}}{{{f5(r['boot_ci95'][1])}}}",
            f"\\newcommand{{\\Hist{n}CIlowNinety}}{{{f5(r['boot_ci90'][0])}}}",
            f"\\newcommand{{\\Hist{n}CIhighNinety}}{{{f5(r['boot_ci90'][1])}}}",
            f"\\newcommand{{\\Hist{n}MDE}}{{{r['mde80']:.5f}}}",
            f"\\newcommand{{\\Hist{n}P}}{{{r['p']:.3f}}}",
            f"\\newcommand{{\\Hist{n}PreReplayMDE}}{{{pre_res[arm]['mde80']:.5f}}}",
            f"\\newcommand{{\\Hist{n}PreReplayDelta}}{{{f5(pre_res[arm]['delta'])}}}",
        ]
    for arm in ALL:
        s, n = sec[arm], name[arm]
        lines += [
            f"\\newcommand{{\\Hist{n}LastTen}}{{{s['last10_reward']:.3f}}}",
            f"\\newcommand{{\\Hist{n}Zvf}}{{{s['mean_zvf']:.3f}}}",
            f"\\newcommand{{\\Hist{n}Rollouts}}{{{s['rollouts']:.0f}}}",
            f"\\newcommand{{\\Hist{n}Hours}}{{{s['wall_clock_h']:.2f}}}",
            f"\\newcommand{{\\Hist{n}Collapse}}{{{s['collapse_rate']:.2f}}}",
        ]
    (OUT / "e1_historical.tex").write_text("\n".join(lines) + "\n")
    for arm in ARMS:
        r, q = main_res[arm], pre_res[arm]
        print(
            arm,
            f"d={r['delta']:+.5f} b95={r['boot_ci95']} b90={r['boot_ci90']} t95={r['t_ci95']} "
            f"mde={r['mde80']:.5f} p={r['p']:.3f} bh={r['bh_reject']} v={r['verdict_by_margin']} | "
            f"pre-replay d={q['delta']:+.5f} mde={q['mde80']:.5f} b90={q['boot_ci90']} v={q['verdict_by_margin']}",
        )
    for arm in ALL:
        print(arm, {k: round(v, 4) for k, v in sec[arm].items()})
    print(replay)


if __name__ == "__main__":
    main()
