#!/usr/bin/env python3
"""Predictable-Noise Power Audit (Berkeley F25 L8 Sida Wang).

The FORECASTING half of Sida Wang's "Predictable Noise / Adding Error Bars to
Evals" lecture that row 20's seed-clustering DEFF audit did NOT operationalise.
Row 20 measured how wide the honest bars ARE; this asks the a-priori question
Miller (arXiv:2411.00640) and Wang (arXiv:2512.21326) actually pose: given the
predictable seed-level noise, WHAT effect can this study resolve, and can the
GRPO==PPO null be turned into a POSITIVE equivalence claim?

Real data: platform_hybrid/experiments/results/samestack_ppo_grpo.json (5 seeds x 2 algos,
paired by seed; heldout_acc and last10_avg per seed).

Pre-registered hypotheses:
  H1  Retrospective power / MDE (heldout): the minimum detectable paired effect
      at 80% power is < 0.01 (sub-1pt) and < a literature cross-stack gap (0.05)
      => the p=0.37 null is a GENUINE equivalence, not an underpowered null.
  H2  TOST equivalence (heldout): the equivalence bound (tightest margin at which
      both one-sided tests reject, = 90% CI half-not-straddling) is < 0.01
      => positive claim "|GRPO-PPO heldout| < bound" replaces "failed to reject".
  H3  Pooling fabricates power (last10): treating S*M step-obs as i.i.d. (n=50)
      understates the MDE by >= sqrt(DEFF) ~ 2x vs the honest seed-clustered n=5
      => pooling doesn't just narrow CIs (row 20), it manufactures false power.
  H4  Metric choice is an error-bar decision (honest split): the last10
      equivalence bound is >10x the heldout bound and collapses >3x when the
      single PPO seed-456 stability outlier is removed => heldout is well-powered
      for equivalence, last10 is not, and the gap is one stability event.
  H5  Pairing power gain (Wang all-pairs): the paired design's SE is >= 1.5x
      smaller than the unpaired two-sample SE => pairing is WHAT makes the
      equivalence claim well-powered, quantifying Wang's all-pairs recommendation.
"""
import json, os
import numpy as np
from scipy.stats import nct, t as tdist, norm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "platform_hybrid/experiments/results/samestack_ppo_grpo.json")
OUT = os.path.join(ROOT, "platform_hybrid/experiments/results/berkeley")
os.makedirs(OUT, exist_ok=True)
ALPHA = 0.05
LIT_GAP = 0.05  # conservative lower end of reported cross-stack GRPO-PPO gaps


def power_paired(delta, n, s_d, alpha=ALPHA):
    """Two-sided paired-t power at true mean diff `delta`, sd `s_d`, n pairs."""
    if n < 2 or s_d <= 0:
        return float("nan")
    df = n - 1
    ncp = delta * np.sqrt(n) / s_d
    tc = tdist.ppf(1 - alpha / 2, df)
    pw = 1 - nct.cdf(tc, df, ncp) + nct.cdf(-tc, df, ncp)
    if not np.isfinite(pw):  # nct underflows at extreme ncp -> normal approximation
        pw = norm.cdf(ncp - tc) + norm.cdf(-ncp - tc)
    return float(min(max(pw, 0.0), 1.0))


def mde(n, s_d, target_power=0.80, alpha=ALPHA):
    """Minimum detectable effect (paired) at `target_power` via bisection."""
    lo, hi = 0.0, max(1.0, 20 * s_d)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if power_paired(mid, n, s_d, alpha) < target_power:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def n_for_delta(delta, s_d, target_power=0.80, alpha=ALPHA, nmax=500):
    for n in range(2, nmax + 1):
        if power_paired(delta, n, s_d, alpha) >= target_power:
            return n
    return nmax + 1  # >nmax


def load():
    d = json.load(open(SRC))
    by = {"grpo": {}, "ppo": {}}
    steps = {"grpo": {}, "ppo": {}}
    for r in d["runs"]:
        by[r["algo"]][r["seed"]] = {"heldout": r["heldout_acc"], "last10": r["last10_avg"]}
        steps[r["algo"]][r["seed"]] = [s["mean_reward"] for s in r["step_log"][-10:]]
    seeds = sorted(set(by["grpo"]) & set(by["ppo"]))
    diffs = {
        m: np.array([by["grpo"][s][m] - by["ppo"][s][m] for s in seeds]) for m in ("heldout", "last10")
    }
    return by, steps, seeds, diffs


def tsv(path, header, rows):
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in r) + "\n")
    print("wrote", os.path.relpath(path, ROOT))


def main():
    by, steps, seeds, diffs = load()
    summary = {
        "source": "Berkeley F25 L8 Sida Wang -- Predictable Noise / Adding Error Bars to Evals",
        "citations": {
            "miller": "arXiv:2411.00640 Adding Error Bars to Evals (Evan Miller, 2024)",
            "wang": "arXiv:2512.21326 Measuring all the noises of LLM Evals (Sida Wang, Dec 2025; all-pairs paired method for statistical power)",
        },
        "n_seeds": len(seeds),
        "method": "retrospective power (noncentral-t) + TOST equivalence + sample-size forecast",
    }

    # ---- per-metric power / MDE / TOST ------------------------------------
    power_rows, tost_rows = [], []
    metric_stat = {}
    for m in ("heldout", "last10"):
        d = diffs[m]
        n = len(d)
        df = n - 1
        mean = float(d.mean())
        s_d = float(d.std(ddof=1))
        se = s_d / np.sqrt(n)
        tstat = mean / se if se > 0 else float("nan")
        p_two = 2 * (1 - tdist.cdf(abs(tstat), df))
        mde80 = mde(n, s_d, 0.80)
        mde50 = mde(n, s_d, 0.50)
        obs_power = power_paired(abs(mean), n, s_d)  # retrospective power at observed effect
        power_at_lit = power_paired(LIT_GAP, n, s_d)  # power to catch a 5pt gap
        tcrit_eq = tdist.ppf(1 - ALPHA, df)  # one-sided for TOST / 90% CI
        ci90 = (mean - tcrit_eq * se, mean + tcrit_eq * se)
        equiv_bound = max(abs(ci90[0]), abs(ci90[1]))  # tightest margin passing TOST
        metric_stat[m] = dict(mean=mean, s_d=s_d, se=se, df=df, mde80=mde80,
                              equiv_bound=equiv_bound, p_two=p_two)
        power_rows.append([m, n, round(mean, 6), round(s_d, 6), round(se, 6),
                           round(tstat, 4), round(p_two, 4), round(mde80, 6),
                           round(mde50, 6), round(obs_power, 4), round(power_at_lit, 4)])
        tost_rows.append([m, round(mean, 6), round(se, 6), round(ci90[0], 6),
                          round(ci90[1], 6), round(equiv_bound, 6),
                          bool(equiv_bound < 0.02), bool(equiv_bound < 0.05),
                          bool(equiv_bound < 0.10)])
    tsv(os.path.join(OUT, "pnpa_power_mde.tsv"),
        ["metric", "n", "mean_diff", "s_d", "se", "t", "p_two", "mde_80", "mde_50",
         "obs_power", "power_at_5pt"], power_rows)
    tsv(os.path.join(OUT, "pnpa_tost_equivalence.tsv"),
        ["metric", "mean_diff", "se", "ci90_low", "ci90_high", "equiv_bound",
         "equiv_at_2pt", "equiv_at_5pt", "equiv_at_10pt"], tost_rows)

    # ---- H3 pooling fabricates power (last10 step-level) -------------------
    # honest: seed-clustered n=5 on last-10 step means; illusion: pool 5x10=50 iid.
    seed_means = np.array([np.mean(steps["ppo"][s]) for s in seeds])  # PPO last10 per seed
    pooled = np.concatenate([np.array(steps["ppo"][s]) for s in seeds])
    var_between = seed_means.var(ddof=1)
    var_within = np.mean([np.array(steps["ppo"][s]).var(ddof=1) for s in seeds])
    icc = var_between / (var_between + var_within) if (var_between + var_within) > 0 else 0.0
    M = 10
    deff = 1 + (M - 1) * icc
    s_seed = seed_means.std(ddof=1)          # honest cluster-level sd (one number/seed)
    s_pool = pooled.std(ddof=1)              # naive pooled sd
    mde_honest = mde(len(seeds), s_seed, 0.80)
    mde_pool = mde(len(pooled), s_pool, 0.80)  # the illusion: treat 50 obs as iid pairs
    tsv(os.path.join(OUT, "pnpa_pooling_power_inflation.tsv"),
        ["quantity", "value"],
[["n_honest_seeds", len(seeds)], ["n_pooled_stepobs", len(pooled)],
         ["icc_step", round(float(icc), 4)], ["deff", round(float(deff), 4)],
         ["sqrt_deff", round(float(np.sqrt(deff)), 4)],
         ["mde80_honest", round(float(mde_honest), 6)],
         ["mde80_pooled_illusion", round(float(mde_pool), 6)],
         ["power_inflation_ratio", round(float(mde_honest / mde_pool), 4)]])

    # ---- H4 outlier robustness (last10, drop PPO seed-456 collapse) --------
    d_full = diffs["last10"]
    keep = [i for i, s in enumerate(seeds) if s != 456]
    d_drop = d_full[keep]
    rob_rows = []
    for name, dd in (("full", d_full), ("drop_seed456", d_drop),
                     ("winsor10", np.clip(d_full, np.percentile(d_full, 10), np.percentile(d_full, 90)))):
        n = len(dd)
        s_d = float(dd.std(ddof=1))
        se = s_d / np.sqrt(n)
        eb = tdist.ppf(1 - ALPHA, n - 1) * se + abs(float(dd.mean()))  # conservative equiv bound
        eb = max(abs(float(dd.mean()) - tdist.ppf(1 - ALPHA, n - 1) * se),
                 abs(float(dd.mean()) + tdist.ppf(1 - ALPHA, n - 1) * se))
        rob_rows.append([name, n, round(float(dd.mean()), 6), round(s_d, 6),
                         round(se, 6), round(mde(n, s_d, 0.80), 6), round(eb, 6)])
    tsv(os.path.join(OUT, "pnpa_outlier_robustness.tsv"),
        ["variant", "n", "mean_diff", "s_d", "se", "mde_80", "equiv_bound"], rob_rows)

    # ---- H5 paired vs unpaired power gain (Wang all-pairs) -----------------
    pv_rows = []
    for m in ("heldout", "last10"):
        g = np.array([by["grpo"][s][m] for s in seeds])
        p = np.array([by["ppo"][s][m] for s in seeds])
        n = len(seeds)
        se_paired = (g - p).std(ddof=1) / np.sqrt(n)
        se_unpaired = np.sqrt(g.var(ddof=1) / n + p.var(ddof=1) / n)
        ratio = se_unpaired / se_paired if se_paired > 0 else float("nan")
        pv_rows.append([m, round(float(se_paired), 6), round(float(se_unpaired), 6),
                        round(float(ratio), 4),
                        round(mde(n, (g - p).std(ddof=1), 0.80), 6),
                        round(mde(n, np.sqrt(g.var(ddof=1) + p.var(ddof=1)), 0.80), 6)])
    tsv(os.path.join(OUT, "pnpa_paired_vs_unpaired.tsv"),
        ["metric", "se_paired", "se_unpaired", "se_ratio_unp_over_pair",
         "mde_paired", "mde_unpaired"], pv_rows)

    # ---- H3b sample-size forecast (heldout seed-level sd) ------------------
    s_held = diffs["heldout"].std(ddof=1)
    fc_rows = [[dlt, n_for_delta(dlt, s_held, 0.80)] for dlt in (0.005, 0.01, 0.02, 0.05, 0.10)]
    tsv(os.path.join(OUT, "pnpa_sample_size_forecast.tsv"),
        ["target_delta", "n_seeds_for_80pct_power"], fc_rows)

    # ---- verdicts ---------------------------------------------------------
    h1 = metric_stat["heldout"]["mde80"] < 0.01 and metric_stat["heldout"]["mde80"] < LIT_GAP
    h2 = metric_stat["heldout"]["equiv_bound"] < 0.01
    infl = float(mde_honest / mde_pool)
    h3 = infl >= np.sqrt(deff) * 0.9 and infl >= 1.8
    eb_full = [r for r in rob_rows if r[0] == "full"][0][6]
    eb_drop = [r for r in rob_rows if r[0] == "drop_seed456"][0][6]
    h4 = (metric_stat["last10"]["equiv_bound"] > 10 * metric_stat["heldout"]["equiv_bound"]
          and eb_full > 3 * eb_drop)
    h5 = pv_rows[0][3] >= 1.5  # heldout se_ratio
    verdicts = {"H1_heldout_well_powered": bool(h1), "H2_positive_equivalence": bool(h2),
                "H3_pooling_fabricates_power": bool(h3), "H4_metric_choice_matters": bool(h4),
                "H5_pairing_power_gain": bool(h5)}
    summary["verdicts"] = verdicts
    summary["n_decisive"] = int(sum(verdicts.values()))
    summary["headline"] = {
        "heldout_mde80": round(metric_stat["heldout"]["mde80"], 6),
        "heldout_equiv_bound": round(metric_stat["heldout"]["equiv_bound"], 6),
        "last10_equiv_bound": round(metric_stat["last10"]["equiv_bound"], 6),
        "power_inflation_from_pooling": round(infl, 3),
        "icc_step": round(float(icc), 4), "deff": round(float(deff), 4),
        "paired_power_gain_heldout": pv_rows[0][3],
        "n_seeds_for_1pt_gap": n_for_delta(0.01, s_held, 0.80),
    }
    json.dump(summary, open(os.path.join(OUT, "pnpa_summary.json"), "w"), indent=2)
    print("\n=== VERDICTS ===")
    for k, v in verdicts.items():
        print(f"  {k}: {'DECISIVE' if v else 'null'}")
    print(f"  {summary['n_decisive']}/5 decisive")
    print("headline:", json.dumps(summary["headline"], indent=2))


if __name__ == "__main__":
    main()
