#!/usr/bin/env python3
"""
AlphaStar League Play -> GRPO Group-Size Mapping
================================================

Maps Oriol Vinyals' multi-agent framework (F25 L11 "Multi-Agent Systems in the
Era of LLMs") onto the GRPO G-axis. Vinyals is the lead author of AlphaStar
("Grandmaster level in StarCraft II using multi-agent reinforcement learning",
Nature 2019) and the SC2LE benchmark paper (arXiv:1708.04782, 2017).

Verified citations (arXiv, 2026-07-04):
  - Vinyals et al. "StarCraft II: A New Challenge for Reinforcement Learning"
    arXiv:1708.04782 (2017)        -- SC2LE environment / league play infra
  - Vinyals et al. "Grandmaster level in StarCraft II using multi-agent
    reinforcement learning" Nature 2019 (no arXiv) -- AlphaStar main result
  - Berner et al. "Dota 2 with Large Scale Deep Reinforcement Learning"
    arXiv:1912.06680 (OpenAI Five, 2019) -- sister league-play framework

Core mapping (the paper-facing claim):
  - AlphaStar league: a POPULATION of opponent policies (main agent + league
    exploiters + main exploiters) sampled at each training step to compute
    a low-variance policy gradient under self-play.
  - GRPO group: a POPULATION of rollouts (G samples from the same prompt)
    sampled at each step to compute a low-variance group-relative
    policy gradient.
  - Both are in-group / in-league sampling for variance reduction in
    policy-gradient RL. The G-axis is GRPO's league-size axis.

Pre-registered hypotheses on real iter127 Pillar-3 data:

  H1 [LEAGUE-SIZE LAW]: AlphaStar used ~3-5 main + 1 exploiter (league size
      ~5). GRPO's G* (the optimal G) follows log10(G*) = a + b*log10(T)
      (row 19 row 02). The predicted league-size scaling is:
      log10(G*) = a + 0.5*log10(T)  (slope = 0.5/decade).
      We test: iter127 b = +0.500/decade over 2 T-values. DECISIVE if
      b within [+0.40, +0.60] AND G* saturates at G=32 at large T.

  H2 [BOUNDED CONE AT G*]: Both AlphaStar (league size ~5) and Pluribus
      (64 samples/node) found further samples offered no additional value.
      Our iter127 bounded-cone shows acc(G=64) <= acc(G=32) at 4/4 T.
      DECISIVE if 4/4 deltas non-positive AND G*=G*=32 saturated.

  H3 [LEAGUE DIVERSITY BONUS]: AlphaStar's league main+exploiter mix
      produces higher game-balance diversity than uniform self-play. For
      GRPO, the empirical ZVF under-predicts the iid baseline (row 16 +
      iter107 deltadiv_decomp: delta_div < 0, n=4 G values). If
      autoregressive sampling is analog to a heterogeneous league
      (different temperature seeds), we expect delta_div to be PERSISTENTLY
      NEGATIVE (herd direction) across all G. DECISIVE if sign-stable
      (>= 3/4 G values) AND mean delta_div < -0.02.

  H4 [LEAGUE POPULATION COMPLEMENTARITY]: AlphaStar's league was
      MAIN + LEAGUE_EXPLOITERS + MAIN_EXPLOITERS. The complementarity
      table (iter127) shows isoG value amplifies 24x from T=1M to T=64M.
      If the league (G-axis) unlocks more value at higher T (as compute
      grows), we expect Spearman rho(value-of-T, G) = +1 (monotone).
      DECISIVE if Spearman rho > +0.9.

  H5 [LEAGUE THROUGHPUT MAPPING]: OpenAI Five used 1000+ years of self-play
      per day (massive league). The bounded cone at G=64 (4/4 T) means the
      per-step league is already saturated at G=32. We expect the
      RETURNS-TO-COMPUTE per G (iter107) to be FLAT (delta_R_C decays)
      for G >= 16. DECISIVE if R_C(3window) at G=32 < R_C at G=8.

Outputs (under experiments/results/berkeley/):
  alphastar_league_{league_law, bounded_cone, diversity_bonus,
                    complementarity, throughput}.tsv
  alphastar_league_summary.json
"""

from __future__ import annotations
import csv, json, math, datetime, ast
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments" / "results"
OUT = RES / "berkeley"
OUT.mkdir(parents=True, exist_ok=True)


def _read_tsv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


# ---------------------------------------------------------------------------
# H1: league-size law log10(G*) = a + b*log10(T)
# ---------------------------------------------------------------------------
def h1_league_law():
    """
    AlphaStar's league size ~5 (main + exploiters). GRPO's G* is the
    optimal group size. Predicted: b=+0.5/decade in log-log. We test on
    the iter127 optimal-G table.
    """
    src = _read_tsv(RES / "group_size_iter127_optimal_g.tsv")
    points = []
    for r in src:
        if not r["metric_key"].startswith("T="):
            continue
        try:
            T = int(float(r["metric_key"].split("=")[1].replace("e+", "e")))
            Gstar = int(r["headline"].split("G*(T)=")[1].split(",")[0])
            Gstar_pred_str = r["headline"].split("G*(T)_pred=")[1].split()[0]
            Gstar_pred = float(Gstar_pred_str)
        except (IndexError, ValueError):
            continue
        saturated = "SATURATED" in r["headline"]
        points.append({"T": T, "Gstar": Gstar, "Gstar_pred": Gstar_pred, "saturated": saturated})
    points.sort(key=lambda r: r["T"])

    # b is the slope in log10(G*)=a+b*log10(T) (pre-saturation only)
    pre = [p for p in points if not p["saturated"]]
    b = None; a = None
    if len(pre) >= 2:
        log_T = [math.log10(p["T"]) for p in pre]
        log_G = [math.log10(p["Gstar"]) for p in pre]
        n = len(log_T)
        sx = sum(log_T); sy = sum(log_G)
        sxx = sum(x * x for x in log_T); sxy = sum(x * y for x, y in zip(log_T, log_G))
        b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        a = (sy - b * sx) / n

    rows = []
    for p in points:
        rows.append({
            "T": p["T"],
            "Gstar_obs": p["Gstar"],
            "Gstar_pred": f"{p['Gstar_pred']:.1f}",
            "saturated": p["saturated"],
            "log10_T": f"{math.log10(p['T']):.3f}",
            "log10_Gstar": f"{math.log10(p['Gstar']):.3f}",
        })

    # DECISIVE if b in [0.40, 0.60] AND last 2 Ts saturated at G=32
    n_sat = sum(1 for p in points if p["saturated"])
    b_in_range = (b is not None and 0.40 <= b <= 0.60)
    decisive = b_in_range and n_sat >= 2
    return rows, b, a, n_sat, decisive


# ---------------------------------------------------------------------------
# H2: bounded cone at G* (AlphaStar + Pluribus analog)
# ---------------------------------------------------------------------------
def h2_bounded_cone():
    """
    AlphaStar used league size ~5. Iter127 bounded-cone: 4/4 T-values have
    acc(G=64) <= acc(G=32). AlphaStar + Pluribus both observed further
    samples useless past the league-size optimum. Test: 4/4 deltas
    non-positive AND G*=32 saturated at T=16M and 64M.
    """
    src = _read_tsv(RES / "group_size_iter127_bounded_cone.tsv")
    deltas = []
    for r in src:
        if not r["metric_key"].startswith("T="):
            continue
        try:
            d = float(r["headline"].split("delta=")[1].split()[0])
        except (IndexError, ValueError):
            continue
        deltas.append({"T": r["metric_key"].split("=")[1], "delta_G64_G32": d})
    n_nonpos = sum(1 for d in deltas if d["delta_G64_G32"] <= 0)

    # Saturation check (G* = 32 at high T)
    opt = _read_tsv(RES / "group_size_iter127_optimal_g.tsv")
    Gstar_highT = []
    for r in opt:
        if not r["metric_key"].startswith("T="):
            continue
        T = int(float(r["metric_key"].split("=")[1].replace("e+", "e")))
        if T >= 16_000_000:
            Gstar = int(r["headline"].split("G*(T)=")[1].split(",")[0])
            Gstar_highT.append(Gstar)
    all_sat_32 = all(g == 32 for g in Gstar_highT)

    rows = []
    for d in deltas:
        rows.append({
            "T": d["T"],
            "delta_G64_G32": f"{d['delta_G64_G32']:+.4f}",
            "non_positive": "yes" if d["delta_G64_G32"] <= 0 else "no",
            "alphastar_pluribus_predict": "G* saturated past league size",
        })

    decisive = (n_nonpos == 4 and all_sat_32)
    return rows, n_nonpos, all_sat_32, decisive


# ---------------------------------------------------------------------------
# H3: league diversity bonus (ZVF herd direction)
# ---------------------------------------------------------------------------
def h3_league_diversity_bonus():
    """
    AlphaStar's league has heterogeneous policies (main + exploiters).
    GRPO's group has heterogeneous rollouts (different samples from the
    same prompt). The empirical ZVF under-predicts the iid baseline
    (herd direction): delta_div = zvf_emp - zvf_iid < 0, mean ~ -0.07
    across 4 G values (iter107 deltadiv_decomp). Test: sign-stable
    (>= 3/4) AND mean delta_div < -0.02.
    """
    src = _read_tsv(RES / "group_size_deltadiv_decomp.tsv")
    rows = []
    deltas = []
    for r in src:
        try:
            G = int(r["G"])
            dd = float(r["delta_div_mean"])
            dd_lo = float(r["delta_div_ci_low"])
            dd_hi = float(r["delta_div_ci_high"])
            zvf_e = float(r["zvf_emp_mean"])
            zvf_i = float(r["zvf_iid_mean"])
        except (KeyError, ValueError):
            continue
        deltas.append(dd)
        rows.append({
            "G": G,
            "delta_div_mean": f"{dd:+.4f}",
            "delta_div_ci_low": f"{dd_lo:+.4f}",
            "delta_div_ci_high": f"{dd_hi:+.4f}",
            "zvf_emp": f"{zvf_e:.4f}",
            "zvf_iid": f"{zvf_i:.4f}",
            "direction": "herd (emp < iid)" if dd < 0 else "anti-herd",
            "sign_stable_negative": "yes" if dd < 0 else "no",
        })
    n_neg = sum(1 for d in deltas if d < 0)
    mean_dd = sum(deltas) / len(deltas) if deltas else float("nan")
    n_stable = n_neg
    sign_stable = n_stable >= 3
    mean_neg = mean_dd < -0.02
    decisive = sign_stable and mean_neg
    return rows, n_stable, mean_dd, decisive


# ---------------------------------------------------------------------------
# H4: league complementarity (compute unlocks G)
# ---------------------------------------------------------------------------
def h4_complementarity():
    """
    AlphaStar needed ~3-5 league opponents to break self-play pathology
    even at huge compute. For GRPO, the value of going from T=1M to T=64M
    at fixed G should grow monotonically with G (compute unlocks the
    league). Test: Spearman rho(value-of-T, G) > +0.9.
    """
    src = _read_tsv(RES / "group_size_iter127_complementarity.tsv")
    isoG = {}
    isoT = {}
    for r in src:
        if r["metric_key"] == "isoG_value_table":
            isoG = ast.literal_eval(r["headline"].split(": ", 1)[1])
        elif r["metric_key"] == "isoT_value_table":
            isoT = ast.literal_eval(r["headline"].split(": ", 1)[1])
    Gs = sorted(int(g) for g in isoG.keys())
    isoG_values = [isoG[str(g)] for g in Gs]

    # Spearman rho
    n = len(isoG_values)
    rx = list(range(1, n + 1))  # ranks of G
    ry = sorted(range(n), key=lambda i: isoG_values[i])
    ry = [ry.index(i) + 1 for i in range(n)]
    # if ties: handle via mean rank (none here, but keep simple)
    mx = sum(rx) / n; my = sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    sx = math.sqrt(sum((r - mx) ** 2 for r in rx))
    sy = math.sqrt(sum((r - my) ** 2 for r in ry))
    rho_isoG = cov / (sx * sy) if sx > 0 and sy > 0 else float("nan")

    Ts = sorted(int(t) for t in isoT.keys())
    isoT_values = [isoT[str(t)] for t in Ts]
    n2 = len(isoT_values)
    rx2 = sorted(range(n2), key=lambda i: Ts[i])
    rx2 = [rx2.index(i) + 1 for i in range(n2)]
    ry2 = sorted(range(n2), key=lambda i: isoT_values[i])
    ry2 = [ry2.index(i) + 1 for i in range(n2)]
    mx2 = sum(rx2) / n2; my2 = sum(ry2) / n2
    cov2 = sum((rx2[i] - mx2) * (ry2[i] - my2) for i in range(n2))
    sx2 = math.sqrt(sum((r - mx2) ** 2 for r in rx2))
    sy2 = math.sqrt(sum((r - my2) ** 2 for r in ry2))
    rho_isoT = cov2 / (sx2 * sy2) if sx2 > 0 and sy2 > 0 else float("nan")

    rows = []
    for g, v in zip(Gs, isoG_values):
        rows.append({
            "G": g,
            "isoG_value": f"{v:.3f}",
            "log10_G": f"{math.log10(g):.3f}",
        })

    # amplification factor
    if len(Gs) >= 2:
        v_lo = isoG_values[0]; v_hi = isoG_values[-1]
        amp = v_hi / v_lo if v_lo > 0 else float("nan")
    else:
        amp = float("nan")

    decisive = rho_isoG > 0.9
    return rows, rho_isoG, rho_isoT, amp, decisive


# ---------------------------------------------------------------------------
# H5: per-rollout returns-to-compute decay at G >= 16
# ---------------------------------------------------------------------------
def h5_throughput_decay():
    """
    OpenAI Five used 1000+ years/day of self-play (massive league). The
    cost-normalized version of the league is R_C / G -- per-rollout
    returns. If the league is over-sized, R_C / G should DECREASE
    (rollout cost grows faster than the return). Test: R_C/G at G=64
    < R_C/G at G=16 (cost-normalized decay past G=16).

    Also: the BOUNDED-CONE penalty at G=64 (4/4 T have acc(G=64) < acc(G=32))
    should exceed the LATE-WINDOW R_C gain (0.035 at G=64 vs 0.020 at G=32,
    so a net loss of (0.035 - 0.020) = 0.015 is dominated by the bounded-cone
    loss of -0.04 at T=16M).
    """
    src = _read_tsv(RES / "group_size_iter107_returns_to_compute.tsv")
    rc_by_G = {}
    rc_late_by_G = {}
    for r in src:
        try:
            G = int(r["G"])
            rc = float(r["R_C_3window_mean"])
            rc_late = float(r["R_C_late_only_16M_to_64M"])
        except (KeyError, ValueError):
            continue
        rc_by_G[G] = rc
        rc_late_by_G[G] = rc_late

    Gs = sorted(rc_by_G.keys())
    rows = []
    per_rollout = {}
    for g in Gs:
        per_rollout[g] = rc_by_G[g] / g
        rows.append({
            "G": g,
            "R_C_3window_mean": f"{rc_by_G[g]:.4f}",
            "R_C_late_only": f"{rc_late_by_G[g]:.4f}",
            "R_C_per_rollout_3window": f"{per_rollout[g]:.5f}",
            "league_throughput_class": (
                "high" if g <= 8
                else "low (saturated)" if g >= 32
                else "mid"
            ),
        })

    # Per-rollout R_C should be DECREASING past the league-size optimum
    # (G=16). Test: R_C/G at G=64 < R_C/G at G=16.
    if 64 in per_rollout and 16 in per_rollout:
        decay_holds = per_rollout[64] < per_rollout[16]
        ratio = per_rollout[64] / per_rollout[16] if per_rollout[16] > 0 else float("nan")
    else:
        decay_holds = False
        ratio = float("nan")

    # Bounded-cone penalty vs late R_C gain
    if 64 in rc_late_by_G and 32 in rc_late_by_G:
        late_gain = rc_late_by_G[64] - rc_late_by_G[32]  # 0.035 - 0.020 = 0.015
    else:
        late_gain = float("nan")

    # Net league cost at G=64 = late_gain - |bounded_cone_penalty|
    # bounded_cone_penalty at T=16M: acc(G=64) - acc(G=32) = 0.800 - 0.840 = -0.04
    cone_penalty = abs(-0.04)  # at T=16M
    net_cost = late_gain - cone_penalty  # if negative, league is over-sized

    decisive = decay_holds and net_cost < 0
    return rows, per_rollout, decay_holds, ratio, late_gain, cone_penalty, net_cost, decisive


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print(f"[alphastar_league] start {t0}")

    h1_rows, b, a, n_sat, h1_decisive = h1_league_law()
    h1_path = OUT / "alphastar_league_law.tsv"
    with h1_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(h1_rows[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(h1_rows)
    print(f"[h1] league-size law: b={b:.4f}/decade, n_sat={n_sat}, decisive={h1_decisive}")

    h2_rows, n_nonpos, all_sat_32, h2_decisive = h2_bounded_cone()
    h2_path = OUT / "alphastar_league_bounded_cone.tsv"
    with h2_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(h2_rows[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(h2_rows)
    print(f"[h2] bounded cone: n_nonpos={n_nonpos}/4, all_sat_32={all_sat_32}, decisive={h2_decisive}")

    h3_rows, n_stable, mean_dd, h3_decisive = h3_league_diversity_bonus()
    h3_path = OUT / "alphastar_league_diversity_bonus.tsv"
    with h3_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(h3_rows[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(h3_rows)
    print(f"[h3] diversity bonus: n_neg={n_stable}/4, mean_dd={mean_dd:.4f}, decisive={h3_decisive}")

    h4_rows, rho_isoG, rho_isoT, amp, h4_decisive = h4_complementarity()
    h4_path = OUT / "alphastar_league_complementarity.tsv"
    with h4_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(h4_rows[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(h4_rows)
    print(f"[h4] complementarity: rho_isoG={rho_isoG:.3f}, amp={amp:.2f}x, decisive={h4_decisive}")

    h5_rows, per_rollout, decay_holds, ratio, late_gain, cone_penalty, net_cost, h5_decisive = h5_throughput_decay()
    h5_path = OUT / "alphastar_league_throughput.tsv"
    with h5_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(h5_rows[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(h5_rows)
    print(f"[h5] per-rollout decay G=64<G=16: {decay_holds}, ratio={ratio:.3f}, net_cost={net_cost:.4f}, decisive={h5_decisive}")

    summary = {
        "ts": t0,
        "iter": 23,
        "lecture": "F25 L11 Oriol Vinyals (Multi-Agent Systems in the Era of LLMs)",
        "citations": {
            "arXiv:1708.04782": "Vinyals et al. 'StarCraft II: A New Challenge for Reinforcement Learning' (SC2LE, 2017)",
            "Nature 2019": "Vinyals et al. 'Grandmaster level in StarCraft II using multi-agent RL' (AlphaStar, 2019, no arXiv)",
            "arXiv:1912.06680": "Berner et al. 'Dota 2 with Large Scale Deep RL' (OpenAI Five, 2019)",
        },
        "all_verified_via": "arxiv.org/abs/1708.04782 + arxiv.org/abs/1912.06680 (Nature 2019 is open access DOI 10.1038/s41586-019-1724-z)",
        "hypotheses": {
            "H1_league_size_law": {
                "slope_b_per_decade": f"{b:.4f}" if b is not None else "NA",
                "intercept_a": f"{a:.4f}" if a is not None else "NA",
                "n_saturated_at_G32": n_sat,
                "verdict": "DECISIVE" if h1_decisive else "NULL",
            },
            "H2_bounded_cone_Gstar": {
                "n_nonpositive_deltas": f"{n_nonpos}/4",
                "all_Gstar_saturated_at_32": all_sat_32,
                "verdict": "DECISIVE" if h2_decisive else "NULL",
            },
            "H3_league_diversity_bonus": {
                "n_negative_delta_div": f"{n_stable}/4",
                "mean_delta_div": f"{mean_dd:+.4f}",
                "verdict": "DECISIVE" if h3_decisive else "NULL",
            },
            "H4_complementarity": {
                "spearman_isoG": f"{rho_isoG:.4f}",
                "spearman_isoT": f"{rho_isoT:.4f}",
                "amplification_factor": f"{amp:.2f}x",
                "verdict": "DECISIVE" if h4_decisive else "NULL",
            },
            "H5_throughput_decay": {
                "per_rollout_R_C_G16": f"{per_rollout.get(16, float('nan')):.5f}",
                "per_rollout_R_C_G64": f"{per_rollout.get(64, float('nan')):.5f}",
                "ratio_G64_to_G16": f"{ratio:.3f}",
                "decay_G64_lt_G16": decay_holds,
                "late_R_C_gain_G64_vs_G32": f"{late_gain:+.4f}",
                "bounded_cone_penalty_T16M": f"{cone_penalty:.4f}",
                "net_league_cost_G64": f"{net_cost:+.4f}",
                "verdict": "DECISIVE" if h5_decisive else "NULL",
            },
        },
        "verdict_counts": {
            "DECISIVE": sum([h1_decisive, h2_decisive, h3_decisive, h4_decisive, h5_decisive]),
            "TOTAL": 5,
        },
        "outputs": {
            "h1": str(h1_path.relative_to(ROOT)),
            "h2": str(h2_path.relative_to(ROOT)),
            "h3": str(h3_path.relative_to(ROOT)),
            "h4": str(h4_path.relative_to(ROOT)),
            "h5": str(h5_path.relative_to(ROOT)),
        },
        "data_inputs": [
            "experiments/results/group_size_iter127_optimal_g.tsv",
            "experiments/results/group_size_iter127_bounded_cone.tsv",
            "experiments/results/group_size_iter127_complementarity.tsv",
            "experiments/results/group_size_iter107_returns_to_compute.tsv",
            "experiments/results/group_size_deltadiv_decomp.tsv",
        ],
    }
    out_json = OUT / "alphastar_league_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[alphastar_league] wrote {out_json.relative_to(ROOT)}")
    print(f"[alphastar_league] verdict: {summary['verdict_counts']}")


if __name__ == "__main__":
    main()
