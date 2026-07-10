#!/usr/bin/env python3
"""Iter 171 — P7 Canonical Headline CI Table.

Builds a SINGLE CI-anchored headline table for the P7 paper on the four-method
N2 reward tensor corpus (grpo, aero, gift, areal x 40 steps). 6 artifacts in
platform_hybrid/experiments/results/p5p8/. Stdlib only. LCG bootstrap B=2000 seed=20260705.
"""
from __future__ import annotations
import csv, glob, json, os, random, statistics

WORKTREE = "/home/claude/tinker-rl-lab-minimax"
DATA_DIR = os.path.join(WORKTREE, "platform_hybrid/experiments/results/n2_reward_tensor_resume")
OUT_DIR = os.path.join(WORKTREE, "platform_hybrid/experiments/results/p5p8")
os.makedirs(OUT_DIR, exist_ok=True)
METHODS = ["grpo", "aero", "gift", "areal"]
G_MENU = [2, 4, 8, 16]; G_BASE = 8; N_PROMPTS = 16
TOST_BOUND = 0.05; B = 2000; SEED = 20260705


def _bci(v, stat_fn=statistics.mean, alpha=0.05, rng=None):
    if rng is None:
        rng = random.Random(SEED)
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    pt = stat_fn(v)
    boots = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(stat_fn([v[i] for i in idx]))
    boots.sort()
    return (pt, boots[int(alpha/2*B)], boots[int((1-alpha/2)*B)], B)


def tost_paired(a, b, bound=TOST_BOUND, rng=None):
    if rng is None:
        rng = random.Random(SEED)
    if len(a) == 0 or len(a) != len(b):
        return False
    _, lo, hi, _ = _bci([a[i]-b[i] for i in range(len(a))], rng=rng)
    return (lo > -bound) and (hi < +bound)


def load_tensors():
    out = {m: [] for m in METHODS}
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*_tensors.jsonl"))):
        method = os.path.basename(path).split("_")[0]
        if method not in METHODS:
            continue
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out[method].append(json.loads(line))
    for m in METHODS:
        out[m].sort(key=lambda r: r["step"])
    return out


def headline_cis(tensors):
    rows = []
    for m in METHODS:
        for metric in ("zvf", "reward_mean", "pcd", "mean_len", "loss"):
            vals = [s[metric] for s in tensors[m] if s[metric] == s[metric]]
            if not vals:
                continue
            pt, lo, hi, n = _bci(vals)
            rows.append({"method": m, "metric": metric, "n": n,
                         "point": round(pt, 4), "lo": round(lo, 4),
                         "hi": round(hi, 4), "ci_hw": round((hi-lo)/2.0, 4)})
    return rows


def y_signature_cis(tensors):
    rows = []
    for m in METHODS:
        y_at_step = {g: [] for g in G_MENU}
        for s in tensors[m]:
            rewards = s["rewards"]
            if not rewards or len(rewards[0]) != G_BASE:
                continue
            ps = [sum(r)/G_BASE for r in rewards]
            for G in G_MENU:
                if G == G_BASE:
                    k_ct = sum(1 for r in rewards if 0 < sum(r) < G_BASE)
                    yvals = [k_ct / max(len(rewards), 1)]
                else:
                    yvals = [1.0 - (p**G) - ((1.0-p)**G) for p in ps]
                y_at_step[G].append(statistics.mean(yvals))
        for G in G_MENU:
            v = y_at_step[G]
            if not v:
                continue
            pt, lo, hi, n = _bci(v)
            rows.append({"method": m, "G": G, "n_steps": len(v),
                         "point": round(pt, 4), "lo": round(lo, 4),
                         "hi": round(hi, 4), "ci_hw": round((hi-lo)/2.0, 4)})
    return rows


def cross_method_tost(tensors):
    rows = []
    for metric in ("zvf", "reward_mean"):
        ps_step = {m: [s[metric] for s in tensors[m] if s[metric] == s[metric]]
                   for m in METHODS}
        L = min(len(ps_step[m]) for m in METHODS)
        for m_a in METHODS:
            for m_b in METHODS:
                if m_a >= m_b:
                    continue
                a = ps_step[m_a][:L]; b = ps_step[m_b][:L]
                pt, lo, hi, _ = _bci([a[i]-b[i] for i in range(L)])
                rows.append({"metric": metric, "pair": f"{m_a}-{m_b}",
                             "n_pairs": L, "delta": round(pt, 4),
                             "lo": round(lo, 4), "hi": round(hi, 4),
                             "tost_pm_005_equiv": bool(tost_paired(a, b))})
    return rows


def ctrl_zvf_triage(s, ps, threshold=0.70, g_alt=12, max_pcd=0.20):
    pcd = s["pcd"] if s["pcd"] == s["pcd"] else 1.0
    return [g_alt]*len(ps) if (pcd <= max_pcd and s["zvf"] >= threshold) else [G_BASE]*len(ps)


def ctrl_dualformer(ps):
    out = []
    for p in ps:
        if p >= 0.95: out.append(2)
        elif p >= 0.85: out.append(4)
        elif p >= 0.70: out.append(8)
        else: out.append(16)
    return out


def ctrl_iso_g(ps):
    out = []
    for p in ps:
        if p in (0.0, 1.0):
            out.append(G_BASE); continue
        best = G_BASE; best_loss = 1e9
        for G in (2, 4, 6, 8, 10, 12, 16):
            y = 1.0 - (p**G) - ((1.0-p)**G)
            if 0.85 <= y <= 0.95:
                best = G; break
            d = min(abs(y - 0.85), abs(y - 0.95))
            if d < best_loss:
                best_loss = d; best = G
        out.append(best)
    return out


def eval_ctrl(steps, cfn):
    """Per step returns rescue_strict, rescue_partial, cost_ratio, contrast_gain."""
    rs_l, rp_l, costs, cg_l = [], [], [], []
    bc = N_PROMPTS * G_BASE
    for s in steps:
        rewards = s["rewards"]
        if not rewards:
            continue
        ps = [sum(r)/G_BASE for r in rewards]
        g_per_p = cfn(s, ps)
        rs_ = rp_ = n_deg = 0; cg = 0.0
        for i, r in enumerate(rewards):
            y_b = 1.0 if 0 < sum(r) < G_BASE else 0.0
            gi = max(2, min(g_per_p[i], 32))
            y_n = 1.0 - (ps[i]**gi) - ((1.0-ps[i])**gi)
            cg += y_n - y_b
            if 0 == sum(r) or G_BASE == sum(r):
                n_deg += 1
                if y_n > 0.5: rs_ += 1
                if y_n > 0.05: rp_ += 1
        rs_l.append(rs_ / max(n_deg, 1)); rp_l.append(rp_ / max(n_deg, 1))
        costs.append(sum(g_per_p) / bc); cg_l.append(cg / len(rewards))
    return rs_l, rp_l, costs, cg_l


def controller_retention_cis(tensors):
    rows = []
    cfns = {
        "C0_fixed_g8": lambda s, ps: [G_BASE]*len(ps),
        "C1_zvf_triage": ctrl_zvf_triage,
        "C2_dualformer_auto": lambda s, ps: ctrl_dualformer(ps),
        "C3_iso_g": lambda s, ps: ctrl_iso_g(ps),
    }
    for m in METHODS:
        for cname, cfn in cfns.items():
            rs, rp, costs, gains = eval_ctrl(tensors[m], cfn)
            if not rs:
                continue
            ptrs, lors, hirs, _ = _bci(rs); ptrp, lorp, hirp, _ = _bci(rp)
            ptg, log, hig, _ = _bci(gains); ptc, loc, hic, _ = _bci(costs)
            rows.append({"method": m, "controller": cname, "n": len(rs),
                         "rescue_strict_pt": round(ptrs, 4),
                         "rescue_strict_lo": round(lors, 4),
                         "rescue_strict_hi": round(hirs, 4),
                         "rescue_partial_pt": round(ptrp, 4),
                         "rescue_partial_lo": round(lorp, 4),
                         "rescue_partial_hi": round(hirp, 4),
                         "contrast_gain_pt": round(ptg, 4),
                         "contrast_gain_lo": round(log, 4),
                         "contrast_gain_hi": round(hig, 4),
                         "cost_ratio_pt": round(ptc, 4),
                         "cost_ratio_lo": round(loc, 4),
                         "cost_ratio_hi": round(hic, 4)})
    return rows


def cross_paper_consistency(ctrl_rows, y_rows):
    """Check iter-163 (cost cap) + iter-167 (gain > fixed-G) + FRONTIER (anti-herding)."""
    rows = []
    c1 = [r for r in ctrl_rows if r["controller"] == "C1_zvf_triage"]
    c0 = {r["method"]: r for r in ctrl_rows if r["controller"] == "C0_fixed_g8"}
    for r in c1:
        rows.append({"iter_ref": "iter-163 row 177",
                     "claim": "zvf-triage cost < 1.50x baseline at step level",
                     "evidence_metric": "cost_ratio_hi",
                     "observed": f'{r["cost_ratio_hi"]:.4f} (method={r["method"]})',
                     "consistent": bool(r["cost_ratio_hi"] < 1.50)})
    for r in c1:
        bg = c0.get(r["method"], {}).get("contrast_gain_pt", 0)
        rows.append({"iter_ref": "iter-167 row 178",
                     "claim": "zvf-triage improves per-prompt contrast vs fixed-G baseline",
                     "evidence_metric": "contrast_gain(C1) > gain(C0) by margin",
                     "observed": f"gain_C1={r['contrast_gain_pt']:.4f} > gain_C0={bg:.4f}",
                     "consistent": bool(r["contrast_gain_pt"] - bg > 0.005)})
    sig = {(r["method"], r["G"]): r for r in y_rows}
    for m in METHODS:
        g8 = sig.get((m, 8), {}).get("point", 0)
        g16 = sig.get((m, 16), {}).get("point", 0)
        rows.append({"iter_ref": "FRONTIER Round 2",
                     "claim": "observed Y(G=8) > iid-projected Y(G=16) [anti-herding bonus]",
                     "evidence_metric": "y_G8_pt > y_proj_G16_pt",
                     "observed": f"Y_obs(G=8)={g8:.4f}, Y_iid(G=16)={g16:.4f}",
                     "consistent": bool(g8 > g16)})
    return rows


def write_tsv(path, rows):
    with open(path, "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    print("[iter171] loading N2 tensors...")
    tensors = load_tensors()
    for m in METHODS:
        print(f"  {m}: {len(tensors[m])} steps")
    hrows = headline_cis(tensors)
    yrows = y_signature_cis(tensors)
    trows = cross_method_tost(tensors)
    crows = controller_retention_cis(tensors)
    write_tsv(os.path.join(OUT_DIR, "p7_iter171_headline_cis.tsv"), hrows)
    write_tsv(os.path.join(OUT_DIR, "p7_iter171_y_at_g.tsv"), yrows)
    write_tsv(os.path.join(OUT_DIR, "p7_iter171_cross_method_tost.tsv"), trows)
    write_tsv(os.path.join(OUT_DIR, "p7_iter171_controller_retention.tsv"), crows)
    print(f"[iter171] 4 artifacts written ({len(hrows)}/{len(yrows)}/{len(trows)}/{len(crows)})")

    krows = cross_paper_consistency(crows, yrows)
    write_tsv(os.path.join(OUT_DIR, "p7_iter171_cross_paper_consistency.tsv"), krows)

    c1 = [r for r in crows if r["controller"] == "C1_zvf_triage"]
    c0m = {r["method"]: r for r in crows if r["controller"] == "C0_fixed_g8"}
    sig = {(r["method"], r["G"]): r for r in yrows}
    h1 = all(float(r["ci_hw"]) < 0.10 for r in hrows if r["metric"] == "zvf")
    h2 = all(float(r["ci_hw"]) < 0.10 for r in yrows if r["G"] == 8)
    h3 = sum(1 for r in trows if r["metric"] == "zvf" and r["tost_pm_005_equiv"]) >= 2
    h3b = sum(1 for r in trows if r["metric"] == "reward_mean" and r["tost_pm_005_equiv"]) >= 4
    h4 = all(r["contrast_gain_pt"] - c0m.get(r["method"], {}).get("contrast_gain_pt", 0) > 0.005 for r in c1)
    h5 = all(r["cost_ratio_hi"] < 1.50 for r in c1)
    h6 = all(r["consistent"] for r in krows)
    h7 = all(sig.get((m, 8), {}).get("point", 0) > sig.get((m, 16), {}).get("point", 0) for m in METHODS)
    summary = {"n_steps_per_method": {m: len(tensors[m]) for m in METHODS},
               "B": B, "seed": SEED, "tost_bound": TOST_BOUND,
               "headline_zvf_cis": [{"method": r["method"], "point": r["point"],
                                     "lo": r["lo"], "hi": r["hi"], "hw": r["ci_hw"]}
                                    for r in hrows if r["metric"] == "zvf"],
               "y_at_g_signature": [{"method": r["method"], "G": r["G"],
                                     "point": r["point"], "lo": r["lo"],
                                     "hi": r["hi"], "hw": r["ci_hw"]}
                                    for r in yrows],
               "tost_equiv_pairs_zvf": sum(1 for r in trows if r["metric"] == "zvf" and r["tost_pm_005_equiv"]),
               "tost_equiv_pairs_reward": sum(1 for r in trows if r["metric"] == "reward_mean" and r["tost_pm_005_equiv"]),
               "consistency_pass": sum(1 for r in krows if r["consistent"]),
               "consistency_total": len(krows),
               "verdicts": {"H1_zvf_ci_hw_below_010_all_methods": h1,
                            "H2_y_at_g8_ci_hw_below_010_all_methods": h2,
                            "H3_at_least_2_of_6_method_pairs_tost_equivalent_zvf": h3,
                            "H3b_at_least_4_of_6_method_pairs_tost_equivalent_reward_mean": h3b,
                            "H4_zvf_triage_contrast_gain_exceeds_fixed_g_baseline": h4,
                            "H5_zvf_triage_cost_hi_below_150_all_methods": h5,
                            "H6_cross_paper_consistency_full_pass": h6,
                            "H7_observed_y_at_g8_exceeds_iid_projection_at_g16": h7}}
    with open(os.path.join(OUT_DIR, "p7_iter171_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[iter171] H1={h1} H2={h2} H3={h3} H3b={h3b} H4={h4} H5={h5} H6={h6} H7={h7}")


if __name__ == "__main__":
    main()
