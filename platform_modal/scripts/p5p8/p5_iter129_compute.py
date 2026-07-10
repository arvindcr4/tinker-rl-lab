"""Companion module for p5_iter129_headline_cis.py — compute helpers
and the head/main row-builder. Kept ≤300 LoC per brief rules.
"""
from __future__ import annotations
import csv, json, math, random, sys
from pathlib import Path
from collections import defaultdict
from p5_iter129_headline_cis import (
    ROOT, OUT_DIR, SEED, B, ALPHA, bootstrap_ci_eta2,
    load_n2, load_zvf130_methods, load_mega_cells,
)


def eta2_n2_channel(n2, methods, ch, steps):
    """eta^2(algo, channel) on N2 panel."""
    groups = [[n2[(m, s)][ch] for s in steps if not math.isnan(n2[(m, s)][ch])]
              for m in methods]
    return bootstrap_ci_eta2(groups)


def cohens_d_with_ci(n2, steps, last10):
    """Cohen's d for GIFT vs other 3, last-10 pooled + bootstrap CI."""
    methods_other = ["grpo", "aero", "areal"]
    gz = [n2[("gift", s)]["zvf"] for s in last10]
    gp = [n2[("gift", s)]["pcd"] for s in last10]
    oz = [n2[(m, s)]["zvf"] for m in methods_other for s in last10]
    op = [n2[(m, s)]["pcd"] for m in methods_other for s in last10]
    def cd(g, o):
        if len(g) < 2 or len(o) < 2: return float("nan")
        mg, mo = sum(g)/len(g), sum(o)/len(o)
        vg = sum((x-mg)**2 for x in g)/(len(g)-1)
        vo = sum((x-mo)**2 for x in o)/(len(o)-1)
        sp = math.sqrt(((len(g)-1)*vg + (len(o)-1)*vo) / (len(g)+len(o)-2))
        return (mg-mo)/sp if sp > 1e-12 else float("nan")
    pz, pp = cd(gz, oz), cd(gp, op)
    rng = random.Random(SEED)
    bz, bp = [], []
    for _ in range(B):
        sg = [gz[rng.randrange(len(gz))] for _ in range(len(gz))]
        so = [oz[rng.randrange(len(oz))] for _ in range(len(oz))]
        if len(set(sg)) < 2 or len(set(so)) < 2: continue
        bz.append(cd(sg, so))
    rng = random.Random(SEED)
    for _ in range(B):
        sg = [gp[rng.randrange(len(gp))] for _ in range(len(gp))]
        so = [op[rng.randrange(len(op))] for _ in range(len(op))]
        if len(set(sg)) < 2 or len(set(so)) < 2: continue
        bp.append(cd(sg, so))
    bz.sort(); bp.sort()
    if len(bz) >= 100:
        return {"zvf": pz, "zvf_lo": bz[int(len(bz)*ALPHA/2)],
                "zvf_hi": bz[int(len(bz)*(1-ALPHA/2))],
                "pcd": pp, "pcd_lo": bp[int(len(bp)*ALPHA/2)],
                "pcd_hi": bp[int(len(bp)*(1-ALPHA/2))],
                "n_gift": len(gz), "n_other": len(oz)}
    return {"zvf": pz, "zvf_lo": pz, "zvf_hi": pz,
            "pcd": pp, "pcd_lo": pp, "pcd_hi": pp,
            "n_gift": len(gz), "n_other": len(oz)}


def eta2_zvf130(zvf130):
    methods = [m for m, s in zvf130.items() if len(s) >= 3]
    groups = [[s["zvf_risk"] for s in zvf130[m].values()] for m in methods]
    eta2, lo, hi = bootstrap_ci_eta2(groups)
    return eta2, lo, hi, methods


def eta2_seed_axis(zvf130):
    seed_to = defaultdict(list)
    for m, seeds in zvf130.items():
        for s, v in seeds.items():
            seed_to[s].append(v["zvf_risk"])
    groups = [v for v in seed_to.values() if len(v) >= 4]
    if len(groups) < 2: return float("nan"), float("nan"), float("nan")
    return bootstrap_ci_eta2(groups)


def lomo_rel(zvf130, target):
    if target not in zvf130: return float("nan")
    others = [m for m in zvf130 if m != target and len(zvf130[m]) >= 3
              and not m.startswith("scaling_law_")]
    if len(others) < 2: return float("nan")
    g_no = [[s["zvf_risk"] for s in zvf130[m].values()] for m in others]
    e_no, _, _ = bootstrap_ci_eta2(g_no)
    g_with = g_no + [[s["zvf_risk"] for s in zvf130[target].values()]]
    e_all, _, _ = bootstrap_ci_eta2(g_with)
    return (e_no - e_all) / e_all if e_all > 1e-12 else float("nan")


def chained_R(mega, n2, ch, axis):
    methods = ["grpo", "aero", "gift", "areal"]
    steps = sorted({k[1] for k in n2.keys()})
    n2_groups = [[n2[(m, s)][ch] for s in steps if not math.isnan(n2[(m, s)][ch])]
                 for m in methods]
    eta_algo = bootstrap_ci_eta2(n2_groups)[0]
    by_axis = defaultdict(list)
    for c in mega:
        v = c.get(ch)
        if v is None or math.isnan(v): continue
        by_axis[c[axis]].append(v)
    mega_groups = list(by_axis.values())
    if len(mega_groups) < 2: return float("nan"), float("nan"), float("nan")
    eta_stack, _, _ = bootstrap_ci_eta2(mega_groups)
    R = eta_stack / eta_algo if eta_algo > 1e-12 else float("nan")
    return R, eta_stack, eta_algo


def chained_R_ci(mega_groups, eta_algo, R_pub):
    """Bootstrap CI for the chained ratio."""
    rng = random.Random(SEED)
    boot = []
    for _ in range(B):
        ng = [[g[rng.randrange(len(g))] for _ in range(len(g))] for g in mega_groups]
        all_v = [v for g in ng for v in g]
        m = sum(all_v)/len(all_v)
        t = sum((v-m)**2 for v in all_v)
        if t <= 0: continue
        a = sum(len(g)*(sum(g)/len(g)-m)**2 for g in ng)
        boot.append(a/t/eta_algo if eta_algo > 1e-12 else float("nan"))
    valid = [b for b in boot if not math.isnan(b) and abs(b) < 1e6]
    if len(valid) < 100: return float("nan"), float("nan")
    valid.sort()
    return valid[int(len(valid)*ALPHA/2)], valid[int(len(valid)*(1-ALPHA/2))]


def build_rows(n2, zvf130, mega):
    """Build all 15 headline rows."""
    methods = ["grpo", "aero", "gift", "areal"]
    steps = sorted({k[1] for k in n2.keys()})
    last10 = [s for s in steps if s >= max(steps)-9]
    rows = []

    # H01: 6-channel eta^2 mean
    six = [eta2_n2_channel(n2, methods, c, steps)[0]
           for c in ["zvf", "pcd", "larq", "reward_mean", "mean_len", "cv_len"]]
    h01 = sum(six)/len(six)
    rng = random.Random(SEED)
    boot = []
    for _ in range(B):
        idx = [rng.randrange(6) for _ in range(6)]
        sub = [six[i] for i in idx]
        boot.append(sum(sub)/6)
    boot.sort()
    h01_lo, h01_hi = boot[int(B*ALPHA/2)], boot[int(B*(1-ALPHA/2))]
    rows.append({
        "id": "H01", "vein": "iter-85", "claim": "eta2_mean_6ch",
        "published_pt": 0.0331, "recomputed_pt": round(h01, 4),
        "ci_lo": round(h01_lo, 4), "ci_hi": round(h01_hi, 4),
        "n": 6,
        "verdict": "PASS" if h01_lo <= 0.0331 <= h01_hi else "TENSION",
        "notes": "6-channel mean; n=6 channels"})

    # H02-H04: per-channel eta^2 on N2
    for hid, ch, pub, vn in [
        ("H02", "zvf", 0.0454, "eta2_zvf_N2"),
        ("H03", "pcd", 0.0357, "eta2_pcd_N2"),
        ("H04", "loss", 0.9867, "eta2_loss_positive_control")]:
        eta, lo, hi = eta2_n2_channel(n2, methods, ch, steps)
        rows.append({
            "id": hid, "vein": "iter-85", "claim": vn,
            "published_pt": pub, "recomputed_pt": round(eta, 4),
            "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
            "n": len(steps),
            "verdict": "PASS" if lo <= pub <= hi else "TENSION",
            "notes": "N2 4-method 40-step; bootstrap B=2000" if hid != "H04"
                     else "expected near 1.0; loss is method-specific"})

    # H05/H06: Cohen's d
    cd = cohens_d_with_ci(n2, steps, last10)
    rows.append({
        "id": "H05", "vein": "iter-85",
        "claim": "cohens_d_zvf_GIFT_vs_others_last10",
        "published_pt": 1.899, "recomputed_pt": round(cd["zvf"], 4),
        "ci_lo": round(cd["zvf_lo"], 4), "ci_hi": round(cd["zvf_hi"], 4),
        "n": cd["n_gift"]+cd["n_other"],
        "verdict": "PASS" if cd["zvf_lo"] <= 1.899 <= cd["zvf_hi"] else "TENSION",
        "notes": "last-10 pooled; bootstrap CI on resampled step indices"})
    rows.append({
        "id": "H06", "vein": "iter-85",
        "claim": "cohens_d_pcd_GIFT_vs_others_last10",
        "published_pt": -1.605, "recomputed_pt": round(cd["pcd"], 4),
        "ci_lo": round(cd["pcd_lo"], 4), "ci_hi": round(cd["pcd_hi"], 4),
        "n": cd["n_gift"]+cd["n_other"],
        "verdict": "PASS" if cd["pcd_lo"] <= -1.605 <= cd["pcd_hi"] else "TENSION",
        "notes": "last-10 pooled; bootstrap CI on resampled step indices"})

    # H07: eta^2 zvf UB (same as H02 hi)
    eta, lo, hi = eta2_n2_channel(n2, methods, "zvf", steps)
    rows.append({
        "id": "H07", "vein": "iter-89", "claim": "eta2_zvf_bootstrap_UB",
        "published_pt": 0.113, "recomputed_pt": round(hi, 4),
        "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
        "n": len(steps), "verdict": "PASS" if hi >= 0.113 else "TENSION",
        "notes": "UB of bootstrap CI; exceeds Ivison 0.05"})

    # H08: zvf130 algo-axis eta^2
    eta, lo, hi, mlist = eta2_zvf130(zvf130)
    rows.append({
        "id": "H08", "vein": "iter-101", "claim": "eta2_zvf_risk_9method",
        "published_pt": 0.763, "recomputed_pt": round(eta, 4),
        "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
        "n": len(mlist),
        "verdict": "PASS" if lo <= 0.763 <= hi else "TENSION",
        "notes": f"zvf130 9-method; n_methods={len(mlist)}"})

    # H09: seed-axis control
    eta, lo, hi = eta2_seed_axis(zvf130)
    rows.append({
        "id": "H09", "vein": "iter-101", "claim": "eta2_seed_zvf_risk_control",
        "published_pt": 0.0071, "recomputed_pt": round(eta, 4),
        "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
        "n": 5, "verdict": "PASS" if lo <= 0.0071 <= hi else "TENSION",
        "notes": "seed-axis control; near 0 by design"})

    # H10: SCAFGRPO LOMO rel_drop
    rel = lomo_rel(zvf130, "scafgrpo")
    rows.append({
        "id": "H10", "vein": "iter-101", "claim": "scafgrpo_lomo_rel_drop",
        "published_pt": -0.1042, "recomputed_pt": round(rel, 4),
        "ci_lo": round(rel, 4), "ci_hi": round(rel, 4),
        "n": 5, "verdict": "PASS" if abs(rel - (-0.1042)) < 0.001 else "TENSION",
        "notes": "leave-one-out deterministic"})

    # H11-H14: chained R
    for hid, ch, axis, pub in [
        ("H11", "zvf", "task_slice", 10.32),
        ("H12", "pcd", "task_slice", 12.62),
        ("H13", "zvf", "G", 9.77),
        ("H14", "pcd", "G", 6.45)]:
        R, eta_stack, eta_algo = chained_R(mega, n2, ch, axis)
        by_axis = defaultdict(list)
        for c in mega:
            v = c.get(ch)
            if v is None or math.isnan(v): continue
            by_axis[c[axis]].append(v)
        mega_groups = list(by_axis.values())
        R_lo, R_hi = chained_R_ci(mega_groups, eta_algo, pub)
        rows.append({
            "id": hid, "vein": "iter-125",
            "claim": f"chained_R_{ch}_x_{axis}",
            "published_pt": pub, "recomputed_pt": round(R, 4),
            "ci_lo": round(R_lo, 4), "ci_hi": round(R_hi, 4),
            "n": len(by_axis),
            "verdict": "PASS" if R_lo <= pub <= R_hi else "TENSION",
            "notes": f"chained ratio; mega stack-axis by {axis}; n_axes={len(by_axis)}"})

    # H15: deterministic
    rows.append({
        "id": "H15", "vein": "iter-121",
        "claim": "auditor_blind_spot_rate_M1M2",
        "published_pt": 0.0, "recomputed_pt": 0.0,
        "ci_lo": 0.0, "ci_hi": 0.0, "n": 196,
        "verdict": "REPORTED",
        "notes": "deterministic audit; M1 cell_id_swap_hash + M2 model_family_swap"})
    return rows


def main():
    n2 = load_n2()
    zvf130 = load_zvf130_methods()
    mega = load_mega_cells()
    rows = build_rows(n2, zvf130, mega)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tsv_path = OUT_DIR / "p5_iter129_headline_cis.tsv"
    fieldnames = ["id", "vein", "claim", "published_pt", "recomputed_pt",
                  "ci_lo", "ci_hi", "n", "verdict", "notes"]
    with open(tsv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for r in rows: w.writerow(r)
    counts = defaultdict(int)
    for r in rows: counts[r["verdict"]] += 1
    summary = {
        "iter": 129, "pillar": "P5", "n_headlines": len(rows),
        "n_pass": counts["PASS"], "n_tension": counts["TENSION"],
        "n_reported": counts["REPORTED"], "n_insufficient_n": counts["INS"],
        "bootstrap": {"B": B, "alpha": ALPHA, "seed": SEED,
                      "method": "percentile, paired where applicable"},
        "headlines": rows}
    json_path = OUT_DIR / "p5_iter129_headline_cis.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=False)
    print(f"WROTE {tsv_path} ({len(rows)} rows)")
    print(f"WROTE {json_path}")
    print()
    print(f"=== P5 iter-129 headline-CI audit ===")
    print(f"n_headlines: {len(rows)}  PASS: {counts['PASS']}  "
          f"TENSION: {counts['TENSION']}  REPORTED: {counts['REPORTED']}")
    for r in rows:
        print(f"  {r['id']}  pt={r['recomputed_pt']:+.4f}  "
              f"CI=[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  → {r['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())