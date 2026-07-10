#!/usr/bin/env python3
"""P6 iter-142 — claim_validation aggregate audit + η²(method) paradox test.

Closes brief vein (a) at the AGGREGATE level. Iter-126 tier-classified per-delta
evidence depth (n_sig / n_panels); iter-106 ledger listed every (delta, metric,
panel) verdict but never aggregated. Iter-142 produces the AGGREGATE summary
that connects:

  - the iter-126 per-delta evidence tier (A/B/D), and
  - the iter-106/iter-118 per-(delta, metric, panel) claim_validation verdicts,
  - and the iter-141 η²(method)=0.0005 same-stack under-identification.

Sharpest test: iter-141 found η²(method)=0.0005 on the N2 same-stack 4-method
reward tensor — implying method-axis is statistically under-identified on the
global step × prompt × rollout variance. Yet three N2 panel claim_validations
return SUPPORTS (e.g. delta_gift.zvf @ n2_same_stack_last10). The audit reconciles
this by separating panel-specific variance (where paired-step deltas ARE non-zero
for some (delta, metric)) from global variance (where method axis has near-zero
explanatory power).

Inputs:
  - registry/entries/delta_*.json (17 variant_delta records; 15 + 2 tool_use
    from iter-138)
  - platform_hybrid/experiments/results/p5p8/p6_iter126_measured_evidence_tier.tsv
    (per-delta tier classification: A/B/C/D)
  - platform_hybrid/experiments/results/p5p8/p5_iter141_anova_eta2.tsv
    (η²(method)=0.0005 anchor)

Outputs:
  - platform_hybrid/experiments/results/p5p8/p6_iter142_verdict_aggregate.tsv
    (per-(delta, metric, panel) rows + verdict aggregate columns)
  - platform_hybrid/experiments/results/p5p8/p6_iter142_tier_x_verdict.tsv
    (cross-tab: tier × verdict counts and rates)
  - platform_hybrid/experiments/results/p5p8/p6_iter142_metric_x_verdict.tsv
    (cross-tab: metric × verdict)
  - platform_hybrid/experiments/results/p5p8/p6_iter142_panel_x_verdict.tsv
    (cross-tab: panel × verdict; tests N2 vs zvf130 SUPPORTS rate)
  - platform_hybrid/experiments/results/p5p8/p6_iter142_sign_concordance.tsv
    (per-delta: predicted_sign matches measured sign? counts)
  - platform_hybrid/experiments/results/p5p8/p6_iter142_eta2_paradox.tsv
    (N2 panel SUPPORTS rate vs zvf130 panel SUPPORTS rate; explained)
  - platform_hybrid/experiments/results/p5p8/p6_iter142_summary.json
"""
import json
import pathlib
import statistics
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
ENT = ROOT / "registry" / "entries"
OUT = ROOT / "experiments" / "results" / "p5p8"
TIER_TSV = OUT / "p6_iter126_measured_evidence_tier.tsv"
ETA2_TSV = OUT / "p5_iter141_anova_eta2.tsv"
OUT.mkdir(parents=True, exist_ok=True)


def _read_tsv(path):
    with path.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        rows = [dict(zip(header, line.rstrip("\n").split("\t"))) for line in f]
    return rows


def _load_tier_map():
    """Return {delta_id: tier_str}."""
    rows = _read_tsv(TIER_TSV)
    return {r["delta_id"]: r["evidence_tier"] for r in rows}


def _load_eta2_method():
    """Return η²(method) point + ci lo + ci hi from iter-141 ANOVA TSV."""
    rows = _read_tsv(ETA2_TSV)
    for r in rows:
        if r.get("factor") == "method":
            return float(r["eta2"]), float(r["ci_lo"]), float(r["ci_hi"])
    return None, None, None


# ---------- load registry ----------
def _load_deltas():
    deltas = {}
    for p in sorted(ENT.glob("delta_*.json")):
        with p.open() as f:
            rec = json.load(f)
        if rec.get("record_type") != "variant_delta":
            continue
        deltas[rec["id"]] = rec
    return deltas


# ---------- sign helpers ----------
def _sign_of_num(x):
    if x is None:
        return 0
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _operator_matches(op, x):
    """Apply predicted_sign operator on measured x."""
    if x is None:
        return None
    if op == ">0":
        return x > 0
    if op == "<0":
        return x < 0
    if op == ">=0":
        return x >= 0
    if op == "<=0":
        return x <= 0
    if op == "=0":
        return abs(x) <= 1e-12
    return None


def _verdict_counts(rows, key_fn):
    """Return Counter of verdicts for rows grouped by key_fn(row)."""
    grouped = defaultdict(Counter)
    for r in rows:
        grouped[key_fn(r)].update([r["verdict"]])
    return grouped


# ---------- main ----------
def main():
    tier_map = _load_tier_map()
    eta2_pt, eta2_lo, eta2_hi = _load_eta2_method()
    deltas = _load_deltas()

    # ----- per-(delta, metric, panel) row extraction -----
    per_rows = []
    for did, rec in sorted(deltas.items()):
        cvs = rec.get("claim_validation") or []
        for cv in cvs:
            per_rows.append({
                "delta_id": did,
                "metric": cv.get("metric", ""),
                "panel": cv.get("panel", ""),
                "predicted_sign": cv.get("predicted_sign") or "",
                "observed_delta": cv.get("observed_delta"),
                "ci_low": cv.get("ci_low"),
                "ci_high": cv.get("ci_high"),
                "significant": cv.get("significant"),
                "verdict": cv.get("verdict", ""),
                "tier": tier_map.get(did, "?"),
            })

    # ----- aggregate: tier × verdict -----
    tv = defaultdict(Counter)
    for r in per_rows:
        tv[r["tier"]].update([r["verdict"]])
    tier_rows = []
    for tier in ("A", "B", "D"):
        c = tv.get(tier, Counter())
        total = sum(c.values())
        for v in ("SUPPORTS", "NEUTRAL", "CONTRADICTS", "UNCLAIMED"):
            tier_rows.append({
                "tier": tier,
                "verdict": v,
                "n": c.get(v, 0),
                "pct": round(100 * c.get(v, 0) / total, 2) if total else 0.0,
                "tier_total_n": total,
            })

    # ----- aggregate: metric × verdict -----
    mv = defaultdict(Counter)
    for r in per_rows:
        mv[r["metric"]].update([r["verdict"]])
    metric_rows = []
    for metric in sorted(mv):
        c = mv[metric]
        total = sum(c.values())
        for v in ("SUPPORTS", "NEUTRAL", "CONTRADICTS", "UNCLAIMED"):
            metric_rows.append({
                "metric": metric,
                "verdict": v,
                "n": c.get(v, 0),
                "pct": round(100 * c.get(v, 0) / total, 2) if total else 0.0,
                "metric_total_n": total,
            })

    # ----- aggregate: panel × verdict -----
    pv = defaultdict(Counter)
    for r in per_rows:
        pv[r["panel"]].update([r["verdict"]])
    panel_rows = []
    for panel in sorted(pv):
        c = pv[panel]
        total = sum(c.values())
        for v in ("SUPPORTS", "NEUTRAL", "CONTRADICTS", "UNCLAIMED"):
            panel_rows.append({
                "panel": panel,
                "verdict": v,
                "n": c.get(v, 0),
                "pct": round(100 * c.get(v, 0) / total, 2) if total else 0.0,
                "panel_total_n": total,
            })

    # ----- sign concordance per delta -----
    sig_conc = []
    for did in sorted(deltas):
        cvs = deltas[did].get("claim_validation") or []
        n_declared = 0
        n_sign_match = 0
        n_sign_match_sig = 0
        for cv in cvs:
            ps = cv.get("predicted_sign")
            od = cv.get("observed_delta")
            sig = cv.get("significant")
            if ps in (None, ""):
                continue
            n_declared += 1
            obs_sign = _sign_of_num(od)
            ps_sign = (">0" if ps in (">0", ">=0") else
"<0" if ps in ("<0", "<=0") else
                      "=0" if ps == "=0" else "")
            if obs_sign == 1 and ps_sign == ">0":
                n_sign_match += 1
                if sig:
                    n_sign_match_sig += 1
            elif obs_sign == -1 and ps_sign == "<0":
                n_sign_match += 1
                if sig:
                    n_sign_match_sig += 1
            elif obs_sign == 0 and ps_sign == "=0":
                n_sign_match += 1
        sig_conc.append({
            "delta_id": did,
            "tier": tier_map.get(did, "?"),
            "n_declared_expected": n_declared,
            "n_sign_match": n_sign_match,
            "n_sign_match_significant": n_sign_match_sig,
            "pct_sign_match": round(100 * n_sign_match / n_declared, 2) if n_declared else 0.0,
            "pct_sign_match_sig": round(100 * n_sign_match_sig / n_declared, 2) if n_declared else 0.0,
        })

    # ----- η²(method) paradox test -----
    # Iter-141 says η²(method) = 0.0005 [0.0001, 0.0049] — method axis under-
    # identified on N2 step × prompt × rollout variance. But N2 panel has its
    # own SUPPORTS rate. We compare:
    n2_supports = 0
    n2_total = 0
    zvf130_supports = 0
    zvf130_total = 0
    for r in per_rows:
        if r["verdict"] == "UNCLAIMED":
            continue
        if r["panel"] == "n2_same_stack_last10":
            n2_total += 1
            if r["verdict"] == "SUPPORTS":
                n2_supports += 1
        elif r["panel"] == "zvf130_5seed":
            zvf130_total += 1
            if r["verdict"] == "SUPPORTS":
                zvf130_supports += 1

    paradox_rows = [
        {
            "panel": "n2_same_stack_last10",
            "n_evaluated": n2_total,
            "n_supports": n2_supports,
            "supports_rate_pct": round(100 * n2_supports / n2_total, 2) if n2_total else 0.0,
        },
        {
            "panel": "zvf130_5seed",
            "n_evaluated": zvf130_total,
            "n_supports": zvf130_supports,
            "supports_rate_pct": round(100 * zvf130_supports / zvf130_total, 2) if zvf130_total else 0.0,
        },
    ]

    # ----- write outputs -----
    def _write_tsv(rows, cols, path):
        with path.open("w") as f:
            f.write("\t".join(cols) + "\n")
            for r in rows:
                f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")

    _write_tsv(
        per_rows,
        ["delta_id", "tier", "metric", "panel", "predicted_sign",
         "observed_delta", "ci_low", "ci_high", "significant", "verdict"],
        OUT / "p6_iter142_verdict_aggregate.tsv",
    )
    _write_tsv(
        tier_rows,
        ["tier", "verdict", "n", "pct", "tier_total_n"],
        OUT / "p6_iter142_tier_x_verdict.tsv",
    )
    _write_tsv(
        metric_rows,
        ["metric", "verdict", "n", "pct", "metric_total_n"],
        OUT / "p6_iter142_metric_x_verdict.tsv",
    )
    _write_tsv(
        panel_rows,
        ["panel", "verdict", "n", "pct", "panel_total_n"],
        OUT / "p6_iter142_panel_x_verdict.tsv",
    )
    _write_tsv(
        sig_conc,
        ["delta_id", "tier", "n_declared_expected", "n_sign_match",
         "n_sign_match_significant", "pct_sign_match", "pct_sign_match_sig"],
        OUT / "p6_iter142_sign_concordance.tsv",
    )
    _write_tsv(
        paradox_rows,
        ["panel", "n_evaluated", "n_supports", "supports_rate_pct"],
        OUT / "p6_iter142_eta2_paradox.tsv",
    )

    # ----- summary JSON -----
    summary = {
        "n_deltas_audited": len(deltas),
        "n_per_rows": len(per_rows),
        "tier_to_verdict": {t: dict(c) for t, c in tv.items()},
        "verdict_counts_global": dict(Counter(r["verdict"] for r in per_rows)),
        "eta2_method_pt_anchor": eta2_pt,
        "eta2_method_lo_anchor": eta2_lo,
        "eta2_method_hi_anchor": eta2_hi,
        "supports_rate_n2_last10": (n2_supports / n2_total if n2_total else 0.0),
        "supports_rate_zvf130_5seed": (zvf130_supports / zvf130_total if zvf130_total else 0.0),
        "n_sign_match_total": sum(r["n_sign_match"] for r in sig_conc),
        "n_sign_match_sig_total": sum(r["n_sign_match_significant"] for r in sig_conc),
        "n_declared_total": sum(r["n_declared_expected"] for r in sig_conc),
    }
    (OUT / "p6_iter142_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )

    # ----- stdout -----
    print("=== iter-142 P6 claim_validation aggregate audit ===")
    print(f"deltas audited: {len(deltas)}")
    print(f"per-(delta, metric, panel) rows: {len(per_rows)}")
    print("global verdict counts:")
    for v, c in Counter(r["verdict"] for r in per_rows).most_common():
        print(f"  {v}: {c}")
    print("tier × verdict (n, %):")
    for r in tier_rows:
        print(f"  tier={r['tier']} verdict={r['verdict']}: n={r['n']} pct={r['pct']}%")
    print("η²(method) anchor (iter-141):")
    print(f"  point={eta2_pt}, CI=[{eta2_lo}, {eta2_hi}]")
    print("panel SUPPORTS rate (paradox test):")
    for r in paradox_rows:
        print(f"  panel={r['panel']}: {r['n_supports']}/{r['n_evaluated']} = {r['supports_rate_pct']}%")


if __name__ == "__main__":
    main()
