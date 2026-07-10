#!/usr/bin/env python3
"""P6 iter-34: populate the new variant_delta_record.measured block from real data.

Two provenanced panels ground each GRPO-variant delta record in measured effect
(variant minus base=grpo), not just claimed component changes:

  * n2_same_stack_last10 -- per-step paired bootstrap on the N2 four-method
    same-stack reward tensors (aero/gift/areal), metrics {zvf, reward_mean},
    last 10 of 40 steps, paired by step index (n_boot=2000, seed=20260704).
  * zvf130_5seed -- normal-approx CI on the 5-seed zvf_iter130 method-risk panel
    (8 variants), metrics {zvf_risk_mean, mean_zvf(=mag_mean)}, delta vs grpo,
    se = sqrt(sd_v^2/n_v + sd_g^2/n_g).

Writes the `measured` array into each registry/entries/delta_*.json for which a
panel carries the variant, validates every entry against the bumped schema, and
emits a TSV. Stdlib + jsonschema only.
"""
import csv
import json
import pathlib
import random
import statistics as st

ROOT = pathlib.Path(__file__).resolve().parents[2]
ENTRIES = ROOT / "registry" / "entries"
N2 = ROOT / "experiments/results/n2_reward_tensor_resume/n2_metrics.tsv"
Z130 = ROOT / "experiments/results/zvf_iter130_method_risk.tsv"
OUT = ROOT / "experiments/results/p5p8"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260704
N_BOOT = 2000
BASE = "grpo"
N2_VARIANTS = ["aero", "gift", "areal"]
Z130_VARIANTS = ["aero", "gift", "areal", "cppo", "ngrpo", "mcgrpo", "es", "scafgrpo"]
LAST_K = 10


def read_tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ---- N2 panel: paired-by-step bootstrap on last-K steps -------------------
def n2_series(rows, method, metric):
    vals = [(int(r["step"]), fnum(r[metric])) for r in rows if r["method"] == method]
    vals = [(s, v) for s, v in vals if v is not None]
    vals.sort()
    return [v for _, v in vals][-LAST_K:]


def paired_boot(dv, dg, n_boot=N_BOOT, seed=SEED):
    # dv, dg aligned per-step series; delta_i = variant_i - base_i
    d = [a - b for a, b in zip(dv, dg)]
    rng = random.Random(seed)
    n = len(d)
    means = []
    for _ in range(n_boot):
        s = [d[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot) - 1]
    return sum(d) / n, lo, hi, n


def n2_measured():
    rows = read_tsv(N2)
    out = {}  # method -> list of measured dicts
    for m in N2_VARIANTS:
        recs = []
        for metric in ("zvf", "reward_mean"):
            vser = n2_series(rows, m, metric)
            gser = n2_series(rows, BASE, metric)
            delta, lo, hi, n = paired_boot(vser, gser)
            recs.append({
                "metric": metric, "panel": "n2_same_stack_last10", "base": BASE,
                "delta": round(delta, 6), "ci_low": round(lo, 6), "ci_high": round(hi, 6),
                "n": n, "significant": (lo > 0) or (hi < 0),
                "ci_method": {"method": "paired_step_bootstrap_pct", "n_boot": N_BOOT,
                              "seed": SEED, "ci_level": 0.95,
                              "source": "platform_modal/scripts/p5p8/p6_measured_delta_block.py"},
                "source": "experiments/results/n2_reward_tensor_resume/n2_metrics.tsv",
                "note": f"last {LAST_K} of 40 steps, G=8, seed 0, same stack",
            })
        out[m] = recs
    return out


# ---- zvf130 panel: 5-seed normal-approx delta vs grpo ---------------------
def z130_measured():
    rows = {r["method"]: r for r in read_tsv(Z130)}
    g = rows[BASE]
    gm_risk, gsd_risk, gn = fnum(g["zvf_risk_mean"]), fnum(g["zvf_risk_sd"]), int(g["n_seeds"])
    gm_mag = fnum(g["mag_mean"])
    # grpo mag has no per-seed sd column beyond risk; use risk sd as conservative proxy? No:
    # mag_mean sd is not stored, so we only CI the risk metric (which has sd), and report
    # mean_zvf(mag) as a point delta with n but no CI-from-sd -> mark not significant unless sd avail.
    out = {}
    for m in Z130_VARIANTS:
        r = rows[m]
        recs = []
        # zvf_risk_mean: has sd for both -> normal approx CI
        vm, vsd, vn = fnum(r["zvf_risk_mean"]), fnum(r["zvf_risk_sd"]), int(r["n_seeds"])
        se = (vsd ** 2 / vn + gsd_risk ** 2 / gn) ** 0.5
        d = vm - gm_risk
        lo, hi = d - 1.96 * se, d + 1.96 * se
        recs.append({
            "metric": "zvf_risk_mean", "panel": "zvf130_5seed", "base": BASE,
            "delta": round(d, 6), "ci_low": round(lo, 6), "ci_high": round(hi, 6),
            "n": vn, "significant": (lo > 0) or (hi < 0),
            "ci_method": {"method": "normal_approx_welch", "n_boot": None, "seed": None,
                          "ci_level": 0.95, "source": "platform_modal/scripts/p5p8/p6_measured_delta_block.py"},
            "source": "experiments/results/zvf_iter130_method_risk.tsv",
            "note": f"5-seed risk index; delta vs grpo (grpo risk={gm_risk:.4f})",
        })
        # mean_zvf (mag_mean): point delta, per-seed sd not stored -> CI = delta,delta, not sig
        dmag = fnum(r["mag_mean"]) - gm_mag
        recs.append({
            "metric": "mean_zvf", "panel": "zvf130_5seed", "base": BASE,
            "delta": round(dmag, 6), "ci_low": round(dmag, 6), "ci_high": round(dmag, 6),
            "n": vn, "significant": False,
            "ci_method": {"method": "point_no_perseed_sd", "n_boot": None, "seed": None,
                          "ci_level": None, "source": "platform_modal/scripts/p5p8/p6_measured_delta_block.py"},
            "source": "experiments/results/zvf_iter130_method_risk.tsv",
            "note": "mag_mean per-seed sd not stored; point estimate only (unmeasurable CI)",
        })
        out[m] = recs
    return out


def main():
    n2 = n2_measured()
    z130 = z130_measured()
    methods = sorted(set(N2_VARIANTS) | set(Z130_VARIANTS))
    tsv_rows = []
    import jsonschema
    schema = json.load(open(ROOT / "registry/schema.json"))
    V = jsonschema.Draft202012Validator(schema)
    written = 0
    for m in methods:
        path = ENTRIES / f"delta_{m}.json"
        rec = json.load(open(path))
        measured = n2.get(m, []) + z130.get(m, [])
        rec["measured"] = measured
        errs = list(V.iter_errors(rec))
        assert not errs, (m, errs[0].message)
        path.write_text(json.dumps(rec, indent=2) + "\n")
        written += 1
        for md in measured:
            tsv_rows.append({"delta_id": f"delta_{m}", "method": m, **{
                k: md[k] for k in ("metric", "panel", "delta", "ci_low", "ci_high",
                                   "n", "significant", "source")}})
    # full-registry re-validation
    ok = bad = 0
    for p in sorted(ENTRIES.glob("*.json")):
        if list(V.iter_errors(json.load(open(p)))):
            bad += 1
        else:
            ok += 1
    tsvp = OUT / "p6_measured_delta_block.tsv"
    with open(tsvp, "w", newline="") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=list(tsv_rows[0].keys()))
        w.writeheader()
        w.writerows(tsv_rows)
    summ = {
        "entries_written": written, "measured_rows": len(tsv_rows),
        "registry_validate": {"pass": ok, "fail": bad, "total": ok + bad},
        "n2_significant": sum(1 for r in tsv_rows if r["panel"] == "n2_same_stack_last10" and r["significant"]),
        "z130_significant": sum(1 for r in tsv_rows if r["panel"] == "zvf130_5seed" and r["significant"]),
        "seed": SEED, "n_boot": N_BOOT,
    }
    json.dump(summ, open(OUT / "p6_measured_delta_block_summary.json", "w"), indent=2)
    print(json.dumps(summ, indent=2))
    print(f"wrote {tsvp}")


if __name__ == "__main__":
    main()
