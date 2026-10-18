"""Candidate v2 reanalysis; historical outputs are retained unchanged.

The per-group plug-in residual is nonnegative algebraically. The EMPIRICAL
per-step residual has sampling plug-in bias minus a heterogeneity gap, so its
expected sign is not fixed. Count overdispersion is consistent with a
heterogeneous-probability assumption but does not identify that mechanism.
Run with --output NEW.json. No training, provider calls, or stored-result edits.
"""

import csv
import argparse
import platform
import scipy
import hashlib
import sys
import json
import math
import pathlib
import statistics as st

from scipy import optimize, stats
from scipy.special import betaln, comb

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "experiments/results"
OUT = RES / "publication_fixes/zvf_confound"
G = 8
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from analysis_validation import mean_preserving_points, trajectory_summary


def h(p, g=G):
    return p**g + (1 - p) ** g


def main(output):
    out = {
        "schema_version": "zvf-delta-v2-candidate",
        "mechanism_claim": "The fitted count model reproduces observed overdispersion under a heterogeneous-probability assumption; it does not uniquely distinguish prompt heterogeneity from other dependence mechanisms.",
    }
    input_paths = [RES / f"tinker_gsm8k_zvf_s{seed}.json" for seed in (42, 123, 456)] + [
        RES / "zvf_contrastive_yield.tsv"
    ]
    out["input_sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in input_paths
    }
    out["dependency_versions"] = {"python": platform.python_version(), "scipy": scipy.__version__}
    out["analysis_sha256"] = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
    out["group_identities"] = []
    # ---------- (1) GSM8K plug-in delta ----------
    ks = []
    for s in (42, 123, 456):
        d = json.load(open(RES / f"tinker_gsm8k_zvf_s{s}.json"))
        for index, pp in enumerate(d["per_problem"]):
            if len(pp["rewards"]) != G or any(
                type(x) not in (bool, int, float) or x not in (0, 1) for x in pp["rewards"]
            ):
                raise ValueError("binary complete group required")
            ks.append(sum(int(x) for x in pp["rewards"]))
            out["group_identities"].append(
                {"seed": s, "prompt_id": pp.get("problem_id", index), "G": G}
            )
    n = len(ks)
    dl = [h(k / G) - (k in (0, G)) for k in ks]
    pbar = sum(ks) / (G * n)
    zvf = sum(k in (0, G) for k in ks) / n
    print(f"GSM8K: n={n} groups, ZVF_obs={zvf:.4f}, pbar={pbar:.4f}")
    print(
        f"(1) plug-in delta: mean={st.mean(dl):+.4f}  min={min(dl):.4f}  frac>0={sum(x > 0 for x in dl) / n:.3f}  "
        f"frac<0={sum(x < 0 for x in dl) / n:.3f}"
    )
    # expected plug-in delta under the exact i.i.d. Bernoulli null, with per-prompt p drawn from the fitted beta
    # and, separately, with every prompt at pbar
    null_hom = sum(stats.binom.pmf(k, G, pbar) * (h(k / G) - (k in (0, G))) for k in range(G + 1))

    def nll(t):
        a, b = math.exp(t[0]), math.exp(t[1])
        return -sum(math.log(comb(G, k)) + betaln(k + a, G - k + b) - betaln(a, b) for k in ks)

    r = optimize.minimize(nll, [0, 0], method="Nelder-Mead")
    if not r.success or not math.isfinite(r.fun):
        raise ValueError("beta-binomial fit did not converge")
    a, b = map(math.exp, r.x)
    bb = [comb(G, k) * math.exp(betaln(k + a, G - k + b) - betaln(a, b)) for k in range(G + 1)]
    null_bb = sum(bb[k] * (h(k / G) - (k in (0, G))) for k in range(G + 1))
    print(f"    E[plug-in delta] under exact iid Bernoulli(pbar): {null_hom:+.4f}")
    print(
        f"    E[plug-in delta] under iid Bernoulli(p_i), p_i ~ Beta({a:.2f},{b:.2f}): {null_bb:+.4f}"
    )
    out["gsm8k_plugin_delta"] = dict(
        n=n,
        mean=st.mean(dl),
        min=min(dl),
        frac_pos=sum(x > 0 for x in dl) / n,
        null_iid_homogeneous=null_hom,
        null_iid_betabinomial=null_bb,
    )

    # ---------- (3) dispersion ----------
    hist = [ks.count(k) for k in range(G + 1)]
    zb = h(pbar)
    zbb = bb[0] + bb[G]
    icc = 1 / (a + b + 1)
    binom = [n * stats.binom.pmf(k, G, pbar) for k in range(G + 1)]
    chi = sum((o - e) ** 2 / e for o, e in zip(hist, binom, strict=False))
    print(
        f"(3) ZVF_obs={zvf:.4f}  homogeneous h_8(pbar)={zb:.4f} (ratio {zvf / zb:.2f})  "
        f"beta-binomial (ICC={icc:.3f}) predicts {zbb:.4f}"
    )
    print(
        f"    k-histogram obs={hist}; chi2 vs Binomial(8,pbar)={chi:.0f} (descriptive discrepancy only)"
    )
    out["gsm8k_dispersion"] = dict(
        zvf_obs=zvf,
        pbar=pbar,
        h_pbar=zb,
        ratio=zvf / zb,
        beta_a=a,
        beta_b=b,
        icc=icc,
        zvf_betabinomial=zbb,
        k_hist=hist,
        chi2_vs_binomial=chi,
        optimizer_converged=bool(r.success),
        fit_status="descriptive_count_fit_not_mechanism_identification",
        expected_count_histogram=[n * p for p in bb],
        chi2_vs_fitted_betabinomial=sum(
            (observed - n * p) ** 2 / (n * p) for observed, p in zip(hist, bb, strict=False)
        ),
    )

    # ---------- (2) empirical per-step residual: descriptive only ----------
    rows = [
        r
        for r in csv.DictReader(
            (l for l in open(RES / "zvf_contrastive_yield.tsv") if not l.startswith("#")),
            delimiter="\t",
        )
        if r["source"] == "groupsize_zvf_sweep"
    ]
    dd = [float(r["delta_div"]) for r in rows]
    summary = trajectory_summary(rows)
    print(
        f"(2) descriptive empirical per-step residual: n={len(dd)} mean={st.mean(dd):+.4f}; whole-seed bootstrap CI={summary['ci']}"
    )
    out["arithmetic_by_group_size"] = {}
    # Latent probabilities only: this illustration is not the empirical plug-in null.
    for Gx in (2, 4, 8, 16):
        sub = [r for r in rows if int(r["G"]) == Gx]
        if not sub:
            continue
        pb = st.mean(float(r["p_x"]) for r in sub)
        # Mean-preserving two-point latent mixture, radius reduced near boundaries
        lo, hi = mean_preserving_points(pb)
        out["arithmetic_by_group_size"][str(Gx)] = trajectory_summary(sub)
        jensen = h(pb, Gx) - 0.5 * (h(lo, Gx) + h(hi, Gx))
        print(
            f"    G={Gx:2d}: n={len(sub)} obs mean delta={st.mean(float(r['delta_div']) for r in sub):+.4f}; "
            f"latent-probability delta for a mean-preserving difficulty spread at pbar={pb:.2f}: {jensen:+.4f}"
        )
    out["arith_step_delta"] = summary
    out["finite_sample_null"] = {
        "status": "not_identified_from_aggregate_rows",
        "reason": "Prompt probabilities and within-step raw group structure unavailable; empirical residual reported descriptively",
        "empirical_sign_guarantee": False,
    }
    out["claim_receipts"] = [
        {
            "claim": key,
            "result_field": "/" + key,
            "analysis_version": out["schema_version"],
            "input_hashes_ref": "/input_sha256",
            "evidence_level": "retained_group_or_trajectory_records_recomputed",
            "rounding": "unrounded JSON",
        }
        for key in (
            "gsm8k_plugin_delta",
            "gsm8k_dispersion",
            "arith_step_delta",
            "arithmetic_by_group_size",
        )
    ]
    with open(output, "x") as stream:
        json.dump(out, stream, indent=2, allow_nan=False)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    main(parser.parse_args().output)
