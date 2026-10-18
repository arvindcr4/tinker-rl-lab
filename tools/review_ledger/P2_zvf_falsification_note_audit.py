"""Recompute every number added to paper_P2_zvf_falsification_note.tex in the 2026-10-18 pass.

Inputs (checked in, no GPU, no provider calls):
  platform_hybrid/experiments/results/tinker_gsm8k_zvf_s{42,123,456}.json   (600 GSM8K groups, G=8)
  platform_hybrid/experiments/results/zvf_contrastive_yield.tsv             (480 groupsize_zvf_sweep step rows)
  platform_hybrid/experiments/results/modal_registry/modal_artifacts/tinkerrl-results/groupsize_zvf/G*_s*.json
Output: platform_hybrid/experiments/results/review_2026-10-18/P2_zvf_falsification_note/audit.json
Usage: python tools/review_ledger/P2_zvf_falsification_note_audit.py
"""

import csv
import json
import math
import pathlib
import statistics as st
from itertools import combinations

import numpy as np
from scipy import optimize, stats
from scipy.special import betaln, comb

ROOT = pathlib.Path(__file__).resolve().parents[2]
RES = ROOT / "platform_hybrid/experiments/results"
OUT = RES / "review_2026-10-18/P2_zvf_falsification_note"
G = 8
SEEDS = (42, 123, 456)


def h(p, g=G):
    return p**g + (1 - p) ** g


def main():
    out = {}
    groups = []  # (seed, question, k)
    for s in SEEDS:
        d = json.load(open(RES / f"tinker_gsm8k_zvf_s{s}.json"))
        out.setdefault("run_metadata", {})[s] = {
            k: d[k] for k in ("model", "n_problems", "group_size", "temperature", "seed")
        }
        for pp in d["per_problem"]:
            assert len(pp["rewards"]) == G
            groups.append((s, pp["question"], int(sum(pp["rewards"]))))
    ks = np.array([g[2] for g in groups])
    n = len(ks)
    pbar = ks.sum() / (G * n)
    zvf = float(np.isin(ks, (0, G)).mean())
    plug = float(np.mean([h(k / G) for k in ks]))
    hist = np.bincount(ks, minlength=G + 1)
    out["gsm8k"] = dict(
        n=n,
        pbar=pbar,
        zvf_obs=zvf,
        zvf_per_seed=[float(np.isin(ks[i * 200 : (i + 1) * 200], (0, G)).mean()) for i in range(3)],
        h_pbar=h(pbar),
        ratio=zvf / h(pbar),
        plugin_mean=plug,
        plugin_delta=plug - zvf,
        pooled_residual=h(pbar) - zvf,
        mixed_share=1 - zvf,
        k_hist=hist.tolist(),
    )

    # beta-binomial MLE (same parameterisation as analysis_publication_fixes/p2_delta_artifact.py)
    def nll(t):
        a, b = math.exp(t[0]), math.exp(t[1])
        return -sum(math.log(comb(G, k)) + betaln(k + a, G - k + b) - betaln(a, b) for k in ks)

    r = optimize.minimize(nll, [0, 0], method="Nelder-Mead")
    a, b = map(math.exp, r.x)
    bb = np.array(
        [comb(G, k) * math.exp(betaln(k + a, G - k + b) - betaln(a, b)) for k in range(G + 1)]
    )
    binp = stats.binom.pmf(np.arange(G + 1), G, pbar)
    e_bb, e_bin = n * bb, n * binp
    pear_bb = (hist - e_bb) ** 2 / e_bb
    with np.errstate(divide="ignore", invalid="ignore"):
        g2_bb = 2 * np.nansum(np.where(hist > 0, hist * np.log(hist / e_bb), 0))
        g2_bin = 2 * np.nansum(np.where(hist > 0, hist * np.log(hist / e_bin), 0))
    # pooled k<=2 vs binomial
    o_p = np.r_[hist[:3].sum(), hist[3:]]
    e_p = np.r_[e_bin[:3].sum(), e_bin[3:]]
    pear_pool = float(((o_p - e_p) ** 2 / e_p).sum())
    # parametric bootstrap of G^2 vs Binomial(8, pbar_hat)
    rng = np.random.default_rng(20261018)
    B = 20000
    exceed = 0
    for _ in range(B):
        kk = rng.binomial(G, pbar, size=n)
        hh = np.bincount(kk, minlength=G + 1)
        pb_ = kk.sum() / (G * n)
        ee = n * stats.binom.pmf(np.arange(G + 1), G, pb_)
        g2 = 2 * np.sum(np.where(hh > 0, hh * np.log(np.where(hh > 0, hh, 1) / ee), 0))
        exceed += g2 >= g2_bin
    var_ratio = float(np.var(ks) / (G * pbar * (1 - pbar)))
    out["fit"] = dict(
        a=a,
        b=b,
        mean_ab=a / (a + b),
        icc=1 / (a + b + 1),
        zvf_bb=float(bb[0] + bb[G]),
        expected_bb=e_bb.tolist(),
        expected_bin=e_bin.tolist(),
        pearson_bb=float(pear_bb.sum()),
        pearson_bb_k0=float(pear_bb[0]),
        df_bb=G + 1 - 1 - 2,
        p_pearson_bb=float(stats.chi2.sf(pear_bb.sum(), 6)),
        g2_bb=float(g2_bb),
        p_g2_bb=float(stats.chi2.sf(g2_bb, 6)),
        g2_bin=float(g2_bin),
        df_bin=7,
        pooled_le2_expected=float(e_p[0]),
        pearson_pooled=pear_pool,
        df_pooled=5,
        bootstrap_reps=B,
        bootstrap_exceed=int(exceed),
        var_ratio=var_ratio,
        null_plugin_hom=float(sum(binp[k] * (h(k / G) - (k in (0, G))) for k in range(G + 1))),
        null_plugin_bb=float(sum(bb[k] * (h(k / G) - (k in (0, G))) for k in range(G + 1))),
    )
    # simulated 95% bands for mean plug-in delta over 600 independent groups
    bands = {}
    for name, pmf in (("binomial", binp), ("betabinomial", bb / bb.sum())):
        sims = []
        for _ in range(20000):
            kk = rng.choice(G + 1, size=n, p=pmf)
            sims.append(
                np.mean([h(k / G) - (k in (0, G)) for k in kk])
                if False
                else float(np.mean(h(kk / G) - np.isin(kk, (0, G))))
            )
        bands[name] = [float(np.quantile(sims, 0.025)), float(np.quantile(sims, 0.975))]
    out["plugin_null_bands"] = bands

    # repeated problems across seeds
    by_q = {}
    for s, q, k in groups:
        by_q.setdefault(q, {})[s] = k
    counts = [len(v) for v in by_q.values()]
    pairs = [(v[s1], v[s2]) for v in by_q.values() for s1, s2 in combinations(sorted(v), 2)]
    x, y = zip(*pairs)
    out["repeats"] = dict(
        distinct=len(by_q),
        in_two=counts.count(2),
        in_three=counts.count(3),
        pairs=len(pairs),
        pearson_r=float(stats.pearsonr(x, y)[0]),
    )

    # synthetic arithmetic step residual (groupsize_zvf_sweep), TSV and raw JSON
    rows = [
        r
        for r in csv.DictReader(
            (l for l in open(RES / "zvf_contrastive_yield.tsv") if not l.startswith("#")),
            delimiter="\t",
        )
        if r["source"] == "groupsize_zvf_sweep"
    ]
    per_g = {
        g: st.mean(float(r["delta_div"]) for r in rows if int(r["G"]) == g) for g in (2, 4, 8, 16)
    }
    raw, meta = [], {}
    for f in sorted(
        (RES / "modal_registry/modal_artifacts/tinkerrl-results/groupsize_zvf").glob("G*_s*.json")
    ):
        d = json.load(open(f))
        meta[f.name] = {
            k: d[k] for k in ("model", "task", "group_size", "seed", "n_steps", "n_prompts")
        }
        g = d["group_size"]
        raw += [h(s["mean_reward"], g) - s["zvf"] for s in d["step_log"]]
    out["arithmetic"] = dict(
        tsv_rows=len(rows),
        tsv_mean=st.mean(float(r["delta_div"]) for r in rows),
        per_G=per_g,
        raw_runs=len(meta),
        raw_steps=len(raw),
        raw_mean=st.mean(raw),
        run_metadata=meta,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.json").write_text(json.dumps(out, indent=2, default=float))
    print(
        json.dumps(
            {k: v for k, v in out.items() if k not in ("run_metadata",)}, indent=1, default=float
        )[:6000]
    )


if __name__ == "__main__":
    main()
