"""ZVF confound analysis on the 368-run W&B `zvf-audit` corpus (P7 U-shape, P5/P6 Table p5-ushape).

Question: is the population "U-shape of ZVF in accuracy" a reward effect, or is it carried by
model identity and configuration (batch, group size G, lr, steps), which are nearly collinear
with model in this corpus?  And how much of step-level ZVF is the binomial identity
h_G(p) = p^G + (1-p)^G evaluated at the step's mean reward?

Input : experiments/results/publication_fixes/zvf_confound/zvf_audit_histories.json
        (pulled by pull_zvf_audit_histories.py; 368 runs, per-step zvf/reward)
Output: experiments/results/publication_fixes/zvf_confound/zvf_confound_results.json
        experiments/results/publication_fixes/zvf_confound/zvf_confound_output.txt (stdout)

Run   : python analysis_publication_fixes/zvf_confound.py
"""

import argparse
import platform
import statsmodels
import hashlib
import sys
import json
import pathlib

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/results/publication_fixes/zvf_confound"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from analysis_validation import robust_incremental


def h(p, G):
    return p**G + (1 - p) ** G


def short(m):
    return m.split("/")[-1]


def main(input_path, output):
    # ---------- tables ----------
    RUNS = json.loads(input_path.read_text())
    run_rows, step_rows, excluded_runs = [], [], []
    seen_runs = set()
    for r in RUNS:
        if not isinstance(r, dict) or r.get("error"):
            raise ValueError("incomplete history export record")
        if r["run_id"] in seen_runs:
            raise ValueError("duplicate history run ID")
        seen_runs.add(r["run_id"])
        c, s = r["config"], r["summary"]
        base = dict(
            run=r["run_id"],
            model=short(r["model"]),
            G=int(c["group"]),
            batch=int(c["batch"]),
            steps=int(c["steps"]),
            lr=float(c["lr"]),
            seed=c.get("seed"),
        )
        hist = [x for x in r["history"] if x.get("zvf") is not None and x.get("reward") is not None]
        if len(hist) != int(c["steps"]) or {x.get("step") for x in hist} != set(
            range(int(c["steps"]))
        ):
            excluded_runs.append(
                {"run_id": r["run_id"], "reason": "incomplete_or_duplicate_step_history"}
            )
            continue
        if not all(
            all(
                type(x[k]) in (float, int) and np.isfinite(x[k]) and 0 <= x[k] <= 1
                for k in ("zvf", "reward")
            )
            for x in hist
        ):
            raise ValueError("nonfinite or out-of-range history")
        if "mean_zvf" in s:
            run_rows.append(
                dict(
                    base,
                    zvf=float(np.mean([x["zvf"] for x in hist])),
                    final_reward=float(s["final_reward"]),
                    mean_step_reward=float(np.mean([x["reward"] for x in hist]))
                    if hist
                    else np.nan,
                    n_hist=len(hist),
                )
            )
        for x in hist:
            step_rows.append(
                dict(
                    base,
                    step=int(x["step"]),
                    zvf=float(x["zvf"]),
                    reward=float(x["reward"]),
                    hG=h(float(x["reward"]), int(c["group"])),
                )
            )
    R = pd.DataFrame(run_rows)
    S = pd.DataFrame(step_rows)
    S["excess"] = S.zvf - S.hG
    res = {
        "schema_version": "zvf-confound-v2-candidate",
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "excluded_runs": excluded_runs,
        "cohort_status": "limited" if excluded_runs else "complete_observed_history",
        "interpretation": "descriptive associations; HC3 does not identify causal mechanisms or resolve model/configuration confounding",
        "reward_window": "same observed step window as mean ZVF",
        "snapshot_binding": "legacy export may mix config/summary and later sampled history; no common snapshot authenticity claim",
        "n_runs": int(len(R)),
        "n_step_rows": int(len(S)),
        "n_runs_with_history": int(S.run.nunique()),
    }
    print(f"runs={len(R)}  step rows={len(S)}  runs with step history={S.run.nunique()}")

    # ---------- 1. per-model table (the 'U-shape' table), all 9 labels ----------
    tab = (
        R.groupby("model")
        .agg(
            n=("run", "size"),
            zvf=("zvf", "mean"),
            final_reward=("final_reward", "mean"),
            mean_step_reward=("mean_step_reward", "mean"),
            G=("G", lambda v: sorted(set(v))),
            batch=("batch", lambda v: sorted(set(v))),
            steps=("steps", lambda v: sorted(set(v))),
        )
        .sort_values("final_reward")
    )
    sm_tab = S.groupby("model").agg(
        step_zvf=("zvf", "mean"), step_hG=("hG", "mean"), step_excess=("excess", "mean")
    )
    tab = tab.join(sm_tab)
    print("\n== 1. Per-model table (summary mean_zvf vs FINAL-step reward; 9 labels) ==")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(tab.round(3).to_string())
    res["per_model"] = json.loads(tab.round(4).to_json(orient="index"))

    # ---------- 2. configuration is collinear with model ----------
    cfg = R.groupby(["model", "batch", "G", "steps"]).size().rename("n").reset_index()
    print("\n== 2. model x (batch, G, steps) cells ==")
    print(cfg.to_string(index=False))
    small = {"Llama-3.2-1B", "Llama-3.2-3B", "Qwen3-4B-Instruct-2507"}
    print(
        "Small-model runs with steps=5,batch=8:",
        int(((R.model.isin(small)) & (R.steps == 5) & (R.batch == 8)).sum()),
        "/",
        int(R.model.isin(small).sum()),
    )
    print(
        "Large-model runs (other 6 labels) with steps=12:",
        int(((~R.model.isin(small)) & (R.steps == 12)).sum()),
        "/",
        int((~R.model.isin(small)).sum()),
    )
    res["config_cells"] = cfg.to_dict(orient="records")

    # ---------- 3. run-level regressions ----------
    def fit(formula, data, groups=None):
        m = smf.ols(formula, data=data)
        f = (
            m.fit(cov_type="cluster", cov_kwds={"groups": data[groups]})
            if groups
            else m.fit(cov_type="HC3")
        )
        return f

    def summ(f, names):
        out = {
            "r2": float(f.rsquared),
            "adj_r2": float(f.rsquared_adj),
            "n": int(f.nobs),
            "covariance_type": f.cov_type,
            "design_rank": int(np.linalg.matrix_rank(f.model.exog)),
            "design_columns": len(f.params),
        }
        for n in names:
            if n in f.params:
                ci = f.conf_int().loc[n]
                out[n] = {
                    "coef": float(f.params[n]),
                    "se": float(f.bse[n]),
                    "p": float(f.pvalues[n]),
                    "ci": [float(ci[0]), float(ci[1])],
                }
        return out

    R["r"] = R.mean_step_reward
    R["r2"] = R.mean_step_reward**2
    forms = {
        "R0_reward_quadratic": "zvf ~ r + r2",
        "R1_model_only": "zvf ~ C(model)",
        "R2_config_only": "zvf ~ C(batch) + C(G) + C(steps) + np.log10(lr)",
        "R3_model+config": "zvf ~ C(model) + C(batch) + C(G) + np.log10(lr)",
        "R4_model+config+reward_quad": "zvf ~ C(model) + C(batch) + C(G) + np.log10(lr) + r + r2",
    }
    print(
        "\n== 3. Run-level OLS (HC3 SE), outcome = run mean ZVF; reward = same-window mean step reward =="
    )
    res["run_level"] = {}
    for k, fm in forms.items():
        f = fit(fm, R)
        res["run_level"][k] = summ(f, ["r", "r2"])
        extra = ""
        if "r2" in f.params:
            extra = (
                f"  r={f.params['r']:+.3f} (p={f.pvalues['r']:.2g})  r^2={f.params['r2']:+.3f} "
                f"(p={f.pvalues['r2']:.2g})"
            )
        print(f"{k:32s} R2={f.rsquared:.3f} adjR2={f.rsquared_adj:.3f}{extra}")
    # Joint reward restrictions use the SAME HC3 model/cohort as surrounding inference.
    res["run_level"]["incremental_reward_given_model_config"] = robust_incremental(
        R, forms["R3_model+config"], forms["R4_model+config+reward_quad"]
    )
    print("Incremental HC3 Wald F:", res["run_level"]["incremental_reward_given_model_config"])
    # U-shape across model means only (9 points)
    M = tab.reset_index()
    fm9 = smf.ols("zvf ~ mean_step_reward + I(mean_step_reward**2)", M).fit()
    print(
        f"Across {int(fm9.nobs)} model means: quad R2={fm9.rsquared:.3f}, coef r^2={fm9.params.iloc[2]:+.2f} p={fm9.pvalues.iloc[2]:.3g} (n={int(fm9.nobs)})"
    )
    res["model_means_quadratic"] = {
        "r2": float(fm9.rsquared),
        "coef_r2": float(fm9.params.iloc[2]),
        "p_r2": float(fm9.pvalues.iloc[2]),
        "n": int(fm9.nobs),
    }

    # ---------- 4. step-level regressions (cluster SE by run) ----------
    S["r"] = S.reward
    S["r2"] = S.reward**2
    sforms = {
        "S0_hG_only": "zvf ~ hG",
        "S1_reward_quadratic": "zvf ~ r + r2",
        "S2_model+batch+G": "zvf ~ C(model) + C(batch) + C(G) + np.log10(lr)",
        "S3_model+batch+G+reward_quad": "zvf ~ C(model) + C(batch) + C(G) + np.log10(lr) + r + r2",
        "S4_model+batch+G+reward_quad+hG": "zvf ~ C(model) + C(batch) + C(G) + np.log10(lr) + r + r2 + hG",
    }
    print("\n== 4. Step-level OLS (SE clustered by run), outcome = step ZVF ==")
    res["step_level"] = {}
    for k, fm in sforms.items():
        f = fit(fm, S, groups="run")
        res["step_level"][k] = summ(f, ["r", "r2", "hG"])
        terms = "  ".join(
            f"{n}={f.params[n]:+.3f}[{f.conf_int().loc[n, 0]:+.3f},{f.conf_int().loc[n, 1]:+.3f}]"
            for n in ("r", "r2", "hG")
            if n in f.params
        )
        print(f"{k:34s} R2={f.rsquared:.3f}  {terms}")
    y, p = S.zvf.values, S.hG.values
    r2_identity = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    print(
        f"Identity line ZVF = h_G(p_step) (no fit): R2={r2_identity:.3f}, corr={np.corrcoef(y, p)[0, 1]:.3f}, "
        f"mean excess={np.mean(y - p):+.3f}"
    )
    res["step_level"]["identity_line"] = {
        "r2": float(r2_identity),
        "corr": float(np.corrcoef(y, p)[0, 1]),
        "mean_excess": float(np.mean(y - p)),
    }

    # ---------- 5. excess over h_G(p_step) = dispersion; what predicts it? ----------
    print(
        "\n== 5. Excess ZVF over h_G(p_step) (descriptive empirical residual; mechanism unidentified) =="
    )
    res["excess"] = {}
    for k, fm in {
        "E0_model": "excess ~ C(model)",
        "E1_model+config": "excess ~ C(model) + C(batch) + C(G) + np.log10(lr)",
        "E2_model+config+reward": "excess ~ C(model) + C(batch) + C(G) + np.log10(lr) + r + r2",
    }.items():
        f = fit(fm, S, groups="run")
        res["excess"][k] = summ(f, ["r", "r2"])
        print(f"{k:26s} R2={f.rsquared:.3f}")
    print("Per-model mean excess (step level):")
    print(S.groupby("model").excess.agg(["mean", "count"]).sort_values("mean").round(3).to_string())

    # ---------- 6. within-model reward slope (does ZVF track reward inside a model?) ----------
    print("\n== 6. Within-model step-level association (model-demeaned) ==")
    S["zvf_dm"] = S.zvf - S.groupby("model").zvf.transform("mean")
    S["r_dm"] = S.r - S.groupby("model").r.transform("mean")
    S["hG_dm"] = S.hG - S.groupby("model").hG.transform("mean")
    within = {}
    for m, g in S.groupby("model"):
        if len(g) < 20 or g.r.std() == 0:
            continue
        within[m] = {
            "n": int(len(g)),
            "corr_zvf_reward": float(np.corrcoef(g.zvf, g.r)[0, 1]),
            "corr_zvf_hG": float(np.corrcoef(g.zvf, g.hG)[0, 1]),
        }
        print(
            f"{m:36s} n={len(g):4d} corr(ZVF,reward)={within[m]['corr_zvf_reward']:+.3f} "
            f"corr(ZVF,h_G(p))={within[m]['corr_zvf_hG']:+.3f}"
        )
    print(
        f"Pooled within-model corr(ZVF,reward)={np.corrcoef(S.zvf_dm, S.r_dm)[0, 1]:+.3f}, "
        f"corr(ZVF,h_G(p))={np.corrcoef(S.zvf_dm, S.hG_dm)[0, 1]:+.3f}"
    )
    res["within_model"] = within
    res["within_model_pooled"] = {
        "corr_zvf_reward": float(np.corrcoef(S.zvf_dm, S.r_dm)[0, 1]),
        "corr_zvf_hG": float(np.corrcoef(S.zvf_dm, S.hG_dm)[0, 1]),
    }

    # ---------- 7. Qwen3-32B G grid provenance ----------
    print("\n== 7. Qwen3-32B group-size grid: runs per cell ==")
    q = (
        R[R.model == "Qwen3-32B"]
        .groupby(["G", "batch", "steps"])
        .agg(n=("run", "size"), zvf=("zvf", "mean"), final_reward=("final_reward", "mean"))
    )
    print(q.round(3).to_string())
    res["qwen3_32b_grid"] = json.loads(q.round(4).reset_index().to_json(orient="records"))

    res["dependency_versions"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "statsmodels": statsmodels.__version__,
    }
    res["analysis_sha256"] = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
    res["claim_receipts"] = [
        {
            "claim": key,
            "result_field": "/" + key,
            "analysis_version": res["schema_version"],
            "input_hash_ref": "/input_sha256",
            "evidence_level": "limited_retained_complete_history_rows_recomputed",
            "cohort_ref": "/run_level/incremental_reward_given_model_config/run_ids",
            "exclusions_ref": "/excluded_runs",
            "rounding": "unrounded inference; descriptive model tables rounded to four decimals",
        }
        for key in (
            "per_model",
            "config_cells",
            "run_level",
            "model_means_quadratic",
            "step_level",
            "excess",
            "within_model",
            "within_model_pooled",
            "qwen3_32b_grid",
        )
    ]
    with open(output, "x") as stream:
        json.dump(res, stream, indent=2, allow_nan=False)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, default=OUT / "zvf_audit_histories.json")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    main(args.input, args.output)
