#!/usr/bin/env python3
"""M18: reproduce P5 Hamming-vs-|delta telemetry| Spearman rhos and run Mantel tests.

Data: first 98 data rows of platform_hybrid/experiments/results/mega_20260704/cells.tsv
(identical, cols 1-17, to the 98-row version committed in fb3f278cc that iter-65 used)
plus the per-cell manifests in mega_20260704/manifests/.

Steps
 1. Re-run the original script's H3 logic (same RNG seed / call order) by importing
    platform_modal/scripts/p5p8/p5_manifest_outcome_coupling.py with paths redirected.
 2. Full-matrix Spearman over all C(98,2)=4753 pairs.
 3. Mantel test: permute cell labels jointly on the Hamming matrix (rows+cols),
    Spearman statistic, 9999 permutations, seed 20261002, one-sided (rho >= obs)
    and two-sided (|rho| >= |obs|) p = (k+1)/(N+1).
"""
from __future__ import annotations

import csv
import importlib.util
import io
import json
import sys
import contextlib
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[4]
MEGA = REPO / "platform_hybrid/experiments/results/mega_20260704"
SCRIPT = REPO / "platform_modal/scripts/p5p8/p5_manifest_outcome_coupling.py"
OUT = Path(__file__).resolve().parent
N_PERM = 9999
SEED = 20261002


def write_98(path: Path) -> None:
    lines = (MEGA / "cells.tsv").read_text().splitlines(keepends=True)
    path.write_text("".join(lines[:99]))


def rerun_original(cells98: Path) -> dict:
    spec = importlib.util.spec_from_file_location("coup", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.CELLS_TSV = cells98
    mod.MANIFEST_DIR = MEGA / "manifests"
    rerun_dir = OUT / "m18_original_rerun"
    mod.OUT_DIR = rerun_dir
    with contextlib.redirect_stdout(io.StringIO()):
        mod.main()
    return json.loads((rerun_dir / "p5_manifest_outcome_coupling_summary.json").read_text())


def spearman_vec(x, y):
    rx, ry = rankdata(x), rankdata(y)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))


def main() -> int:
    cells98 = OUT / "m18_cells98.tsv"
    write_98(cells98)
    orig = rerun_original(cells98)

    # ---- load per-cell data exactly as the original (None zvf/pcd -> 0.0)
    items = ["loss_form", "ref_policy_kl", "sampler_backend_precision", "per_step_zvf_path",
             "group_size_schedule", "heldout_split", "decontamination_notes"]
    rows = list(csv.DictReader(cells98.open(), delimiter="\t"))
    fps, R, Z, P = [], [], [], []
    for r in rows:
        m = json.loads((MEGA / "manifests" / f"{r['cell_id']}.json").read_text())
        fps.append(tuple(str(m.get(k, "MISSING")) for k in items))
        f = lambda v: float(v) if v not in ("", None) else 0.0
        R.append(f(r["mean_reward"])); Z.append(f(r["zvf"])); P.append(f(r["pcd"]))
    n = len(rows)
    F = np.array(fps)
    H = (F[:, None, :] != F[None, :, :]).sum(-1).astype(float)
    iu = np.triu_indices(n, 1)
    outcomes = {"zvf": np.array(Z), "pcd": np.array(P), "mean_reward": np.array(R)}

    rng = np.random.default_rng(SEED)
    perms = [rng.permutation(n) for _ in range(N_PERM)]
    res = {}
    for name, v in outcomes.items():
        D = np.abs(v[:, None] - v[None, :])
        dvec = D[iu]
        rD = rankdata(dvec); rD = (rD - rD.mean()) / np.sqrt(((rD - rD.mean()) ** 2).sum())
        def stat(Hm):
            rh = rankdata(Hm[iu]); rh = rh - rh.mean()
            return float((rh * rD).sum() / np.sqrt((rh ** 2).sum()))
        obs = stat(H)
        null = np.array([stat(H[np.ix_(p, p)]) for p in perms])
        res[name] = {
            "rho_full_4753_pairs": round(obs, 4),
            "mantel_p_one_sided": round(float((np.sum(null >= obs - 1e-12) + 1) / (N_PERM + 1)), 5),
            "mantel_p_two_sided": round(float((np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (N_PERM + 1)), 5),
            "null_mean": round(float(null.mean()), 4),
            "null_q975": round(float(np.quantile(null, 0.975)), 4),
            "null_max": round(float(null.max()), 4),
        }

    # ---- Sensitivity: collapse seed replicates to config means (model, task, G, temp)
    keys = [(r["model"], r["task_slice"], r["G"], r["temperature"]) for r in rows]
    uk = sorted(set(keys))
    idx = {k: [i for i, kk in enumerate(keys) if kk == k] for k in uk}
    Fg = np.array([F[idx[k][0]] for k in uk])
    Fg[:, items.index("per_step_zvf_path")] = [str(i) for i in range(len(uk))]  # stays unique per unit
    Hg = (Fg[:, None, :] != Fg[None, :, :]).sum(-1).astype(float)
    ng = len(uk); iug = np.triu_indices(ng, 1)
    rng2 = np.random.default_rng(SEED + 1)
    permsg = [rng2.permutation(ng) for _ in range(N_PERM)]
    res_g = {}
    for name, v in outcomes.items():
        vg = np.array([v[idx[k]].mean() for k in uk])
        rD = rankdata(np.abs(vg[:, None] - vg[None, :])[iug]); rD = (rD - rD.mean()) / np.sqrt(((rD - rD.mean()) ** 2).sum())
        def stat_g(Hm):
            rh = rankdata(Hm[iug]); rh = rh - rh.mean()
            return float((rh * rD).sum() / np.sqrt((rh ** 2).sum()))
        obs = stat_g(Hg)
        null = np.array([stat_g(Hg[np.ix_(p, p)]) for p in permsg])
        res_g[name] = {"rho": round(obs, 4),
                       "mantel_p_one_sided": round(float((np.sum(null >= obs - 1e-12) + 1) / (N_PERM + 1)), 5),
                       "mantel_p_two_sided": round(float((np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (N_PERM + 1)), 5)}
    sens = {"n_config_units": ng, "n_pairs": int(len(iug[0])), "mantel": res_g}

    # Hamming distance distribution (per_step_zvf_path is unique per cell -> constant +1)
    hv, hc = np.unique(H[iu], return_counts=True)
    summary = {
        "n_cells": n,
        "n_pairs_full": int(len(iu[0])),
        "n_permutations": N_PERM,
        "seed": SEED,
        "statistic": "Spearman rho between upper-triangle Hamming and |delta outcome|; cell labels permuted jointly on rows+cols of the Hamming matrix",
        "hamming_value_counts": {int(a): int(b) for a, b in zip(hv, hc)},
        "original_script_rerun_2000_sampled_pairs": orig["h3"],
        "reported_in_thesis": {"zvf": 0.529, "pcd": 0.541, "mean_reward": 0.251},
        "mantel": res,
        "sensitivity_seed_collapsed": sens,
        "files": {
            "cells": "platform_hybrid/experiments/results/mega_20260704/cells.tsv (first 98 data rows)",
            "manifests": "platform_hybrid/experiments/results/mega_20260704/manifests/<cell_id>.json",
            "original_script": "platform_modal/scripts/p5p8/p5_manifest_outcome_coupling.py",
            "original_summary": "platform_hybrid/experiments/results/p5p8/p5_manifest_outcome_coupling_summary.json",
        },
    }
    (OUT / "m18_mantel_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
