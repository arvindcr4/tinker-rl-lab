"""Recompute the coupling-weight sweep and gate activity from the repository prototype.

Replaces the unsourced Table 2 of spectral_entropy_paper_kdense_revision.tex, whose cited
77-case K-Dense audit bundle is absent from the repository.

Uses only repository code: pilot.objective.spectral_legendre_grpo / entropic_givens_grpo,
pilot.benchmark_spectral_harness.generate_benchmark_fixture (seed 42, float64, CPU),
pilot.entropic_gating.compute_entropy_density.  Gradients are taken with respect to the
fixture log-probability tensor.

Usage (from repo root): PYTHONPATH=zvf-program/flagship python3 tools/review_ledger/Spectral_Entropy_lambda_sweep.py
Writes platform_hybrid/experiments/results/review_2026-10-18/Spectral_Entropy/lambda_sweep.json
"""
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from pilot.benchmark_spectral_harness import generate_benchmark_fixture
from pilot.entropic_gating import compute_entropy_density
from pilot.objective import entropic_givens_grpo, spectral_legendre_grpo

LAMBDAS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def grad(fn, data, lam):
    logps = data["logps"].clone().requires_grad_(True)
    loss, _ = fn(rewards=data["rewards"], logps=logps, old_logps=data["old_logps"],
                 completion_mask=data["completion_mask"], active_rows=data["active_rows"],
                 lambda_entropy=lam)
    (g,) = torch.autograd.grad(loss, logps)
    return g.reshape(-1)


def cos(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


out = {"lambdas": LAMBDAS, "configs": []}
for G in (4, 8, 16):
    for L in (512, 1024, 2048):
        normal, zvf = generate_benchmark_fixture(G, L, dtype=torch.float64)
        g_r = grad(spectral_legendre_grpo, normal, 0.0)
        rows = []
        for lam in LAMBDAS:
            g_h = grad(spectral_legendre_grpo, zvf, lam)
            g_n = grad(spectral_legendre_grpo, normal, lam)
            rows.append({"lambda": lam, "homogeneous_grad_norm": float(g_h.norm()),
                         "heterogeneous_cos_vs_reward": cos(g_n, g_r),
                         "one_minus_cos": 1.0 - cos(g_n, g_r)})
        slope = rows[-1]["homogeneous_grad_norm"]  # lambda = 1
        lin_err = max(abs(r["homogeneous_grad_norm"] - slope * r["lambda"]) / max(slope * r["lambda"], 1e-300)
                      for r in rows if r["lambda"] > 0)
        ent = [float(compute_entropy_density(F.softmax(zvf["logps"][i], dim=-1), dim=-1)) for i in range(G)]
        gated_eq = torch.equal(grad(spectral_legendre_grpo, zvf, 0.1), grad(entropic_givens_grpo, zvf, 0.1))
        out["configs"].append({
            "G": G, "L": L, "reward_grad_norm": float(g_r.norm()), "rows": rows,
            "max_relative_linearity_error": lin_err,
            "gate_row_entropy_min": min(ent), "gate_row_entropy_max": max(ent), "gate_tau": 0.5,
            "log_L": math.log(L), "gated_equals_ungated_homogeneous_grad": gated_eq,
        })
        print(G, L, "|g_r|=%.6g" % g_r.norm(), "slope=%.6g" % slope, "linerr=%.2e" % lin_err,
              "cos@0.1=%.6f cos@1=%.6f" % (rows[4]["heterogeneous_cos_vs_reward"], rows[5]["heterogeneous_cos_vs_reward"]),
              "H=[%.4f,%.4f]" % (min(ent), max(ent)), "gated==ungated", gated_eq)
dst = Path("platform_hybrid/experiments/results/review_2026-10-18/Spectral_Entropy")
dst.mkdir(parents=True, exist_ok=True)
(dst / "lambda_sweep.json").write_text(json.dumps(out, indent=1))
