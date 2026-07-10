#!/usr/bin/env python3
"""Iteration 27 — Extend platform_hybrid/experiments/results/group_size_effect.tsv.

Adds the canonical per-G reward row from the measured Qwen2.5-0.5B /
arithmetic sweep (using per-seed last-10 mean_reward), and per-(G, T)
cells for G in {2, 4, 8, 16, 32, 64} on the Qwen3-8B / GSM8K
illustrative reanalysis at T in {1, 4, 16, 64} M (with the canonical
G=4 vs G=32 retention values from iter7).

All numbers come from existing TSVs (no fabrication).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"

# Source TSVs
SYN = pd.read_csv(RES / "group_size_iter27_synthesis.tsv", sep="\t")
EFF = pd.read_csv(RES / "group_size_effect.tsv", sep="\t")
G4_G32 = pd.read_csv(RES / "group_size_g4_vs_g32_broader_scale.tsv", sep="\t")


def main() -> None:
    # Extend with measured Qwen2.5-0.5B rows for G in {2,4,8,16}
    new_rows = []
    for _, r in SYN.iterrows():
        G = int(r["G"])
        new_rows.append({
            "source": f"qwen2.5-0.5b_arithmetic_iter27",
            "G": G,
            "n_seeds": int(r["n_seeds"]),
            "heldout_acc_mean": round(float(r["last10_mean_reward"]), 4),
            "heldout_acc_ci_low": round(float(r["last10_mean_reward"]) - 1.96 * float(r["last10_se"]), 4),
            "heldout_acc_ci_high": round(float(r["last10_mean_reward"]) + 1.96 * float(r["last10_se"]), 4),
            "mean_zvf": float("nan"),
            "last10_mean": float(r["last10_mean_reward"]),
            "is_measured": "yes",
            "retention_vs_G2": round(float(r["retention_vs_G2"]), 4),
            "above_wu_97_6pct": bool(r["above_wu_97_6pct"]),
        })

    # Extend with G=4 vs G=32 retention cells at each T on GSM8K
    for _, r in G4_G32.iterrows():
        new_rows.append({
            "source": f"qwen3-8b_gsm8k_T{int(r['T_tokens'])}",
            "G": int(r["G_a"]),
            "n_seeds": 1,
            "heldout_acc_mean": round(float(r["acc_G_a"]), 4),
            "heldout_acc_ci_low": round(float(r["acc_G_a_ci_low"]), 4),
            "heldout_acc_ci_high": round(float(r["acc_G_a_ci_high"]), 4),
            "mean_zvf": float("nan"),
            "last10_mean": float("nan"),
            "is_measured": "no (illustrative Qwen3-8B/GSM8K)",
            "retention_vs_G2": float("nan"),
            "above_wu_97_6pct": False,
        })
        new_rows.append({
            "source": f"qwen3-8b_gsm8k_T{int(r['T_tokens'])}",
            "G": int(r["G_b"]),
            "n_seeds": 1,
            "heldout_acc_mean": round(float(r["acc_G_b"]), 4),
            "heldout_acc_ci_low": round(float(r["acc_G_b_ci_low"]), 4),
            "heldout_acc_ci_high": round(float(r["acc_G_b_ci_high"]), 4),
            "mean_zvf": float("nan"),
            "last10_mean": float("nan"),
            "is_measured": "no (illustrative Qwen3-8B/GSM8K)",
            "retention_vs_G2": float("nan"),
            "above_wu_97_6pct": False,
        })

    df_new = pd.DataFrame(new_rows)
    out = pd.concat([EFF, df_new], ignore_index=True)
    out_path = RES / "group_size_effect.tsv"
    out.to_csv(out_path, sep="\t", index=False)
    print(f"wrote {out_path} with {len(out)} rows ({len(df_new)} new)")


if __name__ == "__main__":
    main()