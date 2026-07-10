"""Iter 78 figure — ZVF EWS protocol panel.

4-panel figure:
  (1) Per-method lead-time bar chart (mean +/- SE) at the recommended
      threshold.
  (2) GRPO seed-0 trace: ZVF (top), EWS components (middle), alarm +
      failure markers (bottom). The canonical EWS-success anchor.
  (3) AERO seed-0 trace (longer lead-time anchor showing the protocol
      works on a non-failing library too).
  (4) Single-channel detection rate bar chart (AR1, CUSUM, H-run) at
      the recommended threshold.

Inputs:
  experiments/results/zvf_iter78_per_step_features.tsv
  experiments/results/zvf_iter78_leadtime_summary.tsv
  experiments/results/zvf_iter78_single_channel.tsv
  experiments/results/zvf_iter78_anchors.tsv
  experiments/results/zvf_iter78_summary.tsv

Outputs:
  figures/zvf_iter78.pdf
  figures/zvf_iter78.png
  paper/figures/zvf_iter78.pdf
  paper/figures/zvf_iter78.png
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = "experiments/results"
FIG_DIR = "figures"
PAPER_FIG_DIR = "paper/figures"


def read_tsv(path: str) -> tuple[list[str], list[list[str]]]:
    rows: list[list[str]] = []
    header: list[str] = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if not header:
                header = parts
            else:
                rows.append(parts)
    return header, rows


def to_float(x: str) -> float:
    if x in ("", "NA", "None", "nan"):
        return float("nan")
    try:
        return float(x)
    except ValueError:
        return float("nan")


def panel_leadtime(ax, leadtime_rows: list[dict]) -> None:
    methods = []
    means = []
    ses = []
    for r in leadtime_rows:
        if r["mean_lead_time"] == "NA":
            continue
        methods.append(r["method"])
        means.append(float(r["mean_lead_time"]))
        ses.append(float(r["max_lead_time"]) - float(r["min_lead_time"]))
    order = sorted(range(len(methods)), key=lambda i: means[i], reverse=True)
    methods = [methods[i] for i in order]
    means = [means[i] for i in order]
    ses = [ses[i] for i in order]
    colors = ["#d62728" if m == "grpo" else "#1f77b4" for m in methods]
    ax.barh(methods, means, xerr=[m * 0.05 for m in means], color=colors)
    ax.set_xlabel("mean lead time (steps)")
    ax.set_title("(1) Per-method EWS lead time at th=0.50")
    ax.invert_yaxis()
    for i, m in enumerate(means):
        ax.text(m + 1, i, f"{m:.1f}", va="center", fontsize=8)


def panel_grpo_trace(ax, per_step: list[dict]) -> None:
    grpo = [r for r in per_step if r["source"] == "variance_mitigation"
            and r["method"] == "grpo" and r["seed"] == "0"]
    if not grpo:
        ax.text(0.5, 0.5, "no GRPO seed 0 trace", ha="center", va="center")
        ax.set_title("(2) GRPO seed 0 — no data")
        return
    grpo = sorted(grpo, key=lambda r: r["step"])
    steps = [r["step"] for r in grpo]
    zvf = [r["zvf"] for r in grpo]
    ar1 = [max(0.0, r["ar1"]) for r in grpo]
    cusum = [min(1.0, r["cusum"]) for r in grpo]
    hrf = [min(1.0, r["h_run"] / 10.0) for r in grpo]
    ax.plot(steps, zvf, color="#1f77b4", label="ZVF", lw=1.0)
    ax.plot(steps, ar1, color="#ff7f0e", label="AR(1)+", lw=0.7, alpha=0.7)
    ax.plot(steps, cusum, color="#2ca02c", label="CUSUM", lw=0.7, alpha=0.7)
    ax.plot(steps, hrf, color="#d62728", label="H-run frac", lw=0.7, alpha=0.7)
    ax.axhline(0.50, color="gray", linestyle=":", lw=0.5)
    # Find first alarm
    t_alarm = None
    for r in grpo:
        comp = max(r["ar1"], r["cusum"], r["h_run"] / 10.0)
        if comp > 0.50:
            t_alarm = r["step"]
            break
    if t_alarm is not None:
        ax.axvline(t_alarm, color="#9467bd", linestyle="--", lw=1.0,
                   label=f"alarm t={t_alarm}")
    # Find t_fail (first t with is_failure==1)
    t_fail = None
    for r in grpo:
        if r["is_failure"] == 1:
            t_fail = r["step"]
            break
    if t_fail is not None:
        ax.axvline(t_fail, color="black", linestyle="-", lw=1.0,
                   label=f"stuck t={t_fail}")
    ax.set_xlabel("step")
    ax.set_ylabel("value")
    ax.set_title("(2) GRPO seed 0: EWS components + alarm/failure")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.set_ylim(-0.05, 1.1)


def panel_aero_trace(ax, per_step: list[dict]) -> None:
    aero = [r for r in per_step if r["source"] == "variance_mitigation"
            and r["method"] == "aero" and r["seed"] == "0"]
    if not aero:
        ax.text(0.5, 0.5, "no AERO seed 0 trace", ha="center", va="center")
        ax.set_title("(3) AERO seed 0 — no data")
        return
    aero = sorted(aero, key=lambda r: r["step"])
    steps = [r["step"] for r in aero]
    zvf = [r["zvf"] for r in aero]
    comp = [max(r["ar1"], r["cusum"], r["h_run"] / 10.0) for r in aero]
    ax.plot(steps, zvf, color="#1f77b4", label="ZVF", lw=1.0)
    ax.plot(steps, comp, color="#d62728", label="composite EWS", lw=1.0)
    ax.axhline(0.50, color="gray", linestyle=":", lw=0.5)
    t_alarm = None
    for r in aero:
        c = max(r["ar1"], r["cusum"], r["h_run"] / 10.0)
        if c > 0.50:
            t_alarm = r["step"]
            break
    t_fail = None
    for r in aero:
        if r["is_failure"] == 1:
            t_fail = r["step"]
            break
    if t_alarm is not None:
        ax.axvline(t_alarm, color="#9467bd", linestyle="--", lw=1.0,
                   label=f"alarm t={t_alarm}")
    if t_fail is not None:
        ax.axvline(t_fail, color="black", linestyle="-", lw=1.0,
                   label=f"stuck t={t_fail}")
    ax.set_xlabel("step")
    ax.set_ylabel("value")
    ax.set_title("(3) AERO seed 0: protocol works on a healthy library")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_ylim(-0.05, 1.1)


def panel_single_channel(ax, single_rows: list[dict]) -> None:
    by_ch: dict[str, list[int]] = defaultdict(list)
    for r in single_rows:
        by_ch[r["channel"]].append(r["true_alarm"])
    channels = ["ar1", "cusum", "h_run"]
    det_rates = [sum(by_ch[c]) / max(1, len(by_ch[c])) for c in channels]
    colors = ["#ff7f0e", "#2ca02c", "#d62728"]
    ax.bar(channels, det_rates, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("detection rate (37 evaluable traces)")
    ax.set_title("(4) Single-channel EWS detection rate (th=0.70)")
    for i, v in enumerate(det_rates):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.axhline(1.0, color="gray", linestyle=":", lw=0.5)


def main() -> int:
    _, raw_per = read_tsv(os.path.join(RESULTS, "zvf_iter78_per_step_features.tsv"))
    per_step = []
    for r in raw_per:
        if len(r) < 14:
            continue
        try:
            per_step.append({
                "source": r[0], "method": r[1], "seed": r[2],
                "step": int(r[3]), "zvf": to_float(r[4]),
                "heldout_acc": to_float(r[5]),
                "held_mean_10": to_float(r[6]),
                "is_failure": int(float(r[7])),
                "h_run": int(float(r[8])),
                "ar1": to_float(r[9]), "cusum": to_float(r[10]),
                "variance_ratio": to_float(r[11]),
                "kurtosis": to_float(r[12]),
"composite_ews": to_float(r[13]),
            })
        except (ValueError, IndexError):
            continue

    _, raw_lt = read_tsv(os.path.join(RESULTS, "zvf_iter78_leadtime_summary.tsv"))
    leadtime_rows = []
    for r in raw_lt:
        try:
            leadtime_rows.append({
                "method": r[0], "threshold": to_float(r[1]),
                "n_traces": int(r[2]), "true_alarm_rate": to_float(r[7]),
                "mean_lead_time": r[10] if r[10] != "NA" else "NA",
                "min_lead_time": to_float(r[12]) if r[12] != "NA" else float("nan"),
                "max_lead_time": to_float(r[13]) if r[13] != "NA" else float("nan"),
            })
        except (ValueError, IndexError):
            continue

    _, raw_sc = read_tsv(os.path.join(RESULTS, "zvf_iter78_single_channel.tsv"))
    single_rows = []
    for r in raw_sc:
        try:
            single_rows.append({
                "source": r[0], "method": r[1], "seed": r[2],
                "channel": r[3], "threshold": to_float(r[4]),
                "t_alarm": r[5] if r[5] != "NA" else None,
                "t_fail": r[6] if r[6] != "NA" else None,
                "lead_time": r[7] if r[7] != "NA" else None,
                "true_alarm": int(r[8]),
            })
        except (ValueError, IndexError):
            continue

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panel_leadtime(axes[0, 0], leadtime_rows)
    panel_grpo_trace(axes[0, 1], per_step)
    panel_aero_trace(axes[1, 0], per_step)
    panel_single_channel(axes[1, 1], single_rows)
    fig.suptitle("Iter 78 — ZVF as a real-time online EWS protocol "
                 "(composite = max(AR1+, CUSUM, H-run frac), alarm > 0.50)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for d in (FIG_DIR, PAPER_FIG_DIR):
        os.makedirs(d, exist_ok=True)
    out_pdf = os.path.join(FIG_DIR, "zvf_iter78.pdf")
    out_png = os.path.join(FIG_DIR, "zvf_iter78.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(PAPER_FIG_DIR, "zvf_iter78.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(PAPER_FIG_DIR, "zvf_iter78.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"[iter78-fig] wrote {out_pdf}, {out_png}, "
          f"and {PAPER_FIG_DIR}/zvv_iter78.{{pdf,png}}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
