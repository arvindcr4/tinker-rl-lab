#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import math
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path("/home/claude/tinker-rl-lab/experiments/results")
QUICK_DIR = BASE_DIR / "quick_20260704"
MEGA_DIR = BASE_DIR / "mega_20260704"
FIG_DIR = QUICK_DIR / "figures"
INDEX_PATH = FIG_DIR / "INDEX.md"

POLL_SECONDS = 120
MAX_DURATION_SECONDS = 45 * 60
QUIET_SECONDS = 10 * 60
ROW_DELTA_THRESHOLD = 5

FIG_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


@dataclass(frozen=True)
class Source:
    name: str
    path: Path
    renderer: Callable[[pd.DataFrame, int], list[tuple[str, str]]]
    min_rows: int = 1


figures_generated: dict[str, tuple[str, int, str]] = {}
last_rendered_rows: dict[Path, int] = {}
last_activity = time.time()
start_time = time.time()


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def count_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            line_count = sum(1 for _ in fh)
        return max(0, line_count - 1)
    except OSError:
        return 0


def read_tsv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, sep="\t", on_bad_lines="skip", na_values=["NA", "NaN", ""])
    except Exception as exc:
        print(f"[{now_iso()}] Skipping unreadable TSV {path}: {exc}", flush=True)
        return None


def first_col(df: pd.DataFrame, *names: str) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def numeric_frame(df: pd.DataFrame, required: list[str], optional: list[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    for col in required + (optional or []):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=required)


def save_fig(fig: plt.Figure, name: str, desc: str, row_count: int) -> tuple[str, str]:
    png_path = FIG_DIR / f"{name}.png"
    pdf_path = FIG_DIR / f"{name}.pdf"
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    figures_generated[name] = (desc, row_count, now_iso())
    return name, desc


def update_index() -> None:
    lines = [
        "# Figures Index",
        "",
        "| Figure Name | What It Shows | Rows At Last Render | Timestamp |",
        "|-------------|---------------|---------------------|-----------|",
    ]
    for name in sorted(figures_generated):
        desc, rows, ts = figures_generated[name]
        lines.append(f"| {name} | {desc} | {rows} | {ts} |")
    tmp = INDEX_PATH.with_suffix(".md.tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, INDEX_PATH)


def render_qp12(df: pd.DataFrame, row_count: int) -> list[tuple[str, str]]:
    reward_col = first_col(df, "reward", "reward_mean", "mean_reward")
    if reward_col is None or not {"seed", "step", "zvf"}.issubset(df.columns):
        return []
    clean = numeric_frame(df, ["seed", "step", reward_col, "zvf"])
    if clean.empty:
        return []

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
    for seed, group in clean.sort_values(["seed", "step"]).groupby("seed"):
        label = f"seed {int(seed) if float(seed).is_integer() else seed}"
        axes[0].plot(group["step"], group[reward_col], marker="o", linewidth=1.8, markersize=3, label=label)
        axes[1].plot(group["step"], group["zvf"], marker="o", linewidth=1.8, markersize=3, label=label)
    axes[0].set_title("Reward Trajectory")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Mean reward")
    axes[1].set_title("ZVF Trajectory")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("ZVF")
    axes[1].legend(title="Seed", frameon=False)
    fig.suptitle("qp12: Reward and ZVF by Seed", y=1.03)
    return [save_fig(fig, "qp12_reward_zvf", "Reward and ZVF vs step, one line per seed", row_count)]


def render_qp3(df: pd.DataFrame, row_count: int) -> list[tuple[str, str]]:
    reward_col = first_col(df, "reward", "reward_mean", "mean_reward")
    if reward_col is None or not {"G", "step"}.issubset(df.columns):
        return []
    clean = numeric_frame(df, ["G", "step", reward_col], ["zvf"])
    if clean.empty:
        return []

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for g, group in clean.sort_values(["G", "step"]).groupby("G"):
        label = f"G={int(g) if float(g).is_integer() else g}"
        ax.plot(group["step"], group[reward_col], marker="o", linewidth=2, markersize=3, label=label)
    ax.set_title("qp3: Reward by Group Size")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean reward")
    ax.legend(title="Group size", frameon=False, loc="best")

    if "zvf" in clean.columns and clean["zvf"].notna().any():
        inset = ax.inset_axes([0.58, 0.16, 0.36, 0.36])
        for g, group in clean.dropna(subset=["zvf"]).sort_values(["G", "step"]).groupby("G"):
            inset.plot(group["step"], group["zvf"], linewidth=1.4)
        inset.set_title("ZVF", fontsize=9)
        inset.set_xlabel("Step", fontsize=8)
        inset.set_ylabel("ZVF", fontsize=8)
        inset.tick_params(labelsize=7)
    return [save_fig(fig, "qp3_g4_vs_g8", "Reward vs step for G=4 vs G=8 with ZVF inset", row_count)]


def render_qp4(df: pd.DataFrame, row_count: int) -> list[tuple[str, str]]:
    if not {"model", "cap", "accuracy"}.issubset(df.columns):
        return []
    clean = numeric_frame(df, ["cap", "accuracy"])
    clean = clean[clean["cap"] > 0]
    if clean.empty:
        return []

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for model, group in clean.sort_values(["model", "cap"]).groupby("model"):
        label = str(model).split("/")[-1]
        ax.plot(group["cap"], group["accuracy"], marker="o", linewidth=2, markersize=4, label=label)
    ax.set_xscale("log", base=2)
    ax.set_title("qp4: Accuracy vs Generation Cap")
    ax.set_xlabel("Generation cap (tokens, log scale)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(bottom=0)
    ax.legend(title="Model", frameon=False, loc="best")
    return [save_fig(fig, "qp4_truncation", "Accuracy vs cap on log-x axis, one line per model", row_count)]


def render_qp7(df: pd.DataFrame, row_count: int) -> list[tuple[str, str]]:
    reward_col = first_col(df, "reward", "reward_mean", "mean_reward")
    if reward_col is None or not {"arm", "step", "G"}.issubset(df.columns):
        return []
    clean = numeric_frame(df, ["step", "G", reward_col])
    if clean.empty:
        return []

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.2), sharex=True, height_ratios=[2.0, 1.0])
    for arm, group in clean.sort_values(["arm", "step"]).groupby("arm"):
        axes[0].plot(group["step"], group[reward_col], marker="o", linewidth=2, markersize=3, label=str(arm))
        axes[1].step(group["step"], group["G"], where="post", linewidth=2, label=str(arm))
    axes[0].set_title("qp7: Fixed-G vs Adaptive-G Reward")
    axes[0].set_ylabel("Mean reward")
    axes[0].legend(title="Arm", frameon=False, loc="best")
    axes[1].set_title("Group Size Schedule")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("G")
    axes[1].legend(title="Arm", frameon=False, loc="best")
    return [save_fig(fig, "qp7_adaptive", "Reward vs step by arm with step-G staircase subplot", row_count)]


def render_qp8(df: pd.DataFrame, row_count: int) -> list[tuple[str, str]]:
    if not {"model", "accuracy", "auc"}.issubset(df.columns):
        return []
    clean = df.copy()
    clean["accuracy"] = pd.to_numeric(clean["accuracy"], errors="coerce")
    clean["auc"] = pd.to_numeric(clean["auc"], errors="coerce")
    clean = clean.dropna(subset=["accuracy", "auc"], how="all")
    if clean.empty:
        return []
    clean["label"] = clean["model"].astype(str).str.split("/").str[-1]
    if "split" in clean.columns:
        clean["label"] = clean["label"] + "\n" + clean["split"].astype(str)

    x = np.arange(len(clean))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.bar(x - width / 2, clean["accuracy"], width, label="Accuracy")
    ax.bar(x + width / 2, clean["auc"], width, label="AUC")
    ax.set_title("qp8: Fraud Detection Metrics")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(clean["label"], rotation=20, ha="right")
    ax.set_ylim(0, min(1.05, max(1.0, clean[["accuracy", "auc"]].max().max() * 1.15)))
    ax.legend(frameon=False, loc="best")
    return [save_fig(fig, "qp8_fraud", "Bar chart comparing model and XGBoost accuracy/AUC", row_count)]


def render_mega(df: pd.DataFrame, row_count: int) -> list[tuple[str, str]]:
    if not {"model", "G", "mean_reward", "zvf", "pcd"}.issubset(df.columns):
        return []
    clean = numeric_frame(df, ["G", "mean_reward", "zvf", "pcd"])
    if len(clean) < 50:
        return []

    rendered: list[tuple[str, str]] = []
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    scatter = ax.scatter(clean["mean_reward"], clean["zvf"], c=clean["G"], cmap="viridis", alpha=0.78, s=34)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("G")
    ax.set_title("Mega Campaign: ZVF vs Mean Reward")
    ax.set_xlabel("Mean reward")
    ax.set_ylabel("ZVF")
    rendered.append(save_fig(fig, "mega_zvf_vs_reward", "ZVF vs mean_reward scatter colored by G (U-shape view)", row_count))

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.scatter(clean["mean_reward"], clean["pcd"], alpha=0.7, s=34)
    ax.set_title("Mega Campaign: PCD vs Mean Reward")
    ax.set_xlabel("Mean reward")
    ax.set_ylabel("PCD")
    rendered.append(save_fig(fig, "mega_pcd_vs_reward", "PCD vs mean_reward scatter", row_count))

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    models = list(clean["model"].dropna().astype(str).sort_values().unique())
    gs = sorted(clean["G"].dropna().unique())
    positions = np.arange(len(gs))
    total_width = 0.78
    model_width = total_width / max(1, len(models))
    for m_idx, model in enumerate(models):
        model_df = clean[clean["model"].astype(str) == model]
        offset = -total_width / 2 + model_width * (m_idx + 0.5)
        data = [model_df.loc[model_df["G"] == g, "zvf"].dropna().to_numpy() for g in gs]
        box_positions = positions + offset
        non_empty = [(pos, vals) for pos, vals in zip(box_positions, data) if len(vals)]
        if not non_empty:
            continue
        ax.boxplot(
            [vals for _, vals in non_empty],
            positions=[pos for pos, _ in non_empty],
            widths=model_width * 0.72,
            patch_artist=True,
            showfliers=False,
            boxprops={"alpha": 0.18},
            medianprops={"linewidth": 1.5},
        )
        for pos, vals in non_empty:
            jitter = np.linspace(-model_width * 0.18, model_width * 0.18, len(vals)) if len(vals) > 1 else [0]
            ax.scatter(np.asarray(jitter) + pos, vals, s=18, alpha=0.7, label=model if pos == non_empty[0][0] else None)
    ax.set_title("Mega Campaign: ZVF by G and Model")
    ax.set_xlabel("G")
    ax.set_ylabel("ZVF")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(int(g)) if float(g).is_integer() else str(g) for g in gs])
    ax.legend(title="Model", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    rendered.append(save_fig(fig, "mega_zvf_by_g_model", "ZVF vs G box/strip plot per model", row_count))
    return rendered


SOURCES = [
    Source("qp12", QUICK_DIR / "qp12-zvf-dense.tsv", render_qp12),
    Source("qp3", QUICK_DIR / "qp3-gsweep.tsv", render_qp3),
    Source("qp4", QUICK_DIR / "qp4_truncation.tsv", render_qp4),
    Source("qp7", QUICK_DIR / "qp7_adaptive.tsv", render_qp7),
    Source("qp8", QUICK_DIR / "qp8_fraud.tsv", render_qp8),
    Source("mega", MEGA_DIR / "cells.tsv", render_mega, min_rows=50),
]


def should_render(path: Path, rows: int, is_final: bool) -> bool:
    if rows <= 0:
        return False
    previous = last_rendered_rows.get(path)
    if previous is None:
        return True
    delta = rows - previous
    return delta > 0 if is_final else delta >= ROW_DELTA_THRESHOLD


def poll(is_final: bool = False) -> bool:
    global last_activity
    any_rendered = False
    for source in SOURCES:
        if not source.path.exists():
            continue
        rows = count_rows(source.path)
        if rows < source.min_rows or not should_render(source.path, rows, is_final):
            continue
        df = read_tsv(source.path)
        if df is None:
            continue
        try:
            rendered = source.renderer(df, len(df))
            if rendered:
                last_rendered_rows[source.path] = rows
                any_rendered = True
                names = ", ".join(name for name, _ in rendered)
                print(f"[{now_iso()}] Rendered {source.name} ({len(df)} rows): {names}", flush=True)
        except Exception as exc:
            print(f"[{now_iso()}] Error rendering {source.path}: {exc}", flush=True)
            traceback.print_exc()
    if any_rendered:
        update_index()
        last_activity = time.time()
        print(f"[{now_iso()}] Updated {INDEX_PATH}", flush=True)
    return any_rendered


def main() -> int:
    print(f"[{now_iso()}] Starting figure monitor; output dir: {FIG_DIR}", flush=True)
    poll()

    while True:
        elapsed = time.time() - start_time
        quiet = time.time() - last_activity
        if elapsed >= MAX_DURATION_SECONDS:
            print(f"[{now_iso()}] Max duration reached; doing final render pass.", flush=True)
            poll(is_final=True)
            break
        if quiet >= QUIET_SECONDS:
            print(f"[{now_iso()}] All sources quiet for {math.floor(quiet)}s; doing final render pass.", flush=True)
            poll(is_final=True)
            break
        time.sleep(POLL_SECONDS)
        poll()

    print(f"[{now_iso()}] Monitoring complete.", flush=True)
    print("=== SUMMARY ===", flush=True)
    if figures_generated:
        for name in sorted(figures_generated):
            desc, rows, ts = figures_generated[name]
            print(f"{name}.png/pdf | rows={rows} | {ts} | {desc}", flush=True)
    else:
        print("No figures rendered.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
