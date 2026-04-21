#!/usr/bin/env python3
"""Retrospective early-prefix validation for ZVF/GU diagnostics.

The thesis argument being tested is narrow: if the first few logged
optimization steps already show high zero-variance fraction (ZVF) and no
reward, the run is likely to waste the remaining compute.  This script treats
the first K steps as a frozen "early" observation window and predicts late-run
reward/collapse from existing completed experiment logs.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = [
    ROOT / "experiments" / "master_results.json",
    ROOT / "experiments" / "all_results_consolidated.json",
]
SOURCE_FILES += sorted((ROOT / "experiments" / "tinker-runs" / "results").glob("*.json"))
SOURCE_FILES += sorted((ROOT / "experiments" / "results").glob("*.json"))

OUT_JSON = ROOT / "experiments" / "zvf_predictive_validation_results.json"
OUT_CSV = ROOT / "experiments" / "zvf_predictive_validation_runs.csv"
OUT_MD = ROOT / "experiments" / "zvf_predictive_validation.md"
FIG_PNG = ROOT / "paper" / "figures" / "v2" / "zvf_predictive_validation.png"
FIG_PDF = ROOT / "paper" / "figures" / "v2" / "zvf_predictive_validation.pdf"

EARLY_K = 5
MIN_STEPS = 10
COLLAPSE_THRESHOLD = 0.05
USEFUL_THRESHOLD = 0.25
RULE_ZVF_THRESHOLD = 0.80
RULE_REWARD_THRESHOLD = 0.05


@dataclass
class RunRecord:
    key: str
    source: str
    path: str
    experiment: str
    run_id: str
    model_short: str
    task: str
    seed: str
    rank: str
    lr: str
    batch: str
    group_size: str
    temperature: str
    steps: int
    early_k: int
    early_reward_mean: float
    early_reward_last: float
    early_reward_std: float
    early_zvf_mean: float
    early_zvf_last: float
    early_gu_mean: float
    early_gu_last: float
    late_reward_mean: float
    late_zvf_mean: float
    late_gu_mean: float
    peak_reward: float
    zero_reward_pct: float
    zero_loss_pct: float
    collapse: int
    useful: int
    early_failure_rule: int


def walk_records(obj: Any, path: str = ""):
    if isinstance(obj, dict):
        if isinstance(obj.get("step_log"), list):
            yield path, obj
        for key, value in obj.items():
            next_path = f"{path}.{key}" if path else str(key)
            yield from walk_records(value, next_path)
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            yield from walk_records(value, f"{path}[{index}]")


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def metric_arrays(step_log: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rewards: list[float] = []
    zvfs: list[float] = []
    gus: list[float] = []
    for step in step_log:
        if not isinstance(step, dict):
            continue
        reward = as_float(step.get("reward"))
        zvf = as_float(step.get("zvf"))
        gu = as_float(step.get("gu"))
        if math.isnan(reward):
            continue
        if math.isnan(zvf) and not math.isnan(gu):
            zvf = 1.0 - gu
        if math.isnan(gu) and not math.isnan(zvf):
            gu = 1.0 - zvf
        if math.isnan(zvf) or math.isnan(gu):
            continue
        rewards.append(float(reward))
        zvfs.append(float(zvf))
        gus.append(float(gu))
    return np.array(rewards), np.array(zvfs), np.array(gus)


def record_quality_score(record: RunRecord) -> tuple[int, int, int]:
    has_metadata = sum(
        bool(getattr(record, attr))
        for attr in ["experiment", "model_short", "task", "seed", "rank", "lr"]
    )
    source_preference = 2 if "/tinker-runs/results/" in record.source else 1
    return (record.steps, has_metadata, source_preference)


def make_record(source: Path, subpath: str, rec: dict[str, Any]) -> RunRecord | None:
    step_log = rec.get("step_log")
    if not isinstance(step_log, list):
        return None

    rewards, zvfs, gus = metric_arrays(step_log)
    if len(rewards) < MIN_STEPS:
        return None

    early_len = min(EARLY_K, len(rewards))
    early_rewards = rewards[:early_len]
    early_zvfs = zvfs[:early_len]
    early_gus = gus[:early_len]
    late_rewards = rewards[-10:]
    late_zvfs = zvfs[-10:]
    late_gus = gus[-10:]

    run_id = as_str(rec.get("run_id"))
    experiment = as_str(rec.get("experiment") or rec.get("tag") or rec.get("sweep") or subpath)
    model_short = as_str(rec.get("model_short") or rec.get("model") or "unknown")
    task = as_str(rec.get("task") or ("gsm8k" if "gsm8k" in experiment else "unknown"))
    seed = as_str(rec.get("seed"))
    rank = as_str(rec.get("rank"))
    lr = as_str(rec.get("lr"))
    batch = as_str(rec.get("batch"))
    group_size = as_str(rec.get("group_size") or rec.get("group"))
    temperature = as_str(rec.get("temperature"))
    key = run_id or "|".join(
        [
            experiment,
            model_short,
            task,
            seed,
            rank,
            lr,
            batch,
            group_size,
            temperature,
            str(len(rewards)),
        ]
    )

    late_mean = float(np.mean(late_rewards))
    early_reward_mean = float(np.mean(early_rewards))
    early_zvf_mean = float(np.mean(early_zvfs))
    collapse = int(late_mean <= COLLAPSE_THRESHOLD)
    useful = int(late_mean >= USEFUL_THRESHOLD)
    rule = int(
        early_zvf_mean >= RULE_ZVF_THRESHOLD
        and early_reward_mean <= RULE_REWARD_THRESHOLD
    )

    peak_reward = as_float(rec.get("peak_reward"), as_float(rec.get("peak"), float(np.max(rewards))))
    zero_reward_pct = as_float(rec.get("zero_reward_pct"), float(np.mean(rewards == 0.0) * 100.0))
    zero_loss_pct = as_float(rec.get("zero_loss_pct"))

    return RunRecord(
        key=key,
        source=str(source.relative_to(ROOT)),
        path=subpath,
        experiment=experiment,
        run_id=run_id,
        model_short=model_short,
        task=task,
        seed=seed,
        rank=rank,
        lr=lr,
        batch=batch,
        group_size=group_size,
        temperature=temperature,
        steps=int(len(rewards)),
        early_k=early_len,
        early_reward_mean=early_reward_mean,
        early_reward_last=float(early_rewards[-1]),
        early_reward_std=float(np.std(early_rewards)),
        early_zvf_mean=early_zvf_mean,
        early_zvf_last=float(early_zvfs[-1]),
        early_gu_mean=float(np.mean(early_gus)),
        early_gu_last=float(early_gus[-1]),
        late_reward_mean=late_mean,
        late_zvf_mean=float(np.mean(late_zvfs)),
        late_gu_mean=float(np.mean(late_gus)),
        peak_reward=float(peak_reward),
        zero_reward_pct=float(zero_reward_pct),
        zero_loss_pct=float(zero_loss_pct),
        collapse=collapse,
        useful=useful,
        early_failure_rule=rule,
    )


def load_runs() -> tuple[int, list[RunRecord]]:
    raw_count = 0
    by_key: dict[str, RunRecord] = {}
    for source in SOURCE_FILES:
        if not source.exists():
            continue
        try:
            data = json.loads(source.read_text())
        except json.JSONDecodeError:
            continue
        for subpath, rec in walk_records(data):
            run = make_record(source, subpath, rec)
            if run is None:
                continue
            raw_count += 1
            existing = by_key.get(run.key)
            if existing is None or record_quality_score(run) > record_quality_score(existing):
                by_key[run.key] = run
    runs = sorted(by_key.values(), key=lambda r: (r.task, r.model_short, r.experiment, r.run_id))
    return raw_count, runs


def rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return math.nan
    return pearson(rank_average(x), rank_average(y))


def auc_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return math.nan
    wins = 0.0
    total = float(len(positives) * len(negatives))
    for p in positives:
        wins += float(np.sum(p > negatives))
        wins += 0.5 * float(np.sum(p == negatives))
    return wins / total


def bootstrap_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    fn,
    rng: np.random.Generator,
    n_boot: int = 2000,
) -> tuple[float, float]:
    values: list[float] = []
    n = len(labels)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sample_labels = labels[idx]
        sample_scores = scores[idx]
        value = fn(sample_labels, sample_scores)
        if not math.isnan(value):
            values.append(float(value))
    if not values:
        return (math.nan, math.nan)
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))


def correlation_bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    fn,
    rng: np.random.Generator,
    n_boot: int = 2000,
) -> tuple[float, float]:
    values: list[float] = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        value = fn(x[idx], y[idx])
        if not math.isnan(value):
            values.append(float(value))
    if not values:
        return (math.nan, math.nan)
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))


def confusion(labels: np.ndarray, predictions: np.ndarray) -> dict[str, int]:
    return {
        "tp": int(np.sum((labels == 1) & (predictions == 1))),
        "fp": int(np.sum((labels == 0) & (predictions == 1))),
        "tn": int(np.sum((labels == 0) & (predictions == 0))),
        "fn": int(np.sum((labels == 1) & (predictions == 0))),
    }


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    c = confusion(labels, predictions)
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    accuracy = (tp + tn) / len(labels) if len(labels) else math.nan
    precision = tp / (tp + fp) if (tp + fp) else math.nan
    recall = tp / (tp + fn) if (tp + fn) else math.nan
    specificity = tn / (tn + fp) if (tn + fp) else math.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else math.nan
    return {
        **c,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
    }


def loo_r2(y: np.ndarray, x_columns: list[np.ndarray]) -> float:
    n = len(y)
    if n < 3:
        return math.nan
    predictions = np.empty(n, dtype=float)
    x = np.column_stack([np.ones(n), *x_columns])
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        beta = np.linalg.pinv(x[mask]) @ y[mask]
        predictions[i] = float(x[i] @ beta)
    sse = float(np.sum((y - predictions) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    if sst == 0:
        return math.nan
    return 1.0 - sse / sst


def task_breakdown(runs: list[RunRecord]) -> dict[str, dict[str, int | float]]:
    breakdown: dict[str, dict[str, int | float]] = {}
    for run in runs:
        item = breakdown.setdefault(
            run.task or "unknown",
            {"n": 0, "collapse": 0, "useful": 0, "late_reward_mean": 0.0},
        )
        item["n"] = int(item["n"]) + 1
        item["collapse"] = int(item["collapse"]) + run.collapse
        item["useful"] = int(item["useful"]) + run.useful
        item["late_reward_mean"] = float(item["late_reward_mean"]) + run.late_reward_mean
    for item in breakdown.values():
        item["late_reward_mean"] = float(item["late_reward_mean"]) / int(item["n"])
    return breakdown


def finite(value: float | int) -> float | int | None:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def make_plot(runs: list[RunRecord], metrics: dict[str, Any]) -> None:
    FIG_PNG.parent.mkdir(parents=True, exist_ok=True)

    early_reward = np.array([r.early_reward_mean for r in runs])
    early_zvf = np.array([r.early_zvf_mean for r in runs])
    early_gu = np.array([r.early_gu_mean for r in runs])
    late_reward = np.array([r.late_reward_mean for r in runs])
    collapse = np.array([r.collapse for r in runs])
    useful = np.array([r.useful for r in runs])

    colors = np.where(collapse == 1, "#B42318", np.where(useful == 1, "#1F7A4D", "#546179"))

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), constrained_layout=True)

    ax = axes[0]
    ax.scatter(early_zvf, late_reward, c=colors, s=48, edgecolor="white", linewidth=0.8)
    ax.axvline(RULE_ZVF_THRESHOLD, color="#9A3412", linestyle="--", linewidth=1.2)
    ax.axhline(COLLAPSE_THRESHOLD, color="#9A3412", linestyle=":", linewidth=1.2)
    ax.set_xlabel(f"Early ZVF mean, first {EARLY_K} steps")
    ax.set_ylabel("Late reward mean, last 10 steps")
    ax.set_title("ZVF separates collapsed runs")
    ax.grid(alpha=0.18)

    ax = axes[1]
    ax.scatter(early_gu, late_reward, c=colors, s=48, edgecolor="white", linewidth=0.8)
    ax.axhline(USEFUL_THRESHOLD, color="#166534", linestyle=":", linewidth=1.2)
    ax.set_xlabel(f"Early GU mean, first {EARLY_K} steps")
    ax.set_ylabel("Late reward mean, last 10 steps")
    ax.set_title("GU tracks usable signal")
    ax.grid(alpha=0.18)

    ax = axes[2]
    cm = metrics["early_failure_rule"]["confusion"]
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    im = ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set_xticks([0, 1], labels=["Predict OK", "Predict collapse"])
    ax.set_yticks([0, 1], labels=["Actual OK", "Actual collapse"])
    ax.set_title("Fixed early-failure rule")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="#111827")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.tick_params(axis="x", labelrotation=20)

    fig.suptitle(
        "Pseudo-prospective ZVF validation on existing training logs",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(FIG_PNG, dpi=220)
    fig.savefig(FIG_PDF)
    plt.close(fig)


def write_csv(runs: list[RunRecord]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(run) for run in runs]
    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(raw_count: int, runs: list[RunRecord], metrics: dict[str, Any]) -> None:
    corr = metrics["correlations"]
    rule = metrics["early_failure_rule"]
    aucs = metrics["collapse_auc"]
    regression = metrics["loo_regression"]
    breakdown = metrics["task_breakdown"]

    lines = [
        "# ZVF Predictive Validation",
        "",
        "This is a pseudo-prospective validation over existing completed logs. The predictor only uses the first five logged optimization steps; outcomes are computed from the last ten logged steps.",
        "",
        "## Protocol",
        "",
        f"- Raw candidate records found: {raw_count}",
        f"- Deduplicated independent runs: {len(runs)}",
        f"- Minimum usable run length: {MIN_STEPS} steps",
        f"- Early window: first {EARLY_K} steps",
        f"- Late outcome window: last 10 steps",
        f"- Collapse label: late reward mean <= {COLLAPSE_THRESHOLD}",
        f"- Useful label: late reward mean >= {USEFUL_THRESHOLD}",
        f"- Fixed early-failure rule: early ZVF >= {RULE_ZVF_THRESHOLD} and early reward <= {RULE_REWARD_THRESHOLD}",
        "",
        "## Primary Results",
        "",
        f"- Collapsed runs: {metrics['n_collapse']} / {len(runs)}",
        f"- Useful runs: {metrics['n_useful']} / {len(runs)}",
        f"- Early ZVF vs late reward Spearman: {fmt(corr['early_zvf_vs_late_reward']['spearman'])} [{fmt(corr['early_zvf_vs_late_reward']['spearman_ci'][0])}, {fmt(corr['early_zvf_vs_late_reward']['spearman_ci'][1])}]",
        f"- Early GU vs late reward Spearman: {fmt(corr['early_gu_vs_late_reward']['spearman'])} [{fmt(corr['early_gu_vs_late_reward']['spearman_ci'][0])}, {fmt(corr['early_gu_vs_late_reward']['spearman_ci'][1])}]",
        f"- Collapse AUC using early ZVF: {fmt(aucs['early_zvf']['auc'])} [{fmt(aucs['early_zvf']['ci'][0])}, {fmt(aucs['early_zvf']['ci'][1])}]",
        f"- Collapse AUC using early reward only: {fmt(aucs['early_reward_low']['auc'])} [{fmt(aucs['early_reward_low']['ci'][0])}, {fmt(aucs['early_reward_low']['ci'][1])}]",
        f"- Collapse AUC using ZVF-minus-reward composite: {fmt(aucs['zvf_minus_reward']['auc'])} [{fmt(aucs['zvf_minus_reward']['ci'][0])}, {fmt(aucs['zvf_minus_reward']['ci'][1])}]",
        f"- Fixed rule precision/recall/F1: {fmt(rule['precision'])} / {fmt(rule['recall'])} / {fmt(rule['f1'])}",
        f"- Fixed rule confusion: TP={rule['confusion']['tp']}, FP={rule['confusion']['fp']}, TN={rule['confusion']['tn']}, FN={rule['confusion']['fn']}",
        f"- Leave-one-run-out R^2, early reward only: {fmt(regression['early_reward_only_r2'])}",
        f"- Leave-one-run-out R^2, early reward + early ZVF: {fmt(regression['early_reward_plus_zvf_r2'])}",
        "",
        "## Task Breakdown",
        "",
        "| Task | Runs | Collapsed | Useful | Mean late reward |",
        "|---|---:|---:|---:|---:|",
    ]
    for task, item in sorted(breakdown.items()):
        lines.append(
            f"| {task} | {item['n']} | {item['collapse']} | {item['useful']} | {fmt(float(item['late_reward_mean']))} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "The validation supports the narrow triage claim if the fixed first-five-step ZVF rule separates collapsed runs with high recall and few false positives. It does not by itself prove a universal causal law: the data are still retrospective, small, and drawn from the available completed experiments rather than from a newly randomized prospective campaign.",
        "",
        f"Figure outputs: `{FIG_PNG.relative_to(ROOT)}` and `{FIG_PDF.relative_to(ROOT)}`.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    raw_count, runs = load_runs()
    if not runs:
        raise SystemExit("No usable ZVF/GU step logs found.")

    early_reward = np.array([r.early_reward_mean for r in runs])
    early_zvf = np.array([r.early_zvf_mean for r in runs])
    early_gu = np.array([r.early_gu_mean for r in runs])
    late_reward = np.array([r.late_reward_mean for r in runs])
    collapse = np.array([r.collapse for r in runs], dtype=int)
    useful = np.array([r.useful for r in runs], dtype=int)
    failure_rule = np.array([r.early_failure_rule for r in runs], dtype=int)
    rng = np.random.default_rng(20260422)

    corr_zvf_s = spearman(early_zvf, late_reward)
    corr_gu_s = spearman(early_gu, late_reward)
    corr_reward_s = spearman(early_reward, late_reward)

    metrics: dict[str, Any] = {
        "protocol": {
            "raw_candidate_records": raw_count,
            "deduplicated_runs": len(runs),
            "early_k": EARLY_K,
            "min_steps": MIN_STEPS,
            "late_window_steps": 10,
            "collapse_threshold": COLLAPSE_THRESHOLD,
            "useful_threshold": USEFUL_THRESHOLD,
            "fixed_rule": {
                "early_zvf_mean_gte": RULE_ZVF_THRESHOLD,
                "early_reward_mean_lte": RULE_REWARD_THRESHOLD,
            },
        },
        "n_collapse": int(np.sum(collapse)),
        "n_useful": int(np.sum(useful)),
        "correlations": {
            "early_zvf_vs_late_reward": {
                "pearson": finite(pearson(early_zvf, late_reward)),
                "spearman": finite(corr_zvf_s),
                "spearman_ci": [
                    finite(v)
                    for v in correlation_bootstrap_ci(early_zvf, late_reward, spearman, rng)
                ],
            },
            "early_gu_vs_late_reward": {
                "pearson": finite(pearson(early_gu, late_reward)),
                "spearman": finite(corr_gu_s),
                "spearman_ci": [
                    finite(v)
                    for v in correlation_bootstrap_ci(early_gu, late_reward, spearman, rng)
                ],
            },
            "early_reward_vs_late_reward": {
                "pearson": finite(pearson(early_reward, late_reward)),
                "spearman": finite(corr_reward_s),
                "spearman_ci": [
                    finite(v)
                    for v in correlation_bootstrap_ci(early_reward, late_reward, spearman, rng)
                ],
            },
        },
        "collapse_auc": {},
        "early_failure_rule": {
            **classification_metrics(collapse, failure_rule),
            "confusion": confusion(collapse, failure_rule),
        },
        "loo_regression": {
            "early_reward_only_r2": finite(loo_r2(late_reward, [early_reward])),
            "early_reward_plus_zvf_r2": finite(loo_r2(late_reward, [early_reward, early_zvf])),
        },
        "task_breakdown": task_breakdown(runs),
    }

    auc_scores = {
        "early_zvf": early_zvf,
        "early_gu_low": -early_gu,
        "early_reward_low": -early_reward,
        "zvf_minus_reward": early_zvf - early_reward,
    }
    for name, score in auc_scores.items():
        auc = auc_binary(collapse, score)
        ci = bootstrap_ci(collapse, score, auc_binary, rng)
        metrics["collapse_auc"][name] = {
            "auc": finite(auc),
            "ci": [finite(ci[0]), finite(ci[1])],
        }

    write_csv(runs)
    make_plot(runs, metrics)
    write_markdown(raw_count, runs, metrics)

    OUT_JSON.write_text(
        json.dumps(
            {
                "metrics": metrics,
                "runs": [asdict(run) for run in runs],
                "outputs": {
                    "csv": str(OUT_CSV.relative_to(ROOT)),
                    "markdown": str(OUT_MD.relative_to(ROOT)),
                    "figure_png": str(FIG_PNG.relative_to(ROOT)),
                    "figure_pdf": str(FIG_PDF.relative_to(ROOT)),
                },
            },
            indent=2,
        )
        + "\n"
    )

    print(f"Raw candidate records: {raw_count}")
    print(f"Deduplicated runs: {len(runs)}")
    print(f"Collapsed runs: {metrics['n_collapse']} / {len(runs)}")
    print(f"Useful runs: {metrics['n_useful']} / {len(runs)}")
    print(
        "Early ZVF Spearman vs late reward:",
        fmt(metrics["correlations"]["early_zvf_vs_late_reward"]["spearman"]),
    )
    print(
        "Early GU Spearman vs late reward:",
        fmt(metrics["correlations"]["early_gu_vs_late_reward"]["spearman"]),
    )
    print(
        "Collapse AUC, early ZVF:",
        fmt(metrics["collapse_auc"]["early_zvf"]["auc"]),
    )
    print(
        "Collapse AUC, early reward only:",
        fmt(metrics["collapse_auc"]["early_reward_low"]["auc"]),
    )
    print(
        "Collapse AUC, ZVF-minus-reward:",
        fmt(metrics["collapse_auc"]["zvf_minus_reward"]["auc"]),
    )
    rule = metrics["early_failure_rule"]
    print(
        "Fixed rule P/R/F1:",
        fmt(rule["precision"]),
        fmt(rule["recall"]),
        fmt(rule["f1"]),
    )
    print("Fixed rule confusion:", rule["confusion"])
    print("Wrote:", OUT_JSON.relative_to(ROOT))
    print("Wrote:", OUT_MD.relative_to(ROOT))
    print("Wrote:", FIG_PDF.relative_to(ROOT))


if __name__ == "__main__":
    main()
