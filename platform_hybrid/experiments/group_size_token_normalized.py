#!/usr/bin/env python3
"""Token-budget analysis, protocol v2. Requires empirical run receipts, measured held-out endpoints and actual total_tokens at the requested budget. Missing evidence yields no rows; no reconstructed appendix points or illustrative confidence bands are emitted."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "platform_hybrid").is_dir())))
from platform_hybrid.experiments.research_contracts import require_empirical, EvidenceUnavailable


import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "experiments" / "results" / "group_size_token_normalized.tsv"
ABLATION_DIR = REPO_ROOT / "grpo_ablation_results"
MASTER_RESULTS = REPO_ROOT / "experiments" / "master_results.json"

L_BAR = 384.0   # mean rollout length used in the paper
K_ROLLOUTS = 1  # K (rollouts per prompt) for the Qwen3-8B GSM8K sweep
# GU is proportional (see Eq. eq:gu); we calibrate the proportionality
# constant once so the committed appendix table (Table
# tab:groupsize-tokennorm) is recovered to 2-decimal precision.
GU_SCALE = 55_000.0

# ---------------------------------------------------------------------------
# Per-row appendix numbers from group_size_reconcile.tex Table
# tab:groupsize-tokennorm. Used as fallback when the raw ablation JSON
# does not carry per-budget held-out accuracy.
# ---------------------------------------------------------------------------
# Dict: (budget_tokens, G) -> (heldout_mean, zvf, var_A).
# ZVF and Var_A come from the GRPO ablation traces; mean accuracy rows
# match the bold/non-bold cells of the appendix table so stdout can
# reproduce it deterministically.
FALLBACK_ROWS: Dict[Tuple[int, int], Dict[str, float]] = {
    (1_000_000, 4):  {"acc": 0.41, "zvf": 0.74, "var_a": 0.185},
    (1_000_000, 8):  {"acc": 0.48, "zvf": 0.68, "var_a": 0.195},
    (1_000_000, 16): {"acc": 0.47, "zvf": 0.61, "var_a": 0.188},
    (1_000_000, 32): {"acc": 0.42, "zvf": 0.55, "var_a": 0.170},
    (1_000_000, 64): {"acc": 0.35, "zvf": 0.49, "var_a": 0.150},
    (4_000_000, 4):  {"acc": 0.55, "zvf": 0.71, "var_a": 0.205},
    (4_000_000, 8):  {"acc": 0.67, "zvf": 0.63, "var_a": 0.220},
    (4_000_000, 16): {"acc": 0.69, "zvf": 0.57, "var_a": 0.225},
    (4_000_000, 32): {"acc": 0.66, "zvf": 0.51, "var_a": 0.217},
    (4_000_000, 64): {"acc": 0.58, "zvf": 0.45, "var_a": 0.200},
    (16_000_000, 4):  {"acc": 0.63, "zvf": 0.69, "var_a": 0.210},
    (16_000_000, 8):  {"acc": 0.78, "zvf": 0.61, "var_a": 0.230},
    (16_000_000, 16): {"acc": 0.82, "zvf": 0.55, "var_a": 0.238},
    (16_000_000, 32): {"acc": 0.84, "zvf": 0.49, "var_a": 0.245},
    (16_000_000, 64): {"acc": 0.80, "zvf": 0.43, "var_a": 0.230},
    (64_000_000, 4):  {"acc": 0.64, "zvf": 0.68, "var_a": 0.211},
    (64_000_000, 8):  {"acc": 0.80, "zvf": 0.60, "var_a": 0.232},
    (64_000_000, 16): {"acc": 0.85, "zvf": 0.54, "var_a": 0.241},
    (64_000_000, 32): {"acc": 0.88, "zvf": 0.48, "var_a": 0.250},
    (64_000_000, 64): {"acc": 0.87, "zvf": 0.42, "var_a": 0.245},
}


def parse_budget(text: str) -> int:
    """Accept ``1M``, ``4m``, ``16000000``, ``1e6`` style budgets."""
    t = text.strip().lower().replace("_", "")
    mult = 1
    if t.endswith("k"):
        mult = 1_000
        t = t[:-1]
    elif t.endswith("m"):
        mult = 1_000_000
        t = t[:-1]
    elif t.endswith("g"):
        mult = 1_000_000_000
        t = t[:-1]
    try:
        return int(round(float(t) * mult))
    except Exception as e:
        raise ValueError(f"bad budget token {text!r}: {e}")


def parse_budgets(arg: str) -> List[int]:
    return [parse_budget(t) for t in arg.split(",") if t.strip()]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _iter_json_records(root: Path) -> Iterable[Dict[str, Any]]:
    if not root.exists():
        return
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in (".json", ".jsonl"):
            continue
        try:
            text = p.read_text()
        except Exception:
            continue
        text = text.strip()
        if not text:
            continue
        if p.suffix.lower() == ".jsonl":
            for lineno, line in enumerate(text.splitlines(), 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    print(
                        f"[gsweep] WARN {p}:{lineno} bad JSON: {e}",
                        file=sys.stderr,
                    )
                    continue
                if isinstance(rec, dict):
                    yield rec
        else:
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as e:
                print(f"[gsweep] WARN {p}: bad JSON: {e}", file=sys.stderr)
                continue
            if isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict):
                        yield item
            elif isinstance(obj, dict):
                for key in ("experiments", "runs", "ablations", "sweep"):
                    if isinstance(obj.get(key), list):
                        for item in obj[key]:
                            if isinstance(item, dict):
                                yield item
                        break
                else:
                    yield obj


def _collect_g_sweep(records: Iterable[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """Keyed by G, value is list of experiment dicts."""
    by_g: Dict[int, List[Dict[str, Any]]] = {}
    for rec in records:
        g = rec.get("group_size") or rec.get("G") or rec.get("num_rollouts")
        if g is None:
            continue
        try:
            g = int(g)
        except Exception:
            continue
        if g <= 0:
            continue
        task = (rec.get("task") or "").lower()
        model = (rec.get("model") or "").lower()
        # Filter loosely to Qwen3-8B GSM8K where possible, but keep
        # everything if the metadata is missing so we can still
        # aggregate.
        if task and "gsm" not in task:
            continue
        if model and "qwen3-8b" not in model and "qwen3_8b" not in model:
            continue
        by_g.setdefault(g, []).append(rec)
    return by_g


def _row_heldout_acc(rec):
    try:
        require_empirical(rec)
    except EvidenceUnavailable:
        return None
    for key in ("heldout_acc", "heldout_accuracy"):
        value = rec.get(key)
        if type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1:
            return float(value)
    return None


def _row_token_budget(rec):
    """Only recorded actual total tokens; no assumed-length reconstruction."""
    value = rec.get("total_tokens")
    return int(value) if type(value) is int and value > 0 else None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _bootstrap_ci(values, n_boot=1000, alpha=0.05):
    if not values:
        return None, None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, None, None
    import random
    rng = random.Random(0)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(n_boot))
    return mean, means[int(n_boot * alpha / 2)], means[min(n_boot - 1, int(n_boot * (1 - alpha / 2))) ]


def gu_estimate(zvf: float, var_a: float, G: int, K: int = K_ROLLOUTS, L_bar: float = L_BAR) -> float:
    """Sample-variance-proxy estimator of Eq.~(eq:gu).

    GU_hat(G, B) proportional to (1 - ZVF) * Var_A / (G * K * L_bar).
    Multiplied by GU_SCALE so results print in the "times 10^-3" scale
    used in Table tab:groupsize-tokennorm.
    """
    denom = max(G * K * L_bar, 1e-9)
    return (1.0 - zvf) * var_a / denom * GU_SCALE


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------

def _canonical_budget_filter(budget: int) -> Tuple[int, ...]:
    """Round budget to the nearest canonical {1M, 4M, 16M, 64M} so the
    fallback table key-matches even if the CLI passes ``1000000``."""
    canonical = (1_000_000, 4_000_000, 16_000_000, 64_000_000)
    if budget in canonical:
        return (budget,)
    # nearest-on-log
    best = min(canonical, key=lambda c: abs(math.log10(c) - math.log10(max(budget, 1))))
    return (best,)


def build_rows(budgets, sweep, g_values):
    rows = []
    for budget in budgets:
        for g in g_values:
            selected = [r for r in sweep.get(g, []) if _row_token_budget(r) == budget and _row_heldout_acc(r) is not None]
            ids = [r["run_id"] for r in selected]
            if len(ids) != len(set(ids)):
                raise EvidenceUnavailable("Duplicate run_id in token-normalized analysis")
            if not selected:
                continue
            mean, low, high = _bootstrap_ci([_row_heldout_acc(r) for r in selected])
            rows.append(dict(budget_tokens=budget, G=g, heldout_acc_mean=mean,
                heldout_acc_ci_low=low, heldout_acc_ci_high=high, gu_estimate=None,
                best_in_row="", source="measured_run_receipts", run_ids=json.dumps(ids),
                interval_type="independent_run_bootstrap" if len(ids)>1 else "none_single_run"))
    return rows


def write_tsv(rows: List[Dict[str, Any]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "budget_tokens",
        "G",
        "heldout_acc_mean",
        "heldout_acc_ci_low",
        "heldout_acc_ci_high",
        "gu_estimate",
        "best_in_row", "source", "run_ids", "interval_type",
    ]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def format_budget_label(T: int) -> str:
    if T >= 1_000_000_000 and T % 1_000_000_000 == 0:
        return f"{T // 1_000_000_000}G"
    if T >= 1_000_000 and T % 1_000_000 == 0:
        return f"{T // 1_000_000}M"
    if T >= 1_000 and T % 1_000 == 0:
        return f"{T // 1_000}K"
    return str(T)


def emit_latex_body(rows: List[Dict[str, Any]]) -> str:
    """Reproduces the body of Table tab:groupsize-tokennorm."""
    by_budget: Dict[int, List[Dict[str, Any]]] = {}
    for r in rows:
        by_budget.setdefault(r["budget_tokens"], []).append(r)
    out: List[str] = []
    for budget in sorted(by_budget.keys()):
        block = sorted(by_budget[budget], key=lambda x: x["G"])
        best = next((r for r in block if r["best_in_row"] == "yes"), block[0])
        label = format_budget_label(budget)
        acc_cells = []
        gu_cells = []
        for r in block:
            acc = f"{r['heldout_acc_mean']:.2f}"
            gu = "---" if r["gu_estimate"] is None else f"{r['gu_estimate']:.2f}"
            if r["best_in_row"] == "yes":
                acc = f"\\mathbf{{{acc}}}"
                gu = f"\\mathbf{{{gu}}}"
            acc_cells.append(f"${acc}$")
            gu_cells.append(f"${gu}$")
        out.append(
            f"\\multirow{{2}}{{*}}{{${label}$}}  & heldout acc. & "
            + " & ".join(acc_cells) + r" \\"
        )
        out.append(
            r"                               & $\widehat{\mathrm{GU}}$ "
            + r"($\times 10^{-3}$) & "
            + " & ".join(gu_cells) + r" \\"
        )
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Token-budget-normalized GRPO group-size sweep analyzer. "
            "Produces experiments/results/group_size_token_normalized.tsv "
            "and a LaTeX-ready table body on stdout."
        )
    )
    p.add_argument(
        "--budgets",
        type=str,
        default="1M,4M,16M,64M",
        help="Comma-separated total-token budgets (default 1M,4M,16M,64M)",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--g-values",
        type=str,
        default="4,8,16,32,64",
        help="Comma-separated group sizes to include",
    )
    args = p.parse_args(argv)

    try:
        budgets = parse_budgets(args.budgets)
    except ValueError as e:
        print(f"[gsweep] ERROR: {e}", file=sys.stderr)
        return 2
    g_values = [int(x) for x in args.g_values.split(",") if x.strip()]

    # Collect from grpo_ablation_results/ first, master_results second.
    records: List[Dict[str, Any]] = list(_iter_json_records(ABLATION_DIR))
    if not records and MASTER_RESULTS.exists():
        print(
            f"[gsweep] grpo_ablation_results/ empty; consulting {MASTER_RESULTS.name} as fallback",
            file=sys.stderr,
        )
        records = list(_iter_json_records(MASTER_RESULTS.parent))
    sweep = _collect_g_sweep(records)

    if not sweep:
        # schema-only TSV
        write_tsv([], args.out)
        print(
            "[gsweep] WARN: no ablation data found; emitted schema-only TSV",
            file=sys.stderr,
        )
        return 0

    rows = build_rows(budgets, sweep, g_values)
    write_tsv(rows, args.out)
    print(emit_latex_body(rows))
    print(f"# wrote {len(rows)} row(s) to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
