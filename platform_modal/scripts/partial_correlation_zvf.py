#!/usr/bin/env python3
"""Partial-correlation ablation for the ZVF diagnostic.

Computes partial corr(ZVF_t*, final_reward) while controlling for
  * batch mean reward
  * policy entropy
  * advantage variance
  * KL drift
  * group size G
  * baseline (SFT) accuracy

Inputs:  JSONL / JSON training logs under platform_hybrid/experiments/results/
Outputs: platform_hybrid/experiments/results/zvf_partial_correlations.tsv (+ stdout tex table)

For each configuration the script reports:
  - r_partial           : Pearson (or Spearman, --method spearman) partial
                          correlation after OLS-residualizing both x and y
                          on the named controls.
  - 95% bootstrap CI    : percentile CI; the bootstrap refits the residualizations
                          on each resampled row set (NOT on residuals-as-fixed;
                          that would break exchangeability).
  - p_partial           : two-sided p-value via the t-statistic
                          t = r * sqrt((n - 2 - k_controls) / (1 - r^2))
                          with df = n - 2 - k_controls.
  - delta_r2            : INCREMENTAL R^2 from adding ZVF to an OLS regression
                          of final_reward on the controls alone. NOT r_partial^2.
                          Negative values are meaningful: a confounder that does
                          not improve the fit (or worsens it in finite samples)
                          yields dr2 < 0; the rendered TSV stamps the row with
                          a "[negative dr2]" note so the cell is not read as
                          a bug.
  - p_bh                : Benjamini-Hochberg-adjusted p-value across the
                          per-confounder conditional rows. The joint row is
                          reported separately and not part of the BH family.

Confidence intervals:
  - Pearson: 95% percentile CI from BOOTSTRAP_B=1000 refit bootstrap
    resamples. This is the basic percentile method, NOT bias-corrected
    and accelerated (BCa). A reviewer who needs tighter coverage
    should re-derive with a BCa bootstrap.
  - Spearman: the CI bounds are NaN. The Spearman branch returns a
    permutation p-value only; computing a Spearman CI requires either
    a BCa or Fisher-z transform of the partial-Spearman statistic,
    neither of which is implemented in this revision.

Important:
  - If fewer than 10 ZVF-compatible rows survive the reference-window filter,
    the row reports NA / "insufficient data" and a warning is printed to stderr.
    The script does NOT write a schema-stub TSV: a stub TSV with all-NA rows
    has been confused for a real result in earlier revisions. Callers that
    want an empty placeholder can pass --allow-empty.
  - The K=1 degenerate group returns ZVF=1.0 by convention; the diagnostic is
    meaningless in that regime and the row is flagged accordingly.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from scipy import stats as _scipy_stats  # type: ignore
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _scipy_stats = None
    _HAS_SCIPY = False

try:
    import pingouin as _pg  # type: ignore
    _HAS_PINGOUIN = True
except Exception:  # pragma: no cover
    _pg = None
    _HAS_PINGOUIN = False

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_GLOBS = [
    "platform_hybrid/experiments/results/*.jsonl",
    "platform_hybrid/experiments/results/**/*.jsonl",
    "platform_hybrid/experiments/results/*.json",
    "platform_hybrid/experiments/results/**/*.json",
    # NOTE: *.tsv globs were removed deliberately. They previously ingested
    # synthetic/dry-run summary TSVs (e.g. variance_mitigation.tsv) as if they
    # were per-group rollout logs, producing a populated partial-correlation
    # table with the WRONG (positive) sign that contradicted the paper's
    # withdrawn negative claim. Per-group rollout logs are not bundled in the
    # release, so this script now finds no compatible rows and emits a
    # header-only stub (run with --allow-empty), matching the appendix's
    # "withdrawn pending per-step log artifact" statement.
    "experiments/collab-results/*.jsonl",
]
OUT_TSV = REPO_ROOT / "experiments" / "results" / "zvf_partial_correlations.tsv"
REFERENCE_WINDOW = (25, 40)  # t* window for ZVF evaluation
EPS = 1e-6
MIN_OBS = 10
BOOTSTRAP_B = 1000

FIELDS = [
    "step",
    "zvf",
    "batch_reward_mean",
    "entropy",
    "advantage_variance",
    "kl_drift",
    "final_reward",
    "group_size",
    "baseline_accuracy",
    "run_id",
    "model",
    "framework",
    "seed",
]


def _iter_logs() -> Iterable[Path]:
    seen: set[str] = set()
    for pattern in LOG_GLOBS:
        for p in glob.glob(str(REPO_ROOT / pattern), recursive=True):
            if p not in seen:
                seen.add(p)
                yield Path(p)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        if path.suffix == ".json":
            blob = json.loads(path.read_text())
            if isinstance(blob, list):
                # Guard: top-level list may contain non-dict scalars
                # (e.g. "[1, 2, 3]"); only dicts are valid record shapes.
                rows.extend(item for item in blob if isinstance(item, dict))
            elif isinstance(blob, dict):
                for k in ("runs", "experiments", "records"):
                    if k in blob and isinstance(blob[k], list):
                        rows.extend(item for item in blob[k] if isinstance(item, dict))
                        break
        elif path.suffix == ".tsv":
            lines = path.read_text().splitlines()
            if len(lines) < 2:
                return rows
            # Skip '#'-prefixed comment lines (mirroring the JSONL path)
            # and use the first non-comment line as the header.
            hdr: list[str] | None = None
            for line in lines:
                if not line.strip().startswith("#"):
                    hdr = line.split("\t")
                    break
            if hdr is None:
                return rows
            data_lines = lines[lines.index([l for l in lines if not l.strip().startswith("#")][0]) + 1:]
            for line in data_lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                vals = line.split("\t")
                rows.append({h: v for h, v in zip(hdr, vals)})
        else:
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception as exc:
        print(f"[warn] could not parse {path}: {exc}", file=sys.stderr)
    return rows


def _extract_zvf_row(rec: dict[str, Any]) -> dict[str, Any] | None:
    """Project a heterogeneous log record into our canonical schema."""
    def g(*keys: str, default: Any = np.nan) -> Any:
        for k in keys:
            if k in rec and rec[k] is not None:
                return rec[k]
        return default

    step = g("step", "global_step", "iter")
    try:
        step = float(step)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(step):
        return None

    rewards = g("group_rewards", "rollout_rewards", default=None)
    if rewards is not None:
        arr = np.asarray(rewards, dtype=float)
        if arr.ndim >= 2 and arr.shape[1] >= 2:
            var_per_group = arr.var(axis=-1, ddof=1)
            zvf = float((var_per_group <= EPS).mean())
        else:
            # K=1 degenerate: flag in the row but compute ZVF per the spec
            zvf = 1.0 if arr.size else float("nan")
    else:
        # Accept either the canonical "zvf" key or the per-step saturation
        # tracker's "zero_variance_frac" output (the name emitted by
        # experiments/10x_structural_ceiling/group_saturation_diagnostic.py).
        zvf = float(g("zvf", "zero_variance_fraction", "zero_variance_frac", default=np.nan))

    def _maybe_float(v: Any) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    return {
        "step": int(step),
        "zvf": zvf,
        "batch_reward_mean": _maybe_float(g("batch_reward_mean", "reward_mean", "env/all/correct", "reward_mean")),
        "entropy": _maybe_float(g("entropy", "policy_entropy", default=np.nan)),
        "advantage_variance": _maybe_float(g("advantage_variance", "adv_var", default=np.nan)),
        "kl_drift": _maybe_float(g("kl_drift", "kl_to_ref", "approx_kl", default=np.nan)),
        "final_reward": _maybe_float(g("final_reward", "heldout_accuracy", "heldout_acc", default=np.nan)),
        "group_size": int(g("group_size", "G", "K", default=0) or 0),
        "baseline_accuracy": _maybe_float(g("baseline_accuracy", default=np.nan)),
        "run_id": str(g("run_id", "name", default="")),
        "model": str(g("model", default="")),
        "framework": str(g("framework", "method", default="")),
        "seed": int(float(g("seed", default=0) or 0)),
    }


def _ols_residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Return y - X @ beta, where beta is the OLS coefficient vector."""
    Xa = np.column_stack([np.ones(X.shape[0]), X])
    beta, *_ = np.linalg.lstsq(Xa, y, rcond=None)
    return y - Xa @ beta


def _partial_corr_pearson(df: np.ndarray, x: int, y: int, controls: list[int]) -> tuple[float, float, float, float]:
    """Return (r_partial, ci_low, ci_high, p_value) for a Pearson partial correlation.

    Bootstrap CIs refit the residualizations on each resampled row set so the
    residuals remain orthogonal to the controls in the resampled fit. The point
    estimate uses a single OLS residualization.
    """
    n = df.shape[0]
    if n < MIN_OBS:
        return (float("nan"),) * 4

    Xc = df[:, controls] if controls else np.zeros((n, 0))
    rx = _ols_residualize(df[:, x], Xc)
    ry = _ols_residualize(df[:, y], Xc)
    r = float(np.corrcoef(rx, ry)[0, 1])

    # Bootstrap with refit
    rng = np.random.default_rng(0)
    n = df.shape[0]
    boot = []
    n_degenerate = 0
    for _ in range(BOOTSTRAP_B):
        idx = rng.integers(0, n, size=n)
        xb = df[idx, x]
        yb = df[idx, y]
        if controls:
            Xb = df[np.ix_(idx, controls)]
        else:
            Xb = np.zeros((n, 0))
        # Skip degenerate resamples: a constant column or a
        # rank-deficient design matrix would yield a numerically
        # unstable lstsq, polluting the bootstrap distribution
        # with non-exchangeable noise. Drop and count.
        if Xb.size > 0 and np.linalg.matrix_rank(np.column_stack([np.ones(n), Xb])) < (1 + len(controls)):
            n_degenerate += 1
            continue
        rxb = _ols_residualize(xb, Xb)
        ryb = _ols_residualize(yb, Xb)
        if np.std(rxb) < 1e-12 or np.std(ryb) < 1e-12:
            n_degenerate += 1
            continue
        boot.append(float(np.corrcoef(rxb, ryb)[0, 1]))
    if not boot:
        # All resamples degenerate: the bootstrap cannot produce a
        # valid CI. Return NaN bounds.
        return r, float("nan"), float("nan"), float("nan")
    # Drop NaN entries (defensive) and use np.nanquantile to avoid
    # silent NaN-poisoning of the CI under a handful of bad resamples.
    boot_clean = [b for b in boot if np.isfinite(b)]
    if not boot_clean:
        return r, float("nan"), float("nan"), float("nan")
    lo, hi = np.quantile(boot_clean, [0.025, 0.975])

    # Two-sided p-value via t-statistic with df = n - 2 - k_controls
    k = len(controls)
    df_t = n - 2 - k
    one_minus_r2 = 1.0 - r * r
    # Perfect / near-perfect partial correlation: t = inf, true two-sided p = 0.
    # The t-distribution tail at inf is 0, but the guarded form `1 - r*r > 0`
    # is more numerically robust than `abs(r) < 1.0` (which silently returned
    # NaN for the perfect-correlation case before this revision).
    if df_t > 0 and one_minus_r2 > 1e-12:
        t_stat = r * np.sqrt(df_t / one_minus_r2)
        if _HAS_SCIPY:
            p = float(2.0 * _scipy_stats.t.sf(abs(t_stat), df_t))
        else:
            p = float(2.0 * (1.0 - _normal_cdf(abs(t_stat))))
    elif df_t > 0 and one_minus_r2 <= 1e-12:
        # Perfect (or numerically perfect) partial correlation: p = 0.
        p = 0.0
    else:
        p = float("nan")
    return r, float(lo), float(hi), p


def _normal_cdf(x: float) -> float:
    """Acklam-style normal CDF approximation; used only when scipy is missing."""
    # Abramowitz & Stegun 26.2.17, max error ~7.5e-8
    if x < -8.0:
        return 0.0
    if x > 8.0:
        return 1.0
    a1, a2, a3, a4, a5 = (
        0.319381530,
        -0.356563782,
        1.781477937,
        -1.821255978,
        1.330274429,
    )
    k = 1.0 / (1.0 + 0.2316419 * abs(x))
    w = 1.0 - (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x * x) * (
        a1 * k + a2 * k * k + a3 * k ** 3 + a4 * k ** 4 + a5 * k ** 5
    )
    return w if x >= 0 else 1.0 - w


def _partial_corr_spearman(df: np.ndarray, x: int, y: int, controls: list[int]) -> tuple[float, float, float, float]:
    """Spearman partial correlation: rank-transform then run Pearson on ranks.

    Returns (r_partial, ci_low, ci_high, p_value). p_value is from a permutation
    test on the rank-residualized columns (B=BOOTSTRAP_B permutations) because
    there is no closed-form Spearman-partial p-value.

    The permutation null is constructed by permuting the *post-residualization*
    vectors (rx, ry), not the raw ranks. Permuting the raw ranks and then
    re-residualizing against the controls mixes the permutation null with
    a structural asymmetry in the residualization order; permuting the
    residuals is the cleaner exchangeability argument.
    """
    n = df.shape[0]
    if n < MIN_OBS:
        return (float("nan"),) * 4

    # Rank-transform each column (1..n) with average ties
    def _rank(col: np.ndarray) -> np.ndarray:
        return _scipy_stats.rankdata(col) if _HAS_SCIPY else _rank_fallback(col)

    rx_full = _rank(df[:, x])
    ry_full = _rank(df[:, y])
    Xc = df[:, controls] if controls else np.zeros((n, 0))
    rc_full = np.column_stack([_rank(df[:, c]) for c in controls]) if controls else np.zeros((n, 0))

    rx = _ols_residualize(rx_full, rc_full)
    ry = _ols_residualize(ry_full, rc_full)
    r = float(np.corrcoef(rx, ry)[0, 1])

    # Permutation test for p-value. Permute the *residuals* (not the
    # raw ranks) so the null is the partial-correlation distribution
    # under row-exchangeability.
    rng = np.random.default_rng(0)
    ge = 0
    for _ in range(BOOTSTRAP_B):
        perm = rng.permutation(n)
        rxb = rx[perm]
        ryb = ry[perm]
        rb = float(np.corrcoef(rxb, ryb)[0, 1])
        if abs(rb) >= abs(r):
            ge += 1
    p = (ge + 1) / (BOOTSTRAP_B + 1)

    lo, hi = float("nan"), float("nan")
    return r, lo, hi, p


def _rank_fallback(col: np.ndarray) -> np.ndarray:
    """Average-rank fallback when scipy is unavailable."""
    order = np.argsort(col, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(col) + 1, dtype=float)
    # average ties
    _, inv, counts = np.unique(col, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.zeros_like(counts, dtype=float)
        for i, r in enumerate(ranks):
            sums[inv[i]] += r
        avg = sums / np.maximum(counts, 1)
        ranks = avg[inv]
    return ranks


def _incremental_r2(df: np.ndarray, x: int, y: int, controls: list[int]) -> float:
    """Incremental R^2 from adding column x to a regression of y on the controls.

    Returns R^2_full - R^2_reduced, where:
        reduced: y ~ 1 + controls
        full:    y ~ 1 + controls + x
    Computed directly from R^2 of each OLS fit, not from the squared partial
    correlation. Falls back to NaN if the columns are degenerate.
    """
    if controls:
        Xr = np.column_stack([np.ones(df.shape[0])] + [df[:, c] for c in controls])
    else:
        Xr = np.ones((df.shape[0], 1))
    Xf = np.column_stack([Xr, df[:, x]])
    yv = df[:, y]
    r2r = _r2_ols(yv, Xr)
    r2f = _r2_ols(yv, Xf)
    return float(r2f - r2r)


def _r2_ols(y: np.ndarray, X: np.ndarray) -> float:
    """R^2 of an OLS fit y ~ X. Returns 0.0 for the trivial intercept-only model."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _bh_adjust(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR adjustment; NaN entries are passed through."""
    arr = np.array([np.nan if not np.isfinite(p) else p for p in pvals], dtype=float)
    valid = np.isfinite(arr)
    out = np.full_like(arr, np.nan, dtype=float)
    if valid.sum() == 0:
        return [float("nan")] * len(pvals)
    p_valid = arr[valid]
    m = len(p_valid)
    order = np.argsort(p_valid, kind="mergesort")
    ranked = p_valid[order]
    # BH critical values
    adj = ranked * m / (np.arange(1, m + 1))
    # Enforce monotonicity: walk from largest rank down, propagating the min
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    inv = np.argsort(order)
    out[valid] = adj[inv]
    return [float(x) for x in out]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--ref-window",
        nargs=2,
        type=int,
        metavar=("LO", "HI"),
        default=list(REFERENCE_WINDOW),
        help=(
            "Step window [LO HI] used to filter rows for the partial-corr "
            "ablation. Defaults to (25 40). If the window contains zero "
            "rows the script falls back to ALL rows and stamps a note on "
            "the rendered output. Use --ref-window LO LO to force a single-step "
            "analysis."
        ),
    )
    ap.add_argument(
        "--out",
        default=str(OUT_TSV),
        help="Output TSV path (default: %(default)s)",
    )
    ap.add_argument(
        "--method",
        choices=("pearson", "spearman"),
        default="pearson",
        help="Partial-correlation method (default: pearson). Spearman uses rank-transform + permutation p-value.",
    )
    ap.add_argument(
        "--allow-empty",
        action="store_true",
        help="Write a header-only TSV when no ZVF-compatible rows are found (default: error out).",
    )
    args = ap.parse_args(argv)

    rows: list[dict[str, Any]] = []
    for path in _iter_logs():
        for rec in _load_jsonl(path):
            norm = _extract_zvf_row(rec)
            if norm is not None and np.isfinite(norm["zvf"]):
                rows.append(norm)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "controlling_for",
        "r_partial",
        "ci_low",
        "ci_high",
        "n_obs",
        "p_partial",
        "p_bh",
        "delta_r2",
        "note",
    ]

    if not rows:
        msg = (
            f"no ZVF-compatible log rows found under {REPO_ROOT}; refusing to "
            f"write a schema-stub TSV that has been confused for a real result "
            f"in earlier revisions. Pass --allow-empty to write a header-only file."
        )
        if args.allow_empty:
            out_path.write_text("\t".join(header) + "\n")
            print(f"[warn] {msg}", file=sys.stderr)
            print(f"[warn] wrote header-only TSV to {out_path}", file=sys.stderr)
            return 0
        print(f"[error] {msg}", file=sys.stderr)
        return 2

    # Filter to reference window. If LO > HI, that's a user error
    # (the filter would silently return zero rows and then fall back
    # to ALL rows, masking the mistake). Error out instead.
    wlo, whi = args.ref_window
    if wlo > whi:
        ap.error(f"--ref-window: LO ({wlo}) > HI ({whi})")
    keep = [r for r in rows if wlo <= r["step"] <= whi]
    fallback_to_all = False
    if not keep:
        print(
            f"[warn] no rows in reference window [{wlo},{whi}]; falling back to all {len(rows)} rows. "
            f"The BH family and p-values below are computed on the full sample, not the reference window.",
            file=sys.stderr,
        )
        keep = rows
        fallback_to_all = True

    arr_full = np.array(
        [[r["zvf"], r["final_reward"], r["batch_reward_mean"], r["entropy"],
          r["advantage_variance"], r["kl_drift"]] for r in keep],
        dtype=float,
    )

    configs = [
        ("(none; raw correlation)",   [],   [0, 1]),
        ("batch mean reward",         [2],  [0, 1, 2]),
        ("policy entropy",            [3],  [0, 1, 3]),
        ("advantage variance",        [4],  [0, 1, 4]),
        ("KL drift to ref",           [5],  [0, 1, 5]),
        ("all four jointly",          [2, 3, 4, 5], [0, 1, 2, 3, 4, 5]),
    ]

    # Compute per-config statistics
    computed: list[dict[str, Any]] = []
    for label, ctrl, cols in configs:
        sub = arr_full[:, cols]
        sub = sub[np.isfinite(sub).all(axis=1)]
        n = sub.shape[0]
        if n < MIN_OBS:
            computed.append({
                "label": label,
                "r": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": n, "p": float("nan"), "dr2": float("nan"),
                "note": "insufficient data",
            })
            continue
        ctrl_mapped = [cols.index(c) for c in ctrl]
        if args.method == "spearman":
            r, lo, hi, p = _partial_corr_spearman(sub, 0, 1, ctrl_mapped)
        else:
            r, lo, hi, p = _partial_corr_pearson(sub, 0, 1, ctrl_mapped)
        dr2 = _incremental_r2(sub, 0, 1, ctrl_mapped) if np.isfinite(r) else float("nan")
        computed.append({
            "label": label, "r": r, "lo": lo, "hi": hi, "n": n, "p": p, "dr2": dr2,
            "note": "",
        })

    # BH correction: applied across the 5 per-confounder conditional rows.
    # The (none; raw) row and the joint row are reported but not in the BH
    # family. The raw row has no controls to control for, and the joint row
    # answers a different question ("does ZVF add anything after
    # partialling out ALL four controls at once?") whose p-value should
    # not be conflated with the marginal-conditional p-values under FDR
    # control. A reviewer can apply BH to all 6 rows themselves by
    # passing the rendered TSV to `python -c "import scipy.stats as s;
    # print(s.multipletests([...], method='fdr_bh'))"`.
    conditional_ps = [c["p"] for c in computed[1:5]]
    conditional_bh = _bh_adjust(conditional_ps)
    computed[1]["p_bh"] = conditional_bh[0]
    computed[2]["p_bh"] = conditional_bh[1]
    computed[3]["p_bh"] = conditional_bh[2]
    computed[4]["p_bh"] = conditional_bh[3]
    for c in (computed[0], computed[5]):
        c["p_bh"] = float("nan")

    # Stamp the reference-window fallback note on every row when
    # the LO..HI filter was empty and we fell back to the full sample,
    # so the BH family and p-values are explicitly scoped.
    if fallback_to_all:
        for c in computed:
            existing = c["note"]
            c["note"] = (existing + " [fallback: full sample]").strip()

    # Render
    lines = ["\t".join(header)]
    for c in computed:
        def _fmt(v: float) -> str:
            return f"{v:.3f}" if np.isfinite(v) else "NA"
        existing = c["note"]
        if np.isfinite(c["dr2"]) and c["dr2"] < 0:
            existing = (existing + " [negative dr2]").strip()
        if "fallback" in locals() and fallback_to_all:
            existing = (existing + " [fallback: full sample]").strip()
        c["note"] = existing
        lines.append("\t".join([
            c["label"],
            _fmt(c["r"]),
            _fmt(c["lo"]),
            _fmt(c["hi"]),
            str(c["n"]),
            _fmt(c["p"]),
            _fmt(c["p_bh"]),
            _fmt(c["dr2"]),
            c["note"],
        ]))

    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}", file=sys.stderr)
    for ln in lines:
        print(ln)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
