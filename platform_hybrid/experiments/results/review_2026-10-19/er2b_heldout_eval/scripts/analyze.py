#!/usr/bin/env python3
"""Held-out pass@k and training-reproduction summary for DeepReview #494 (P10 E-R2b).

Reads pass@k JSONs from tinker_passk.py and the July / rerun training logs.
Missing adapter evals are skipped. Numpy and the standard library only.

  python3 analyze.py [--json OUT]
  python3 analyze.py --raw-dir DIR --extra-eval FILE.json --json OUT
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

KS = (1, 8, 32)
DELTA_KS = (1, 32)
N_RESAMPLES = 10_000
SEED = 0

# Same stem order as the held-out sampling jobs.
ADAPTER_STEMS = (
    "er2b_rerun_g2_s123",
    "er2b_rerun_g2_s456",
    "er2b_rerun_g16_s123",
    "er2b_rerun_g16_s456",
)
ARMS = (
    (2, 123),
    (2, 456),
    (16, 123),
    (16, 456),
)
ARM_RE = re.compile(r"^er2b_rerun_g(\d+)_s(\d+)$")


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k. Byte-identical to tinker_passk.pass_at_k."""
    if n - c < k:
        return 1.0
    r = 1.0
    for i in range(k):
        r *= (n - c - i) / (n - i)
    return 1.0 - r


def repo_root() -> Path:
    here = Path(__file__).resolve()
    marker = Path("platform_hybrid") / "experiments" / "tinker-runs" / "results"
    for cand in here.parents:
        if (cand / marker).is_dir():
            return cand
    raise SystemExit("cannot locate platform_hybrid/experiments/tinker-runs/results from " + str(here))


def clean(value: float) -> float:
    number = float(value)
    if number == 0.0:
        return 0.0
    return number


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return clean(float(np.mean(np.asarray(values, dtype=float))))


def percentile_ci(samples: np.ndarray) -> list[float]:
    try:
        lo, hi = np.percentile(samples, [2.5, 97.5], method="linear")
    except TypeError:
        lo, hi = np.percentile(samples, [2.5, 97.5], interpolation="linear")
    return [clean(lo), clean(hi)]


def problem_pass(n: int, counts: list[int], k: int) -> np.ndarray:
    return np.array([pass_at_k(n, c, k) for c in counts], dtype=float)


def paired_delta(model: np.ndarray, base: np.ndarray, n_boot: int, n_perm: int, seed: int) -> dict:
    """Mean paired difference, percentile bootstrap CI, two-sided sign-flip p.

    Both generators are constructed with `seed`, so the two procedures do not
    share a stream. The p-value is (#{|T*| >= |T|} + 1) / (n_perm + 1).
    """
    if model.shape != base.shape:
        raise SystemExit(f"paired lengths differ: {model.shape} vs {base.shape}")
    delta = model - base
    observed = clean(float(delta.mean()))
    n = int(delta.size)
    if n == 0:
        raise SystemExit("no problems to compare")
    boot_rng = np.random.default_rng(seed)
    boot_idx = boot_rng.integers(0, n, size=(n_boot, n))
    ci = percentile_ci(delta[boot_idx].mean(axis=1))
    perm_rng = np.random.default_rng(seed)
    signs = perm_rng.integers(0, 2, size=(n_perm, n), dtype=np.int8) * np.int8(2) - np.int8(1)
    null = (signs * delta).mean(axis=1)
    n_extreme = int(np.count_nonzero(np.abs(null) >= abs(observed)))
    p_value = clean((n_extreme + 1) / (n_perm + 1))
    return {"delta": observed, "ci95": ci, "p": p_value}


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}: invalid JSON ({exc})") from exc


def load_eval(path: Path) -> dict:
    doc = load_json(path)
    if "per_problem_c" not in doc or "n_per_problem" not in doc:
        raise SystemExit(f"{path}: need per_problem_c and n_per_problem")
    n = int(doc["n_per_problem"])
    counts = [int(c) for c in doc["per_problem_c"]]
    if n < 1 or not counts:
        raise SystemExit(f"{path}: empty eval")
    if any(c < 0 or c > n for c in counts):
        raise SystemExit(f"{path}: per_problem_c outside 0..n_per_problem")
    for key in ("truncation_rate", "mean_tokens"):
        if key not in doc:
            raise SystemExit(f"{path}: missing {key}")
    curves = {k: problem_pass(n, counts, k) for k in KS}
    return {
        "name": path.stem,
        "path": str(path),
        "n_problems": len(counts),
        "n_per_problem": n,
        "truncation_rate": clean(doc["truncation_rate"]),
        "mean_tokens": clean(doc["mean_tokens"]),
        "curves": curves,
        "pass_at_k": {str(k): clean(float(curves[k].mean())) for k in KS},
    }


def discover_evals(raw_dir: Path, extras: list[Path]) -> tuple[dict, list[str]]:
    base_path = raw_dir / "base.json"
    if not base_path.is_file():
        raise SystemExit(f"missing base eval: {base_path}")
    found: dict[str, dict] = {"base": load_eval(base_path)}
    missing: list[str] = []
    for stem in ADAPTER_STEMS:
        path = raw_dir / f"{stem}.json"
        if path.is_file():
            found[stem] = load_eval(path)
        else:
            missing.append(stem)
    for path in extras:
        if not path.is_file():
            raise SystemExit(f"missing extra eval: {path}")
        loaded = load_eval(path)
        found[loaded["name"]] = loaded
        if loaded["name"] in missing:
            missing.remove(loaded["name"])
    base_n = found["base"]["n_problems"]
    for name, model in found.items():
        if model["n_problems"] != base_n:
            raise SystemExit(
                f"{name}: {model['n_problems']} problems, base has {base_n} (order must match)"
            )
    return found, missing


def model_row(model: dict, base: dict, n_boot: int, n_perm: int, seed: int) -> dict:
    deltas = {}
    for k in DELTA_KS:
        deltas[str(k)] = paired_delta(model["curves"][k], base["curves"][k], n_boot, n_perm, seed)
    return {
        "name": model["name"],
        "path": model["path"],
        "n_problems": model["n_problems"],
        "n_per_problem": model["n_per_problem"],
        "pass_at_k": model["pass_at_k"],
        "truncation_rate": model["truncation_rate"],
        "mean_tokens": model["mean_tokens"],
        "delta_vs_base": deltas,
    }


def pooled_row(group: int, models: list[dict], base: dict, n_boot: int, n_perm: int, seed: int) -> dict:
    """Unweighted mean, across seeds, of each problem's unbiased pass@k."""
    curves = {}
    pass_at = {}
    for k in KS:
        stacked = np.stack([model["curves"][k] for model in models], axis=0)
        curves[k] = stacked.mean(axis=0)
        pass_at[str(k)] = clean(float(curves[k].mean()))
    deltas = {
        str(k): paired_delta(curves[k], base["curves"][k], n_boot, n_perm, seed) for k in DELTA_KS
    }
    seeds = []
    for model in models:
        match = ARM_RE.match(model["name"])
        seeds.append(int(match.group(2)) if match else None)
    return {
        "arm": f"G={group}",
        "group_size": group,
        "n_seeds": len(models),
        "models": [model["name"] for model in models],
        "seeds": seeds,
        "pass_at_k": pass_at,
        "truncation_rate": clean(float(np.mean([model["truncation_rate"] for model in models]))),
        "mean_tokens": clean(float(np.mean([model["mean_tokens"] for model in models]))),
        "delta_vs_base": deltas,
    }


def reward_series(doc: dict) -> list[float]:
    trace = doc.get("reward_trace")
    if isinstance(trace, list) and trace:
        return [float(x) for x in trace]
    log = doc.get("step_log") or []
    return [float(row["reward"]) for row in log if "reward" in row and row["reward"] is not None]


def zvf_series(doc: dict) -> list[float]:
    log = doc.get("step_log") or []
    return [float(row["zvf"]) for row in log if "zvf" in row and row["zvf"] is not None]


def train_side(path: Path | None) -> dict | None:
    if path is None or not path.is_file():
        return None
    doc = load_json(path)
    rewards = reward_series(doc)
    zvfs = zvf_series(doc)
    planned = int(doc["steps"]) if doc.get("steps") is not None else None
    n_log = len(doc.get("step_log") or [])
    completed = n_log if n_log else len(rewards)
    if "last10_avg" in doc and doc["last10_avg"] is not None:
        last10 = clean(doc["last10_avg"])
    else:
        last10 = mean_or_none(rewards[-10:])
    window = zvfs[-20:]
    status = doc.get("status")
    incomplete = False
    if status is not None and str(status).lower() != "completed":
        incomplete = True
    if planned is not None and completed < planned:
        incomplete = True
    return {
        "path": str(path),
        "status": status,
        "group_size": doc.get("group_size"),
        "seed": doc.get("seed"),
        "steps_planned": planned,
        "steps_completed": completed,
        "n_reward": len(rewards),
        "n_step_log": n_log,
        "last10": last10,
        "last10_n": min(10, len(rewards)) if rewards else 0,
        "zvf_final20": mean_or_none(window),
        "zvf_final20_n": len(window),
        "incomplete": incomplete,
    }


def training_rows(train_dir: Path, july_dir: Path) -> list[dict]:
    rows = []
    for group, seed in ARMS:
        july = train_side(july_dir / f"er2b_g{group}_s{seed}.json")
        rerun = train_side(train_dir / f"er2b_rerun_g{group}_s{seed}.json")
        rows.append(
            {
                "arm": f"G={group} s={seed}",
                "group_size": group,
                "seed": seed,
                "july": july,
                "rerun": rerun,
                "rerun_incomplete": True if rerun is None else bool(rerun["incomplete"]),
            }
        )
    return rows


def build_report(evals: dict, missing: list[str], train_dir: Path, july_dir: Path, n_boot: int, n_perm: int, seed: int) -> dict:
    base = evals["base"]
    order = ["base", *[stem for stem in ADAPTER_STEMS if stem in evals]]
    order.extend(sorted(name for name in evals if name not in order))
    models = [model_row(evals[name], base, n_boot, n_perm, seed) for name in order]
    pooled = []
    for group in (2, 16):
        members = []
        for name in order:
            match = ARM_RE.match(name)
            if match and int(match.group(1)) == group and name in evals:
                members.append(evals[name])
        if members:
            pooled.append(pooled_row(group, members, base, n_boot, n_perm, seed))
    return {
        "protocol": {
            "pass_at_k": "mean over problems of tinker_passk.pass_at_k (unbiased)",
            "delta": "adapter minus base, paired by problem index",
            "ci": "percentile bootstrap of the paired mean, 2.5 and 97.5",
            "p": "two-sided sign-flip, (count + 1) / (n_perm + 1)",
            "pool": "unweighted mean across seeds of per-problem pass@k",
            "n_bootstrap": n_boot,
            "n_permutation": n_perm,
            "seed": seed,
            "ks": list(KS),
            "delta_ks": list(DELTA_KS),
        },
        "missing_eval": missing,
        "models": models,
        "pooled": pooled,
        "training": training_rows(train_dir, july_dir),
    }


def fmt(value, digits: int = 6) -> str:
    if value is None:
        return "—"
    number = float(value)
    if number == 0.0:
        return "0"
    return f"{number:.{digits}f}"


def fmt_ci(bounds: list[float]) -> str:
    lo, hi = bounds
    if lo == 0.0 and hi == 0.0:
        return "[0, 0]"
    return f"[{fmt(lo)}, {fmt(hi)}]"


def fmt_window(value, n: int, full: int) -> str:
    if value is None:
        return "—"
    text = fmt(value)
    if n != full:
        return f"{text} (n={n})"
    return text


def markdown(report: dict) -> str:
    lines = [
        "# E-R2b held-out evaluation",
        "",
        "Pass@k is recomputed from `per_problem_c` with the unbiased estimator in `tinker_passk.pass_at_k`, then averaged over problems. Deltas are adapter minus base on the same problem order. The 95% interval is a paired percentile bootstrap over problems "
        f"({report['protocol']['n_bootstrap']} resamples, seed {report['protocol']['seed']}). "
        "The p-value is a two-sided sign-flip test over problems "
        f"({report['protocol']['n_permutation']} flips, seed {report['protocol']['seed']}), "
        "`(#{|T*| >= |T|} + 1) / (n_perm + 1)`.",
        "",
        "## Held-out",
        "",
        "| model | n | pass@1 | pass@8 | pass@32 | truncation_rate | mean_tokens | Δpass@1 | 95% CI | p | Δpass@32 | 95% CI | p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---:|",
    ]
    for row in report["models"]:
        d1 = row["delta_vs_base"]["1"]
        d32 = row["delta_vs_base"]["32"]
        lines.append(
            "| {name} | {n} | {p1} | {p8} | {p32} | {tr} | {mt} | {a} | {aci} | {ap} | {b} | {bci} | {bp} |".format(
                name=row["name"],
                n=row["n_per_problem"],
                p1=fmt(row["pass_at_k"]["1"]),
                p8=fmt(row["pass_at_k"]["8"]),
                p32=fmt(row["pass_at_k"]["32"]),
                tr=fmt(row["truncation_rate"]),
                mt=fmt(row["mean_tokens"], 4),
                a=fmt(d1["delta"]),
                aci=fmt_ci(d1["ci95"]),
                ap=fmt(d1["p"]),
                b=fmt(d32["delta"]),
                bci=fmt_ci(d32["ci95"]),
                bp=fmt(d32["p"]),
            )
        )
    if report["missing_eval"]:
        lines.append("")
        lines.append("Skipped missing adapter evals: " + ", ".join(f"`{name}`" for name in report["missing_eval"]) + ".")
    lines.extend(
        [
            "",
            "## Pooled arms",
            "",
            "G=2 pools seeds 123 and 456. G=16 pools seeds 123 and 456. For each problem the arm score is the unweighted mean of the seeds' unbiased pass@k, then the same paired delta, interval, and sign-flip test are taken against base.",
            "",
        ]
    )
    if not report["pooled"]:
        lines.append("No adapter evals to pool.")
    else:
        lines.append(
            "| arm | seeds | pass@1 | pass@8 | pass@32 | truncation_rate | mean_tokens | Δpass@1 | 95% CI | p | Δpass@32 | 95% CI | p |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---:|")
        for row in report["pooled"]:
            d1 = row["delta_vs_base"]["1"]
            d32 = row["delta_vs_base"]["32"]
            seed_text = ", ".join(str(s) for s in row["seeds"])
            lines.append(
                "| {arm} | {seeds} | {p1} | {p8} | {p32} | {tr} | {mt} | {a} | {aci} | {ap} | {b} | {bci} | {bp} |".format(
                    arm=row["arm"],
                    seeds=seed_text,
                    p1=fmt(row["pass_at_k"]["1"]),
                    p8=fmt(row["pass_at_k"]["8"]),
                    p32=fmt(row["pass_at_k"]["32"]),
                    tr=fmt(row["truncation_rate"]),
                    mt=fmt(row["mean_tokens"], 4),
                    a=fmt(d1["delta"]),
                    aci=fmt_ci(d1["ci95"]),
                    ap=fmt(d1["p"]),
                    b=fmt(d32["delta"]),
                    bci=fmt_ci(d32["ci95"]),
                    bp=fmt(d32["p"]),
                )
            )
        present = {row["arm"] for row in report["pooled"]}
        for arm in ("G=2", "G=16"):
            if arm not in present:
                lines.append("")
                lines.append(f"{arm}: no adapter evals.")
    lines.extend(
        [
            "",
            "## Training reproduction",
            "",
            "Last-10 train reward is `last10_avg` when that field is present, otherwise the mean of the last 10 `reward_trace` values (or `step_log[].reward` if the trace is absent). Final-20 ZVF is the mean of the last 20 `step_log[].zvf` values. A shorter log is averaged over the steps that exist and marked with `n`. A rerun is incomplete when `status` is not `completed` or logged steps are below planned `steps`.",
            "",
            "| arm | G | seed | july last10 | rerun last10 | july ZVF final20 | rerun ZVF final20 | july steps | rerun steps | rerun incomplete |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["training"]:
        july = row["july"] or {}
        rerun = row["rerun"] or {}
        lines.append(
            "| {arm} | {g} | {seed} | {jl} | {rl} | {jz} | {rz} | {js} | {rs} | {flag} |".format(
                arm=row["arm"],
                g=row["group_size"],
                seed=row["seed"],
                jl=fmt_window(july.get("last10"), july.get("last10_n") or 0, 10) if july else "—",
                rl=fmt_window(rerun.get("last10"), rerun.get("last10_n") or 0, 10) if rerun else "—",
                jz=fmt_window(july.get("zvf_final20"), july.get("zvf_final20_n") or 0, 20) if july else "—",
                rz=fmt_window(rerun.get("zvf_final20"), rerun.get("zvf_final20_n") or 0, 20) if rerun else "—",
                js=_steps(july) if july else "—",
                rs=_steps(rerun) if rerun else "—",
                flag="yes" if row["rerun_incomplete"] else "no",
            )
        )
    lines.append("")
    return "\n".join(lines)


def _steps(side: dict) -> str:
    done = side.get("steps_completed")
    planned = side.get("steps_planned")
    if done is None:
        return "—"
    if planned is None:
        return str(done)
    return f"{done}/{planned}"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    here = Path(__file__).resolve()
    eval_root = here.parents[1]
    parser = argparse.ArgumentParser(description="E-R2b held-out pass@k and training reproduction (DeepReview #494)")
    parser.add_argument("--json", dest="json_out", type=Path, default=None, help="Write the same numbers as JSON")
    parser.add_argument("--raw-dir", type=Path, default=eval_root / "raw")
    parser.add_argument("--train-dir", type=Path, default=eval_root / "train")
    parser.add_argument("--july-dir", type=Path, default=None)
    parser.add_argument("--extra-eval", type=Path, nargs="*", default=[])
    parser.add_argument("--n-boot", type=int, default=N_RESAMPLES)
    parser.add_argument("--n-perm", type=int, default=N_RESAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.n_boot < 1 or args.n_perm < 1:
        raise SystemExit("--n-boot and --n-perm must be positive")
    july_dir = args.july_dir if args.july_dir is not None else repo_root() / "platform_hybrid" / "experiments" / "tinker-runs" / "results"
    evals, missing = discover_evals(args.raw_dir, list(args.extra_eval))
    report = build_report(evals, missing, args.train_dir, july_dir, args.n_boot, args.n_perm, args.seed)
    sys.stdout.write(markdown(report))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
