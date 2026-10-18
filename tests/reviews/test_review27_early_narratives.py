"""Temporary-only regressions for early scaling evidence and output semantics."""
import csv
import hashlib
import importlib
import json
import shutil
import math
from pathlib import Path
import subprocess
import sys

import matplotlib.figure
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "platform_modal/scripts"


def load(iteration):
    return importlib.import_module(f"platform_modal.scripts.scaling_law_iter{iteration}")


def read_table(path):
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def source_rows(root, count=6):
    rows = []
    steps = [8, 13, 11, 19, 15, 24]
    for i in range(count):
        p = [2, 4, 8, 16, 32, 64][i]
        t = np.arange(1, steps[i] + 1)
        rewards = (.25 + .05 * i) * (1 - np.exp(-.4 * t))
        source = root / f"trace_{i}.json"
        run_id = f"00{i}"  # numeric-looking identity must remain an exact string
        source.write_text(json.dumps({"run_id": run_id, "reward_trace": rewards.tolist()}))
        rows.append({
            "model": f"Model_{i}", "params_B": p, "arch": "dense" if i < 3 else "moe",
            "family": "fixture", "n_steps": len(rewards), "r_first": rewards[0], "r_final": rewards[-1],
            "r_mean": float(rewards.mean()), "r_peak": max(rewards), "r_var": float(rewards.var()),
            "early_mean": float(rewards[:4].mean()), "late_mean": float(rewards[-4:].mean()), "zero_frac": 0,
            "trace_file": source.name, "trace_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "run_id": run_id,
        })
    return rows


def candidate_row(root):
    row = source_rows(root, 1)[0]
    row.update(C_proxy=row["params_B"] * row["n_steps"] * 512,
               log10_P=math.log10(row["params_B"]),
               log10_C=math.log10(row["params_B"] * row["n_steps"] * 512),
               R_max=.25, **{"lambda": .4}, fit_ok=True, evidence_kind="measured_trace_fit")
    return row


@pytest.mark.parametrize("iteration", [45, 49, 57])
def test_package_and_direct_imports_have_no_mkdir(iteration, tmp_path):
    # Fresh processes isolate the package and direct-script import paths.
    for module, path in ((f"platform_modal.scripts.scaling_law_iter{iteration}", ROOT),
                         (f"scaling_law_iter{iteration}", SCRIPTS)):
        program = f"""
import sys
sys.path.insert(0, {str(path)!r})
import matplotlib.pyplot, scipy.optimize
from pathlib import Path
Path.mkdir = lambda *a, **k: (_ for _ in ()).throw(AssertionError('import wrote a directory'))
import importlib
importlib.import_module({module!r})
"""
        subprocess.run([sys.executable, "-c", program], check=True, cwd=tmp_path, capture_output=True)


def test_iter45_end_to_end_preserves_identity_and_slope_schema(tmp_path, monkeypatch):
    module = load(45)
    monkeypatch.setattr(module, "B_BOOT", 19)
    rows = source_rows(tmp_path)
    write_table(tmp_path / "experiments/results/scaling_law_extended_frontier.tsv", rows)
    output = tmp_path / "candidate"
    module.main(output_dir=output, source_root=tmp_path)
    comp = read_table(output / "scaling_law_iter45_compute_proxy_v2.tsv")
    assert [r["run_id"] for r in comp] == [f"00{i}" for i in range(6)]
    fits = read_table(output / "scaling_law_iter45_scaling_v2.tsv")
    assert all("semi_log_reward_slope" in r and "alpha" not in r for r in fits)
    preds = read_table(output / "scaling_law_iter45_predictions_v2.tsv")
    assert preds[-1]["status"] == "unavailable" and preds[-1]["pass"] == ""
    summary = read_table(output / "scaling_law_iter45_summary_v2.tsv")[0]
    assert summary["allocation_status"] == "not_identified"
    text = (output / "sections/scaling_law_iter45_v2.tex").read_text()
    assert "Chinchilla" not in text and "pre-registered" not in text
    assert "semi_log" not in summary and "semi_log_reward_slope_dense" in summary
    assert (output / "figures/scaling_law_iter45_v2.png").exists()
    manifest = json.loads((output / "scaling_law_iter45_manifest.json").read_text())
    assert len(manifest["inputs"]) == 7 and all(r["sha256"] for r in manifest["inputs"])


def test_iter45_mixed_missing_trace_rows_keep_fixed_schema(tmp_path, monkeypatch):
    module = load(45)
    monkeypatch.setattr(module, "B_BOOT", 9)
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", lambda *a, **k: None)
    rows = source_rows(tmp_path, 4)
    rows[0].update(trace_file="", trace_sha256="", run_id="")
    write_table(tmp_path / "experiments/results/scaling_law_extended_frontier.tsv", rows)
    module.main(output_dir=tmp_path / "candidate", source_root=tmp_path)
    rows = read_table(tmp_path / "candidate/scaling_law_iter45_compute_proxy_v2.tsv")
    assert rows[0]["evidence_kind"] == "unavailable"
    assert rows[1]["evidence_kind"] == "measured_trace_fit"


def test_iter49_end_to_end_outputs_descriptive_scores_only(tmp_path, monkeypatch):
    m45, m49 = load(45), load(49)
    monkeypatch.setattr(m45, "B_BOOT", 9)
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", lambda *a, **k: None)
    write_table(tmp_path / "experiments/results/scaling_law_extended_frontier.tsv", source_rows(tmp_path))
    output = tmp_path / "candidate"
    m45.main(output_dir=output, source_root=tmp_path)
    m49.main(output_dir=output, source_root=tmp_path)
    scores = read_table(output / "scaling_law_iter49_budget_scores_v2.tsv")
    assert len(scores) == 50 * 6
    assert all(r["allocation_status"] == "not_identified" for r in scores)
    assert not (output / "scaling_law_iter49_optimal_anchor_v2.tsv").exists()
    assert (output / "scaling_law_iter49_loo_residuals.tsv").exists()
    preds = read_table(output / "scaling_law_iter49_predictions_v2.tsv")
    assert preds[-1]["status"] == "unavailable"
    assert "semi_log_reward_slope_logP" in read_table(output / "scaling_law_iter49_two_param_v2.tsv")[0]


@pytest.mark.parametrize("mutation", ["synthetic_kind", "synthetic_source", "missing_id", "missing_hash", "missing_path", "nan_id", "duplicate", "nonfinite", "hash_mismatch"])
def test_iter49_rejects_unverified_or_synthetic_rows(tmp_path, monkeypatch, mutation):
    module = load(49)
    row = candidate_row(tmp_path)
    rows = [row]
    if mutation == "synthetic_kind":
        row["evidence_kind"] = "synthetic_trace_fit"
    elif mutation == "synthetic_source":
        source = tmp_path / row["trace_file"]
        record = json.loads(source.read_text()); record["synthetic"] = True
        source.write_text(json.dumps(record))
        row["trace_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    elif mutation == "missing_id": row["run_id"] = ""
    elif mutation == "missing_hash": row["trace_sha256"] = ""
    elif mutation == "missing_path": row["trace_file"] = ""
    elif mutation == "nan_id": row["run_id"] = "NaN"
    elif mutation == "duplicate": rows.append(dict(row))
    elif mutation == "nonfinite": row["R_max"] = "nan"
    elif mutation == "hash_mismatch": row["trace_sha256"] = "0" * 64
    path = tmp_path / "candidate.tsv"; write_table(path, rows)
    monkeypatch.setattr(module, "SRC", path); monkeypatch.setattr(module, "REPO", tmp_path)
    with pytest.raises(ValueError): module.load_iter45()


def test_iter49_valid_exact_identity_and_budget_invariance(tmp_path, monkeypatch):
    module = load(49)
    path = tmp_path / "candidate.tsv"; write_table(path, [candidate_row(tmp_path)])
    monkeypatch.setattr(module, "SRC", path); monkeypatch.setattr(module, "REPO", tmp_path)
    assert module.load_iter45()[0]["run_id"] == "000"
    a = module.budget_scores([1, 2, 3], 4, -.2, .8, .1)
    b = module.budget_scores([1, 2, 3], 10, -.2, .8, .1)
    assert np.array_equal(np.argsort(a), np.argsort(b))
    assert b - a == pytest.approx([4.8] * 3)


def test_iter49_rank_deficient_model_is_unavailable():
    with pytest.raises(ValueError, match="rank deficient"):
        load(49).ols_3var([1, 2, 3], [2, 3, 4], [.2, .3, .4])


def test_iter57_loglog_exponent_and_no_offset(monkeypatch):
    module = load(57); monkeypatch.setattr(module, "N_BOOT", 99)
    params = np.array([1., 2., 4., 8., 16.])
    records = [{"model": str(p), "params_B": p, "reward_summary": .01 * p ** .2} for p in params]
    fit = module.log_summary_fit(records)
    assert fit["alpha"] == pytest.approx(.2)
    assert fit["alpha_lo"] == pytest.approx(.2)
    assert fit["alpha_hi"] == pytest.approx(.2)
    assert 0 < fit["n_boot_success"] <= 99
    again = module.log_summary_fit(records)
    assert fit["alpha_lo"] == again["alpha_lo"]


def test_iter57_zero_summaries_are_excluded_and_named(monkeypatch):
    module = load(57); monkeypatch.setattr(module, "N_BOOT", 9)
    result = module.log_summary_fit([{"model": "zero", "params_B": 1, "reward_summary": 0}])
    assert result["excluded_nonpositive"] == ["zero"]
    assert result["status"] == "unavailable" and result["n_boot_success"] == 0


@pytest.mark.parametrize("trace", [[], [1, float("nan")], [[1, 2], [3, 4]]])
def test_iter57_invalid_summary_rejected(trace):
    with pytest.raises(ValueError): load(57).reward_summary(trace)


def test_iter57_fit_failure_never_returns_initial_parameters(monkeypatch):
    module = load(57)
    def fail(*args, **kwargs): raise RuntimeError("fixture failure")
    monkeypatch.setattr(module, "curve_fit", fail)
    with pytest.raises(ValueError, match="fit failed"):
        module.fit_canonical(np.array([.1, .2, .3]))


def test_iter57_predicates_reverse_and_equality_is_included():
    module = load(57)
    fit = dict(alpha_lo=-.2, alpha_hi=.3)
    rows = [{"model": name, "fit_status": "ok", "lam_at_bound": name != "Nemotron-120B",
             "gap_canonical_minus_summary": gap}
            for name, gap in [("Nemotron-120B", -.2), ("A", -.1), ("B", 0)]]
    preds = module.descriptive_criteria(rows, fit)
    assert preds[0]["pass_fail"] == preds[1]["pass_fail"] == "PASS"
    assert preds[1]["observed_value"] == 3
    rows[1]["gap_canonical_minus_summary"] = .1
    rows[1]["lam_at_bound"] = False
    preds = module.descriptive_criteria(rows, fit)
    assert preds[0]["pass_fail"] == preds[1]["pass_fail"] == "FAIL"
    rows[1]["fit_status"] = "unavailable"
    preds = module.descriptive_criteria(rows, fit)
    assert preds[0]["pass_fail"] == preds[1]["pass_fail"] == "UNAVAILABLE"


def test_iter57_end_to_end_plot_and_text_follow_changed_data(tmp_path, monkeypatch):
    module = load(57)
    monkeypatch.setattr(module, "N_BOOT", 49)
    names = {"Other_A": ("a.json", 1.), "Nemotron-120B": ("b.json", 2.), "Other_C": ("c.json", 4.)}
    extras = {"Extra_D": ("d.json", 8.)}
    monkeypatch.setattr(module, "POOL_A", names); monkeypatch.setattr(module, "POOL_B", extras)
    trace_dir = tmp_path / "experiments/tinker-runs/results"; trace_dir.mkdir(parents=True)
    for name, (filename, params) in {**names, **extras}.items():
        reward = .1 * params ** -.2
        (trace_dir / filename).write_text(json.dumps({"run_id": name, "reward_trace": [reward] * 8}))
    saved = []
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", lambda fig, *a, **k: saved.append(fig))
    output = tmp_path / "candidate"
    module.main(output_dir=output, source_root=tmp_path)
    bounds = read_table(output / "scaling_law_iter57_lam_bound.tsv")
    assert all("phase_pred_from_bound" not in r for r in bounds)
    assert all(r["optimizer_status"] in {"at_upper_bound", "interior_to_upper_bound"} for r in bounds)
    summary = read_table(output / "scaling_law_iter57_robust_rmax.tsv")
    assert "reward_summary" in summary[0] and "r_max_robust" not in summary[0]
    # The chart reads named fitted/summary values, not overwritten positional rows.
    bars = saved[0].axes[1].patches
    assert [p.get_height() for p in bars[:3]] == pytest.approx([float(r["R_max_canonical"]) for r in summary])
    assert [p.get_height() for p in bars[3:]] == pytest.approx([float(r["reward_summary"]) for r in summary])
    points = saved[0].axes[2].collections[1].get_offsets()
    assert points[-1, 1] == pytest.approx(math.log10(.1 * 8 ** -.2))
    text = (output / "sections/scaling_law_iter57.tex").read_text()
    assert "-0.2000" in text
    assert "positive but small" not in text and "only anchor whose" not in text
    assert "ordinary mean, not a trimmed mean" in text
    assert "not establish equivalence" in text
    assert r"Other\_A" in text and r"P2\_summary\_ge\_canonical" in text


def test_iter57_report_compiles_when_latex_format_is_installed(tmp_path):
    if not shutil.which("pdflatex") or not shutil.which("kpsewhich"):
        pytest.skip("LaTeX tools are not installed")
    format_check = subprocess.run(["kpsewhich", "pdflatex.fmt"], capture_output=True, text=True)
    if format_check.returncode != 0 or not format_check.stdout.strip():
        pytest.skip("pdflatex exists but its format is not installed")
    module = load(57)
    record = {"model": "Model_A", "fit_status": "ok", "lam_at_bound": False,
              "gap_canonical_minus_summary": -.2}
    fit = {"alpha": -.2, "alpha_lo": -.4, "alpha_hi": -.1, "r2": .9, "n": 3,
           "n_boot_success": 5, "n_boot_attempted": 9, "excluded_nonpositive": []}
    path = tmp_path / "section.tex"
    module.write_section([record], {"pool_A": fit}, module.descriptive_criteria([record], fit), path)
    text = path.read_text()
    # Compile a standalone document only when the optional toolchain is complete.
    document = tmp_path / "report.tex"
    document.write_text("\\documentclass{article}\n\\usepackage{amsmath}\n\\begin{document}\n" + text + "\n\\end{document}\n")
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", str(document)],
                   cwd=tmp_path, check=True, capture_output=True)


def test_iter45_explicit_synthetic_source_is_rejected(tmp_path, monkeypatch):
    module = load(45)
    rows = source_rows(tmp_path, 1)
    source = tmp_path / rows[0]["trace_file"]
    record = json.loads(source.read_text()); record["synthetic"] = True
    source.write_text(json.dumps(record))
    rows[0]["trace_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    write_table(tmp_path / "experiments/results/scaling_law_extended_frontier.tsv", rows)
    with pytest.raises(ValueError, match="synthetic|reconstruct"):
        module.main(output_dir=tmp_path / "candidate", source_root=tmp_path)


def test_iter45_summary_values_cannot_change_observed_fit():
    module = load(45)
    trace = .4 * (1 - np.exp(-.3 * np.arange(1, 15)))
    a = module.fit_saturation(*([0] * 9), observed_trace=trace)
    b = module.fit_saturation(*([999] * 9), observed_trace=trace)
    assert a == pytest.approx(b)
    assert a[:2] == pytest.approx((.4, .3))
