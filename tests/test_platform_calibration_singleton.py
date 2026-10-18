"""Exercise the calibration CLI with synthetic runs, never stored results."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

DIRECTORY = (Path(__file__).resolve().parents[1]
             / "reports/project_finish_2026-10-11/validation/samestack_v2")
FAMILY = {"g8-ppo", "g8-ppo_8x8", "g8-g2"}


def run_calibration(tmp_path, rows):
    for arm, seed, correct in rows:
        (tmp_path / f"{arm}_s{seed}.json").write_text(json.dumps({
            "arm": arm, "seed": seed,
            "pre_correct": [1] * 5 + [0] * 5,
            "post_correct": [1] * correct + [0] * (10 - correct),
        }))
    return subprocess.run(
        [sys.executable, "-I", str(DIRECTORY / "platform_calibration.py"),
         str(DIRECTORY), str(tmp_path)],
        cwd=tmp_path, text=True, capture_output=True, check=False,
    )


def unavailable(c, n):
    assert c["status"] == "unavailable"
    assert c["reason"] == "insufficient_pairs"
    assert c["n"] == n
    for key in ("p_t", "p_signflip", "tost_p", "holm_p_t", "holm_p_signflip",
                "holm_tost_p", "t_ci", "tost_ci90", "boot_ci",
                "difference_ci_95", "equivalence_ci_90", "within_margin_unadjusted"):
        assert c[key] is None


def test_singleton_primary_contrast_is_unavailable(tmp_path):
    result = run_calibration(tmp_path, [("grpo_g8", 1, 7), ("ppo", 1, 6)])
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    for metric in ("post", "gain"):
        contrasts = report[f"contrasts_{metric}"]
        assert set(contrasts) == FAMILY
        c = contrasts["g8-ppo"]
        unavailable(c, 1)
        assert c["mean"] == pytest.approx(0.1)
        assert c["platform_of_seed"] == {"1": "A10G"}
        for name in FAMILY - {"g8-ppo"}:
            unavailable(contrasts[name], 0)
            assert contrasts[name]["mean"] is None


def test_no_shared_seed_is_unavailable(tmp_path):
    result = run_calibration(tmp_path, [("grpo_g8", 1, 7), ("ppo", 2, 6)])
    assert result.returncode == 0, result.stderr
    for metric in ("post", "gain"):
        for c in json.loads(result.stdout)[f"contrasts_{metric}"].values():
            unavailable(c, 0)


def test_empty_input_has_explicit_error_not_index_error(tmp_path):
    result = run_calibration(tmp_path, [])
    assert result.returncode != 0
    assert "expected Modal A10G plus one Colab platform, got []" in result.stderr
    assert "Traceback" not in result.stderr


def test_multiple_pairs_keep_full_holm_family_with_missing_or_singleton_arms(tmp_path):
    rows = []
    for seed, g8, ppo in [(1, 7, 5), (2, 8, 5), (3, 9, 5)]:
        rows.extend([("grpo_g8", seed, g8), ("ppo", seed, ppo)])
    rows.append(("ppo_8x8", 1, 6))
    result = run_calibration(tmp_path, rows)
    assert result.returncode == 0, result.stderr
    spec = importlib.util.spec_from_file_location("calibration_test_stats", DIRECTORY / "stats_lib.py")
    statistics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(statistics)
    for metric in ("post", "gain"):
        contrasts = json.loads(result.stdout)[f"contrasts_{metric}"]
        c = contrasts["g8-ppo"]
        assert c["n"] == 3
        assert c["mean"] == pytest.approx(0.3)
        for stat in ("p_t", "p_signflip", "tost_p"):
            expected = statistics.holm({"tested": c[stat], "singleton": 1., "missing": 1.})
            assert c[f"holm_{stat}"] == pytest.approx(expected["tested"])
        unavailable(contrasts["g8-ppo_8x8"], 1)
        unavailable(contrasts["g8-g2"], 0)
    # The consumer guard must not weaken the statistical helper's contract.
    with pytest.raises(ValueError, match="at least two shared seeds"):
        statistics.paired({1: 0.1}, {1: 0.0})


def test_complete_family_retains_multi_pair_inference(tmp_path):
    rows = [(arm, seed, correct)
            for seed in (1, 2, 3)
            for arm, correct in (("grpo_g8", 6 + seed), ("ppo", 5),
                                 ("ppo_8x8", 6), ("grpo_g2", 4))]
    result = run_calibration(tmp_path, rows)
    assert result.returncode == 0, result.stderr
    for metric in ("post", "gain"):
        contrasts = json.loads(result.stdout)[f"contrasts_{metric}"]
        assert set(contrasts) == FAMILY
        for c in contrasts.values():
            assert c["n"] == 3
            assert c.get("status") != "unavailable"
            assert len(c["difference_ci_95"]) == 2
        for stat in ("p_t", "p_signflip", "tost_p"):
            ordered = sorted(contrasts, key=lambda name: contrasts[name][stat])
            running = 0.0
            for index, name in enumerate(ordered):
                running = max(running, min(1., (3 - index) * contrasts[name][stat]))
                assert contrasts[name][f"holm_{stat}"] == pytest.approx(running)
