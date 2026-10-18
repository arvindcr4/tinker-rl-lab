import unittest
from protocol_stats import pearson, clustered_correlation, finite_pool_summary, fingerprint
import pytest


class ProspectiveStatsTests(unittest.TestCase):
    def test_degenerate_correlation_is_not_evidence(self):
        for xs, ys in [([], []), ([1], [2]), ([1, 1], [2, 3]), ([float("nan"), 2], [1, 2])]:
            assert pearson(xs, ys) is None
        with pytest.raises(ValueError):
            pearson([1, 2], [1])

    def test_seed_clusters_and_unrounded_estimates(self):
        rows = [{"seed": s, "x": i, "y": i + s} for s in range(3) for i in range(5)]
        result = clustered_correlation(rows, "x", "y", 50)
        assert result["n_independent_seeds"] == 3
        assert len(result["per_seed"]) == 3
        assert result == clustered_correlation(rows, "x", "y", 50)

    def test_hierarchical_bootstrap_binary_and_degenerate_pools(self):
        result = finite_pool_summary([[0] * 8, [1] * 8], 2, 30, 20)
        assert result["emp_ci95"] == [1, 1]
        assert result["original_rollouts"] == 16
        assert result["true_p_prediction"] is None
        mixed = finite_pool_summary([[0, 1] * 4] * 4, 2, 30, 20)
        assert mixed["plugin_ci95"][0] < mixed["plugin_ci95"][1]
        with pytest.raises(ValueError):
            finite_pool_summary([[float("nan")]], 2)

    def test_fingerprint_changes_with_configuration(self):
        assert (
            fingerprint(__file__, {"G": 2})["sha256"] != fingerprint(__file__, {"G": 3})["sha256"]
        )


if __name__ == "__main__":
    unittest.main()


class ControllerRegressionTests(unittest.TestCase):
    def test_drop_disabled_really_preserves_controller_population(self):
        import ast
        from pathlib import Path

        tree = ast.parse(Path(__file__).with_name("e6_live_triage.py").read_text())
        wanted = {"adaptive_fn", "_group_zerovar", "Decision", "ZVFController"}
        tree.body = [
            n
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted
        ]
        scope = dict(WINDOW=5, DROP_K=3, STOP_K=4, ZVF_MAX=0.85, EPS_LO=0.05, G0=4, GMIN=2, GMAX=12)
        exec(compile(tree, "controller", "exec"), scope)
        for enabled in [False, True]:
            ctrl = scope["ZVFController"](True, enable_drop=enabled)
            for _ in range(4):
                ctrl.step([0, 0], [7, 7])
            assert ctrl.dropped == ({7} if enabled else set())
