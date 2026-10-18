import importlib.util
import math
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "platform_modal/scripts"
sys.path.insert(0, str(SCRIPTS))
from _review27_rank import kendall, rank_correlation, spearman


def test_double_tied_kendall_is_one():
    assert kendall([0, 0, 1], [0, 0, 1]) == pytest.approx(1)


def test_constant_spearman_never_becomes_significant():
    rho, p = spearman([1, 1, 1], [1, 2, 3], b=19)
    assert math.isnan(rho)
    assert math.isnan(p)


def test_tied_rank_permutation_is_reproducible():
    assert spearman([0, 0, 1, 2], [2, 2, 1, 0], b=39) == spearman([0, 0, 1, 2], [2, 2, 1, 0], b=39)
    assert rank_correlation([0, 0, 1, 2], [2, 2, 1, 0]) == pytest.approx(-1)


@pytest.mark.parametrize("values", [[1, float("nan")], [1, float("inf")]])
def test_nonfinite_pairs_rejected(values):
    with pytest.raises(ValueError, match="finite"):
        spearman(values, [1, 2])


def test_actual_iter49_constant_input_regression():
    spec = importlib.util.spec_from_file_location(
        "review27_iter49", SCRIPTS / "scaling_law_iter49.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rho, p = module.spearman_pvalue([1, 1, 1], [1, 2, 3], b=19)
    assert math.isnan(rho)
    assert math.isnan(p)
