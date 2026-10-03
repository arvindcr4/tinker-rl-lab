"""Pin the shared platform_modal analysis primitives (byte-level TSV output, seeded stats)."""

import random
from math import erf, sqrt
from pathlib import Path
from statistics import NormalDist

import numpy as np

from tests._shared_fakes import load_module

ROOT = Path(__file__).resolve().parents[1]
ac = load_module("_analysis_common", ROOT / "platform_modal/scripts/_analysis_common.py")


def test_write_dict_tsv_bytes(tmp_path):
    out = tmp_path / "sub" / "a.tsv"
    ac.write_dict_tsv(out, [{"a": 1, "b": "x y", "extra": 9}, {"a": 2.5, "b": "q\tr"}], ["a", "b"])
    assert out.read_bytes() == b'a\tb\r\n1\tx y\r\n2.5\t"q\tr"\r\n'


def test_write_rows_tsv_bytes(tmp_path, capsys):
    out = tmp_path / "r.tsv"
    ac.write_rows_tsv(out, ["k", "v"], [["a", 1], ["b", 0.25]])
    assert out.read_bytes() == b"k\tv\r\na\t1\r\nb\t0.25\r\n"
    assert capsys.readouterr().out == f"wrote {out}\n"


def test_write_header_and_hash_commented_tsv_bytes(tmp_path):
    h = tmp_path / "h.tsv"
    ac.write_header_tsv(h, ["x", "y"], [{"x": 1}, {"x": 2, "y": 0.5}])
    assert h.read_bytes() == b"x\ty\n1\t\n2\t0.5\n"
    c = tmp_path / "c.tsv"
    ac.write_hash_commented_tsv(c, [], "line one\nline two")
    assert c.read_bytes() == b"# line one\n# line two\n(empty)\n"


def test_paired_bootstrap_is_seed_deterministic():
    g, d = [0.1, 0.2, 0.3, 0.4], [0.2, 0.25, 0.5, 0.45]
    a = ac.paired_bootstrap(g, d, 500, random.Random(56))
    b = ac.paired_bootstrap(g, d, 500, random.Random(56))
    assert a == b
    assert a["n_pairs"] == 4
    assert a["mean_diff"] == 0.1
    assert ac.paired_bootstrap([], [], 10, random.Random(0))["p_le0"] == 1.0


def test_auc_rank_and_ccf():
    labels = np.array([0, 0, 1, 1])
    assert ac.auc_rank(labels, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert np.isnan(ac.auc_rank(np.array([0, 0]), np.array([0.1, 0.2])))
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ccf = ac.ccf_at_lags(x, x, 1)
    assert ccf.shape == (3,)
    assert abs(ccf[1] - 1.0) < 1e-12


def test_normaldist_cdf_matches_erf_closure_bitwise():
    cdf = NormalDist().cdf
    for i in range(-2000, 2001):
        x = i / 100
        assert cdf(x) == 0.5 * (1 + erf(x / sqrt(2)))


def test_auroc_argsort_ranks_tie_example_and_bootstrap():
    # Ties are ranked in argsort order, not averaged (a known quirk kept for provenance).
    assert ac.auroc_argsort_ranks(np.array([1, 0, 0, 1]), np.full(4, 0.5)) == 0.5
    assert ac.auroc_argsort_ranks(np.array([0, 1]), np.array([0.5, 0.5])) == 1.0
    assert np.isnan(ac.auroc_argsort_ranks(np.array([1, 1]), np.array([0.1, 0.2])))
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9])
    a = ac.bootstrap_ci(y, s, np.random.default_rng(134), B=200)
    assert a == ac.bootstrap_ci(y, s, np.random.default_rng(134), B=200)
    assert 0.0 <= a[0] <= a[1] <= 1.0


def test_rankdata_avg_averages_ties():
    assert ac.rankdata_avg([3.0, 1.0, 3.0, 2.0]) == [3.5, 1.0, 3.5, 2.0]
