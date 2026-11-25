import numpy as np
from anomaly import *


def test_truth_count():
    _, t = make_data()
    assert t.sum() == 15


def test_zscore_flags_obvious_outlier():
    x = np.r_[np.random.default_rng(1).normal(0, 1, 200), 50.0]
    assert zscore(x)[-1]


def test_iqr_flags_obvious_outlier():
    assert iqr(np.r_[np.arange(20.0), 500.0])[-1]


def test_prf_perfect():
    t = np.array([True, False, True])
    assert prf(t, t) == (1.0, 1.0, 1.0)


def test_iforest_shape():
    x, _ = make_data()
    assert iforest(x).shape == x.shape
