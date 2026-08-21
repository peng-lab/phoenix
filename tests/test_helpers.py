import numpy as np
import pytest
from scipy.sparse import csr_matrix

from phoenix.helpers.demo_plot import norm01, to_dense


def test_to_dense_passes_through_dense_arrays():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = to_dense(x)
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, x)


def test_to_dense_converts_sparse_matrices():
    x = csr_matrix(np.array([[0.0, 2.0], [3.0, 0.0]]))
    out = to_dense(x)
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, [[0.0, 2.0], [3.0, 0.0]])


def test_norm01_rescales_to_unit_range():
    x = np.array([0.0, 1.0, 2.0, 3.0, 100.0])  # 100.0 is an outlier clipped by the quantile
    out = norm01(x, q=0.8)
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)
    assert (out >= 0).all() and (out <= 1).all()


def test_norm01_handles_constant_input():
    x = np.full(5, 3.0)
    out = norm01(x)
    # lo == hi is nudged apart internally so this shouldn't divide by zero
    assert np.isfinite(out).all()
