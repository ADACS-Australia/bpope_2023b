import jax.numpy as jnp
import numpy as np
import pytest
from jaxoplanet.experimental.starry.light_curve.utils import right_project
from jaxoplanet.test_utils import assert_allclose


@pytest.mark.parametrize("l_max", [10, 7, 5, 4])
def test_compare_starry_right_project(l_max):
    """Comparison test with starry OpsYlm.right_project"""
    starry = pytest.importorskip("starry")
    theta = np.arange(0.1, np.pi, 0.1)
    inc = 0
    obl = 90
    np.random.seed(l_max)
    n_max = l_max**2 + 2 * l_max + 1
    M = np.random.rand(theta.shape[0], n_max)
    M = jnp.array(M)
    m = starry._core.core.OpsYlm(l_max, 0, 0, 1)
    expected = m.right_project(M, inc, obl, theta)
    calc = right_project(M, inc, obl, theta)
    assert_allclose(calc, expected)
