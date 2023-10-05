import warnings

import jax.numpy as jnp
import numpy as np
import pytest
from jaxoplanet.experimental.starry.light_curve.utils import right_project
from jaxoplanet.test_utils import assert_allclose


@pytest.mark.parametrize("l_max", [10, 7, 5, 4, 3, 2, 1, 0])
def test_compare_starry_right_project(l_max):
    """Comparison test with starry OpsYlm.right_project"""
    starry = pytest.importorskip("starry")
    theano = pytest.importorskip("theano")
    theano.config.gcc__cxxflags += " -fexceptions"
    starry.config.lazy = False
    n_t = 50

    inc = 0
    obl = jnp.pi / 2

    theta = jnp.linspace(0, jnp.pi, n_t)

    np.random.seed(l_max)
    n_max = l_max**2 + 2 * l_max + 1
    M = np.random.rand(n_t, n_max)
    M = jnp.array(M)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = starry.Map(l_max)
        expected = m.ops.right_project(M, inc, obl, theta).squeeze()
    calc = right_project(l_max, M, inc, obl, theta)
    print("shape of result: ", expected.shape)
    assert_allclose(calc, expected)


@pytest.mark.parametrize("l_max", [5, 4, 3, 2, 1, 0])
@pytest.mark.parametrize("theta", [np.pi / 4, 0])
@pytest.mark.parametrize("n_t", [1, 50])
def test_compare_starry_right_project_edge_case(l_max, theta, n_t):
    """Comparison test with starry OpsYlm.right_project"""
    starry = pytest.importorskip("starry")
    theano = pytest.importorskip("theano")
    theano.config.gcc__cxxflags += " -fexceptions"
    starry.config.lazy = False

    inc = 0
    obl = np.pi / 2

    np.random.seed(l_max)
    n_max = l_max**2 + 2 * l_max + 1
    M = np.random.rand(n_t, n_max)
    # M = jnp.array(M)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = starry.Map(l_max)
        expected = m.ops.right_project(M, inc, obl, theta).squeeze()

    calc = right_project(l_max, M, inc, obl, theta)
    print("shape of result: ", calc.shape)
    print("shape of expected: ", expected.shape)
    assert_allclose(calc, expected)
