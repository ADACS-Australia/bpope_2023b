import warnings

import jax.numpy as jnp
import numpy as np
import pytest
from jaxoplanet.experimental.starry.light_curve.ylm import light_curve
from jaxoplanet.test_utils import assert_allclose


@pytest.mark.parametrize("l_max", [5, 4, 3, 2, 1, 0])
@pytest.mark.parametrize("n", [500, 1, 0])
def test_compare_starry_light_curve(l_max, n):
    starry = pytest.importorskip("starry")
    starry.config.lazy = False
    theano = pytest.importorskip("theano")
    theano.config.gcc__cxxflags += " -fexceptions"

    ro = 0.1
    xo = jnp.linspace(0, ro + 2, n)
    yo = jnp.zeros(n)
    zo = jnp.ones(n)
    inc = 0
    obl = np.pi / 2
    theta = jnp.linspace(0, np.pi, n)

    # starry map.flux takes theta in the unit of degree
    jnp.linspace(0, 180, n)
    n_max = (l_max + 1) ** 2
    y = np.random.uniform(0, 1, n_max)
    y[0] = 1.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = starry.Map(l_max)
        expect = m.ops.flux(theta, xo, yo, zo, ro, inc, obl, y, m._u, m._f) * (
            0.5 * np.sqrt(np.pi)
        )

    calc = light_curve(l_max, inc, obl, y, xo, yo, zo, ro, theta)

    assert_allclose(calc, expect)
