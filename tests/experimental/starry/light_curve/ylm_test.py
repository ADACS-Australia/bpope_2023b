import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.config import config
from jaxoplanet.experimental.starry.custom_jvp_rules import (
    zero_safe_arctan2,
    zero_safe_sqrt,
)
from jaxoplanet.experimental.starry.light_curve.ylm import design_matrix, light_curve
from jaxoplanet.test_utils import assert_allclose

config.update("jax_enable_x64", True)


@pytest.mark.parametrize("l_max", [5, 4, 3, 2, 1, 0])
@pytest.mark.parametrize("n", [500, 1])
def test_compare_starry_design_matrix(l_max, n):
    starry = pytest.importorskip("starry")
    starry.config.lazy = False
    theano = pytest.importorskip("theano")
    theano.config.gcc__cxxflags += " -fexceptions"

    ro = 0.1
    xo = jnp.linspace(0, ro + 2, n)
    yo = jnp.zeros(n)
    zo = jnp.linspace(-2, 2, n)
    inc = 0
    obl = np.pi / 2
    theta = jnp.linspace(0, np.pi, n)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = starry.Map(l_max)
        expect = m.ops.X(theta, xo, yo, zo, ro, inc, obl, m._u, m._f) * (
            0.5 * np.sqrt(np.pi)
        )
        # print("expect: ", expect)

    # prepare inputs
    bo = zero_safe_sqrt(xo**2 + yo**2)
    theta_z = zero_safe_arctan2(xo, yo)

    calc = design_matrix(l_max, inc, obl)(bo, zo, ro, theta, theta_z)
    assert_allclose(expect, calc)


@pytest.mark.parametrize("l_max", [5, 4, 3, 2, 1, 0])
@pytest.mark.parametrize("n", [500, 1])
def test_compare_starry_light_curve(l_max, n):
    starry = pytest.importorskip("starry")
    starry.config.lazy = False
    theano = pytest.importorskip("theano")
    theano.config.gcc__cxxflags += " -fexceptions"

    ro = 0.1
    xo = jnp.linspace(0, ro + 2, n)
    yo = jnp.zeros(n)
    zo = jnp.linspace(-2, 2, n)
    inc = 0
    obl = np.pi / 2
    theta = jnp.linspace(0, np.pi, n)

    n_max = (l_max + 1) ** 2
    y = np.random.uniform(0, 1, n_max)
    y[0] = 1.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = starry.Map(l_max)
        expect = m.ops.flux(theta, xo, yo, zo, ro, inc, obl, y, m._u, m._f) * (
            0.5 * np.sqrt(np.pi)
        )

    # prepare inputs
    bo = zero_safe_sqrt(xo**2 + yo**2)
    theta_z = zero_safe_arctan2(xo, yo)

    calc = light_curve(l_max, inc, obl, bo, zo, ro, theta, theta_z, y)
    assert_allclose(calc, expect)


# @pytest.mark.parametrize("l_max", list(range(0,17,3)))
@pytest.mark.parametrize("theta", [0.0, np.pi / 4, np.pi / 2, np.pi])
@pytest.mark.parametrize(
    "xyz", [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, -1.0), (1.0, 1.0, 10.0)]
)
@pytest.mark.parametrize("ro", [0.01, 0.1, 0.5, 1.1, 2.0])
def test_grad(theta, xyz, ro):
    l_max = 5
    inc = 0.1
    obl = np.pi / 2
    n_max = (l_max + 1) ** 2
    y = np.random.uniform(0, 1, n_max)
    y[0] = 1.0

    xo, yo, zo = xyz

    bo = zero_safe_sqrt(xo**2 + yo**2)
    theta_z = zero_safe_arctan2(xo, yo)

    for n in range(1, 9):
        lc_grad = jax.grad(light_curve, argnums=n)(
            l_max,
            inc,
            obl,
            bo,
            zo,
            ro,
            theta,
            theta_z,
            y,
        )
        assert np.all(np.isfinite(lc_grad))
