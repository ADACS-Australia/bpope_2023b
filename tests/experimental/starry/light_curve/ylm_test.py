import warnings

import numpy as np
import pytest
from jaxoplanet.experimental.starry.light_curve.ylm import X, light_curve
from jaxoplanet.test_utils import assert_allclose


@pytest.mark.parametrize("l_max", [5, 4, 3, 2, 1, 0])
def test_compare_starry_X(l_max):
    starry = pytest.importorskip("starry")
    theano = pytest.importorskip("theano")
    theano.config.gcc__cxxflags += " -fexceptions"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        m = starry.Map(l_max)

        theta = np.linspace(0, 360, 10)
        xo = np.zeros(10)
        yo = np.zeros(10)
        zo = np.ones(10)
        ro = 0.0

        inc = 90
        obl = 0

        theta_tt, xo_tt, yo_tt, zo_tt = m._math.vectorize(theta, xo, yo, zo)
        theta_tt, xo_tt, yo_tt, zo_tt, ro_tt = m._math.cast(
            theta_tt, xo_tt, yo_tt, zo_tt, ro
        )
        expect = m.ops.X(
            theta_tt, xo_tt, yo_tt, zo_tt, ro_tt, inc, obl, m._u, m._f
        ).eval() * (0.5 * np.sqrt(np.pi))

        calc = X(l_max, theta, xo, yo, zo, ro, inc, obl)

        assert_allclose(calc, expect)


@pytest.mark.parametrize("l_max", [5, 4, 3, 2, 1, 0])
def test_compare_starry_light_curve(l_max):
    starry = pytest.importorskip("starry")
    starry.config.lazy = False
    theano = pytest.importorskip("theano")
    theano.config.gcc__cxxflags += " -fexceptions"

    theta = np.linspace(0, 360, 10)
    xo = np.zeros(10)
    yo = np.zeros(10)
    zo = np.ones(10)
    ro = 0.0

    inc = 90
    obl = 0

    N = (l_max + 1) ** 2
    y = np.random.uniform(-5, 5, N)
    y[0] = 1.0

    m = starry.Map(l_max)

    # theta_tt, xo_tt, yo_tt, zo_tt, y_tt = m._math.vectorize(theta, xo, yo, zo, y)
    # theta_tt, xo_tt, yo_tt, zo_tt, ro_tt = m._math.cast(
    #         theta_tt, xo_tt, yo_tt, zo_tt, ro
    # )
    expect = m.ops.flux(theta, xo, yo, zo, ro, inc, obl, y, m._u, m._f) * (
        0.5 * np.sqrt(np.pi)
    )
    print("expect: ", expect)
    calc = light_curve(l_max, theta, xo, yo, zo, ro, inc, obl, y)

    assert_allclose(calc, expect)
