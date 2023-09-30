import warnings

import numpy as np
import pytest
from jaxoplanet.experimental.starry.light_curve.inference import (
    lnlike,
    set_data,
    set_prior,
    solve,
)

"""
Cases tested against starry
---------------------------

set_data(flux, C=None, cho_C=None)

    - tested with C as a scalar, vector
    - NOT tested with C as a matrix
    - NOT tested with cho_C

set_prior(lmax, mu, L)

    - tested 3x different lmax
    - tested mu as default, scalar, vector
    - tested L as scalar, vector
    - NOT tested L as a matrix

solve(lmax, flux, C, bodies, design_matrix=None, t=None)

    - tested 3x different lmax
    - tested with design_matrix
    - NOT tested with t (NOT YET implemented)

lnlike(lmax, flux, C, bodies, design_matrix=None, t=None, woodbury=True)

    - tested 3x different lmax
    - tested with design_matrix
    - NOT tested with t (NOT YET implemented)
    - tested woodbury as [True, False]
"""


# TODO: get design matrix test passing
# @pytest.mark.parametrize("l_max", [5, 4, 3, 2, 1, 0])
# def test_compare_starry_design_matrix(l_max):
#     starry = pytest.importorskip("starry")
#     starry.config.lazy = False
#     theano = pytest.importorskip("theano")
#     theano.config.gcc__cxxflags += " -fexceptions"

#     ro = 0.1
#     xo = jnp.linspace(0, ro + 2, 500)
#     yo = jnp.zeros(500)
#     zo = jnp.linspace(0, ro + 2, 500)
#     inc = 0
#     obl = np.pi / 2
#     theta = jnp.linspace(0, np.pi, 500)
#     n_max = (l_max + 1) ** 2
#     y = np.random.uniform(0, 1, n_max)
#     y[0] = 1.0

#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         m = starry.Map(l_max)
#         expect = m.ops.X(theta, xo, yo, zo, ro, inc, obl, m._u, m._f)

#     calc = design_matrix(l_max, inc, obl, y, xo, yo, zo, ro, theta)

#     np.testing.assert_allclose(calc, expect, atol=1e-12)


# @pytest.fixture(autouse=True)
# def data():

#     # Generate a synthetic light curve with just a little noise
#     l_max = 5
#     ro = 0.1
#     xo = np.linspace(0, ro + 2, 500)
#     yo = np.zeros(500)
#     zo = np.zeros(500)
#     inc = 0
#     obl = np.pi / 2
#     theta = np.linspace(0, np.pi, 500)
#     n_max = (l_max + 1) ** 2
#     y = np.random.uniform(0, 1, n_max)
#     y[0] = 1.0
#     # kwargs = dict(theta=theta, xo=xo, yo=yo, zo=zo)

#     true_flux = light_curve(l_max, inc, obl, y, xo, yo, zo, ro, theta)

#     sigma = 1e-5
#     np.random.seed(1)
#     syn_flux = true_flux + np.random.randn(len(theta)) * sigma

#     # Get design matrix from starry (not yet in jaxoplanet)
#     starry.config.lazy = False
#     m = starry.Map(l_max)
#     # starry_flux = m.ops.flux(theta, xo, yo, zo, ro, inc, obl, y, m._u, m._f) * (
#     #     0.5 * np.sqrt(np.pi)
#     # )
#     # X = m.design_matrix()

#     X = m.design_matrix(theta=theta, xo=xo, yo=yo, zo=zo, ro=ro)

#     return (l_max, true_flux, syn_flux, sigma, y, X)


# # val = ["scalar"]
# # woodbury = [False, True]
# # solve_inputs: val
# # lnlike_inputs = itertools.product(val, woodbury)


# # @pytest.mark.parametrize("L,C", solve_inputs)
# def test_map_solve(data):

#     l_max, true_flux, syn_flux, sigma, y, X = data

#     # Place a generous prior on the map coefficients
#     (calc_mu, calc_L) = set_prior(l_max, L=1)

#     # Provide the dataset
#     (calc_flux, calc_C) = set_data(syn_flux, C=sigma ** 2)

#     # Solve the linear problem
#     mu, cho_cov = map_solve(X, syn_flux, calc_C[1], calc_mu, calc_L[2])

#     # Ensure the likelihood of the true value is close to that of
#     # the MAP solution
#     cov = np.dot(cho_cov, np.transpose(cho_cov))
#     LnL0 = multivariate_normal.logpdf(mu, mean=mu, cov=cov)
#     LnL = multivariate_normal.logpdf(y, mean=mu, cov=cov)
#     assert LnL0 - LnL < 5.00

#     # Check that we can draw from the posterior
#     # map.draw()


@pytest.mark.parametrize("lmax", [10, 7, 5])
@pytest.mark.parametrize("C", [2.5e-07, "vector"])
def test_compare_starry_set_data(lmax, C):
    starry = pytest.importorskip("starry")
    Ny = (lmax + 1) * (lmax + 1)

    if C == "vector":
        np.random.seed(12)
        C = np.random.default_rng().uniform(low=1e-9, high=1e-6, size=Ny)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        starry.config.lazy = False
        starry.config.quiet = True

        pri = starry.Primary(starry.Map(ydeg=lmax))
        sec = starry.Secondary(starry.Map(ydeg=lmax), porb=1.00)
        sys = starry.System(pri, sec)
        t = np.linspace(-2.5, 2.5, 1000)
        flux = sys.flux(t)

        sys.set_data(flux, C=C)
        expect_flux = sys._flux
        expect_C = (
            sys._C.value,
            sys._C.cholesky,
            sys._C.inverse,
            sys._C.lndet,
            sys._C.kind,
            sys._C.N,
        )

    (calc_flux, calc_C) = set_data(flux, C=C)

    kind = {"cholesky": 0, "scalar": 1, "vector": 2, "matrix": 3}
    np.testing.assert_allclose(calc_flux, expect_flux, atol=1e-12)  # flux
    np.testing.assert_allclose(calc_C[0], expect_C[0], atol=1e-12)  # value
    np.testing.assert_allclose(calc_C[1], expect_C[1], atol=1e-12)  # cholesky
    np.testing.assert_allclose(calc_C[2], expect_C[2], atol=1e-12)  # inverse
    np.testing.assert_allclose(calc_C[3], expect_C[3], atol=1e-12)  # lndet
    np.testing.assert_allclose(calc_C[4], kind[expect_C[4]], atol=1e-12)  # kind
    np.testing.assert_allclose(calc_C[5], expect_C[5], atol=1e-12)  # N


@pytest.mark.parametrize("lmax", [10, 7, 5])
@pytest.mark.parametrize("pri_mu", [None, 0.1, "vector"])
@pytest.mark.parametrize("pri_L", [1e-2, "vector"])
def test_compare_starry_set_prior(lmax, pri_mu, pri_L):
    starry = pytest.importorskip("starry")
    Ny = (lmax + 1) * (lmax + 1)

    if pri_mu == "vector":
        np.random.seed(12)
        pri_mu = np.random.default_rng().uniform(low=0.0, high=0.3, size=Ny)
        pri_mu[0] = 1.0

    if pri_L == "vector":
        pri_L = np.tile(1e-2, Ny)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        starry.config.lazy = False
        starry.config.quiet = True
        pri = starry.Primary(starry.Map(ydeg=lmax))
        sec = starry.Secondary(starry.Map(ydeg=lmax), porb=1.00)
        starry.System(pri, sec)

        pri.map.set_prior(mu=pri_mu, L=pri_L)
        expect_mu = pri.map._mu
        expect_L = (
            pri.map._L.value,
            pri.map._L.cholesky,
            pri.map._L.inverse,
            pri.map._L.lndet,
            pri.map._L.kind,
            pri.map._L.N,
        )

    (calc_mu, calc_L) = set_prior(lmax, mu=pri_mu, L=pri_L)

    kind = {"cholesky": 0, "scalar": 1, "vector": 2, "matrix": 3}
    np.testing.assert_allclose(calc_mu, expect_mu, atol=1e-12)  # mu
    np.testing.assert_allclose(calc_L[0], expect_L[0], atol=1e-12)  # value
    np.testing.assert_allclose(calc_L[1], expect_L[1], atol=1e-12)  # cholesky
    np.testing.assert_allclose(calc_L[2], expect_L[2], atol=1e-12)  # inverse
    np.testing.assert_allclose(calc_L[3], expect_L[3], atol=1e-12)  # lndet
    np.testing.assert_allclose(calc_L[4], kind[expect_L[4]], atol=1e-12)  # kind
    np.testing.assert_allclose(calc_L[5], expect_L[5], atol=1e-12)  # N


@pytest.mark.parametrize("lmax", [10, 7, 5])
def test_compare_starry_solve(lmax):
    starry = pytest.importorskip("starry")
    Ny = (lmax + 1) * (lmax + 1)
    pri_mu = None
    pri_L = np.tile(1e-2, Ny)
    C = 2.5e-07

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        starry.config.lazy = False
        starry.config.quiet = True

        pri = starry.Primary(starry.Map(ydeg=lmax))
        sec = starry.Secondary(starry.Map(ydeg=lmax), porb=1.00)
        sys = starry.System(pri, sec)

        pri.map.set_prior(mu=pri_mu, L=pri_L)

        t = np.linspace(-2.5, 2.5, 1000)
        flux = sys.flux(t)
        sys.set_data(flux, C=C)
        expect_C = (
            sys._C.value,
            sys._C.cholesky,
            sys._C.inverse,
            sys._C.lndet,
            sys._C.kind,
            sys._C.N,
        )

        (expect_x, expect_cho_cov) = sys.solve(t=t)

        dm = sys.design_matrix(t=t)

    pri_L = dict(
        value=pri.map._L.value,
        cholesky=pri.map._L.cholesky,
        inverse=pri.map._L.inverse,
        lndet=pri.map._L.lndet,
        N=pri.map._L.N,
        kind=pri.map._L.kind,
    )
    pri_body = dict(
        mu=pri.map._mu,
        L=pri_L,
        amp=pri.map.amp,
        y=pri.map.y,
        n_max=pri.map.Ny,
        l_max=pri.map.ydeg,
    )
    sec_body = dict(
        mu=sec.map._mu,
        L=None,
        amp=sec.map.amp,
        y=sec.map.y,
        n_max=sec.map.Ny,
        l_max=sec.map.ydeg,
    )
    bodies = [pri_body, sec_body]

    (calc_x, calc_cho_cov) = solve(lmax, flux, expect_C, bodies, design_matrix=dm)

    np.testing.assert_allclose(calc_x, expect_x, atol=1e-9)
    np.testing.assert_allclose(calc_cho_cov, expect_cho_cov, atol=1e-11)


@pytest.mark.parametrize("lmax", [10, 7, 5])
@pytest.mark.parametrize("woodbury", [True, False])
def test_compare_starry_lnlike(lmax, woodbury):
    starry = pytest.importorskip("starry")
    Ny = (lmax + 1) * (lmax + 1)
    pri_mu = None
    pri_L = np.tile(1e-2, Ny)
    C = 2.5e-07

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        starry.config.lazy = False
        starry.config.quiet = True

        pri = starry.Primary(starry.Map(ydeg=lmax))
        sec = starry.Secondary(starry.Map(ydeg=lmax), porb=1.00)
        sys = starry.System(pri, sec)

        t = np.linspace(-2.5, 2.5, 1000)
        flux = sys.flux(t)
        sys.set_data(flux, C=C)
        expect_C = (
            sys._C.value,
            sys._C.cholesky,
            sys._C.inverse,
            sys._C.lndet,
            sys._C.kind,
            sys._C.N,
        )

        pri.map.set_prior(mu=pri_mu, L=pri_L)

        expect_ln = sys.lnlike(t=t, woodbury=woodbury)

        dm = sys.design_matrix(t=t)

    pri_L = dict(
        value=pri.map._L.value,
        cholesky=pri.map._L.cholesky,
        inverse=pri.map._L.inverse,
        lndet=pri.map._L.lndet,
        N=pri.map._L.N,
        kind=pri.map._L.kind,
    )
    pri_body = dict(
        mu=pri.map._mu, L=pri_L, amp=pri.map.amp, y=pri.map.y, n_max=pri.map.Ny
    )
    sec_body = dict(
        mu=sec.map._mu, L=None, amp=sec.map.amp, y=sec.map.y, n_max=sec.map.Ny
    )
    bodies = [pri_body, sec_body]

    calc_ln = lnlike(lmax, flux, expect_C, bodies, design_matrix=dm, woodbury=woodbury)

    np.testing.assert_allclose(calc_ln, expect_ln, atol=1e-12)
