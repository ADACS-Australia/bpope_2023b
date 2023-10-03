import itertools
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxoplanet.experimental.starry.light_curve.inference import (
    cast,
    design_matrix,
    get_lnlike,
    get_lnlike_woodbury,
    lnlike,
    map_solve,
    set_data,
    set_prior,
    solve,
)
from jaxoplanet.experimental.starry.light_curve.ylm import light_curve
from jaxoplanet.test_utils import assert_allclose
from scipy.stats import multivariate_normal

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


@pytest.fixture(autouse=True)
def data():
    # Generate a synthetic light curve with just a little noise
    l_max = 1
    ro = 0.1
    xo = np.linspace(0, ro + 2, 500)
    yo = np.zeros(500)
    zo = np.linspace(0, ro + 2, 500)
    inc = np.pi / 3
    obl = np.pi / 2
    theta = np.linspace(0, np.pi, 500)
    n_max = (l_max + 1) ** 2
    y = np.random.uniform(0, 1, n_max)
    y[0] = 1.0
    kwargs = dict(l_max=l_max, obl=obl, y=y, xo=xo, yo=yo, zo=zo, ro=ro, theta=theta)

    true_flux = light_curve(l_max, inc, obl, y, xo, yo, zo, ro, theta)

    sigma = 1e-5
    np.random.seed(1)
    syn_flux = true_flux + np.random.randn(len(theta)) * sigma

    X = design_matrix(l_max, inc, obl, y, xo, yo, zo, ro, theta)

    return (l_max, n_max, syn_flux, sigma, y, X, kwargs)


# Parameter combinations used in tests.
vals = [0, 1, 2, 3]
woodbury = [False, True]
solve_inputs = itertools.product(vals, vals)
lnlike_inputs = itertools.product(vals, vals, woodbury)


@pytest.mark.parametrize("L,C", solve_inputs)
def test_map_solve(L, C, data):
    l_max, n_max, syn_flux, sigma, y, X, _ = data

    # Place a generous prior on the map coefficients
    if L == 1:  # scalar
        calc_mu, calc_L = set_prior(l_max, L=1)
    elif L == 2:  # vector
        calc_mu, calc_L = set_prior(l_max, L=np.ones(n_max))
    elif L == 3:  # matrix
        calc_mu, calc_L = set_prior(l_max, L=np.eye(n_max))
    elif L == 0:  # cholesky
        calc_mu, calc_L = set_prior(l_max, cho_L=np.eye(n_max))

    # Provide the dataset
    if C == 1:  # scalar
        _, calc_C = set_data(syn_flux, C=sigma**2)
    elif C == 2:  # vector
        _, calc_C = set_data(syn_flux, C=np.ones(len(syn_flux)) * sigma**2)
    elif C == 3:  # matrix
        _, calc_C = set_data(syn_flux, C=np.eye(len(syn_flux)) * sigma**2)
    elif C == 0:  # cholesky
        _, calc_C = set_data(syn_flux, cho_C=np.eye(len(syn_flux)) * sigma)

    # Solve the linear problem
    mu, cho_cov = map_solve(X, syn_flux, calc_C[1], calc_mu, calc_L[2])

    # Ensure the likelihood of the true value is close to that of
    # the MAP solution
    cov = np.dot(cho_cov, np.transpose(cho_cov))
    LnL0 = multivariate_normal.logpdf(mu, mean=mu, cov=cov)
    LnL = multivariate_normal.logpdf(y, mean=mu, cov=cov)
    assert LnL0 - LnL < 5.00


def test_map_solve_scalar(data):
    l_max, _, syn_flux, sigma, y, X, _ = data

    # Place a generous prior on the map coefficients
    calc_mu, calc_L = set_prior(l_max, L=1)

    # Provide the dataset
    _, calc_C = set_data(syn_flux, C=sigma**2)

    # Solve the linear problem
    mu, cho_cov = map_solve(X, syn_flux, calc_C[1], calc_mu, calc_L[2])

    # Ensure the likelihood of the true value is close to that of
    # the MAP solution
    cov = np.dot(cho_cov, np.transpose(cho_cov))
    LnL0 = multivariate_normal.logpdf(mu, mean=mu, cov=cov)
    LnL = multivariate_normal.logpdf(y, mean=mu, cov=cov)
    assert LnL0 - LnL < 5.00


@pytest.mark.parametrize("L,C,woodbury", lnlike_inputs)
def test_lnlike(L, C, woodbury, data):
    """Test the log marginal likelihood method."""

    l_max, n_max, syn_flux, sigma, _, _, kwargs = data

    # Place a generous prior on the map coefficients
    # and compute prior covariance and inverse covariance matrices
    if L == 1:  # scalar
        calc_mu, calc_L = set_prior(l_max, L=1)
        L = calc_L[0] * jnp.ones(n_max)
        LInv = calc_L[2] * jnp.ones(n_max)
    elif L == 2:  # vector
        calc_mu, calc_L = set_prior(l_max, L=np.ones(n_max))
        LInv = calc_L[2] * jnp.ones(n_max)
        L = calc_L[0] * jnp.ones(n_max)
    elif L == 3:  # matrix
        calc_mu, calc_L = set_prior(l_max, L=np.eye(n_max))
        L = jax.scipy.linalg.block_diag(*[calc_L[0] * jnp.eye(n_max)])
        LInv = jax.scipy.linalg.block_diag(*[calc_L[2] * jnp.eye(n_max)])
    elif L == 0:  # cholesky
        calc_mu, calc_L = set_prior(l_max, cho_L=np.eye(n_max))
        L = jax.scipy.linalg.block_diag(*[calc_L[0] * jnp.eye(n_max)])
        LInv = jax.scipy.linalg.block_diag(*[calc_L[2] * jnp.eye(n_max)])

    # Provide the dataset
    if C == 1:  # scalar
        _, calc_C = set_data(syn_flux, C=sigma**2)
    elif C == 2:  # vector
        _, calc_C = set_data(syn_flux, C=np.ones(len(syn_flux)) * sigma**2)
    elif C == 3:  # matrix
        _, calc_C = set_data(syn_flux, C=np.eye(len(syn_flux)) * sigma**2)
    elif C == 0:  # cholesky
        _, calc_C = set_data(syn_flux, cho_C=np.eye(len(syn_flux)) * sigma)

    # Compute the marginal log likelihood for different inclinations
    incs = [0, np.pi / 12, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2]
    ll = np.zeros_like(incs, dtype=float)
    for i, inc in enumerate(incs):
        X = design_matrix(inc=inc, **kwargs)
        if woodbury is True:
            lndetL = cast([calc_L[3]])
            ll[i] = get_lnlike_woodbury(
                X, syn_flux, calc_C[2], calc_mu, LInv, calc_C[3], lndetL
            )
        else:
            ll[i] = get_lnlike(X, syn_flux, calc_C[0], calc_mu, L)

    # Verify that we get the correct inclination
    assert incs[np.argmax(ll)] == np.pi / 3
    assert_allclose(ll[np.argmax(ll)], 5002.211, rtol=1e-5)  # benchmarked


@pytest.mark.parametrize("woodbury", [True, False])
def test_lnlike_scalar(woodbury, data):
    l_max, n_max, syn_flux, sigma, _, _, kwargs = data

    # Place a generous prior on the map coefficients
    calc_mu, calc_L = set_prior(l_max, L=1)
    L = calc_L[0] * jnp.ones(n_max)
    LInv = jnp.concatenate([calc_L[2] * jnp.ones(n_max)])
    lndetL = cast([calc_L[3]])

    # Provide the dataset
    _, calc_C = set_data(syn_flux, C=sigma**2)

    # Compute the marginal log likelihood for different inclinations
    incs = [0, np.pi / 12, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2]
    ll = np.zeros_like(incs, dtype=float)
    for i, inc in enumerate(incs):
        X = design_matrix(inc=inc, **kwargs)
        if woodbury is True:
            ll[i] = get_lnlike_woodbury(
                X, syn_flux, calc_C[2], calc_mu, LInv, calc_C[3], lndetL
            )
        else:
            ll[i] = get_lnlike(X, syn_flux, calc_C[0], calc_mu, L)

    # Verify that we get the correct inclination
    assert incs[np.argmax(ll)] == np.pi / 3
    assert_allclose(ll[np.argmax(ll)], 5002.211, rtol=1e-5)  # benchmarked


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
    assert_allclose(calc_flux, expect_flux)  # flux
    assert_allclose(calc_C[0], expect_C[0])  # value
    assert_allclose(calc_C[1], expect_C[1])  # cholesky
    assert_allclose(calc_C[2], expect_C[2])  # inverse
    assert_allclose(calc_C[3], expect_C[3])  # lndet
    assert_allclose(calc_C[4], kind[expect_C[4]])  # kind
    assert_allclose(calc_C[5], expect_C[5])  # N


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
    assert_allclose(calc_mu, expect_mu)  # mu
    assert_allclose(calc_L[0], expect_L[0])  # value
    assert_allclose(calc_L[1], expect_L[1])  # cholesky
    assert_allclose(calc_L[2], expect_L[2])  # inverse
    assert_allclose(calc_L[3], expect_L[3])  # lndet
    assert_allclose(calc_L[4], kind[expect_L[4]])  # kind
    assert_allclose(calc_L[5], expect_L[5])  # N


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

    assert_allclose(calc_x, expect_x)
    assert_allclose(calc_cho_cov, expect_cho_cov)


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

    assert_allclose(calc_ln, expect_ln)
