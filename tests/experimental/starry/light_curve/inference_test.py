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

    np.testing.assert_allclose(calc_flux, expect_flux, atol=1e-12)  # flux
    np.testing.assert_allclose(calc_C[0], expect_C[0], atol=1e-12)  # value
    np.testing.assert_allclose(calc_C[1], expect_C[1], atol=1e-12)  # cholesky
    np.testing.assert_allclose(calc_C[2], expect_C[2], atol=1e-12)  # inverse
    np.testing.assert_allclose(calc_C[3], expect_C[3], atol=1e-12)  # lndet
    np.testing.assert_allclose(calc_C[5], expect_C[5], atol=1e-12)  # kind
    np.testing.assert_string_equal(calc_C[4], expect_C[4])  # N


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

    np.testing.assert_allclose(calc_mu, expect_mu, atol=1e-12)  # mu
    np.testing.assert_allclose(calc_L[0], expect_L[0], atol=1e-12)  # value
    np.testing.assert_allclose(calc_L[1], expect_L[1], atol=1e-12)  # cholesky
    np.testing.assert_allclose(calc_L[2], expect_L[2], atol=1e-12)  # inverse
    np.testing.assert_allclose(calc_L[3], expect_L[3], atol=1e-12)  # lndet
    np.testing.assert_string_equal(calc_L[4], expect_L[4])  # N
    np.testing.assert_allclose(calc_L[5], expect_L[5], atol=1e-12)  # kind


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

        design_matrix = sys.design_matrix(t=t)

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

    (calc_x, calc_cho_cov) = solve(
        lmax, flux, expect_C, bodies, design_matrix=design_matrix
    )

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

        design_matrix = sys.design_matrix(t=t)

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

    calc_ln = lnlike(
        lmax, flux, expect_C, bodies, design_matrix=design_matrix, woodbury=woodbury
    )

    np.testing.assert_allclose(calc_ln, expect_ln, atol=1e-12)
