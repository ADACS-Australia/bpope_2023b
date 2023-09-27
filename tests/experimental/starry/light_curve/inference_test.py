import warnings

import numpy as np
import pytest
from jaxoplanet.experimental.starry.light_curve.inference import (
    set_data,
    set_prior,
    solve,
)


@pytest.mark.parametrize("lmax", [10, 7, 5])
def test_compare_starry_set_data(lmax):
    starry = pytest.importorskip("starry")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        starry.config.lazy = False
        starry.config.quiet = True
        map = starry.Map(ydeg=lmax)
        map.add_spot(amp=-0.075, sigma=0.1, lat=0, lon=-30)
        A_y = np.array(map.y[1:])
        map.reset()
        map.add_spot(amp=-0.075, sigma=0.1, lat=-30, lon=60)
        B_y = np.array(map.y[1:])
        pri = starry.Primary(
            starry.Map(ydeg=lmax, udeg=2, inc=80.0, amp=1.0),
            r=1.0,
            m=1.0,
            prot=1.25,
        )
        pri.map[1:] = [0.40, 0.25]
        pri.map[1:, :] = A_y
        sec = starry.Secondary(
            starry.Map(ydeg=lmax, udeg=2, inc=80.0, amp=0.1),
            r=0.7,
            m=0.7,
            porb=1.00,
            prot=0.625,
            t0=0.15,
            inc=80.0,
        )
        sec.map[1:] = [0.20, 0.05]
        sec.map[1:, :] = B_y
        sys = starry.System(pri, sec)
        t = np.linspace(-2.5, 2.5, 1000)
        flux_true = sys.flux(t)
        sigma = 0.0005
        flux = flux_true + sigma * np.random.randn(len(t))
        sys.set_data(flux, C=sigma**2)
        expect_flux = sys._flux
        expect_C = (
            sys._C.value,
            sys._C.cholesky,
            sys._C.inverse,
            sys._C.lndet,
            sys._C.kind,
            sys._C.N,
        )
    (calc_flux, calc_C) = set_data(flux, C=sigma**2)
    np.testing.assert_allclose(calc_flux, expect_flux, atol=1e-12)  # flux
    np.testing.assert_allclose(calc_C[0], expect_C[0], atol=1e-12)  # value
    np.testing.assert_allclose(calc_C[1], expect_C[1], atol=1e-12)  # cholesky
    np.testing.assert_allclose(calc_C[2], expect_C[2], atol=1e-12)  # inverse
    np.testing.assert_allclose(calc_C[3], expect_C[3], atol=1e-12)  # lndet
    np.testing.assert_allclose(calc_C[5], expect_C[5], atol=1e-12)  # kind
    np.testing.assert_string_equal(calc_C[4], expect_C[4])  # N


@pytest.mark.parametrize("lmax", [10, 7, 5])
def test_compare_starry_set_prior(lmax):
    starry = pytest.importorskip("starry")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        starry.config.lazy = False
        starry.config.quiet = True
        map = starry.Map(ydeg=lmax)
        map.add_spot(amp=-0.075, sigma=0.1, lat=0, lon=-30)
        A_y = np.array(map.y[1:])
        map.reset()
        map.add_spot(amp=-0.075, sigma=0.1, lat=-30, lon=60)
        B_y = np.array(map.y[1:])
        pri = starry.Primary(
            starry.Map(ydeg=lmax, udeg=2, inc=80.0, amp=1.0),
            r=1.0,
            m=1.0,
            prot=1.25,
        )
        pri.map[1:] = [0.40, 0.25]
        pri.map[1:, :] = A_y
        sec = starry.Secondary(
            starry.Map(ydeg=lmax, udeg=2, inc=80.0, amp=0.1),
            r=0.7,
            m=0.7,
            porb=1.00,
            prot=0.625,
            t0=0.15,
            inc=80.0,
        )
        sec.map[1:] = [0.20, 0.05]
        sec.map[1:, :] = B_y
        sys = starry.System(pri, sec)
        t = np.linspace(-2.5, 2.5, 1000)
        flux_true = sys.flux(t)
        sigma = 0.0005
        flux = flux_true + sigma * np.random.randn(len(t))
        sys.set_data(flux, C=sigma**2)
        pri_mu = np.zeros(pri.map.Ny)
        pri_mu[0] = 1.0
        pri_L = np.zeros(pri.map.Ny)
        pri_L[0] = 1e-2
        pri_L[1:] = 1e-2
        pri.map.set_prior(mu=pri_mu, L=pri_L)
        expect_L = (
            pri.map._L.value,
            pri.map._L.cholesky,
            pri.map._L.inverse,
            pri.map._L.lndet,
            pri.map._L.kind,
            pri.map._L.N,
        )
        expect_mu = pri.map._mu
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        starry.config.lazy = False
        starry.config.quiet = True
        map = starry.Map(ydeg=lmax)
        map.add_spot(amp=-0.075, sigma=0.1, lat=0, lon=-30)
        A_y = np.array(map.y[1:])
        map.reset()
        map.add_spot(amp=-0.075, sigma=0.1, lat=-30, lon=60)
        B_y = np.array(map.y[1:])
        pri = starry.Primary(
            starry.Map(ydeg=lmax, udeg=2, inc=80.0, amp=1.0),
            r=1.0,
            m=1.0,
            prot=1.25,
        )
        pri.map[1:] = [0.40, 0.25]
        pri.map[1:, :] = A_y
        sec = starry.Secondary(
            starry.Map(ydeg=lmax, udeg=2, inc=80.0, amp=0.1),
            r=0.7,
            m=0.7,
            porb=1.00,
            prot=0.625,
            t0=0.15,
            inc=80.0,
        )
        sec.map[1:] = [0.20, 0.05]
        sec.map[1:, :] = B_y
        sys = starry.System(pri, sec)
        t = np.linspace(-2.5, 2.5, 1000)
        flux_true = sys.flux(t)
        sigma = 0.0005
        flux = flux_true + sigma * np.random.randn(len(t))
        sys.set_data(flux, C=sigma**2)
        expect_flux = sys._flux
        expect_C = (
            sys._C.value,
            sys._C.cholesky,
            sys._C.inverse,
            sys._C.lndet,
            sys._C.kind,
            sys._C.N,
        )
        pri_mu = np.zeros(pri.map.Ny)
        pri_mu[0] = 1.0
        pri_L = np.zeros(pri.map.Ny)
        pri_L[0] = 1e-2
        pri_L[1:] = 1e-2
        pri.map.set_prior(mu=pri_mu, L=pri_L)
        (expect_x, expect_cho_cov) = sys.solve(t=t)
        design_matrix = sys.design_matrix(t=t)
        bodies = [pri, sec]
    (calc_x, calc_cho_cov) = solve(
        lmax, expect_flux, expect_C, bodies, design_matrix=design_matrix
    )
    np.testing.assert_allclose(calc_x, expect_x, atol=1e-8)
    np.testing.assert_allclose(calc_cho_cov, expect_cho_cov, atol=1e-10)
