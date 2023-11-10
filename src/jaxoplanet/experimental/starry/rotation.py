from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import block_diag
from scipy.special import factorial

from jaxoplanet.experimental.starry.custom_jvp_rules import (
    zero_safe_arctan2,
    zero_safe_power,
    zero_safe_sqrt,
)
from jaxoplanet.types import Array


def axis_to_euler(u1: float, u2: float, u3: float, theta: float):
    """Returns euler angles (zxz convention) associated to a given axis-angle rotation

    Parameters
    ----------
    u1 : float
        x component of the axis-rotation vector
    u2 : float
        y component of the axis-rotation vector
    u3 : float
        z component of the axis-rotation vector
    theta : float
        rotation angle in radians

    Returns
    -------
    tuple
        the three euler angles in the zyz convention
    """
    tol = jnp.finfo(jax.dtypes.result_type(1.0)).eps * 10
    theta = jnp.where(theta == 0, tol, theta)
    u1u2_null = jnp.logical_and(u1 == 0, u2 == 0)
    u1 = jnp.where(u1u2_null, tol, u1)
    u2 = jnp.where(u1u2_null, tol, u2)

    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    P01 = u1 * u2 * (1 - cos_theta) - u3 * sin_theta
    P02 = u1 * u3 * (1 - cos_theta) + u2 * sin_theta
    P11 = cos_theta + u2 * u2 * (1 - cos_theta)
    P12 = u2 * u3 * (1 - cos_theta) - u1 * sin_theta
    P20 = u3 * u1 * (1 - cos_theta) - u2 * sin_theta
    P21 = u3 * u2 * (1 - cos_theta) + u1 * sin_theta
    P22 = cos_theta + u3 * u3 * (1 - cos_theta)

    norm1 = zero_safe_sqrt(P20 * P20 + P21 * P21)
    norm2 = zero_safe_sqrt(P02 * P02 + P12 * P12)

    alpha0 = jnp.arctan2(0.0, 1.0)
    beta0 = jnp.arctan2(0.0, -1.0)
    gamma0 = zero_safe_arctan2(P01, P11)

    alpha1 = jnp.arctan2(0.0, 1.0)
    beta1 = jnp.arctan2(0.0, 1.0)
    gamma1 = zero_safe_arctan2(-P01, P11)

    alpha2 = zero_safe_arctan2(P12 / norm2, P02 / norm2)
    beta2 = zero_safe_arctan2(zero_safe_sqrt(1 - P22**2), P22)
    gamma2 = zero_safe_arctan2(P21 / norm1, -P20 / norm1)

    case1 = jnp.logical_and((P22 < -1 + tol), (P22 > -1 - tol))
    case2 = jnp.logical_and((P22 < 1 + tol), (P22 > 1 - tol))

    alpha = jnp.where(case1, alpha0, jnp.where(case2, alpha1, alpha2))
    beta = jnp.where(case1, beta0, jnp.where(case2, beta1, beta2))
    gamma = jnp.where(case1, gamma0, jnp.where(case2, gamma1, gamma2))

    return alpha, beta, gamma


# TODO: type hinting callable
### method 1: Rl
def Rl(l: int):
    """Rotation matrix of the spherical harmonics map order l

    Parameters
    ----------
    l : int
        order

    Returns
    -------
    Array
        rotation matrix
    """
    tol = jnp.finfo(jax.dtypes.result_type(1.0)).eps * 10
    # U
    U = np.zeros((2 * l + 1, 2 * l + 1), dtype=np.complex_)
    Ud1 = np.ones(2 * l + 1) * 1j
    Ud1[l + 1 : :] = (-1) ** np.arange(1, l + 1)
    np.fill_diagonal(U, Ud1)
    np.fill_diagonal(np.fliplr(U), -1j * Ud1)
    U[l, l] = np.sqrt(2)
    U *= 1 / np.sqrt(2)
    U = jnp.array(U)

    # dlm
    m, mp = np.indices((2 * l + 1, 2 * l + 1)) - l
    k = np.arange(0, 2 * l + 2)[:, None, None]

    denom = (
        factorial(k)
        * factorial(l + m - k)
        * factorial(l - mp - k)
        * factorial(mp - m + k)
    )

    mask = denom != 0

    numer = (
        jnp.power(-1 + 0j, mp + m)
        * jnp.sqrt(
            factorial(l - m) * factorial(l + m) * factorial(l - mp) * factorial(l + mp)
        )
        * (-1) ** k
    )

    # add tol to elements with zero value in denominator;
    # change the value of corresponding elements in numerator to zero
    fac = (numer * mask) / (denom + ~mask * tol)

    @jax.jit
    def _Rl(alpha: float, beta: float, gamma: float):
        # if beta_tol is too small, it may result in nan value when taking
        # gradient; the following value is tweaked for x64 enabled and
        # l in range (0, 17)
        beta_tol = 1.0e-8
        beta = jnp.where(beta % jnp.pi == 0, beta + beta_tol, beta)

        dlm = (
            fac
            * zero_safe_power(jnp.cos(beta / 2), (2 * l + m - mp - 2 * k))
            * zero_safe_power(jnp.sin(beta / 2), (-m + mp + 2 * k))
        )

        dlm = jnp.nansum(dlm, 0)
        Dlm = jnp.exp(-1j * (mp * alpha + m * gamma)) * dlm

        return jnp.real(jnp.linalg.solve(U, Dlm.T) @ U)

    return _Rl


"""
### method 2: Rl
@partial(jax.jit, static_argnames=("l"))
def Rl(l: int, alpha: float, beta: float, gamma: float):
    tol = jnp.finfo(jax.dtypes.result_type(1.0)).eps * 10

    # U
    U = jnp.zeros((2 * l + 1, 2 * l + 1), dtype=jnp.complex_)
    Ud1 = jnp.ones(2 * l + 1) * 1j
    indices = jnp.arange(l + 1, 2 * l + 1)
    values = (-1) ** jnp.arange(1, l + 1)
    Ud1 = Ud1.at[indices].set(values)
    u_ind = jnp.arange(2 * l + 1)
    U = U.at[u_ind, u_ind].set(Ud1)
    U = U.at[u_ind, -u_ind - 1].set(-1j * Ud1)
    U = U.at[l, l].set(jnp.sqrt(2))
    U_ = U * 1 / jnp.sqrt(2)
    # print("type U: ", type(U_))
    # Dlm
    m, mp = np.indices((2 * l + 1, 2 * l + 1)) - l
    k = np.arange(0, 2 * l + 2)[:, None, None]

    denom = (
        factorial(k)
        * factorial(l + m - k)
        * factorial(l - mp - k)
        * factorial(mp - m + k)
    )

    mask = denom != 0

    numer = (
        jnp.power(-1 + 0j, mp + m)
        * jnp.sqrt(
            factorial(l - m) * factorial(l + m) * factorial(l - mp) * factorial(l + mp)
        )
        * (-1) ** k
    )

    fac = (numer * mask) / (denom + ~mask * tol)

    # if beta_tol is too small, it may result in nan value when taking
    # gradient; the following value is tweaked for x64 enabled and
    # l in range (0, 17)
    beta_tol = 1.0e-8
    beta = jnp.where(beta % jnp.pi == 0, beta + beta_tol, beta)

    dlm = (
        fac
        * zero_safe_power(jnp.cos(beta / 2), (2 * l + m - mp - 2 * k))
        * zero_safe_power(jnp.sin(beta / 2), (-m + mp + 2 * k))
    )

    dlm = jnp.nansum(dlm, 0)
    Dlm = jnp.exp(-1j * (mp * alpha + m * gamma)) * dlm

    return jnp.real(jnp.linalg.solve(U_, Dlm.T) @ U_)
"""


### method 1: R_full
def R_full(l_max: int, u: Array) -> Callable[[Array], Array]:
    """Full Wigner rotation matrix of an axis-angle rotation angle theta about vector u

    Parameters
    ----------
    l_max : int
        maximum order of the spherical harmonics map
    u : Array
        axis-rotation vector

    Returns
    -------
    Callable[[Array], Array]
        a jax.vmap function of theta returning the Wigner matrix for this angle
    """
    Rls = [Rl(l) for l in range(l_max + 1)]
    n_max = l_max**2 + 2 * l_max + 1

    @partial(jnp.vectorize, signature=f"()->({n_max},{n_max})")
    def _R(theta: Array) -> Array:
        alpha, beta, gamma = axis_to_euler(u[0], u[1], u[2], theta)
        full = block_diag(*[rl(alpha, beta, gamma) for rl in Rls])
        return jnp.where(theta != 0, full, jnp.eye(l_max * (l_max + 2) + 1))

    return _R


'''
### method 2: R_full
def R_full(l_max: int, u):
    """Full Wigner rotation matrix of an axis-angle rotation angle theta about vector u

    Parameters
    ----------
    l_max : int
        maximum order of the spherical harmonics map
    u : Array
        axis-rotation vector

    Returns
    -------
    Callable[[Array], Array]
        a jax.vmap function of theta returning the Wigner matrix for this angle
    """
    n_max = l_max**2 + 2 * l_max + 1

    # @partial(jnp.vectorize, signature=f"()->({n_max},{n_max})")
    def _R(theta: Array) -> Array:
        alpha, beta, gamma = axis_to_euler(u[0], u[1], u[2], theta)
        c = [Rl(l, alpha, beta, gamma) for l in range(l_max + 1)]
        full = block_diag(*c)
        return jnp.where(theta != 0, full, jnp.eye(l_max * (l_max + 2) + 1))

    return _R
'''


### method 1: Rdot
def Rdot(l_max: int, u: Array) -> Callable[[Array], Array]:
    """Dot product R@y of the rotation matrix R with a vector y

    Parameters
    ----------
    l_max : int
        maximum order of the spherical harmonics map
    u : Array
        axis-rotation vector

    Returns
    -------
    Callable[[Array], Array]
        a jax.vmap function of (y, theta) returning the product R@y where
        - y is a vector of spherical harmonics coefficients
        - theta is the rotation angle in radians
    """
    Rls = [Rl(l) for l in range(l_max + 1)]
    n_max = l_max**2 + 2 * l_max
    idxs = jnp.cumsum(jnp.array([2 * l + 1 for l in range(l_max + 1)]))[0:-1]

    @partial(jnp.vectorize, signature=f"({n_max}),()->({n_max})")
    def R(y: Array, theta: Array) -> Array:
        yls = jnp.split(y, idxs)
        alpha, beta, gamma = axis_to_euler(u[0], u[1], u[2], theta)
        return jnp.hstack([rl(alpha, beta, gamma) @ yl for rl, yl in zip(Rls, yls)])

    return R


### method 1: dotR
def dotR(l_max: int, u: Array) -> Callable[[Array], Array]:
    """Dot product M@R of a matrix M with the rotation matrix R

    Parameters
    ----------
    l_max : int
        maximum order of the spherical harmonics map
    u : Array
        axis-rotation vector

    Returns
    -------
    Callable[[Array], Array]
        a jax.vmap function of (M, theta) returning the product M@R where
        - M is a matrix (Array)
        - theta is the rotation angle in radians
    """
    n_max = l_max**2 + 2 * l_max + 1
    Rls = [Rl(l) for l in range(l_max + 1)]

    @partial(jnp.vectorize, signature=f"({n_max}),()->({n_max})")
    def R(M: Array, theta: Array) -> Array:
        alpha, beta, gamma = axis_to_euler(u[0], u[1], u[2], theta)
        return jnp.hstack(
            [
                M[l**2 : (l + 1) ** 2] @ Rls[l](alpha, beta, gamma)
                for l in range(l_max + 1)
            ]
        )

    return R
