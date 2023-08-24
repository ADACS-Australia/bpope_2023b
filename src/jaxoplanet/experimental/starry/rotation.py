from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import block_diag
from scipy.special import factorial

# import bpope_2023b.src.jaxoplanet as jaxoplanet
from jaxoplanet.types import Array

""" Functions:
    ---------
    
    axis_to_euler(u1, u2, u3, theta)
    
                    Returns euler angles (zxz convention) associated with a
                    given axis-angle rotation.

    Rl(l: int)     
                    Rotation matrix of the spherical harmonics map order l.
                    
    R_full(l_max, u)
        
                    Full Wigner rotation matrix of an axis-angle rotation angle
                    theta about vector u.
    
    Rdot(l_max, u)
    
                    Dot product R@y of the rotation matrix R with a vector y.
    
    dotR(l_max, u)
    
                    Dot product M@R of a matrix M with the rotation matrix R.
                    
"""

@jax.jit # sets up Just In Time compilation of a JAX function
         # to be efficiently executed in XLA.
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
    tol = 1e-16
    
    # if theta==0 then theta=tol else theta=theta
    theta = jnp.where(theta == 0, tol, theta)
    
    # return True if x,y components of rotation vector == 0
    u1u2_null = jnp.logical_and(u1 == 0, u2 == 0)
    
    # if x,y components == 0: set them to tol
    u1 = jnp.where(u1u2_null, tol, u1)
    u2 = jnp.where(u1u2_null, tol, u2)
    
    # define cosine and sine of rotation angle
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    
    # computes the elements of P (3d cartesian roation matrix for axis-angle
    # rotation)
    P01 = u1 * u2 * (1 - cos_theta) - u3 * sin_theta
    P02 = u1 * u3 * (1 - cos_theta) + u2 * sin_theta
    P11 = cos_theta + u2 * u2 * (1 - cos_theta)
    P12 = u2 * u3 * (1 - cos_theta) - u1 * sin_theta
    P20 = u3 * u1 * (1 - cos_theta) - u2 * sin_theta
    P21 = u3 * u2 * (1 - cos_theta) + u1 * sin_theta
    P22 = cos_theta + u3 * u3 * (1 - cos_theta)
    
    # computes the two norms (sqrt(a^2+b^2)) in the set of equations derived
    # from equating P with Q (the euler angle rotation matrix)
    norm1 = jnp.sqrt(P20 * P20 + P21 * P21)
    norm2 = jnp.sqrt(P02 * P02 + P12 * P12)

    # not really sure where this is from
    # arctan2 returns angle in radians of x,y position.
    alpha0 = jnp.arctan2(0.0, 1.0) # 0
    beta0 = jnp.arctan2(0.0, -1.0) # pi
    gamma0 = jnp.arctan2(P01, P11)

    alpha1 = jnp.arctan2(0.0, 1.0) # 0
    beta1 = jnp.arctan2(0.0, 1.0)  # 0
    gamma1 = jnp.arctan2(-P01, P11)

    # arctan2(sin(alpha),cos(alpha))
    alpha2 = jnp.arctan2(P12 / norm2, P02 / norm2)
    beta2 = jnp.arctan2(jnp.sqrt(1 - P22**2), P22)
    gamma2 = jnp.arctan2(P21 / norm1, -P20 / norm1)

    # case1: P22 ~ -1 (within a tolerance)
    case1 = jnp.logical_and((P22 < -1 + tol), (P22 > -1 - tol))
    # case2: P22 ~ 1 (within a tolerance)
    case2 = jnp.logical_and((P22 < 1 + tol), (P22 > 1 - tol))

    # alpha = (if case1 then alpha0 elif case2 then alpha1 else alpha2)
    # alpha = (if P22~-1 then 0 elif P22~1 then 0 else alpha2)
    alpha = jnp.where(case1, alpha0, jnp.where(case2, alpha1, alpha2))
    # beta = (if P22~-1 then pi elif P22~1 then 0 else beta2)
    beta = jnp.where(case1, beta0, jnp.where(case2, beta1, beta2))
    gamma = jnp.where(case1, gamma0, jnp.where(case2, gamma1, gamma2))

    return alpha, beta, gamma


# todo: type hinting callable
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
    # U: describes transformation from complex to real
    U = np.zeros((2 * l + 1, 2 * l + 1), dtype=np.complex_) # initiate U
    # j represents complex i
    Ud1 = np.ones(2 * l + 1) * 1j # array of 0+1j complex numbers
    # np.arange returns list of numbers from (start, end+1)
    Ud1[l + 1 : :] = (-1) ** np.arange(1, l + 1)
    np.fill_diagonal(U, Ud1)
    np.fill_diagonal(np.fliplr(U), -1j * Ud1)
    U[l, l] = np.sqrt(2)
    U *= 1 / np.sqrt(2)
    U = jnp.array(U) # create array on JAX's default device

    # D: rotation matric for complex spherical harmonics
    # dlm
    # creates two matrices, 1st populated with row indices, 2nd with column.
    m, mp = np.indices((2 * l + 1, 2 * l + 1)) - l # m: row indices, mp: column
    # array of arrays-within-arrays of numbers from 0 to 2l+1
    k = np.arange(0, 2 * l + 2)[:, None, None]

    @jax.jit # set up Just In Time compilation
    def _Rl(alpha: float, beta: float, gamma: float):
        dlm = (
            jnp.power(-1 + 0j, mp + m)                       # (-1)^(m'+m)
            * jnp.sqrt(
                factorial(l - m)
                * factorial(l + m)
                * factorial(l - mp)
                * factorial(l + mp)
            )                                                # sqrt term
            * (-1) ** k                                      # (-1)^k
            * jnp.cos(beta / 2) ** (2 * l + m - mp - 2 * k)  # cos term
            * jnp.sin(beta / 2) ** (-m + mp + 2 * k)         # sin term
            / (
                factorial(k)
                * factorial(l + m - k)
                * factorial(l - mp - k)
                * factorial(mp - m + k)
            )                                                # division term
        )
        # returns an array of 2l+1 arrays of length 2l+1
        dlm = jnp.nansum(dlm, 0) # computes sum along axis 0 (with nans accepted)
        Dlm = jnp.exp(-1j * (mp * alpha + m * gamma)) * dlm # first exp term
        
        # solves R = U_inv@D@U
        # returns real component of complex argument
        # solves Ux=Dlm^T -> x=u_inv@D, then x@U = u_inv@D@U
        return jnp.real(jnp.linalg.solve(U, Dlm.T) @ U)

    # returns a function Rl(alpha,beta,gamma) that computes the Rl matrix for
    # the given Euler angles.
    return _Rl

# Returns a function of theta, _R(theta), which returns the rotation matrix of
# the rotation angle about the rotation vector
# - called as: R_full(l_max, u)(theta)
# - Callable specifies input arguments and return type
# - -> is a return annotation: this function returns a callable function
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
    # jax.vmap function is used to vectorise a function across a given axis,
    # meaning the function can be applied to multiple inputs at the same time.
    # (useful for matrix multiplication, since mm results in a matrix populated
    # with dot products of pairs of vectors from the multiplied matrices)
    
    # compute the Rl matrices to populate the full R rotation matrix
    Rls = [Rl(l) for l in range(l_max + 1)]
    n_max = l_max**2 + 2 * l_max + 1

    # - jnp.vectorize defines function that is automatically repeated across any
    #   leading dimension (without concern of how to handle higher dimensions)
    # - signature: generalised universal function signature for vectorised
    #   matrix-vector multiplication. Function will be called with and expected
    #   to return arrays with shapes given.
    # - returns vectorised version of the given function.
    @partial(jnp.vectorize, signature=f"()->({n_max},{n_max})")
    # - underscore is a syntax hint that the object (function) is used internally
    #   (this is not enforced, just a note to programmer)
    def _R(theta: Array) -> Array:
        # get euler angles of axis-rotation vector and angle
        alpha, beta, gamma = axis_to_euler(u[0], u[1], u[2], theta)
        # creates a block diagonal matrix populated with Rl matrices
        full = block_diag(*[rl(alpha, beta, gamma) for rl in Rls])
        # if theta!=0 return full, else return matrix w/ diagonal of 1s
        #   (if the angle of rotation = 0 then no rotation)
        return jnp.where(theta != 0, full, jnp.eye(l_max * (l_max + 2) + 1))

    # returns a function, _R(theta) that returns rotation matrix R
    return _R

# rotates vector of spherical coeffients, y, to y' via rotation matrix R
# - called as: Rdot(l_max, u)(y, theta)
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
    
    # compute the Rl matrices that populate the full R rotation matrix
    Rls = [Rl(l) for l in range(l_max + 1)]
    n_max = l_max**2 + 2 * l_max
    # returns an array of l_max squares (i.e. 1,4,9,16,25,...)
    idxs = jnp.cumsum(jnp.array([2 * l + 1 for l in range(l_max + 1)]))[0:-1]

    @partial(jnp.vectorize, signature=f"({n_max}),()->({n_max})")
    # why is theta an array?
    def R(y: Array, theta: Array) -> Array:
        # jnp.split splits an array into multiple sub-arrays, with split
        # locations indicated by the idxs array.
        # - (sub-arrays are views of original array, not copies, which means
        #   modification of sub-array views modifies original array!)
        yls = jnp.split(y, idxs)
        # get Euler angles
        alpha, beta, gamma = axis_to_euler(u[0], u[1], u[2], theta)
        # for each Rl and sub-array of y, compute Rl(alpha,beta,gamma)
        # and then matrix multiply Rl(angles)@sub-array
        # and then stacks (concatenates) the resulting arrays along the second
        # dimension (returns one array of the rotated y)
        return jnp.hstack([rl(alpha, beta, gamma) @ yl for rl, yl in zip(Rls, yls)])
    
    # return a function R(y,theta) that returns a rotated y'
    return R

# To perform a set of rotations on y.
# - called as: dotR(l_max, u)(M, theta)
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
    
    # compute the Rl matrices that populate the full R rotation matrix
    Rls = [Rl(l) for l in range(l_max + 1)]
    n_max = l_max**2 + 2 * l_max + 1
    
    # Forces M to be a (n_max,n_max) matrix.
    #@partial(jnp.vectorize, signature=f"({n_max},{n_max}),()->({n_max},{n_max})")
    @partial(jnp.vectorize, signature=f"(m,{n_max}),()->(m,{n_max})")
    def R(M: Array, theta: Array) -> Array:
        # get Euler angles
        alpha, beta, gamma = axis_to_euler(u[0], u[1], u[2], theta)
        
        return np.hstack(
            # for each l:
            # - get sub-matrix of columns (with same column indices as yls
            #   split in Rdot),
            # - get rotation matrix Rl(alpha,beta,gamma)
            # - then matrix multiply: sub-matrix@rotation matrix
            [
                M[:, l**2 : (l + 1) ** 2] @ Rls[l](alpha, beta, gamma)
                for l in range(l_max + 1)
            ]
            # stack (column-wise concatenatation) resulting matrices
        )
    
    # return a function R(M,theta) that returns M@R
    return R

# (m,n),(n,p)->(m,p)
