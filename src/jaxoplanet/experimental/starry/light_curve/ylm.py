from functools import partial

import jax
import jax.numpy as jnp

from jaxoplanet.experimental.starry.basis import A1, A
from jaxoplanet.experimental.starry.custom_jvp_rules import (
    zero_safe_arctan2,
    zero_safe_sqrt,
)
from jaxoplanet.experimental.starry.light_curve.utils import right_project
from jaxoplanet.experimental.starry.rotation import R_full
from jaxoplanet.experimental.starry.solution import rT_solution_vector, solution_vector
from jaxoplanet.types import Array


@partial(jax.jit, static_argnames=("l_max", "order"))
def light_curve(
    l_max: int,
    inc: float,
    obl: float,
    y: Array,
    xo: Array,
    yo: Array,
    zo: Array,
    ro: Array,
    theta: Array,
    order=10,
) -> Array:
    b_ = zero_safe_sqrt(xo**2 + yo**2)
    b = jnp.atleast_1d(b_)

    theta_z_ = zero_safe_arctan2(xo, yo)
    theta_z = jnp.atleast_1d(theta_z_)

    # occultation mask
    cond_rot = (b >= (1.0 + ro)) | (zo <= 0.0) | (ro == 0.0)
    cond_rot_ = cond_rot[:, jnp.newaxis]

    sT = solution_vector(l_max, order=order)(b, ro)
    sTA = sT @ A(l_max)
    sTAR = jax.vmap(jnp.dot, in_axes=(0, 0))(sTA, R_full(l_max, [0, 0, 1])(theta_z))

    # rotational phase
    rTA1 = rT_solution_vector(l_max) @ A1(l_max)
    rTA1_ = jnp.broadcast_to(rTA1, sTAR.shape)

    # applying occultation mask
    sTAR_ = jnp.where(~cond_rot_, sTAR, rTA1_)

    # get the design matrix
    X = right_project(l_max, sTAR_, inc, obl, theta)
    print("X shape: ", X.shape)
    print("shape of y: ", y.shape)
    if y.shape == (1,):
        return jnp.dot(X, y[0])
    return jnp.dot(X, y)


# @partial(jax.jit, static_argnames=("l_max", "order"))
# def light_curve(
#     l_max: int, order=10) -> callable:
#     l_max = l_max


#     @jax.jit
#     def _lc(
#     inc: float,
#     obl: float,
#     y: Array,
#     xo: Array,
#     yo: Array,
#     zo: Array,
#     ro: Array,
#     theta: Array,) -> Array:
#         sT_func = solution_vector(l_max, order=order)
#         a = A(l_max)
#         rTA1 = rT_solution_vector(l_max) @ A1(l_max)
#         tol = 10 * jnp.finfo(jnp.float64).eps
#         b_ = zero_safe_sqrt(xo**2 + yo**2)
#         # b2 = xo**2 + yo**2
#         # cond_b = jnp.less(b2, tol)
#         # b2_ = jnp.where(cond_b, 1, b2)
#         # b_ = jnp.where(cond_b, 0, jnp.sqrt(b2_))
#         b = jnp.atleast_1d(b_)

#         cond_y = jnp.logical_and(yo > -tol, yo < tol)
#         yo_ = jnp.where(cond_y, 1, yo)
#         theta_z_ = jnp.where(cond_y, 0, jnp.arctan2(xo, yo_))
#         theta_z = jnp.atleast_1d(theta_z_)

#         # occultation mask
#         cond_rot = (b >= (1.0 + ro)) | (zo <= 0.0) | (ro == 0.0)
#         cond_rot_ = cond_rot[:, jnp.newaxis]

#         # sT = solution_vector(l_max, order=order)(b, ro)
#         sT = sT_func(b, ro)
#         sTA = sT @ a
#         sTAR = jax.vmap(jnp.dot, in_axes=(0, 0))(sTA, R_full(l_max, [0, 0, 1])
#               (theta_z))

#         # rotational phase
#         # rTA1 = rT_solution_vector(l_max) @ A1(l_max)
#         rTA1_ = jnp.broadcast_to(rTA1, sTAR.shape)

#         # applying occultation mask
#         sTAR_ = jnp.where(~cond_rot_, sTAR, rTA1_)

#         # get the design matrix
#         X = right_project(l_max, sTAR_, inc, obl, theta)
#         print("X shape: ", X.shape)
#         print("shape of y: ", y.shape)
#         if y.shape == (1,):
#             return jnp.dot(X, y[0])
#         return jnp.dot(X, y)
#     return _lc
