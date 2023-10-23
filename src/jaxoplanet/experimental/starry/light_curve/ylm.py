from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from jaxoplanet.experimental.starry.basis import A1, A
from jaxoplanet.experimental.starry.rotation import R_full
from jaxoplanet.experimental.starry.solution import rT_solution_vector, solution_vector
from jaxoplanet.types import Array

# @partial(jax.jit, static_argnames=("l_max", "order"))
# def light_curve(
#     l_max: int,
#     inc: float,
#     obl: float,
#     y: Array,
#     xo: Array,
#     yo: Array,
#     zo: Array,
#     ro: Array,
#     theta: Array,
#     order=10,
# ) -> Array:
#     b_ = zero_safe_sqrt(xo**2 + yo**2)
#     b = jnp.atleast_1d(b_)

#     theta_z_ = zero_safe_arctan2(xo, yo)
#     theta_z = jnp.atleast_1d(theta_z_)

#     # occultation mask
#     cond_rot = (b >= (1.0 + ro)) | (zo <= 0.0) | (ro == 0.0)
#     cond_rot_ = cond_rot[:, jnp.newaxis]

#     sT = solution_vector(l_max, order=order)(b, ro)
#     print("st shape: ", sT.shape)
#     sTA = sT @ A(l_max)
#     print("A shape: ", A(l_max).shape)
#     print("sta shape: ", sTA.shape)
#     sTAR = jax.vmap(jnp.dot, in_axes=(0, 0))(sTA, R_full(l_max, [0, 0, 1])(theta_z))

#     # rotational phase
#     rTA1 = rT_solution_vector(l_max) @ A1(l_max)
#     rTA1_ = jnp.broadcast_to(rTA1, sTAR.shape)

#     # applying occultation mask
#     sTAR_ = jnp.where(~cond_rot_, sTAR, rTA1_)

#     # get the design matrix
#     X = right_project(l_max, sTAR_, inc, obl, theta)
#     print("X shape: ", X.shape)
#     print("shape of y: ", y.shape)
#     if y.shape == (1,):
#         return jnp.dot(X, y[0])
#     return jnp.dot(X, y)


def design_matrix(
    l_max: int,
    inc: float,
    obl: float,
    order=10,
) -> Callable:
    (l_max + 1) ** 2

    def _rotation_vector():
        return rT_solution_vector(l_max) @ A1(l_max)

    rTA1 = _rotation_vector()
    # print("rTA1: ", rTA1)

    A_ = A(l_max)

    def _get_R_frames() -> Tuple[Array, Array]:
        cos_obl = jnp.cos(obl)
        sin_obl = jnp.sin(obl)

        # rotation axes
        uo = [-cos_obl, -sin_obl, 0.0]
        uz = [0, 0, 1]
        ux = [1, 0, 0]

        # rotate angles
        a1 = -((0.5 * jnp.pi) - inc)
        a2 = obl
        a3 = -0.5 * jnp.pi

        # rotate to the sky frame
        R_sky = R_full(l_max, uo)(a1) @ R_full(l_max, uz)(a2) @ R_full(l_max, ux)(a3)

        # polar frame
        R_polar = R_full(l_max, ux)(0.5 * jnp.pi)

        return R_sky, R_polar

    R_sky, R_polar = _get_R_frames()

    # @jax.jit
    @jax.vmap
    # @partial(jnp.vectorize, signature=f"(),(),(),(),()->({n_max})")
    def _X(b: Array, z: Array, r: Array, theta: Array, theta_z: Array):
        # occultation mask
        cond_rot = (b >= (1.0 + r)) | (z <= 0.0) | (r == 0.0)
        # cond_rot_ = cond_rot[:, jnp.newaxis]

        sT = solution_vector(l_max, order=order)(b, r)
        # sT = solution_vector(l_max, b, r, order=order)
        # A_ = A(l_max)
        R_z = R_full(l_max, [0, 0, 1])(theta_z)
        sTAR = sT @ A_ @ R_z
        # rTA1 = rT_solution_vector(l_max) @ A1(l_max)

        sTAR_ = jnp.where(~cond_rot, sTAR, rTA1)

        # R_sky, R_polar = _get_R_frames()

        R_cor = R_full(l_max, [0, 0, 1])(theta)

        return sTAR_ @ R_sky @ R_cor @ R_polar

    return _X


@partial(jax.jit, static_argnames=("l_max", "order"))
def light_curve(
    l_max: int,
    inc: float,
    obl: float,
    b: Array,
    z: Array,
    r: Array,
    theta: Array,
    theta_z: Array,
    y: Array,
    order=10,
):
    x_func = design_matrix(l_max, inc, obl, order=order)
    # opt_x_func = jax.jit(jax.vmap(x_func))
    x = x_func(b, z, r, theta, theta_z)
    # x = opt_x_func(b, z, r, theta, theta_z)
    # return opt_x_func
    return x @ y
