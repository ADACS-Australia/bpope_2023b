from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from jaxoplanet.experimental.starry.basis import A1, A
from jaxoplanet.experimental.starry.custom_jvp_rules import (
    zero_safe_arctan2,
    zero_safe_sqrt,
)
from jaxoplanet.experimental.starry.rotation import R_full
from jaxoplanet.experimental.starry.solution import rT_solution_vector, solution_vector
from jaxoplanet.types import Array


@partial(jax.jit, static_argnames=("l_max", "order"))
def light_curve(
    l_max: int,
    inc: float,
    obl: float,
    xo: Array,
    yo: Array,
    zo: Array,
    r: Array,
    theta: Array,
    y: Array,
    order=10,
):
    # R_sky, R_polar = get_R_frames(l_max, inc, obl)
    x_func = design_matrix(l_max, inc, obl, order=order)
    x = x_func(xo, yo, zo, r, theta)
    return x @ y


def get_R_frames(l_max: int, inc: float, obl: float) -> Tuple[Array, Array]:
    cos_obl = jnp.cos(obl)
    sin_obl = jnp.sin(obl)

    # rotation axes
    uo = [-cos_obl, -sin_obl, 0.0]
    uz = [0, 0, 1]
    ux = [1, 0, 0]

    a1 = -((0.5 * jnp.pi) - inc)
    a2 = obl
    a3 = -0.5 * jnp.pi

    # rotate to the sky frame
    R_sky = R_full(l_max, uo)(a1) @ R_full(l_max, uz)(a2) @ R_full(l_max, ux)(a3)

    # polar frame
    R_polar = R_full(l_max, ux)(0.5 * jnp.pi)

    return R_sky, R_polar


def rotation_vector(l_max):
    return rT_solution_vector(l_max) @ A1(l_max)


def design_matrix(
    l_max: int,
    inc: float,
    obl: float,
    order=10,
) -> Callable:
    n_max = (l_max + 1) ** 2

    rTA1 = rotation_vector(l_max)
    A_ = A(l_max)
    R_sky, R_polar = get_R_frames(l_max, inc, obl)

    # @jax.jit
    # @partial(jax.vmap, in_axes=(0,0,None,0,0))
    # @jax.vmap
    @partial(jnp.vectorize, signature=f"(),(),(),(),()->({n_max})")
    def _X(x: Array, y: Array, z: Array, r: Array, theta: Array):
        b = zero_safe_sqrt(x**2 + y**2)
        theta_z = zero_safe_arctan2(x, y)

        # occultation mask
        cond_rot = (b >= (1.0 + r)) | (z <= 0.0) | (r == 0.0)

        sT = solution_vector(l_max, order=order)(b, r)
        # rot_z_func = dotR(l_max, [0, 0, 1])
        # sTAR = rot_z_func(sT @ A_, theta_z)

        R_z = R_full(l_max, [0, 0, 1])(theta_z)

        sTAR = sT @ A_ @ R_z
        sTAR_ = jnp.where(~cond_rot, sTAR, rTA1)
        R_cor = R_full(l_max, [0, 0, 1])(theta)

        return sTAR_ @ R_sky @ R_cor @ R_polar

    return _X
