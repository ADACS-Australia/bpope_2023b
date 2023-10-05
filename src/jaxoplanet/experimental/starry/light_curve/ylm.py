from functools import partial

import jax
import jax.numpy as jnp

from jaxoplanet.experimental.starry.basis import A1, A
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
    b_ = jnp.sqrt(xo**2 + yo**2)
    b = jnp.atleast_1d(b_)
    theta_z_ = jnp.arctan2(xo, yo)
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

    return X @ y
