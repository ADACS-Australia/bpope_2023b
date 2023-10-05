import jax.numpy as jnp

from jaxoplanet.experimental.starry.basis import A
from jaxoplanet.experimental.starry.light_curve.utils import right_project
from jaxoplanet.types import Array


def light_curve():
    # light_curve = jnp.dot(X(),y)
    pass


def X(
    lmax: int, theta: Array, xo: Array, yo: Array, zo: Array, ro, inc, obl, fproj, u, f
):
    """Compute the light curve design matrix."""

    # Compute the occultation mask
    bo = jnp.sqrt(xo**2 + yo**2)
    thetao = jnp.arctan2(xo, yo)

    # Occultation + phase curve operator
    sT = oblate_solution_vector(lmax, fproj, thetao, bo, ro, inc)

    # TODO: Limb darkening

    # Rotate to Green's basis
    sTA = jnp.dot(sT, A(lmax))

    # Projection onto the sky
    sTAR = right_project(sTA, inc, obl, theta)

    # TODO: Gravity darkening

    return sTAR


def oblate_solution_vector(lmax, f, theta, bo, ro, inc):
    pass
