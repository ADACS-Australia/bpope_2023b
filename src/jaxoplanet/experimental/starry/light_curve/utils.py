import jax.numpy as jnp

from jaxoplanet.experimental.starry.rotation import dotR
from jaxoplanet.types import Array


def right_project(M: Array, inc: int, obl: int, theta: Array):
    r"""Apply the projection operator on the right.

    Specifically, this method returns the dot product :math:`M \cdot R`,
    where ``M`` is an input matrix and ``R`` is the Wigner rotation matrix
    that transforms a spherical harmonic coefficient vector in the
    input frame to a vector in the observer's frame.
    """

    l_max = (jnp.sqrt(M.shape[1]) - 1).astype(int)

    cos_obl = jnp.cos(obl)
    sin_obl = jnp.sin(obl)

    # Rotate to the sky frame.
    M = M.at[:].set(dotR(l_max, [-cos_obl, -sin_obl, 0.0])(M, -((0.5 * jnp.pi) - inc)))
    M = M.at[:].set(dotR(l_max, [0.0, 0.0, 1.0])(M, obl))
    M = M.at[:].set(dotR(l_max, [1.0, 0.0, 0.0])(M, (-0.5 * jnp.pi)))

    # Rotate to the correct phase.
    M = M.at[:].set(dotR(l_max, [0.0, 0.0, 1.0])(M, theta))

    # Rotate to the polar frame.
    M = M.at[:].set(dotR(l_max, [1.0, 0.0, 0.0])(M, 0.5 * jnp.pi))

    return M
