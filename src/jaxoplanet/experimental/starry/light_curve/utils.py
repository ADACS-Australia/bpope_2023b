from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp

from jaxoplanet.experimental.starry.rotation import R_full
from jaxoplanet.types import Array, Scalar


def right_project(l_max: int, M: Array, inc: float, obl: float, theta: Array):
    r"""Apply the projection operator on the right.

    Specifically, this method returns the dot product :math:`M \cdot R`,
    where ``M`` is an input matrix and ``R`` is the Wigner rotation matrix
    that transforms a spherical harmonic coefficient vector in the
    input frame to a vector in the observer's frame. `inc`, `obl` and `theta`
    are in radians.
    """

    theta_ = jnp.atleast_1d(theta)

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

    # Rotate to the sky frame
    R_sky = R_full(l_max, uo)(a1) @ R_full(l_max, uz)(a2) @ R_full(l_max, ux)(a3)
    M_sky = M @ R_sky

    # Rotate to the correct phase
    if theta_.shape[0] == 1:
        theta_b = jnp.broadcast_to(theta_, (M.shape[0],))
        R_theta = R_full(l_max, uz)(theta_b)
    if theta_.shape[0] == M.shape[0]:
        R_theta = R_full(l_max, uz)(theta_)
    # TODO: raise error for all other cases

    M_cor = jax.vmap(jnp.dot, in_axes=(0, 0))(M_sky, R_theta)

    # Rotate to the polar frame
    M_ = M_cor @ R_full(l_max, ux)(0.5 * jnp.pi)

    return M_


class RotationPhase(NamedTuple):
    period: Scalar
    t_0: Scalar
    theta_0: Scalar

    @classmethod
    def init(
        cls,
        period: Scalar,
        t_0: Optional[Scalar] = None,
        theta_0: Optional[Scalar] = None,
    ) -> "RotationPhase":
        if not t_0:
            t_0 = 0.0
        if not theta_0:
            theta_0 = 0.0

        return cls(period=period, t_0=t_0, theta_0=theta_0)

    def phase_curve(self, t: Array) -> Array:
        return 2 * jnp.pi * (t - self.t_0) / self.period + self.theta_0
