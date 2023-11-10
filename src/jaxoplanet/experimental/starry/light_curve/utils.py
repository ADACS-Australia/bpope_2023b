from typing import NamedTuple, Optional

import jax.numpy as jnp

from jaxoplanet.types import Array, Scalar

"""
def get_R_frames(l_max: int, inc: float, obl: float) -> Tuple[Array, Array]:

        cos_obl = jnp.cos(obl)
        sin_obl = jnp.sin(obl)

        # rotation axes
        uo = [-cos_obl, -sin_obl, 0.0]
        uz = [0, 0, 1]
        ux = [1, 0, 0]
        print("type ux: ", type(ux))
        # jax.debug.print("🤯 {ux} 🤯", ux=ux)
        # rotate angles
        a1 = -((0.5 * jnp.pi) - inc)
        a2 = obl
        a3 = -0.5 * jnp.pi

        # rotate to the sky frame
        R_sky = R_full(l_max, uo)(a1) @ R_full(l_max, uz)(a2) @ R_full(l_max, ux)(a3)

        # polar frame
        R_p_func = R_full(l_max, ux)
        print("rp func type: ", type(R_p_func))
        R_polar = R_full(l_max, ux)(0.5 * jnp.pi)
        print("rp type: ", type(R_polar))
        # jax.debug.print("🤯 {dr} 🤯", dr=R_polar.shape)
        return R_sky, R_polar
"""


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
