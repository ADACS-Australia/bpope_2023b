from typing import NamedTuple, Optional

import jax.numpy as jnp

from jaxoplanet.types import Array, Scalar


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
