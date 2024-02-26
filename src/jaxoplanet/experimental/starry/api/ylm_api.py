from typing import NamedTuple, Optional

import jax.numpy as jnp
import numpy as np

from jaxoplanet.experimental.starry.light_curve.utils import RotationPhase
from jaxoplanet.experimental.starry.light_curve.ylm import light_curve
from jaxoplanet.proto import LightCurveOrbit
from jaxoplanet.types import Array, Scalar


class YlmLightCurve(NamedTuple):
    l_max: int
    inc: Scalar
    obl: Scalar
    y: Array

    @classmethod
    def init(
        cls,
        l_max: int,
        inc: Optional[Scalar] = None,
        obl: Optional[Scalar] = None,
        y: Optional[Array] = None,
    ) -> "YlmLightCurve":
        if inc is None:
            inc = jnp.pi / 2
        if obl is None:
            obl = 0.0
        if y is None:
            n_max = (l_max + 1) ** 2
            y = np.zeros(n_max)
            y[0] = 1.0

        # TODO: add input checks and raise errors
        return cls(l_max=l_max, inc=inc, obl=obl, y=y)

    def light_curve(
        self, orbit: LightCurveOrbit, phase: RotationPhase, t: Array
    ) -> Array:
        r_star = orbit.central_radius
        xo, yo, zo = orbit.relative_position(t)

        # convert to match dimensions
        xo_ = xo.flatten()
        yo_ = yo.flatten()
        zo_ = zo.flatten()

        ro = orbit.radius / r_star

        theta = phase.phase_curve(t)
        # b = zero_safe_sqrt(xo_**2 + yo_**2)
        # theta_z = zero_safe_arctan2(xo_, yo_)

        # lc_func = partial(light_curve, self.l_max, self.inc, self.obl, self.y)
        lc = light_curve(
            self.l_max, self.inc, self.obl, xo_, yo_, zo_, ro, theta, self.y
        )

        # lc = lc_func(xo_, yo_, zo_, ro, theta)
        return lc
