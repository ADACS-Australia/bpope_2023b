import jax.numpy as jnp
import numpy as np

from jaxoplanet.experimental.starry.basis import A1, A
from jaxoplanet.experimental.starry.light_curve.utils import right_project
from jaxoplanet.experimental.starry.rotation import dotR
from jaxoplanet.experimental.starry.solution import rT_solution_vector, solution_vector
from jaxoplanet.types import Array


def light_curve(
    l_max: int,
    theta: Array,
    xo: Array,
    yo: Array,
    zo: Array,
    ro: float,
    inc: float,
    obl: float,
    y: Array,
) -> Array:
    design_matrix = X(l_max, theta, xo, yo, zo, ro, inc, obl)
    # y = _get_y(l_max)
    return design_matrix @ y


def X(
    l_max: int,
    theta: Array,
    xo: Array,
    yo: Array,
    zo: Array,
    ro: float,
    inc: float,
    obl: float,
) -> Array:
    # Determine shapes
    cols = (l_max + 1) ** 2
    rows = theta.shape[0]
    X = np.zeros((rows, cols))

    # Compute the occultation mask
    b = np.sqrt(xo**2 + yo**2)
    cond_rot = (b >= (1.0 + ro)) | (zo <= 0.0) | (ro == 0.0)
    cond_occ = ~cond_rot
    i_rot = np.where(cond_rot)[0]
    i_occ = np.where(cond_occ)[0]

    # Rotational operator
    if len(i_rot) > 0:
        print("rotational operator")
        rTA1 = rT_solution_vector(l_max) @ A1(l_max)
        # Check for dimension
        rTA1 = jnp.broadcast_to(rTA1, (rows, cols))
        X[i_rot] = right_project(rTA1, inc, obl, theta[i_rot])

    # Occulation operator
    if len(i_occ) > 0:
        print("occultation operator")
        sT_func = solution_vector(l_max)
        sT = sT_func(b[i_occ], ro)
        sTA = sT @ A(l_max)
        theta_z = np.arctan2(xo[i_occ], yo[i_occ])
        sTAR = dotR(sTA, theta_z)
        X[i_occ] = right_project(sTAR, inc, obl, theta[i_occ])

    return X


def _get_y(l_max: int) -> Array:
    rows = (l_max + 1) ** 2
    y = np.zeros((rows, 1))
    y[0, 0] = 1
    return y


'''
def flux(self, theta, xo, yo, zo, ro, inc, obl, y, u, f):
        """Compute the light curve."""
        return tt.dot(self.X(theta, xo, yo, zo, ro, inc, obl, u, f), y)

def X(self, theta, xo, yo, zo, ro, inc, obl, u, f):
        """Compute the light curve design matrix."""
        # Determine shapes
        rows = theta.shape[0]
        cols = self.rTA1.shape[1]
        X = tt.zeros((rows, cols))

        # Compute the occultation mask
        b = tt.sqrt(xo ** 2 + yo ** 2)
        b_rot = tt.ge(b, 1.0 + ro) | tt.le(zo, 0.0) | tt.eq(ro, 0.0)
        b_occ = tt.invert(b_rot)
        i_rot = tt.arange(b.size)[b_rot]
        i_occ = tt.arange(b.size)[b_occ]



        # Rotation operator

        rTA1 = self.rTA1
        rTA1 = tt.tile(rTA1, (theta[i_rot].shape[0], 1))
        X = tt.set_subtensor(
            X[i_rot], self.right_project(rTA1, inc, obl, theta[i_rot])
        )

        # Occultation + rotation operator
        sT = self.sT(b[i_occ], ro)
        sTA = ts.dot(sT, self.A)
        theta_z = tt.arctan2(xo[i_occ], yo[i_occ])
        sTAR = self.tensordotRz(sTA, theta_z)

        X = tt.set_subtensor(
            X[i_occ], self.right_project(sTAR, inc, obl, theta[i_occ])
        )

        return X
'''
