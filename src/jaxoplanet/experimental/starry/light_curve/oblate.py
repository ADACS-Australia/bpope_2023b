import jax.numpy as jnp
import numpy as np

from jaxoplanet.experimental.starry.basis import A
from jaxoplanet.experimental.starry.light_curve.utils import right_project
from jaxoplanet.types import Array

"""

Notes
-----

    f vs fproj as inputs

"""


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


STARRY_ROOT_MAX_ITER = 100
STARRY_ROOT_TOL_HIGH = 1e-12
STARRY_COMPLETE_OCC_TOL = 1e-12
STARRY_GRAZING_TOL = 1e-12
STARRY_NO_OCC_TOL = 1e-12
STARRY_THETA_UNIT_RADIUS_TOL = 1e-12

STARRY_BO_EQUALS_RO_TOL = 1e-12
STARRY_BO_EQUALS_RO_EQUALS_HALF_TOL = 1e-12
STARRY_BO_EQUALS_ZERO_TOL = 1e-12
STARRY_RO_EQUALS_ZERO_TOL = 1e-12
STARRY_BO_EQUALS_ONE_MINUS_RO_TOL = 1e-12
STARRY_ROOT_TOL_THETA_PI_TWO = 1e-12
STARRY_T_TOL = 1e-12
STARRY_MIN_F = 1e-12


def oblate_solution_vector(lmax, f, theta, bo, ro, inc):
    ncoeff = (lmax + 1) * (lmax + 1)
    sT = np.zeros(ncoeff)

    # Compute projected oblateness.
    # fproj = np.sqrt((1 - f) ** 2 * np.cos(inc) ** 2 + np.sin(inc) ** 2)

    # Nudge inputs away from singular points.
    f, theta, costheta, sintheta, bo, ro = nudge_inputs(f, theta, bo, ro)

    # No occultation.
    if ro == 0:
        return sT

    # Compute angles of intersection.
    phi1, phi2, xi1, xi2 = get_angles(f, theta, sintheta, costheta, bo, ro)

    # Compute sT for special cases.
    if phi1 == 0.0 and phi2 == 0.0 and xi1 == 0.0 and xi2 == 2 * np.pi:
        # Complete occulation
        return (1 - f) * sT0(lmax)
    elif phi1 == 0.0 and phi2 == 0.0 and xi1 == 0.0 and xi2 == 0.0:
        # No occulation
        return sT

    # Compute M integral analytically for even mu.
    M = compute_M(lmax, theta, sintheta, costheta, bo, ro, phi1, phi2)

    # Compute pT numerically for odd mu.
    pTodd = compute_pTodd(
        lmax, f, theta, sintheta, costheta, bo, ro, phi1, phi2
    )  # TODO

    # Compute L (t integral).
    Lt = compute_L(lmax, xi1, xi2)

    # Compute surface integrals.
    n = 0
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            mu = l - m
            nu = l + m

            # Compute pT and tT integrals.
            if nu % 2 == 0:
                # Case 1
                pT = (1 - f) ** -(nu / 2) * M[(mu + 2) // 2, nu // 2]
                tT = (1 - f) * Lt[mu // 2 + 2, nu // 2]
            elif l == 1 and m == 0:
                # Case 2
                pT = pTodd[n]
                tT = (1 - f) * (xi2 - xi1) / 3
            else:
                # Cases 3, 4, and 5 are equivalent.
                pT = pTodd[n]
                tT = 0.0

            # Compute surface integral.
            sT[n] = pT + tT
            n += 1

    return sT


def nudge_inputs(f, theta, bo, ro):
    if abs(bo - ro) < STARRY_BO_EQUALS_RO_TOL:
        if bo > ro:
            bo = ro + STARRY_BO_EQUALS_RO_TOL
        else:
            bo = ro - STARRY_BO_EQUALS_RO_TOL
    if (
        abs(bo - ro) < STARRY_BO_EQUALS_RO_EQUALS_HALF_TOL
        and abs(ro - 0.5) < STARRY_BO_EQUALS_RO_EQUALS_HALF_TOL
    ):
        bo = ro + STARRY_BO_EQUALS_RO_EQUALS_HALF_TOL
    if abs(bo) < STARRY_BO_EQUALS_ZERO_TOL:
        bo = STARRY_BO_EQUALS_ZERO_TOL
    if 0 < ro < STARRY_RO_EQUALS_ZERO_TOL:
        ro = STARRY_RO_EQUALS_ZERO_TOL
    if abs(1 - bo - ro) < STARRY_BO_EQUALS_ONE_MINUS_RO_TOL:
        bo = 1 - ro + STARRY_BO_EQUALS_ONE_MINUS_RO_TOL

    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    if abs(theta - 0.5 * np.pi) < STARRY_ROOT_TOL_THETA_PI_TWO:
        if theta > 0.5 * np.pi:
            theta += STARRY_ROOT_TOL_THETA_PI_TWO
        else:
            theta -= STARRY_ROOT_TOL_THETA_PI_TWO
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
    elif abs(theta + 0.5 * np.pi) < STARRY_ROOT_TOL_THETA_PI_TWO:
        if theta > -0.5 * np.pi:
            theta += STARRY_ROOT_TOL_THETA_PI_TWO
        else:
            theta -= STARRY_ROOT_TOL_THETA_PI_TWO
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
    elif abs(sintheta) < STARRY_T_TOL:
        if sintheta > 0:
            sintheta = STARRY_T_TOL
        else:
            sintheta = -STARRY_T_TOL
        if costheta > 0:
            theta = 0
        else:
            theta = np.pi
    if f < STARRY_MIN_F:
        f = STARRY_MIN_F

    return f, theta, costheta, sintheta, bo, ro


def get_angles(f, theta, sintheta, costheta, bo, ro):
    b = 1 - f

    # Enforce positive bo.
    if bo < 0:
        bo = -bo
        theta -= np.pi

    # Trivial cases.
    if bo <= ro - 1 + STARRY_COMPLETE_OCC_TOL:
        # Complete occultation.
        phi1 = phi2 = xi1 = 0.0
        xi2 = 2 * np.pi
        return phi1, phi2, xi1, xi2

    if bo + ro + f <= 1 + STARRY_GRAZING_TOL:
        # Regular occultation, but occultor doesn't touch the limb.
        phi1 = xi1 = xi2 = 0.0
        phi2 = 2 * np.pi
        return phi1, phi2, xi1, xi2

    if bo >= 1 + ro - STARRY_NO_OCC_TOL:
        # No occultation.
        phi1 = phi2 = xi1 = xi2 = 0.0
        return phi1, phi2, xi1, xi2

    # Avoid the grazing configuration that leads to instabilities in the root solver.
    if 1 - ro - STARRY_GRAZING_TOL <= bo <= 1 - ro + STARRY_GRAZING_TOL:
        bo = 1 - ro + STARRY_GRAZING_TOL

    # The eigensolver doesn't converge when ro = 1 and theta = pi / 2.
    if (abs(1 - ro) < STARRY_THETA_UNIT_RADIUS_TOL) and (
        abs(costheta) < STARRY_THETA_UNIT_RADIUS_TOL
    ):
        if costheta > 0:
            costheta += STARRY_THETA_UNIT_RADIUS_TOL
        else:
            costheta -= STARRY_THETA_UNIT_RADIUS_TOL

    # Get the points of intersection between the circle & ellipse.
    # These are the roots to a quartic equation.
    xo = bo * sintheta
    yo = bo * costheta
    x, y = get_roots(b, theta, costheta, sintheta, bo, ro)

    # No intersections between the circle and the ellipse.
    if len(x) == 0:
        # Is the center of the circle outside the ellipse?
        if abs(xo) > 1 or abs(yo) > b * (1 - xo**2) ** 0.5:
            # Is the center of the ellipse outside the circle?
            if bo > ro:
                # No occultation.
                phi1 = phi2 = xi1 = xi2 = 0.0
            else:
                # Complete occultation.
                phi1 = phi2 = xi1 = 0.0
                xi2 = 2 * np.pi
        else:
            # Regular occultation, but occultor doesn't touch the limb.
            phi1 = xi1 = xi2 = 0.0
            phi2 = 2 * np.pi

    elif len(x) == 1:
        pass  # Grazing configuration.

    # Regular occultation.
    elif len(x) == 2:
        xo = bo * sintheta
        yo = bo * costheta

        # Get angles.
        phi1 = theta + np.arctan2(y[0] - yo, x[0] - xo)
        phi2 = theta + np.arctan2(y[1] - yo, x[1] - xo)
        term = 1 - x[0] ** 2
        xi1 = np.arctan2((term if term >= 0 else 0) ** 0.5, x[0])

        if y[0] < 0:
            xi1 *= -1

        term = 1 - x[1] ** 2
        xi2 = np.arctan2((term if term >= 0 else 0) ** 0.5, x[1])

        if y[1] < 0:
            xi2 *= -1

        # Wrap and sort angles.
        phi1 = phi1 % (2 * np.pi)
        phi2 = phi2 % (2 * np.pi)
        xi1 = xi1 % (2 * np.pi)
        xi2 = xi2 % (2 * np.pi)

        # xi is always counter-clockwise.
        if xi1 > xi2:
            xi1, xi2 = xi2, xi1
            phi1, phi2 = phi2, phi1

        # Ensure T integral takes us through the inside of the occultor.
        mid = 0.5 * (xi1 + xi2)
        xm = np.cos(mid)
        ym = b * np.sin(mid)

        if (xm - xo) ** 2 + (ym - yo) ** 2 >= ro**2:
            xi1, xi2 = xi2, xi1
            xi2 += 2 * np.pi

        # Ensure P integral takes us through the inside of the star.
        mid = 0.5 * (phi1 + phi2)
        xm = xo + ro * np.cos(theta - mid)
        ym = yo - ro * np.sin(theta - mid)

        if ym**2 > b**2 * (1 - xm**2):
            if phi1 < phi2:
                phi1 += 2 * np.pi
            else:
                phi2 += 2 * np.pi

        # phi is always counter-clockwise.
        if phi1 > phi2:
            phi1, phi2 = phi2, phi1

    else:
        pass  # Pathological case.

    return phi1, phi2, xi1, xi2


# Polynomial root finder using an eigensolver.
def eigen_roots(coeffs):
    matsz = len(coeffs) - 1
    vret = []
    companion_mat = np.zeros((matsz, matsz), dtype=complex)

    for n in range(matsz):
        for m in range(matsz):
            if n == m + 1:
                companion_mat[n, m] = 1.0

            if m == matsz - 1:
                companion_mat[n, m] = -coeffs[matsz - n] / coeffs[0]

    eig = np.linalg.eigvals(companion_mat)
    vret.extend(eig)

    return vret


def get_roots(b, theta, sintheta, costheta, bo, ro):
    """Compute the points of intersection between a circle and an ellipse
    in the frame where the ellipse is centered at the origin,
    the semi-major axis of the ellipse is aligned with the x axis,
    and the circle is centered at `(xo, yo)`.
    """

    x = np.zeros(4)
    y = np.zeros(4)

    xo = bo * sintheta
    yo = bo * costheta

    b2 = b * b
    b4 = b2 * b2
    ro2 = ro * ro
    xo2 = xo * xo
    yo2 = yo * yo

    # Get quartic coefficients.
    coeffs = [
        (1 - b2) ** 2,
        -4 * xo * (1 - b2),
        -2 * (b4 + ro2 - 3 * xo2 - yo2 - b2 * (1 + ro2 - xo2 + yo2)),
        -4 * xo * (b2 - ro2 + xo2 + yo2),
        b4 - 2 * b2 * (ro2 - xo2 + yo2) + (ro2 - xo2 - yo2) ** 2,
    ]

    # TODO: Throw an exception if root eigensolver did not converge.
    roots = eigen_roots(coeffs)

    vret = []

    # for root in roots:
    for n in len(roots):
        root = roots[n]
        minerr = float("inf")  # == np.inf

        for _k in range(STARRY_ROOT_MAX_ITER):
            root2 = root**2
            root3 = root2 * root
            root4 = root3 * root
            f = (
                coeffs[0] * root4
                + coeffs[1] * root3
                + coeffs[2] * root2
                + coeffs[3] * root
                + coeffs[4]
            )
            absf = abs(f)

            if absf <= minerr:
                minerr = absf
                minxc = root

                if minerr <= STARRY_ROOT_TOL_HIGH:
                    break

            # Take a step.
            df = (
                4.0 * coeffs[0] * root3
                + 3.0 * coeffs[1] * root2
                + 2.0 * coeffs[2] * root
                + coeffs[3]
            )

            if df == 0.0:
                break

            root -= f / df

        root = minxc

        # Keep the root if it is real.
        if abs(root.imag) < STARRY_ROOT_TOL_HIGH:
            # Nudge the root away from the endpoints.
            if root.real > 1:
                root = 1.0 - STARRY_ROOT_TOL_HIGH
            elif root.real < -1:
                root = -1.0 + STARRY_ROOT_TOL_HIGH
            elif root.real < xo - ro:
                root = xo - ro + STARRY_ROOT_TOL_HIGH
            elif root.real > xo + ro:
                root = xo + ro - STARRY_ROOT_TOL_HIGH

            # Determine the y value of the point on the ellipse
            # corresponding to each root and the signs of the
            # functions describing the intersecting circle &
            # ellipse segments.
            fA = b * np.sqrt(1.0 - root * root)
            fB = np.sqrt(ro2 - (root - xo) * (root - xo))
            diff = [
                abs(fA - (yo + fB)),
                abs(fA - (yo - fB)),
                abs(-fA - (yo + fB)),
                abs(-fA - (yo - fB)),
            ]

            idx = np.argmin(diff)
            if idx < 2:
                s0 = 1.0
            else:
                s0 = -1.0
            if idx % 2 == 0:
                s1 = 1.0
            else:
                s1 = -1.0

            # Save the root.
            x[len(vret)] = root.real
            y[len(vret)] = s0 * fA.real

            # Compute the root's derivatives.
            if len(vret) > 0:
                # if N > 0:  # TODO: What is N?
                p = np.sqrt(1 - root.real**2)
                q = np.sqrt(ro2 - (root.real - xo) ** 2)
                v = (root.real - xo) / q
                w = b / p
                t = 1.0 / (w * root.real - (s1 * s0) * v)

                dxdb = t * p
                dxdtheta = -(s1 * costheta * v - sintheta) * (bo * t * s0)
                dxdbo = -(costheta + s1 * sintheta * v) * (t * s0)
                dxdro = -ro * t / q * s1 * s0

                # TODO: Derivatives?
                x[len(vret)].derivatives = (
                    dxdb * b.derivatives()
                    + dxdtheta * theta.derivatives()
                    + dxdbo * bo.derivatives()
                    + dxdro * ro.derivatives()
                )

                y[len(vret)] = s0 * b * np.sqrt(1.0 - x[len(vret)] * x[len(vret)])

            vret.append((x[len(vret)], y[len(vret)]))

    return x[: len(vret)], y[: len(vret)]


# Compute matrix L for a complete occultation.
def compute_L0(lmax):
    nmax = lmax + 3
    L = np.zeros((nmax, nmax), dtype=float)

    # Set lower boundary.
    L[0, 0] = 2 * np.pi

    # Recurse
    for u in range(0, nmax, 2):
        for v in range(2, nmax, 2):
            fac = (v - 1.0) / (u + v)
            L[u, v] = fac * L[u, v - 2]
            L[v, u] = fac * L[v - 2, u]

    return L


# Compute sT for a complete occultation.
def sT0(lmax):
    ncoeff = (lmax + 1) * (lmax + 1)
    sT0 = np.zeros(ncoeff)

    Lt = compute_L0(lmax)

    n = 0
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            mu = l - m
            nu = l + m
            if nu % 2 == 0:
                # Case 1
                sT0[n] = Lt[mu // 2 + 2, nu // 2]
            elif l == 1 and m == 0:
                # Case 2
                sT0[n] = 2 * np.pi / 3
            n += 1

    return sT0


# Compute the L integral.
def compute_L(lmax, phip1, phip2):
    nmax = lmax + 3
    cp1 = np.cos(phip1)
    cp2 = np.cos(phip2)
    sp1 = np.sin(phip1)
    sp2 = np.sin(phip2)
    L = np.zeros((nmax, nmax), dtype=float)

    # Set lower boundaries.
    L[0, 0] = phip2 - phip1
    L[1, 0] = sp2 - sp1
    L[0, 1] = cp1 - cp2
    L[1, 1] = 0.5 * (cp1**2 - cp2**2)

    # Set recursion coefficients.
    A0 = cp1 * sp1
    B0 = cp2 * sp2
    C0 = cp2 * sp2
    D0 = cp1 * sp1

    # Recurse
    for u in range(nmax):
        A1, B1, C1, D1 = A0, B0, C0, D0
        for v in range(2, nmax):
            fac = 1.0 / (u + v)
            L[u, v] = fac * (A1 - B1 + (v - 1) * L[u, v - 2])
            L[v, u] = fac * (C1 - D1 + (v - 1) * L[v - 2, u])
            A1 *= sp1
            B1 *= sp2
            C1 *= cp2
            D1 *= cp1
        A0 *= cp1
        B0 *= cp2
        C0 *= sp2
        D0 *= sp1

    return L


# Compute the M integral analytically for even mu.
def compute_M(lmax, theta, sintheta, costheta, bo, ro, phi1, phi2):
    S = np.zeros((lmax + 3, lmax + 3))
    C = np.zeros((lmax + 3, lmax + 3))
    fac0 = 1.0

    for i in range(lmax + 3):
        facs = ro * fac0
        facc = fac0
        for j in range(lmax + 3):
            if j < i + 1:
                S[i, lmax + 2 + j - i] = facs
                C[lmax + 2 + j - i, i] = facc
                fac = bo * (i - j) / (ro * (j + 1.0))
                facs *= fac * sintheta
                facc *= fac * costheta
        fac0 *= ro

    Lp = compute_L(lmax, phi1 - theta, phi2 - theta)
    M = np.dot(
        np.dot(S[: lmax + 2, : lmax + 2], Lp[::-1, : lmax + 2]), C[:, : lmax + 2]
    )

    return M


def compute_pTodd(lmax, f, theta, sintheta, costheta, bo, ro, phi1, phi2):
    pass
