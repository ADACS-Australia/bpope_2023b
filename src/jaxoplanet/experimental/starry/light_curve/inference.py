from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import config

from jaxoplanet.experimental.starry.basis import A1, A
from jaxoplanet.experimental.starry.light_curve.utils import right_project
from jaxoplanet.experimental.starry.rotation import dotR
from jaxoplanet.experimental.starry.solution import rT_solution_vector, solution_vector
from jaxoplanet.types import Array

config.update("jax_enable_x64", True)


"""

Notes
-----

    - Currently, solve() and lnlike() accept the parameter "bodies",
      which is a list of starry.Primary- and starry.Secondary-like objects.
    - solve() and lnlike() can't yet accept a time series as input over a
      design matrix, since jaxoplanet does not currently have a function
      for constructing a design matrix.


Still to do
-----------

    check if jax has any functions we can use
    jit/partial jit functions
    add signatures
    add input types
    construct an 'eclipsing binary' file from these functions.

"""


@partial(jax.jit, static_argnames=("l_max", "order"))
def design_matrix(
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
    b = jnp.sqrt(xo**2 + yo**2)
    theta_z = jnp.arctan2(xo, yo)

    # occultation mask
    cond_rot = (b >= (1.0 + ro)) | (zo <= 0.0) | (ro == 0.0)
    cond_rot_ = cond_rot[:, jnp.newaxis]

    sT = solution_vector(l_max, order=order)(b, ro)
    sTA = sT @ A(l_max)
    sTAR = dotR(l_max, [0, 0, 1])(sTA, theta_z)

    # rotational phase
    rTA1 = rT_solution_vector(l_max) @ A1(l_max)
    rTA1_ = jnp.broadcast_to(rTA1, sTAR.shape)

    # applying occultation mask
    sTAR_ = jnp.where(~cond_rot_, sTAR, rTA1_)

    # get the design matrix
    X = right_project(l_max, sTAR_, inc, obl, theta)

    return X


def cast(*args):
    if len(args) == 1:
        return jnp.asarray(args[0], dtype="float64")
    else:
        return [jnp.asarray(arg, dtype="float64") for arg in args]


def cho_solve(A: Array, b: Array) -> Array:
    b_ = jax.scipy.linalg.solve_triangular(A, b, lower=True)
    return jax.scipy.linalg.solve_triangular(jnp.transpose(A), b_, lower=False)


def get_covariance(
    C: float | Array = None, cho_C: float | Array = None, N: int = None
) -> Tuple[Array, Array, Array, Array, str, int]:
    """A container for covariance matrices.

    Args:
        C (scalar, vector, or matrix, optional): The covariance.
            Defaults to None.
        cho_C (matrix, optional): The lower Cholesky factorization of
            the covariance. Defaults to None.
        N (int, optional): The number of rows/columns in the covariance
            matrix, required if ``C`` is a scalar. Defaults to None.
    """

    # User provided the Cholesky factorization.
    if cho_C is not None:
        cholesky = cast(cho_C)
        value = jnp.dot(cholesky, jnp.transpose(cholesky))
        inverse = cho_solve(cholesky, jnp.eye(cholesky.shape[0]))
        lndet = 2 * jnp.sum(jnp.log(jnp.diag(cholesky)))
        kind = "cholesky"
        N = cho_C.shape[0]

    # User provided the covariance as a scalar, vector, or matrix.
    elif C is not None:
        C = cast(C)

        if hasattr(C, "ndim"):
            if C.ndim == 0:
                assert N is not None, "Please provide a matrix size `N`."
                cholesky = jnp.sqrt(C)
                inverse = cast(1.0 / C)
                lndet = cast(N * jnp.log(C))
                value = C
                kind = "scalar"

            elif C.ndim == 1:
                cholesky = jnp.sqrt(C)
                inverse = 1.0 / C
                lndet = jnp.sum(jnp.log(C))
                value = C
                kind = "vector"
                N = C.shape[0]

            else:
                cholesky = jax.scipy.linalg.cholesky(C, lower=True)
                inverse = cho_solve(cholesky, jnp.eye(C.shape[0]))
                lndet = 2 * jnp.sum(jnp.log(jnp.diag(cholesky)))
                value = C
                kind = "matrix"
                N = C.shape[0]

        # Assume it's a scalar
        else:
            assert N is not None, "Please provide a matrix size `N`."
            cholesky = jnp.sqrt(C)
            inverse = cast(1.0 / C)
            lndet = cast(N * jnp.log(C))
            value = C
            kind = "scalar"

    else:
        raise ValueError(
            "Either the covariance or its Cholesky factorization must be provided."
        )

    return value, cholesky, inverse, lndet, kind, N


def set_data(
    flux: Array, C: float | Array = None, cho_C: float | Array = None
) -> Tuple[Array, tuple]:
    """Set the data vector and covariance matrix.

    This method is required by the :py:meth:`solve` method, which
    analytically computes the posterior over surface maps for all bodies
    in the system given a dataset and a prior, provided both are described
    as multivariate Gaussians.

    Args:
        flux (vector): The observed system light curve.
        C (scalar, vector, or matrix): The data covariance. This may be
            a scalar, in which case the noise is assumed to be
            homoscedastic, a vector, in which case the covariance
            is assumed to be diagonal, or a matrix specifying the full
            covariance of the dataset. Default is None. Either `C` or
            `cho_C` must be provided.
        cho_C (matrix): The lower Cholesky factorization of the data
            covariance matrix. Defaults to None. Either `C` or
            `cho_C` must be provided.
    """
    flux = cast(flux)
    C = get_covariance(C=C, cho_C=cho_C, N=flux.shape[0])

    return flux, C


def set_prior(
    lmax: int,
    mu: float | Array = None,
    L: float | Array = None,
    cho_L: float | Array = None,
) -> Tuple[Array, tuple]:
    """Set the prior mean and covariance of the spherical harmonic coefficients.

    This method is required by the :py:meth:`solve` method, which
    analytically computes the posterior over surface maps given a
    dataset and a prior, provided both are described as multivariate
    Gaussians.

    Note that the prior is placed on the **amplitude-weighted** coefficients,
    i.e., the quantity ``x = map.amp * map.y``. Because the first spherical
    harmonic coefficient is fixed at unity, ``x[0]`` is
    the amplitude of the map. The actual spherical harmonic coefficients
    are given by ``x / map.amp``.

    This convention allows one to linearly fit for an arbitrary map normalization
    at the same time as the spherical harmonic coefficients, while ensuring
    the ``starry`` requirement that the coefficient of the :math:`Y_{0,0}`
    harmonic is always unity.

    Args:
        lmax (scalar): The maximum degree of spherical harmonic coefficients.
        mu (scalar or vector): The prior mean on the amplitude-weighted
            spherical harmonic coefficients. Default is `1.0` for the
            first term and zero for the remaining terms. If this is a vector,
            it must have length equal to :py:attr:`Ny`.
        L (scalar, vector, or matrix): The prior covariance. This may be
            a scalar, in which case the covariance is assumed to be
            homoscedastic, a vector, in which case the covariance
            is assumed to be diagonal, or a matrix specifying the full
            prior covariance. Default is None. Either `L` or
            `cho_L` must be provided.
        cho_L (matrix): The lower Cholesky factorization of the prior
            covariance matrix. Defaults to None. Either `L` or
            `cho_L` must be provided.

    """
    Ny = (lmax + 1) * (lmax + 1)

    if mu is None:
        mu = cast(jnp.zeros(Ny).at[0].set(1.0))

    _mu = cast(mu) * cast(jnp.ones(Ny))
    _L = get_covariance(C=L, cho_C=cho_L, N=Ny)

    return _mu, _L


def map_solve(
    X: Array, flux: Array, cho_C: float | Array, mu: Array, LInv: float | Array
) -> Tuple[Array, Array]:
    """
    Compute the maximum a posteriori (MAP) prediction for the
    spherical harmonic coefficients of a map given a flux timeseries.

    Args:
        X (matrix): The flux design matrix.
        flux (array): The flux timeseries.
        cho_C (scalar/vector/matrix): The lower cholesky factorization
            of the data covariance.
        mu (array): The prior mean of the spherical harmonic coefficients.
        LInv (scalar/vector/matrix): The inverse prior covariance of the
            spherical harmonic coefficients.

    Returns:
        The vector of spherical harmonic coefficients corresponding to the
        MAP solution and the Cholesky factorization of the corresponding
        covariance matrix.

    """
    # Compute C^-1 . X
    if cho_C.ndim == 0:
        CInvX = X / cho_C**2
    elif cho_C.ndim == 1:
        CInvX = jnp.dot(jnp.diag(1 / cho_C**2), X)
    else:
        CInvX = cho_solve(cho_C, X)

    # Compute W = X^T . C^-1 . X + L^-1
    W = jnp.dot(jnp.transpose(X), CInvX)
    # If LInv is a scalar or a 1-dimensional array, increment the
    # diagonal elements of W with the values from LInv.
    if LInv.ndim == 0 or LInv.ndim == 1:
        W = W.at[jnp.diag_indices_from(W)].set(W[jnp.diag_indices_from(W)] + LInv)
        LInvmu = mu * LInv
    # If LInv is a matrix, directly add LInv to W.
    else:
        W += LInv
        LInvmu = jnp.dot(LInv, mu)

    # Compute the max like y and its covariance matrix
    cho_W = jax.scipy.linalg.cholesky(W, lower=True)
    M = cho_solve(cho_W, jnp.transpose(CInvX))
    yhat = jnp.dot(M, flux) + cho_solve(cho_W, LInvmu)
    ycov = cho_solve(cho_W, jnp.eye(cho_W.shape[0]))
    cho_ycov = jax.scipy.linalg.cholesky(ycov, lower=True)

    return yhat, cho_ycov


def solve(
    lmax: int,
    flux: Array,
    C: tuple,
    bodies: list,
    design_matrix: Array = None,
    t: Array = None,
) -> Tuple[Array, Array]:
    """Solve the least-squares problem for the posterior over maps for all bodies.

    This method solves the generalized least squares problem given a system
    light curve and its covariance (set via the :py:meth:`set_data` method)
    and a Gaussian prior on the spherical harmonic coefficients
    (set via the :py:meth:`set_prior` method). The map amplitudes and
    coefficients of each of the bodies in the system are then set to the
    maximum a posteriori (MAP) solution.

    Args:
        lmax (scalar): The maximum degree of spherical harmonic coefficients.
        flux (array): The flux timeseries.
        C (tuple): A container of covariance matrices (value, cholesky,
            inverse, lndet, kind, N).
        bodies (list): The bodies of the system passed in as dictionaries
            of the bodies' attributes.
        design_matrix (matrix, optional): The flux design matrix, the
            quantity returned by :py:meth:`design_matrix`. Default is
            None, in which case this is computed based on ``kwargs``.
        t (vector, optional): The vector of times at which to evaluate
            :py:meth:`design_matrix`, if a design matrix is not provided.
            Default is None.

    Returns:
        The posterior mean for the spherical harmonic \
        coefficients `l > 0` and the Cholesky factorization of the \
        posterior covariance of all of the bodies in the system, \
        stacked in order (primary, followed by each of the secondaries \
        in the order they were provided.)

    .. note::
        Users may call the :py:meth:`draw` method of this class to draw
        from the posterior after calling :py:meth:`solve`.
    """

    Ny = (lmax + 1) * (lmax + 1)

    # Get the full design matrix
    if design_matrix is None:
        raise ValueError("Design matrix construction not yet implemented.")
        # assert t is not None, "Please provide a time vector `t`."
        # design_matrix = design_matrix(t)
    X = cast(design_matrix)

    # Get the data vector
    f = cast(flux)

    # Check for bodies whose priors are set
    solved_bodies = []
    inds = []
    dense_L = False
    for k, body in enumerate(bodies):
        # If no priors have been set on this body
        if body["mu"] is None or body["L"] is None:
            # Subtract out this term from the data vector,
            # since it is fixed
            f -= body["amp"] * jnp.dot(X[:, jnp.arange(Ny) + Ny * k], body["y"])

        else:
            # Add to our list of indices/bodies to solve for
            inds.extend(jnp.arange(Ny) + Ny * k)
            solved_bodies.append(body)
            if body["L"]["kind"] in ["matrix", "cholesky"]:
                dense_L = True

    # Do we have at least one body?
    if len(solved_bodies) == 0:
        raise ValueError("Please provide a prior for at least one body.")

    # Keep only the terms we'll solve for
    X = X[:, inds]

    # Stack our priors
    mu = jnp.concatenate([body["mu"] for body in solved_bodies])

    if not dense_L:
        # We can just concatenate vectors
        LInv = jnp.concatenate(
            [body["L"]["inverse"] * jnp.ones(body["n_max"]) for body in solved_bodies]
        )
    else:
        # FACT: The inverse of a block diagonal matrix
        # is the block diagonal matrix of the inverses.
        LInv = jax.scipy.linalg.block_diag(
            *[body["L"]["inverse"] * jnp.eye(body["n_max"]) for body in solved_bodies]
        )

    # Compute the MAP solution
    x, cho_cov = map_solve(X, f, C[1], mu, LInv)

    # Set all the map vectors
    n = 0
    for body in solved_bodies:
        inds = slice(n, n + body["n_max"])
        body["amp"] = x[inds][0]
        if body["l_max"] > 0:
            body["y"][1:] = x[inds][1:] / body["amp"]
        n += body["n_max"]

    # Return the mean and covariance
    return (x, cho_cov)


def get_lnlike(
    X: Array, flux: Array, C: float | Array, mu: Array, L: float | Array
) -> Array:
    """
    Compute the log marginal likelihood of the data given a design matrix.

    Args:
        X (matrix): The flux design matrix.
        flux (array): The flux timeseries.
        C (scalar/vector/matrix): The data covariance matrix.
        mu (array): The prior mean of the spherical harmonic coefficients.
        L (scalar/vector/matrix): The prior covariance of the spherical
            harmonic coefficients.

    Returns:
        The log marginal likelihood of the `flux` vector conditioned on
        the design matrix `X`. This is the likelihood marginalized over
        all possible spherical harmonic vectors, which is analytically
        computable for the linear `starry` model.

    """
    # Compute the GP mean
    gp_mu = jnp.dot(X, mu)

    # Compute the GP covariance
    if L.ndim == 0:
        XLX = jnp.dot(X, jnp.transpose(X)) * L
    elif L.ndim == 1:
        XLX = jnp.dot(jnp.dot(X, jnp.diag(L)), jnp.transpose(X))
    else:
        XLX = jnp.dot(jnp.dot(X, L), jnp.transpose(X))
    # If C is a scalar or a 1-dimensional array, increment the
    # diagonal elements of XLX with the values from C.
    if C.ndim == 0 or C.ndim == 1:
        gp_cov = XLX.at[jnp.diag_indices_from(XLX)].set(
            XLX[jnp.diag_indices_from(XLX)] + C
        )
    # If C is a matrix, directly add C to XLX.
    else:
        gp_cov = C + XLX

    cho_gp_cov = jax.scipy.linalg.cholesky(gp_cov, lower=True)

    # Compute the marginal likelihood
    N = X.shape[0]
    r = jnp.reshape(flux - gp_mu, (-1, 1))
    lnlike = -0.5 * jnp.dot(jnp.transpose(r), cho_solve(cho_gp_cov, r))
    lnlike -= jnp.sum(jnp.log(jnp.diag(cho_gp_cov)))
    lnlike -= 0.5 * N * jnp.log(2 * jnp.pi)

    return lnlike[0, 0]


def get_lnlike_woodbury(
    X: Array,
    flux: Array,
    CInv: float | Array,
    mu: Array,
    LInv: float | Array,
    lndetC: float | Array,
    lndetL: float | Array,
) -> Array:
    """
    Compute the log marginal likelihood of the data given a design matrix
    using the Woodbury identity.

    Args:
        X (matrix): The flux design matrix.
        flux (array): The flux timeseries.
        CInv (scalar/vector/matrix): The inverse data covariance matrix.
        mu (array): The prior mean of the spherical harmonic coefficients.
        LInv (scalar/vector/matrix): The inverse prior covariance of the
            spherical harmonic coefficients.
        lndetC: ...
        lndetL: # TODO: describe last 3 inputs.

    Returns:
        The log marginal likelihood of the `flux` vector conditioned on
        the design matrix `X`. This is the likelihood marginalized over
        all possible spherical harmonic vectors, which is analytically
        computable for the linear `starry` model.

    """
    # Compute the GP mean
    gp_mu = jnp.dot(X, mu)

    # Residual vector
    r = jnp.reshape(flux - gp_mu, (-1, 1))

    # Inverse of GP covariance via Woodbury identity
    if CInv.ndim == 0:
        U = X * CInv
    elif CInv.ndim == 1:
        U = jnp.dot(jnp.diag(CInv), X)
    else:
        U = jnp.dot(CInv, X)

    if LInv.ndim == 0:
        W = jnp.dot(jnp.transpose(X), U) + LInv * jnp.eye(U.shape[1])
    elif LInv.ndim == 1:
        W = jnp.dot(jnp.transpose(X), U) + jnp.diag(LInv)
    else:
        W = jnp.dot(jnp.transpose(X), U) + LInv
    cho_W = jax.scipy.linalg.cholesky(W, lower=True)

    if CInv.ndim == 0:
        SInv = CInv * jnp.eye(U.shape[0]) - jnp.dot(
            U, cho_solve(cho_W, jnp.transpose(U))
        )
    elif CInv.ndim == 1:
        SInv = jnp.diag(CInv) - jnp.dot(U, cho_solve(cho_W, jnp.transpose(U)))
    else:
        SInv = CInv - jnp.dot(U, cho_solve(cho_W, jnp.transpose(U)))

    # Determinant of GP covariance
    lndetW = 2 * jnp.sum(jnp.log(jnp.diag(cho_W)))
    lndetS = lndetW + lndetC + lndetL

    # Compute the marginal likelihood
    N = X.shape[0]
    lnlike = -0.5 * jnp.dot(jnp.transpose(r), jnp.dot(SInv, r))
    lnlike -= 0.5 * lndetS
    lnlike -= 0.5 * N * jnp.log(2 * jnp.pi)

    return lnlike[0, 0]


def lnlike(
    lmax: int,
    flux: Array,
    C: tuple,
    bodies: list,
    design_matrix: Array = None,
    t: Array = None,
    woodbury: bool = True,
) -> Array:
    """Returns the log marginal likelihood of the data given a design matrix.

    This method computes the marginal likelihood (marginalized over the
    spherical harmonic coefficients of all bodies) given a system
    light curve and its covariance (set via the :py:meth:`set_data` method)
    and a Gaussian prior on the spherical harmonic coefficients
    (set via the :py:meth:`set_prior` method).

    Args:
        lmax (scalar): The maximum degree of spherical harmonic coefficients.
        flux (array): The flux timeseries.
        C (tuple): A container of covariance matrices (value, cholesky,
            inverse, lndet, kind, N).
        bodies (list): The bodies of the system passed in as dictionaries
            of the bodies' attributes.
        design_matrix (matrix, optional): The flux design matrix, the
            quantity returned by :py:meth:`design_matrix`. Default is
            None, in which case this is computed based on ``kwargs``.
        t (vector, optional): The vector of times at which to evaluate
            :py:meth:`design_matrix`, if a design matrix is not provided.
            Default is None.
        woodbury (bool, optional): Solve the linear problem using the
            Woodbury identity? Default is True. The
            `Woodbury identity <https://en.wikipedia.org/wiki/Woodbury_matrix_identity>`_
            is used to speed up matrix operations in the case that the
            number of data points is much larger than the number of
            spherical harmonic coefficients. In this limit, it can
            speed up the code by more than an order of magnitude. Keep
            in mind that the numerical stability of the Woodbury identity
            is not great, so if you're getting strange results try
            disabling this. It's also a good idea to disable this in the
            limit of few data points and large spherical harmonic degree.

    Returns:
        lnlike: The log marginal likelihood.
    """

    Ny = (lmax + 1) * (lmax + 1)

    # Get the full design matrix
    if design_matrix is None:
        raise ValueError("Design matrix construction not yet implemented.")
        # assert t is not None, "Please provide a time vector `t`."
        # design_matrix = design_matrix(t)
    X = cast(design_matrix)

    # Get the data vector
    f = cast(flux)

    # Check for bodies whose priors are set
    solved_bodies = []
    inds = []
    dense_L = False
    for k, body in enumerate(bodies):
        # If no priors have been set on this body
        if body["mu"] is None or body["L"] is None:
            # Subtract out this term from the data vector,
            # since it is fixed
            f -= body["amp"] * jnp.dot(X[:, jnp.arange(Ny) + Ny * k], body["y"])

        else:
            # Add to our list of indices/bodies to solve for
            inds.extend(jnp.arange(Ny) + Ny * k)
            solved_bodies.append(body)
            if body["L"]["kind"] in ["matrix", "cholesky"]:
                dense_L = True

    # Do we have at least one body?
    if len(solved_bodies) == 0:
        raise ValueError("Please provide a prior for at least one body.")

    # Keep only the terms we'll solve for
    X = X[:, inds]

    # Stack our priors
    mu = jnp.concatenate([body["mu"] for body in solved_bodies])

    # Compute the likelihood
    if woodbury:
        if not dense_L:
            # We can just concatenate vectors
            LInv = jnp.concatenate(
                [
                    body["L"]["inverse"] * jnp.ones(body["n_max"])
                    for body in solved_bodies
                ]
            )
        else:
            LInv = jnp.block_diag(
                *[
                    body["L"]["inverse"] * jnp.eye(body["n_max"])
                    for body in solved_bodies
                ]
            )
        lndetL = cast([body["L"]["lndet"] for body in solved_bodies])
        return get_lnlike_woodbury(X, f, C[2], mu, LInv, C[3], lndetL)
    else:
        if not dense_L:
            # We can just concatenate vectors
            L = jnp.concatenate(
                [body["L"]["value"] * jnp.ones(body["n_max"]) for body in solved_bodies]
            )
        else:
            L = jnp.block_diag(
                *[body["L"]["value"] * jnp.eye(body["n_max"]) for body in solved_bodies]
            )
        return get_lnlike(X, f, C[0], mu, L)
