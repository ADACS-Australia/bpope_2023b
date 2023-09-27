import jax
import jax.numpy as jnp
import numpy as np
from jax import config

config.update("jax_enable_x64", True)


"""

TESTED against starry results:

    set_data(flux, C=sigma ** 2)
    -> get_covariance()
    -> cast()
    -> cho_solve()

    set_prior(lmax=5, mu=pri_mu, L=pri_L)
    -> same as above

"""


def cast(*args):
    if len(args) == 1:
        return jnp.asarray(args[0], dtype="float64")
    else:
        return [jnp.asarray(arg, dtype="float64") for arg in args]


def cho_solve(A, b):
    b_ = jax.scipy.linalg.solve_triangular(A, b, lower=True)
    return jax.scipy.linalg.solve_triangular(jnp.transpose(A), b_, lower=False)


def get_covariance(C=None, cho_C=None, N=None):
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


def set_data(flux, C=None, cho_C=None):
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


def set_prior(lmax, mu=None, L=None, cho_L=None):
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
        mu = np.zeros(Ny)
        mu[0] = 1.0
        mu = cast(mu)

    _mu = cast(mu) * cast(np.ones(Ny))
    _L = get_covariance(C=L, cho_C=cho_L, N=Ny)

    return _mu, _L


def map_solve(X, flux, cho_C, mu, LInv):
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


def solve(lmax, flux, C, bodies, design_matrix=None, t=None):
    """Solve the least-squares problem for the posterior over maps for all bodies.

    This method solves the generalized least squares problem given a system
    light curve and its covariance (set via the :py:meth:`set_data` method)
    and a Gaussian prior on the spherical harmonic coefficients
    (set via the :py:meth:`set_prior` method). The map amplitudes and
    coefficients of each of the bodies in the system are then set to the
    maximum a posteriori (MAP) solution.

    Args:
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
        assert t is not None, "Please provide a time vector `t`."
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
        if body.map._mu is None or body.map._L is None:
            # Subtract out this term from the data vector,
            # since it is fixed
            f -= body.map.amp * jnp.dot(X[:, jnp.arange(Ny) + Ny * k], body.map.y)

        else:
            # Add to our list of indices/bodies to solve for
            inds.extend(jnp.arange(Ny) + Ny * k)
            solved_bodies.append(body)
            if body.map._L.kind in ["matrix", "cholesky"]:
                dense_L = True

    # Do we have at least one body?
    if len(solved_bodies) == 0:
        raise ValueError("Please provide a prior for at least one body.")

    # Keep only the terms we'll solve for
    X = X[:, inds]

    # Stack our priors
    mu = jnp.concatenate([body.map._mu for body in solved_bodies])

    if not dense_L:
        # We can just concatenate vectors
        LInv = jnp.concatenate(
            [body.map._L.inverse * jnp.ones(body.map.Ny) for body in solved_bodies]
        )
    else:
        # FACT: The inverse of a block diagonal matrix
        # is the block diagonal matrix of the inverses.
        LInv = jax.scipy.linalg.block_diag(
            *[body.map._L.inverse * jnp.eye(body.map.Ny) for body in solved_bodies]
        )

    # Compute the MAP solution
    x, cho_cov = map_solve(X, f, C[1], mu, LInv)

    # Set all the map vectors
    n = 0
    for body in solved_bodies:
        inds = slice(n, n + body.map.Ny)
        body.map.amp = x[inds][0]
        if body.map.ydeg > 0:
            body.map[1:, :] = x[inds][1:] / body.map.amp
        n += body.map.Ny

    # Return the mean and covariance
    return (x, cho_cov)
