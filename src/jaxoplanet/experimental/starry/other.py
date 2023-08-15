import jax
import jax.numpy as jnp
import numpy as np

from scipy.special import gamma
from typing import Callable, Tuple
from functools import partial

from jax import jit, vmap, lax
import bpope_2023b.src.jaxoplanet as jaxoplanet
from jaxoplanet.types import Array
from jaxoplanet.test_utils import assert_allclose

from jaxoplanet.experimental.starry.basis import (A1,A2_inv)
from jaxoplanet.experimental.starry.rotation import (dotR)
from jaxoplanet.experimental.starry.solution import (solution_vector)

# -----------------------------------------------------------------------------

np.set_printoptions(suppress = True) # Display arrays in decimal notation, not scientific.
jax.config.update("jax_enable_x64", True) # Set double precision.

# -----------------------------------------------------------------------------

@jax.jit
def compute_cos_sin_nt(cos_nt,sin_nt,n):
    
    # Compute nth columns of cos_nt/sin_nt matrices
    cos_nt_n = 2.0*cos_nt[:,n-1] * cos_nt[:,1] - cos_nt[:,n-2]
    sin_nt_n = 2.0*sin_nt[:,n-1] * cos_nt[:,1] - sin_nt[:,n-2]
    
    return cos_nt_n, sin_nt_n

# Compute the Rz (tensor) rotation matrix.
@jax.jit
def computeRz(M: Array, theta: Array) -> Tuple[Array, Array]:
    
    n_max = M.shape[1]
    l_max = int(np.sqrt(n_max)-1)  # Degree of spherical harmonics.

    npts  = theta.size             # Length of timeseries.
    
    if npts == 0:
        return jnp.zeros((npts,n_max)), jnp.zeros((npts,n_max))
    
    # Initialise z rotation vectors
    cos_nt = jnp.zeros((npts,np.max(np.array([[2,l_max+1]]))))
    sin_nt = jnp.zeros((npts,np.max(np.array([[2,l_max+1]]))))
    
    cos_nt = cos_nt.at[:,0].set(1)
    cos_nt = cos_nt.at[:,1].set(jnp.cos(theta))
    sin_nt = sin_nt.at[:,1].set(jnp.sin(theta))
    
    # Compute cos & sin vectors for the z_hat rotation.
    for n in range(2,l_max+1):
        # Compute nth column of cos_nt/sin_nt matrices.
        cos_temp, sin_temp = compute_cos_sin_nt(cos_nt,sin_nt,n)
        cos_nt = cos_nt.at[:,n].set(cos_temp)
        sin_nt = sin_nt.at[:,n].set(sin_temp)
    
    # Construct m with for-loop (cannot jaxify as m.shape depends on i).
    m = jnp.array([0])
    for i in range(1,l_max+1):
        m = jnp.append(m, jnp.arange(-i,i+1))
    
    # Construct cos(m theta) and sin(m theta) matrix.
    cos_mt = jnp.where(m<0, cos_nt[:,-m], cos_nt[:,m])
    sin_mt = jnp.where(m<0, -sin_nt[:,-m], sin_nt[:,m])
    
    return cos_mt, sin_mt

# -----------------------------------------------------------------------------

# Two versions of tensordotRz, one returns a function of M(theta), other a value.

# Computes the tensor dot product M.Rz(theta).
# Returns an Array.
# Example: sTAR = tensordotRz(sTA)(theta_z)
def tensordotRz_callable(M: Array) -> Callable[[Array], Array]:
    
    # Check shapes.
    Nr   = M.shape[1]
    degr = int(np.sqrt(Nr)-1) # Degree of spherical harmonics.
    
    # Construct j with for-loop. (cannot jaxify as j shape depends on degr/i)
    j = np.array([0])
    for i in range(1,degr+1):
        j = np.append(j,np.arange((i+1)*(i+1)-1,i*i-1,-1))
    
    # Dot them in.
    @jax.jit
    def compute_tensordotRz(theta: Array) -> Array:
        cos_mt, sin_mt = computeRz(M, theta)
        # Elementwise multiplication of matrices.
        return M*cos_mt + M[:,j]*sin_mt

    return compute_tensordotRz
# -----------------------------------------
# Returns a Callable.
# Example: sTAR = tensordotRz(sTA, theta_z)
@jax.jit
def tensordotRz(M: Array, theta: Array) -> Array:
    
    # Check shapes.
    Nr   = M.shape[1]
    degr = int(np.sqrt(Nr)-1) # Degree of spherical harmonics.
    
    # Construct j with for-loop. (cannot jaxify as j shape depends on degr/i)
    j = np.array([0])
    for i in range(1,degr+1):
        j = np.append(j,np.arange((i+1)*(i+1)-1,i*i-1,-1))
    
    # Dot them in.
    cos_mt, sin_mt = computeRz(M, theta)
        
    # Elementwise multiplication of matrices.
    return M*cos_mt + M[:,j]*sin_mt

#tensordotRz_callable(sTA)(theta_z)
#tensordotRz(sTA, theta_z)
# -----------------------------------------------------------------------------
# Only partially jitted

@jax.jit
def compute_tensordotRz_b(M,bMRz,cos_mt,sin_mt,j,m):
    
    # Dot the sines, cosines in.
    tmp_c = bMRz * cos_mt          # Not sure if this is needed with jax?
    tmp_s = bMRz * sin_mt
    
    # --- d/d_theta ---
    tensordotRz_btheta = jnp.sum(m*(M[:,j]*tmp_c - M*tmp_s), axis=1)
     
    # --- d/d_M     ---
    # Separate cases if M is a row vector or not, since
    #          shape(tensordotRz_bM) == (M.shape[0],Nr).
    if M.shape[0]==1:
        tensordotRz_bM = jnp.sum(tmp_c, axis=0) + jnp.sum(tmp_s[:,j], axis=0)
    else:
        tensordotRz_bM = tmp_c + tmp_s[:,j]
    
    # Alternatively:
    # tensordotRz_bM = jnp.where(M.shape[0]==1,
    #                            jnp.sum(tmp_c, axis=0) + jnp.sum(tmp_s[:,j], axis=0),
    #                            tmp_c + tmp_s[:,j])
    
    return tensordotRz_btheta, tensordotRz_bM
    
def tensordotRzGrad(M: Array, theta: Array, bMRz: Array):
    
    # Shape checks
    npts = theta.size
    Nr   = M.shape[1]
    degr = int(np.sqrt(Nr)-1) # check this
    
    # Initial/Initialise(?) gradients.
    tensordotRz_btheta = np.zeros(npts)
    tensordotRz_bM     = np.zeros((M.shape[0],Nr))
    
    if npts==0 or M.shape[0]==0:
        return tensordotRz_btheta, tensordotRz_bM
    
    # l*l+2*l-j
    j = np.array([0])
    for i in range(1,degr+1):
        j = np.append(j,np.arange((i+1)*(i+1)-1,i*i-1,-1))
    
    # j-l
    m = np.array([0])
    for i in range(1,degr+1):
        m = np.append(m,np.arange(-i,i+1))
        
    # Compute sin_mt, cos_mt.
    cos_mt, sin_mt = computeRz(degr, theta)
    
    tensordotRz_btheta, tensordotRz_bM = compute_tensordotRz_b(M,bMRz,
                                                               cos_mt,sin_mt,
                                                               j,m)

    return tensordotRz_btheta, tensordotRz_bM

# -----------------------------------------------------------------------------

#@partial(jit, static_argnums=(1,2)) # Error: a value becomes a tracer in Rl().
def right_project(M: Array, inc: int, obl: int, theta: Array):
    r"""Apply the projection operator on the right.

    Specifically, this method returns the dot product :math:`M \cdot R`,
    where ``M`` is an input matrix and ``R`` is the Wigner rotation matrix
    that transforms a spherical harmonic coefficient vector in the
    input frame to a vector in the observer's frame.
    """
    l_max = int(jnp.sqrt(M.shape[1])-1)
    
    cos_obl = jnp.cos(obl)
    sin_obl = jnp.sin(obl)
    
    # Rotate to the sky frame.
    M = M.at[:].set(dotR(l_max, [-cos_obl,-sin_obl,0.0])(M, -((0.5*jnp.pi)-inc)))
    M = M.at[:].set(dotR(l_max, [0.0,0.0,1.0])(M, obl))
    M = M.at[:].set(dotR(l_max, [1,0,0])(M, (-0.5*jnp.pi)))
    
    # # Rotate to the correct phase.
    M = M.at[:].set(tensordotRz(M, theta))
    
    # Rotate to the polar frame.
    M = M.at[:].set(dotR(l_max,[1.0,0.0,0.0])(M,0.5*jnp.pi))

    return M

# -----------------------------------------------------------------------------

# Compute rT analytically.
# (from starry paper and https://starry.readthedocs.io/en/latest/notebooks/LDNormalization/)
def rT_n(n):
    """Compute the n^th term in the rotation solution vector `r` analytically."""
    l  = np.floor(np.sqrt(n)).astype(int)
    m  = n - l*l - l
    mu = l - m
    nu = l + m
    
    if (((mu/2)%2 == 0) and ((nu/2)%2 == 0)):
        return gamma(mu/4+0.5)*gamma(nu/4+0.5)/gamma((mu+nu)/4+2)
    elif ((((mu-1)/2)%2 == 0) and (((nu-1)/2)%2 == 0)):
        return (0.5*np.sqrt(np.pi)*gamma(mu/4+0.25)*gamma(nu/4+0.25)/gamma((mu+nu)/4+2))
    else:
        return 0

def rT(n_max):
    
    rT_vec = np.zeros(n_max)
    
    for n in range(n_max):
        rT_vec[n] = rT_n(n)
        
    return rT_vec


# This function and the translated starry function below produce slightly
# different arrays. They return the same values but in a different order.

# ---------------------------------------
# C++ starry code that is different to starry paper calculation.
# (translated via ChatGPT).

# Compute rT numerically.
def compute_rT(lmax):
    def set_phase_element(rT, l, m, value):
        rT[l * l + l + m] = value
        rT[l * l + l - m] = value
    
    pi = np.pi
    rT = np.zeros((lmax + 1) * (lmax + 1), dtype=float)
    amp0 = pi
    lfac1 = 1.0
    lfac2 = 2.0 / 3.0
    
    for l in range(0, lmax + 1, 4):
        amp = amp0
        for m in range(0, l + 1, 4):
            mu = l - m
            nu = l + m
            set_phase_element(rT, l, m, amp * lfac1)
            if l < lmax:
                set_phase_element(rT, l + 1, m + 1, amp * lfac2)
            amp *= (nu + 2.0) / (mu - 2.0)
        
        lfac1 /= (l // 2 + 2) * (l // 2 + 3)
        lfac2 /= (l // 2 + 2.5) * (l // 2 + 3.5)
        amp0 *= 0.0625 * (l + 2) * (l + 2)
    
    amp0 = 0.5 * pi
    lfac1 = 0.5
    lfac2 = 4.0 / 15.0
    
    for l in range(2, lmax + 1, 4):
        amp = amp0
        for m in range(2, l + 1, 4):
            mu = l - m
            nu = l + m
            set_phase_element(rT, l, m, amp * lfac1)
            if l < lmax:
                set_phase_element(rT, l + 1, m + 1, amp * lfac2)
            amp *= (nu + 2.0) / (mu - 2.0)
        
        lfac1 /= (l // 2 + 2) * (l // 2 + 3)
        lfac2 /= (l // 2 + 2.5) * (l // 2 + 3.5)
        amp0 *= 0.0625 * l * (l + 4)
    
    return rT

# -----------------------------------------------------------------------------

# Computes the design matrix
def X(l_max, theta, xo, yo, zo, ro, inc, obl, u, f):
    '''
    u: vector of limb-darkening coefficients
    f: multiplicative filter (relevant for radial velocity, reflected light,
                              and oblate maps)
    
    For oblate:
        f = The oblateness of the spheroid. This is the ratio of the difference
        between the equatorial and polar radii to the equatorial radius, and
        must be in the range [0, 1).
    For sphere:
        f=0
    '''
    filter = (np.any(u>0) or f>0)  # limb darkening or oblate?
    
    # Determine shapes.
    npts = theta.shape[0]           # How many phases?
    n_max = l_max**2 + 2 * l_max + 1
    
    # Initialise light curve design matrix
    X = np.zeros((npts,n_max))
    
    # Compute occultation mask ---
    
    # Distance between centres as occultor moves along its trajectory.
    b = np.sqrt(xo**2 + yo**2)
    
    #      (no occultation)(behind star)  (no size)
    b_rot = (b >= 1.0+ro) | (zo <= 0.0) | (ro==0.0)  # Any true -> no occultation
    b_occ = np.invert(b_rot)                         # Distances of occultation pts
    i_rot = np.arange(b.size)[b_rot]                 # Indexes of pts w/ no occultation
    i_occ = np.arange(b.size)[b_occ]                 # Indexes of pts of occultation

    # Compute A
    A1_mat = A1(l_max)
    A      = np.linalg.inv(A2_inv(l_max))@A1_mat

    # Compute solution vector for non-occulting phases
    rT_vec = rT(n_max)

    if filter:
        # TODO: What does F() do?
        F    = F(u,f)                                # <--- ???
        rT_f = np.dot(rT_vec, F)
        rTA1 = np.dot(rT_f, A1_mat)
    else:
        rTA1 = np.dot(rT_vec, A1_mat)

    rTA1 = np.tile(rTA1, (theta[i_rot].shape[0],1))

    # Compute sTAR
    sT       = solution_vector(l_max)(b[i_occ], ro)  # -> (npts,N)
    sTA      = np.dot(sT, A)                         # -> (npts,N)
    theta_z  = np.arctan2(xo[i_occ], yo[i_occ])      # -> (npts,)
    sTAR     = tensordotRz(sTA, theta_z)             # -> (npts,N)

    if filter:
        A1Inv    = np.linalg.inv(A1)
        A1InvF   = np.dot(A1Inv, F)
        A1InvFA1 = np.dot(A1InvF, A1)
        sTAR     = np.dot(sTAR, A1InvFA1)

    # Fill design matrix with occulting and non-occulting points (sTARR,rTA1R)
    X[i_occ] = right_project(sTAR, inc, obl, theta[i_occ])  # Occulting phases
    X[i_rot] = right_project(rTA1, inc, obl, theta[i_rot])  # Non-occulting phases

    return X

