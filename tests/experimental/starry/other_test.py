import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax import jit, vmap, lax
import bpope_2023b.src.jaxoplanet as jaxoplanet

from jaxoplanet.types import Array
from jaxoplanet.test_utils import assert_allclose

from bpope_2023b.src.jaxoplanet.experimental.starry.other import (tensordotRz)

# Display arrays in decimal notation, not scientific.
np.set_printoptions(suppress = True)


@pytest.mark.parametrize("l_max", [10, 7, 5, 4])
@pytest.mark.parametrize("npts", [1, 10, 100, 1000])
def test_tensordotRz(l_max, npts):
    """Test tensordotRz against symbolic one."""
    
    # Import sympy for test.
    pytest.importorskip("sympy")
    
    # Set test parameters.
    np.random.seed(l_max)
    n_max = l_max**2 + 2*l_max + 1
    M     = np.random.rand(n_max, npts)
    theta = np.random.random(npts)
    
    # Compute expected and calculated results.
    expected = np.array(tensordotRz_symbolic(M, theta)).astype(float)
    calc     = tensordotRz(M, theta)
    
    # Check equivalence.
    assert_allclose(calc, expected)


@pytest.mark.parametrize("l_max", [10, 7, 5, 4])
@pytest.mark.parametrize("npts", [1, 10, 100, 1000])
def test_compare_starry_tensordotRz(l_max, npts):
    """Comparison test with starry ops.tensordotRz."""
    
    # Import starry for test.
    starry = pytest.importorskip("starry")
    
    # Set test parameters.
    np.random.seed(l_max)
    n_max = l_max**2 + 2*l_max + 1
    M     = np.random.rand(n_max, npts)
    theta = np.random.random(npts)
    
    # Initiate starry object (ydeg=l_max, udeg=0, fdeg=0, nw=1).
    test_map = starry._core.core.OpsYlm(l_max, 0, 0, 1)
    
    # Compute expected and calculated results.
    expected = test_map.ops.tensordotRz(M, theta)
    calc     = tensordotRz(M, theta)
    
    # Check equivalence.
    assert_allclose(calc, expected)


def tensordotRz_symbolic(M, theta):
    import sympy as sm
    
    # ...
    
    return