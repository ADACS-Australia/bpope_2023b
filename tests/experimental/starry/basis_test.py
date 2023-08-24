import warnings

import numpy as np
import pytest

from jaxoplanet.experimental.starry.basis import A1, A2_inv
# import bpope_2023b.src.jaxoplanet as jaxoplanet
# from jaxoplanet.src.jaxoplanet.experimental.starry.basis import A1, A2_inv

"""
Notes on functions, symbols, modules, etc.
------------------------------------------

@:
    - decorator.
    - design pattern that allows new functionality to be added to an
      existing object without modifying the object's structure.
    - Useful for modular and reusable code; define decorator once and then
      apply it to any number of functions.

@pytest.mark.parametrize("lmax", [10, 7, 5, 4, 3, 2, 1, 0]):
    - allows definition of multiple sets of arguments at the test function.
    - test_A1 will be run 8 times with different lmax values.
    
pytest.importorskip("sympy"):
    - imports and returns the module called ("sympy").
    - skips test if module can't be imported.

np.testing.assert_allclose(calc, expected, atol=5e-12):
    - raises an error if computed and expected results are not equal.
      up to the specified tolerance, atol.

sympy:
    - library for symbolic mathematics (algebra).
    - manipulates equations/expressions in symbolic form, rather than numerical.

expression.coeff(term):
    - returns the coefficients of the specified term in the expression.

ylm.subs(sm.sqrt(1 - x**2 - y**2), 0):
    - substitute all matching variables/expressions in ylm with specified value.

sm.lambdify([], A1(lmax))():
    - lambdify translates SymPy expressions into Python functions.
    - acts like a lambda function.
    - in this case, numerically evaluates A1 for lmax.
"""

# set lmax values for eight tests
@pytest.mark.parametrize("lmax", [10, 7, 5, 4, 3, 2, 1, 0])
def test_A1(lmax):
    pytest.importorskip("sympy") # import sympy module used in A1_symbolic
    expected = A1_symbolic(lmax) # compute sympy (expected) result
    calc = A1(lmax)              # compute jaxoplanet (calculated) result
    np.testing.assert_allclose(calc, expected, atol=5e-12) # test equivalence


@pytest.mark.parametrize("lmax", [10, 7, 5, 4, 3, 2, 1, 0])
def test_A2_inv(lmax):
    pytest.importorskip("sympy")
    expected = A2_inv_symbolic(lmax)
    calc = A2_inv(lmax)
    np.testing.assert_allclose(calc, expected)


@pytest.mark.parametrize("lmax", [10, 7, 5, 4, 3, 2, 1, 0])
def test_compare_starry_A1(lmax):
    starry = pytest.importorskip("starry") # import starry
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")    # never print warnings
        m = starry.Map(lmax)               # generate starry Map
                                           # generate A1 from Map
        expect = m.ops.A1.eval().toarray() * (0.5 * np.sqrt(np.pi))
    calc = A1(lmax)                        # compute jaxoplanet result
    np.testing.assert_allclose(calc, expect, atol=5e-12) # test equivalence


@pytest.mark.parametrize("lmax", [10, 7, 5, 4, 3, 2, 1, 0])
def test_compare_starry_A2_inv(lmax):
    starry = pytest.importorskip("starry")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = starry.Map(lmax)               # generate starry Map
                                           # compute starry A2 from A@A1Inv
        A2 = m.ops.A.eval().toarray() @ m.ops.A1Inv.eval().toarray()
    inv = A2_inv(lmax)                     # compute jaxoplanet result
    np.testing.assert_allclose(inv @ A2, np.eye(len(inv)), atol=5e-12) # test


# computes the symbolic result
def A1_symbolic(lmax):
    """The sympy implementation of the A1 matrix from the starry paper"""
    import math

    import sympy as sm
    from sympy.functions.special.tensor_functions import KroneckerDelta

    # creates instances of symbols class with variable names
    x, y = sm.symbols("x y")

    # call example: Coefficient(ylm, ptilde(n,x,y))
    # ylm = l-mth spherical harmonic, ptilde(n,x,y) = nth poly basis term
    def Coefficient(expression, term):
        """Return the coefficient multiplying `term` in `expression`."""
        # returns the coefficients of the specified term in the expression
        coeff = expression.coeff(term)
        # strip x,y,z away to return only the coefficient
        # substitue (this variable/expression, with this value)
        coeff = coeff.subs(sm.sqrt(1 - x**2 - y**2), 0).subs(x, 0).subs(y, 0)
        return coeff

    def ptilde(n, x, y):
        """Return the n^th term in the polynomial basis."""
        l = math.floor(math.sqrt(n))
        m = n - l * l - l
        mu = l - m
        nu = l + m
        if nu % 2 == 0:
            i = mu // 2
            j = nu // 2
            k = 0
        else:
            i = (mu - 1) // 2
            j = (nu - 1) // 2
            k = 1
        # returns (x^i)(y^j)(z^k)
        return x**i * y**j * sm.sqrt(1 - x**2 - y**2) ** k

    def A(l, m):
        """A spherical harmonic normalization constant."""
        return sm.sqrt(
            (2 - KroneckerDelta(m, 0))
            * (2 * l + 1)
            * sm.factorial(l - m)
            / (4 * sm.pi * sm.factorial(l + m))
        )

    def B(l, m, j, k):
        """Another spherical harmonic normalization constant."""
        try:
            ratio = sm.factorial((l + m + k - 1) / 2) / sm.factorial(
                (-l + m + k - 1) / 2
            )
        except ValueError:
            ratio = 0
        return (
            2**l
            * sm.factorial(m)
            / (
                sm.factorial(j)
                * sm.factorial(k)
                * sm.factorial(m - j)
                * sm.factorial(l - m - k)
            )
            * ratio
        )

    def C(p, q, k):
        """Return the binomial theorem coefficient `C`."""
        return sm.factorial(k / 2) / (
            sm.factorial(q / 2) * sm.factorial((k - p) / 2) * sm.factorial((p - q) / 2)
        )

    def Y(l, m, x, y):
        """Return the spherical harmonic of degree `l` and order `m`."""
        # returns scalar1*(x^i)(y^j)(z^k) + scalar2*...
        res = 0
        z = sm.sqrt(1 - x**2 - y**2)
        if m >= 0:
            for j in range(0, m + 1, 2):
                for k in range(0, l - m + 1, 2):
                    for p in range(0, k + 1, 2):
                        for q in range(0, p + 1, 2):
                            res += (
                                (-1) ** ((j + p) // 2)
                                * A(l, m)
                                * B(l, m, j, k)
                                * C(p, q, k)
                                * x ** (m - j + p - q)
                                * y ** (j + q)
                            )
                for k in range(1, l - m + 1, 2):
                    for p in range(0, k, 2):
                        for q in range(0, p + 1, 2):
                            res += (
                                (-1) ** ((j + p) // 2)
                                * A(l, m)
                                * B(l, m, j, k)
                                * C(p, q, k - 1)
                                * x ** (m - j + p - q)
                                * y ** (j + q)
                                * z
                            )
        else:
            for j in range(1, abs(m) + 1, 2):
                for k in range(0, l - abs(m) + 1, 2):
                    for p in range(0, k + 1, 2):
                        for q in range(0, p + 1, 2):
                            res += (
                                (-1) ** ((j + p - 1) // 2)
                                * A(l, abs(m))
                                * B(l, abs(m), j, k)
                                * C(p, q, k)
                                * x ** (abs(m) - j + p - q)
                                * y ** (j + q)
                            )
                for k in range(1, l - abs(m) + 1, 2):
                    for p in range(0, k, 2):
                        for q in range(0, p + 1, 2):
                            res += (
                                (-1) ** ((j + p - 1) // 2)
                                * A(l, abs(m))
                                * B(l, abs(m), j, k)
                                * C(p, q, k - 1)
                                * x ** (abs(m) - j + p - q)
                                * y ** (j + q)
                                * z
                            )
        return res

    def p_Y(l, m, lmax):
        """Return the l-m^th column vector of A1."""
        # compute l-m^th spherical harmonic (scalar1*(x^i)(y^j)(z^k) + scalar2*...)
        ylm = Y(l, m, x, y)
        # assign first element in column vector (constant term)
        # substitute x,y,z with 0 to get constant term
        res = [ylm.subs(sm.sqrt(1 - x**2 - y**2), 0).subs(x, 0).subs(y, 0)]
        # assign remaining elements (non-constant terms)
        for n in range(1, (lmax + 1) ** 2):
            # append coefficients of ptilde(n,x,y) in ylm
            # append coefficients of nth poly basis term in l-mth spheric harmonic
            res.append(Coefficient(ylm, ptilde(n, x, y)))
        return res

    def A1(lmax):
        """Return the change-of-basis matrix from spherical harmonics to polynomial."""
        res = sm.zeros((lmax + 1) ** 2, (lmax + 1) ** 2) # array of arrays
        n = 0
        for l in range(lmax + 1):
            for m in range(-l, l + 1):
                res[n] = p_Y(l, m, lmax) # compute column vectors
                n += 1
        return res

    # numerically evaluates A1
    return sm.lambdify([], A1(lmax))()

def A2_inv_symbolic(lmax):
    """The sympy implementation of the A2 matrix from the starry paper"""
    import math

    import sympy as sm

    x, y = sm.symbols("x y")

    def Coefficient(expression, term):
        """Return the coefficient multiplying `term` in `expression`."""
        coeff = expression.coeff(term)
        coeff = coeff.subs(sm.sqrt(1 - x**2 - y**2), 0).subs(x, 0).subs(y, 0)
        return coeff

    def ptilde(n, x, y):
        """Return the n^th term in the polynomial basis."""
        l = math.floor(math.sqrt(n))
        m = n - l * l - l
        mu = l - m
        nu = l + m
        if nu % 2 == 0:
            i = mu // 2
            j = nu // 2
            k = 0
        else:
            i = (mu - 1) // 2
            j = (nu - 1) // 2
            k = 1
        return x**i * y**j * sm.sqrt(1 - x**2 - y**2) ** k

    def gtilde(n, x, y):
        """Return the n^th term in the Green's basis."""
        l = math.floor(math.sqrt(n))
        m = n - l * l - l
        mu = l - m
        nu = l + m
        if nu % 2 == 0:
            I = [mu // 2]
            J = [nu // 2]
            K = [0]
            C = [(mu + 2) // 2]
        elif (l == 1) and (m == 0):
            I = [0]
            J = [0]
            K = [1]
            C = [1]
        elif (mu == 1) and (l % 2 == 0):
            I = [l - 2]
            J = [1]
            K = [1]
            C = [3]
        elif mu == 1:
            I = [l - 3, l - 1, l - 3]
            J = [0, 0, 2]
            K = [1, 1, 1]
            C = [-1, 1, 4]
        else:
            I = [(mu - 5) // 2, (mu - 5) // 2, (mu - 1) // 2]
            J = [(nu - 1) // 2, (nu + 3) // 2, (nu - 1) // 2]
            K = [1, 1, 1]
            C = [(mu - 3) // 2, -(mu - 3) // 2, -(mu + 3) // 2]

        res = 0
        for i, j, k, c in zip(I, J, K, C):
            res += c * x**i * y**j * sm.sqrt(1 - x**2 - y**2) ** k
        return res

    def p_G(n, lmax):
        """Return the n^th column vector of A2."""
        g = gtilde(n, x, y)
        res = [g.subs(sm.sqrt(1 - x**2 - y**2), 0).subs(x, 0).subs(y, 0)]
        for n in range(1, (lmax + 1) ** 2):
            res.append(Coefficient(g, ptilde(n, x, y)))
        return res

    def A2_inv(lmax):
        """Return the change-of-basis matrix from polynomial to Green's."""
        res = sm.zeros((lmax + 1) ** 2, (lmax + 1) ** 2)
        n = 0
        for l in range(lmax + 1):
            for _m in range(-l, l + 1):
                res[n] = p_G(n, lmax)
                n += 1
        return res

    return sm.lambdify([], A2_inv(lmax))()
