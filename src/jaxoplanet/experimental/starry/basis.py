import math
from collections import defaultdict

import numpy as np
from scipy.special import gamma

""" Functions:
    ---------
    
    A1(lmax)         
                    Computes the change-of-basis matrix from spherical
                    harmonics to polynomial.
                    Calls ptilde to calculate the mth term of the polynomial
                    basis in terms of x,y,z.
                    Calls p_Y to compute the mth column vector of A1.

    A2_inv(lmax)     
                    Computes the inverse of the change-of-basis matrix from
                    polynomial to Green's.
                    Calls ptilde to calculate the nth term of the polynomial
                    basis in terms of powers of x,y,z.
                    Calls p_G to compute the nth column vector of A2_inv.
                    
    ptilde(n)
                    Computes the nth term of the polynomial basis in terms
                    of x,y,z.
                    (i,j,k) for (x^i)*(y^j)*(z^k)
                    
    Alm(l,m)
                    A spherical harmonic normalization constant, called by Ylm.
                    degree l, order m
                    
    Blmjk(l,m,j,k)
                    A spherical harmonic normalization constant, called by Ylm.
                    degree l, order m
                    j,k : Ylm summation indexes
                    
    Cpqk(p,q,k)
                    The binomial theorem coefficient `C`, called by Ylm.
                    p,q,k : Ylm summation indexes

    Ylm(l,m)
                    Calulates l-mth term of the spherical harmonic basis.
                    ACCEPTS:    degree l, order m
                    RETURNS:    dictionary of {(i,j,k): scalar}
                                (i,j,k) are powers of x,y,z
                                Ylm(x,y) = SUM(scalar*(x^i)*(y^j)*(z^k))
                    
    p_Y(p,l,m,res)
                    Computes the mth column vector of the change-of-basis
                    matrix A1.
                    p : the mth term of the polynomial basis in terms of
                    powers of x,y,z (i,j,k) where 
                    degree l, order m
                    res : mth column 
                    
    gtilde(n)
                    Computes the nth term of the Green's basis vector in
                    terms of powers of x,y,z: SUM(scalar*(x^i)*(y^j)*(z^k))
                    
    p_G(p,n,res)
                    Computes the nth column vector for the inverse of the
                    change-of-basis matrix A2 (from polynomial to Green's).
                    p : dictionary from ptilde
                    n : which column of A2_inv
                    res : dictionary for computing nth Green's basis term.
                          {(i,j,k):scalar}
                          component = SUM(scalar*(x^i)(y^j)(z^k))
"""

def A1(lmax):
    """
    Computes the change-of-basis matrix from spherical harmonics to polynomial.
    Calls ptilde to calculate the mth term of the polynomial basis in terms of
    powers of x,y,z.
    Calls p_Y to compute the mth column vector of A1.
    
    Note
    ----------
    The normalization here matches the starry paper, but not the
    code. To get the code's normalization, multiply the result by 2 /
    sqrt(pi).

    Parameters
    ----------
    lmax : integer
        The largest degree up to which the spherical harmonics are computed.

    Returns
    -------
    array of arrays of scalars
        The change-of-basis matrix from spherical harmonics to polynomial.

    """
    n = (lmax + 1) ** 2
    res = np.zeros((n, n))
    p = {ptilde(m): m for m in range(n)}
    n = 0
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            p_Y(p, l, m, res[:, n])
            n += 1
    return res


def A2_inv(lmax):
    """
    Computes the inverse of the change-of-basis matrix from polynomial to
    Green's.
    Calls ptilde to calculate the mth term of the polynomial basis in terms of
    powers of x,y,z.
    Calls p_G to compute the nth column vector of A2_inv.

    Parameters
    ----------
    lmax : integer
        The largest degree up to which the spherical harmonics are computed.

    Returns
    -------
    array of arrays of scalars
        The inverse of the change-of-basis matrix from polynomial to Green's.

    """
    n = (lmax + 1) ** 2
    res = np.zeros((n, n))
    p = {ptilde(m): m for m in range(n)}
    n = 0
    for l in range(lmax + 1):
        for _ in range(-l, l + 1):
            p_G(p, n, res[:, n])
            n += 1
    return res


def ptilde(n):
    """
    Calculates the nth term of the polynomial basis in terms of powers of
    x,y,z: (x^i)*(y^j)*(z^k)

    Parameters
    ----------
    n : integer
        nth component.

    Returns
    -------
    i : integer              <---- ?
        power of x.
    j : integer
        power of y.
    k : integer
        power of z.
        k in [0,1]

    """
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
    return (i, j, k)


def Alm(l, m):
    """
    Computes the scalar value of the spherical harmonic normalization constant
    called by Ylm(l,m).

    Parameters
    ----------
    l : integer
        degree of spherical harmonic
    m : integer
        order of spherical harmonic

    Returns
    -------
    float
        Scalar result.

    """
    return math.sqrt(
        (2 - int(m == 0))
        * (2 * l + 1)
        * math.factorial(l - m)
        / (4 * math.pi * math.factorial(l + m))
    )


def Blmjk(l, m, j, k):
    """
    Computes the scalar value of the spherical harmonic normalization constant
    called by Ylm(l,m).

    Parameters
    ----------
    l : integer
        Degree of spherical harmonic.
    m : integer
        Order of spherical harmonic.
    j : integer
        Index of summation in Ylm(l,m).
    k : integer
        Index of summation in Ylm(l,m).

    Returns
    -------
    numpy.float64
        Scalar result.

    """
    a = l + m + k - 1
    b = -l + m + k - 1
    if (b < 0) and (b % 2 == 0):
        return 0
    else:
        ratio = gamma(0.5 * a + 1) / gamma(0.5 * b + 1)
    return (
        2**l
        * math.factorial(m)
        / (
            math.factorial(j)
            * math.factorial(k)
            * math.factorial(m - j)
            * math.factorial(l - m - k)
        )
        * ratio
    )


def Cpqk(p, q, k):
    """
    Computes the scalar value of the binomial theorem coefficient `C`, called
    by Ylm(l,m).

    Parameters
    ----------
    p : integer
        Index of summation in Ylm(l,m).
    q : integer
        Index of summation in Ylm(l,m).
    k : integer
        Index of summation in Ylm(l,m).

    Returns
    -------
    float
        Scalar result.

    """
    return math.factorial(k // 2) / (
        math.factorial(q // 2)
        * math.factorial((k - p) // 2)
        * math.factorial((p - q) // 2)
    )


def Ylm(l, m):
    """
    Computes the lmth term of the spherical harmonic basis y_tilde.

    Parameters
    ----------
    l : integer
        degree of spherical harmonic.
    m : integer
        order of spherical harmonic.

    Returns
    -------
    dictionary
    
        KEY: integers (i,j,k)
        VALUE: scalar
        
        Ylm(x,y) = SUM(scalar*(x^i)*(y^j)*(z^k))
        with z = sqrt(1-x^2-y^2)
        and  k in [0,1]

    """
    res = defaultdict(lambda: 0)
    A = Alm(l, abs(m))
    for j in range(int(m < 0), abs(m) + 1, 2):
        for k in range(0, l - abs(m) + 1, 2):
            B = Blmjk(l, abs(m), j, k)
            if not B:
                continue
            factor = A * B
            for p in range(0, k + 1, 2):
                for q in range(0, p + 1, 2):
                    ind = (abs(m) - j + p - q, j + q, 0)
                    res[ind] += (
                        (-1) ** ((j + p - (m < 0)) // 2) * factor * Cpqk(p, q, k)
                    )
        for k in range(1, l - abs(m) + 1, 2):
            B = Blmjk(l, abs(m), j, k)
            if not B:
                continue
            factor = A * B
            for p in range(0, k, 2):
                for q in range(0, p + 1, 2):
                    ind = (abs(m) - j + p - q, j + q, 1)
                    res[ind] += (
                        (-1) ** ((j + p - (m < 0)) // 2) * factor * Cpqk(p, q, k - 1)
                    )

    return dict(res)


def p_Y(p, l, m, res):
    """
    Computes the mth column vector of the change-of-basis matrix A1 by
    calling Ylm(l,m) to compute the l-mth term of the spherical harmonic basis.
    
    The mth column of A1 corresponds to the nth spherical harmonic
    where n = l^2 + l + m.

    Parameters
    ----------
    p : dictionary
        KEY:    integers (i,j,k)
                powers of mth term of polynomial basis.
                (x^i)*(y^j)*(z^k)
        VALUE:  integer m
                mth term of polynomial basis.
    l : integer
        Degree of spherical harmonic.
    m : integer
        Order of spherical harmonic.
    res : array
        Column vector of change-of-basis matrix A1.

    Returns
    -------
    res : array of integers
        mth column vector of change-of-basis matrix A1.

    """
    for k, v in Ylm(l, m).items():
        if k not in p:
            continue
        res[p[k]] = v
    return res


def gtilde(n):
    """
    Computes the nth term of the Green's basis vector in terms of powers of
    x,y,z: SUM(scalar*(x^i)*(y^j)*(z^k))

    Parameters
    ----------
    n : integer
        nth column of the A2 change-of-basis matrix from polynomial to Green's.

    Returns
    -------
    res : dictionary
    
        KEY: integers (i,j,k)
        VALUE: scalar
        
        gtilde(n) = SUM(scalar*(x^i)*(y^j)*(z^k))

    """
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
    res = {}
    for i, j, k, c in zip(I, J, K, C):
        res[(i, j, k)] = c
    return res


def p_G(p, n, res):
    """
    Computes the nth column vector for the inverse of the change-of-basis
    matrix A2 (from polynomial to Green's).

    Parameters
    ----------
    p : dictionary
            KEY: integers (i,j,k)
            VALUE: scalar
    n : integer
        denotes the nth column of the inverse of the change-of-basis matrix A2.
    res : array of scalars
        column vector zeros representing the yet-to-be-populated nth column
        of the inverse of the change-of-basis matrix A2.

    Returns
    -------
    res : column vector of scalars
        nth column vector of the inverse of the change-of-basis matrix A2.

    """
    for k, v in gtilde(n).items():
        if k not in p:
            continue
        res[p[k]] = v
    return res
