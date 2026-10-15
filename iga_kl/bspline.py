"""Univariate B-spline utilities (Piegl & Tiller, The NURBS Book)."""

import numpy as np


def find_span(n, p, u, U):
    """Knot span index i such that U[i] <= u < U[i+1] (A2.1). n = number of basis functions."""
    if u >= U[n]:
        return n - 1
    if u <= U[p]:
        return p
    low, high = p, n
    mid = (low + high) // 2
    while u < U[mid] or u >= U[mid + 1]:
        if u < U[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    return mid


def basis_funs_ders(i, u, p, U, nders):
    """Nonzero basis functions and derivatives at u (A2.3).

    Returns ders[k, j] = d^k N_{i-p+j, p}(u) / du^k, k = 0..nders, j = 0..p.
    """
    nders = min(nders, p) if p > 0 else 0
    ndu = np.zeros((p + 1, p + 1))
    ndu[0, 0] = 1.0
    left = np.zeros(p + 1)
    right = np.zeros(p + 1)
    for j in range(1, p + 1):
        left[j] = u - U[i + 1 - j]
        right[j] = U[i + j] - u
        saved = 0.0
        for r in range(j):
            ndu[j, r] = right[r + 1] + left[j - r]
            temp = ndu[r, j - 1] / ndu[j, r]
            ndu[r, j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        ndu[j, j] = saved
    ders = np.zeros((nders + 1, p + 1))
    ders[0, :] = ndu[:, p]
    a = np.zeros((2, p + 1))
    for r in range(p + 1):
        s1, s2 = 0, 1
        a[0, 0] = 1.0
        for k in range(1, nders + 1):
            d = 0.0
            rk = r - k
            pk = p - k
            if r >= k:
                a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                d = a[s2, 0] * ndu[rk, pk]
            j1 = 1 if rk >= -1 else -rk
            j2 = k - 1 if r - 1 <= pk else p - r
            for j in range(j1, j2 + 1):
                a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                d += a[s2, j] * ndu[rk + j, pk]
            if r <= pk:
                a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                d += a[s2, k] * ndu[r, pk]
            ders[k, r] = d
            s1, s2 = s2, s1
    r = p
    for k in range(1, nders + 1):
        ders[k, :] *= r
        r *= p - k
    return ders


def uniform_open_knots(p, n_el):
    """Open knot vector on [0,1] with n_el uniform spans and degree p (n = n_el + p functions)."""
    interior = np.linspace(0.0, 1.0, n_el + 1)[1:-1]
    return np.concatenate([np.zeros(p + 1), interior, np.ones(p + 1)])


def n_basis(U, p):
    """Number of basis functions for knot vector U and degree p."""
    return len(U) - p - 1


def unique_spans(U):
    """Nonzero knot spans as (start, end) pairs."""
    Uu = np.unique(U)
    return [(Uu[k], Uu[k + 1]) for k in range(len(Uu) - 1)]


def gauss_legendre(n):
    """Gauss-Legendre points and weights on [-1, 1]."""
    return np.polynomial.legendre.leggauss(n)


def bezier_degree_elevate(Pw, t=1):
    """Degree-elevate a single-span (Bezier) curve given by homogeneous control points Pw[n, d] by t.

    Q_i = (i/(p+1)) P_{i-1} + (1 - i/(p+1)) P_i, applied t times.
    """
    Pw = np.asarray(Pw, dtype=float)
    for _ in range(t):
        p = Pw.shape[0] - 1
        Q = np.zeros((p + 2, Pw.shape[1]))
        for i in range(p + 2):
            a = i / (p + 1)
            if i > 0:
                Q[i] += a * Pw[i - 1]
            if i <= p:
                Q[i] += (1.0 - a) * Pw[i]
        Pw = Q
    return Pw


def insert_knot(U, p, Pw, u, r=1):
    """Insert knot u r times into curve (U, p, Pw) (A5.1). Returns (U_new, Pw_new)."""
    U = np.asarray(U, dtype=float)
    Pw = np.asarray(Pw, dtype=float)
    n = len(Pw) - 1
    k = find_span(n + 1, p, u, U)
    s = int(np.sum(np.isclose(U, u)))
    if r + s > p:
        raise ValueError("knot multiplicity would exceed degree")
    mp = n + p + 1
    UQ = np.zeros(mp + r + 1)
    UQ[: k + 1] = U[: k + 1]
    UQ[k + 1 : k + r + 1] = u
    UQ[k + r + 1 :] = U[k + 1 :]
    Qw = np.zeros((n + r + 1, Pw.shape[1]))
    Qw[: k - p + 1] = Pw[: k - p + 1]
    Qw[k - s + r :] = Pw[k - s :]
    Rw = Pw[k - p : k - s + 1].copy()
    L = 0
    for j in range(1, r + 1):
        L = k - p + j
        for i in range(p - j - s + 1):
            alpha = (u - U[L + i]) / (U[i + k + 1] - U[L + i])
            Rw[i] = alpha * Rw[i + 1] + (1.0 - alpha) * Rw[i]
        Qw[L] = Rw[0]
        Qw[k + r - j - s] = Rw[p - j - s]
    for i in range(L + 1, k - s):
        Qw[i] = Rw[i - L]
    return UQ, Qw


def refine_knots(U, p, Pw, knots):
    """Insert a sequence of knots one at a time."""
    for u in knots:
        U, Pw = insert_knot(U, p, Pw, u)
    return U, Pw
