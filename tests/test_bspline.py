import numpy as np

from iga_kl import bspline as bs


def _all_basis(U, p, u, nders=2):
    n = bs.n_basis(U, p)
    i = bs.find_span(n, p, u, U)
    d = bs.basis_funs_ders(i, u, p, U, nders)
    full = np.zeros((nders + 1, n))
    full[: d.shape[0], i - p : i + 1] = d
    return full


def test_partition_of_unity_and_derivative_sums():
    for p in [1, 2, 3, 5]:
        U = bs.uniform_open_knots(p, 4)
        for u in np.linspace(0, 1, 23):
            f = _all_basis(U, p, u)
            assert np.isclose(f[0].sum(), 1.0)
            assert np.allclose(f[1:].sum(axis=1), 0.0, atol=1e-10)
            assert np.all(f[0] >= -1e-14)


def test_derivatives_finite_difference():
    p = 4
    U = np.array([0, 0, 0, 0, 0, 0.2, 0.5, 0.5, 0.8, 1, 1, 1, 1, 1.0])
    h = 1e-5
    for u in [0.1, 0.35, 0.51, 0.7, 0.9]:
        f = _all_basis(U, p, u)
        fp = _all_basis(U, p, u + h)
        fm = _all_basis(U, p, u - h)
        assert np.allclose((fp[0] - fm[0]) / (2 * h), f[1], atol=1e-6)
        assert np.allclose((fp[0] - 2 * f[0] + fm[0]) / h**2, f[2], atol=1e-4)


def test_fig1_knot_vector():
    U = np.array([0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1.0])
    assert bs.n_basis(U, 3) == 7
    f = _all_basis(U, 3, 0.0)
    assert np.isclose(f[0, 0], 1.0)
    f = _all_basis(U, 3, 1.0)
    assert np.isclose(f[0, -1], 1.0)


def test_gauss_exactness():
    for n in range(1, 7):
        x, w = bs.gauss_legendre(n)
        for k in range(2 * n):
            exact = 0.0 if k % 2 else 2.0 / (k + 1)
            assert np.isclose(np.sum(w * x**k), exact)


def test_bezier_elevation_preserves_curve():
    Pw = np.array([[0, 0, 0, 1], [1, 2, 0, 0.7], [3, 1, 0, 1.0]])
    U2 = np.array([0, 0, 0, 1, 1, 1.0])
    Q = bs.bezier_degree_elevate(Pw, 2)
    U4 = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1.0])
    for u in np.linspace(0, 1, 11):
        c2 = _all_basis(U2, 2, u)[0] @ Pw
        c4 = _all_basis(U4, 4, u)[0] @ Q
        assert np.allclose(c2[:3] / c2[3], c4[:3] / c4[3])


def test_knot_insertion_preserves_curve():
    p = 3
    U = np.array([0, 0, 0, 0, 0.5, 1, 1, 1, 1.0])
    rng = np.random.default_rng(0)
    Pw = rng.random((5, 4)) + 0.5
    Un, Qw = bs.refine_knots(U, p, Pw, [0.25, 0.5, 0.75, 0.9])
    assert bs.n_basis(Un, p) == Qw.shape[0]
    for u in np.linspace(0, 1, 31):
        c0 = _all_basis(U, p, u)[0] @ Pw
        c1 = _all_basis(Un, p, u)[0] @ Qw
        assert np.allclose(c0, c1)
