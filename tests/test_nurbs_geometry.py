import numpy as np
import pytest

from iga_kl import geometry
from iga_kl.mesh import Mesh


def _random_params(n=30, seed=1):
    rng = np.random.default_rng(seed)
    return rng.random((n, 2))


def test_local_indices_consistent_with_cp_index():
    s = geometry.cylinder(1.0, 2.0, 0.0, np.pi / 2).k_refine(3, 4)
    ien, R, _, _ = s.basis(0.3, 0.7)
    # interpolation must reproduce the surface point computed independently from Pw
    P = s.control_points.reshape(-1, 3, order="F")
    assert np.allclose(R @ P[ien], s.point(0.3, 0.7))
    # global index a = aa + (p+1)*bb matches i + n*j
    i, j = 2, 1
    assert s.cp_index(i, j) == i + s.n * j


@pytest.mark.parametrize("p,n_el", [(2, 2), (3, 4), (5, 32)])
def test_refinement_invariance_and_exactness(p, n_el):
    R = 25.0
    for coarse in [geometry.cylinder(R, 50.0, np.radians(50), np.radians(90)), geometry.hemisphere_quarter(R)]:
        fine = coarse.k_refine(p, n_el)
        assert fine.n == n_el + p and fine.m == n_el + p
        for xi, eta in _random_params():
            S0, dS0, ddS0 = coarse.eval(xi, eta)
            S1, dS1, ddS1 = fine.eval(xi, eta)
            assert np.allclose(S0, S1, atol=1e-10)
            assert np.allclose(dS0, dS1, atol=1e-8)
            assert np.allclose(ddS0, ddS1, atol=1e-6)
            if coarse.m == 3:  # sphere
                assert np.isclose(np.linalg.norm(S1), R)
            else:
                assert np.isclose(np.hypot(S1[0], S1[1]), R)


def test_surface_derivatives_finite_difference():
    s = geometry.hemisphere_quarter(3.0).k_refine(3, 3)
    h = 1e-5
    for xi, eta in _random_params(10, 2):
        xi, eta = 0.1 + 0.8 * xi, 0.1 + 0.8 * eta
        S, dS, ddS = s.eval(xi, eta)
        Sxp, Sxm = s.point(xi + h, eta), s.point(xi - h, eta)
        Syp, Sym = s.point(xi, eta + h), s.point(xi, eta - h)
        assert np.allclose((Sxp - Sxm) / (2 * h), dS[0], atol=1e-6)
        assert np.allclose((Syp - Sym) / (2 * h), dS[1], atol=1e-6)
        assert np.allclose((Sxp - 2 * S + Sxm) / h**2, ddS[0], atol=1e-3)
        assert np.allclose((Syp - 2 * S + Sym) / h**2, ddS[1], atol=1e-3)
        Spp = s.point(xi + h, eta + h)
        Spm = s.point(xi + h, eta - h)
        Smp = s.point(xi - h, eta + h)
        Smm = s.point(xi - h, eta - h)
        assert np.allclose((Spp - Spm - Smp + Smm) / (4 * h * h), ddS[2], atol=1e-3)


def test_edge_and_corner_helpers():
    s = geometry.cylinder(1.0, 2.0, 0.0, np.pi / 2).k_refine(2, 3)
    P = s.control_points.reshape(-1, 3, order="F")
    assert np.allclose(P[s.edge_cps("eta0")][:, 2], 0.0)
    assert np.allclose(P[s.edge_cps("eta1")][:, 2], 2.0)
    assert np.allclose(P[s.edge_cps("xi0")][:, 1], 0.0)  # phi = 0 edge: y = 0
    assert np.allclose(P[s.edge_cps("xi1")][:, 0], 0.0)  # phi = 90: x = 0
    assert np.allclose(s.point(1.0, 1.0), P[s.corner_cp((1, 1))])
    assert np.allclose(s.point(0.0, 1.0), P[s.corner_cp((0, 1))])


def _area(mesh):
    P = mesh.X_e
    G1 = np.einsum("ega,eak->egk", mesh.dR[..., 0], P)
    G2 = np.einsum("ega,eak->egk", mesh.dR[..., 1], P)
    J = np.linalg.norm(np.cross(G1, G2), axis=-1)
    return np.sum(mesh.w * J)


def test_mesh_areas():
    R, L = 25.0, 50.0
    cyl = geometry.cylinder(R, L, np.radians(50), np.radians(90)).k_refine(3, 4)
    assert np.isclose(_area(Mesh(cyl)), R * L * np.radians(40))
    hemi = geometry.hemisphere_quarter(10.0).k_refine(3, 8)
    assert np.isclose(_area(Mesh(hemi)), np.pi * 100.0 / 2, rtol=1e-8)
    m = Mesh(cyl)
    assert m.n_el == 16 and m.n_gp == 16 and m.n_en == 16
    assert np.allclose(m.R.sum(axis=-1), 1.0)
