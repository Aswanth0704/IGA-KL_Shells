import numpy as np
from scipy.spatial.transform import Rotation

from iga_kl import geometry
from iga_kl.constraints import clamp, symmetry
from iga_kl.material import Material
from iga_kl.model import ShellModel


def test_finite_rigid_rotation_is_stress_free():
    surf = geometry.hemisphere_quarter(2.0).k_refine(3, 2)
    model = ShellModel(surf, Material(100.0, 0.3, 0.1))
    Q = Rotation.from_rotvec([0.7, -1.1, 0.4]).as_matrix()
    P = surf.control_points.reshape(-1, 3, order="F")
    u = (P @ Q.T - P + np.array([0.3, -0.2, 0.9])).ravel()
    K, f_int = model.tangent(u)
    assert np.allclose(f_int, 0.0, atol=1e-10)
    assert np.isclose(model.energy(u), 0.0, atol=1e-16)


def test_internal_force_and_tangent_by_finite_differences():
    surf = geometry.cylinder(1.0, 1.5, 0.0, 1.0).k_refine(3, 2)
    model = ShellModel(surf, Material(100.0, 0.3, 0.05))
    rng = np.random.default_rng(7)
    u = 0.05 * rng.standard_normal(model.n_dof)
    K, f_int = model.tangent(u)
    K = K.toarray()
    assert np.allclose(K, K.T, atol=1e-9 * np.abs(K).max())
    h = 1e-6
    idx = rng.choice(model.n_dof, 12, replace=False)
    for i in idx:
        e = np.zeros(model.n_dof)
        e[i] = h
        Wp, Wm = model.energy(u + e), model.energy(u - e)
        assert np.isclose((Wp - Wm) / (2 * h), f_int[i], rtol=1e-5, atol=1e-8 * np.abs(f_int).max())
        fp, fm = model.tangent(u + e)[1], model.tangent(u - e)[1]
        assert np.allclose((fp - fm) / (2 * h), K[:, i], rtol=1e-5, atol=1e-6 * np.abs(K).max())


def test_first_newton_iterate_equals_linear_solution():
    surf = geometry.cylinder(300.0, 300.0, 0.0, np.pi / 2).k_refine(3, 4)
    model = ShellModel(surf, Material(3.0e6, 0.3, 3.0))
    c = model.constraints
    c.fix_cps(surf.edge_cps("eta0"), (0, 1))
    symmetry(c, surf, "eta1", 2)
    symmetry(c, surf, "xi0", 1)
    symmetry(c, surf, "xi1", 0)
    model.add_point_load((0.0, 1.0), [-0.25, 0.0, 0.0])
    u_lin = model.solve_linear()
    u_nl, hist = model.solve_nonlinear(n_steps=1, tol=1e-10)
    assert len(hist[0]) <= 4
    assert np.allclose(u_lin, u_nl, rtol=1e-6, atol=1e-6 * np.abs(u_lin).max())


def test_large_deflection_cantilever():
    # Tip point load with PL^2/EI = 10: Mattiasson (1981): w/L = 0.81061, u/L = 0.55500
    L, b, t, E = 10.0, 1.0, 0.1, 1.2e4
    I = b * t**3 / 12
    P = 10.0 * E * I / L**2
    surf = geometry.plate(L, b).k_refine(4, 8, 1)
    model = ShellModel(surf, Material(E, 0.0, t))
    clamp(model.constraints, surf, "xi0", 2)
    model.add_edge_load("xi1", [0.0, 0.0, -P / b])
    u, hist = model.solve_nonlinear(n_steps=10, tol=1e-9)
    tip = model.displacement_at(u, 1.0, 0.5)
    assert np.isclose(-tip[2] / L, 0.81061, rtol=2e-3)
    assert np.isclose(-tip[0] / L, 0.55500, rtol=2e-3)
    # quadratic convergence at the end of every step; the linear predictor overshoots on this
    # slender beam so early iterations are spent recovering from spurious membrane stretching
    assert all(len(h) <= 13 for h in hist)
    K, f_int = model.tangent(u)
    r = model.T.T @ (model.f_ext - f_int)
    assert np.linalg.norm(r) < 1e-8 * np.linalg.norm(model.T.T @ model.f_ext)
