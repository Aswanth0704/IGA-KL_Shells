import numpy as np
import pytest

from iga_kl import geometry
from iga_kl.constraints import Constraints, clamp, rigid_diaphragm, symmetry
from iga_kl.material import Material
from iga_kl.model import ShellModel


def test_constraint_builder_chains():
    c = Constraints(9)
    c.fix(0)
    c.couple(1, 0)
    c.couple(2, 1)
    c.couple(3, 4)
    c.couple(4, 5)
    c.fix(7, 2.0)
    T, u_p = c.build()
    assert T.shape == (9, 3)  # classes: {3,4,5}, {6}, {8}
    assert np.allclose(T[[0, 1, 2, 7]].toarray(), 0.0)
    assert np.allclose(T[[3, 4, 5]].toarray().sum(axis=0), [3, 0, 0]) or np.allclose(
        np.sort(T[[3, 4, 5]].toarray().sum(axis=0)), [0, 0, 3]
    )
    assert u_p[7] == 2.0 and np.count_nonzero(u_p) == 1
    with pytest.raises(ValueError):
        c.fix(8, 1.0)
        c.couple(8, 7)


@pytest.mark.parametrize(
    "surf",
    [
        geometry.plate(2.0, 1.0).k_refine(2, 3),
        geometry.cylinder(1.0, 2.0, 0.0, 1.0).k_refine(3, 2),
        geometry.hemisphere_quarter(1.0).k_refine(2, 3),
    ],
)
def test_stiffness_symmetric_with_six_rigid_modes(surf):
    model = ShellModel(surf, Material(100.0, 0.3, 0.05))
    K = model.stiffness().toarray()
    assert np.allclose(K, K.T, atol=1e-9 * np.abs(K).max())
    ev = np.linalg.eigvalsh(K)
    scale = ev[-1]
    assert np.all(np.abs(ev[:6]) < 1e-9 * scale)
    assert ev[6] > 1e-6 * scale


def test_membrane_patch_test():
    surf = geometry.plate(3.0, 2.0).k_refine(3, 3)
    model = ShellModel(surf, Material(50.0, 0.25, 0.1))
    A = np.array([[0.01, 0.004, 0.0], [-0.002, 0.005, 0.0], [0.0, 0.0, 0.0]])
    P = surf.control_points.reshape(-1, 3, order="F")
    u_exact = (P @ A.T).ravel()
    for edge in ["xi0", "xi1", "eta0", "eta1"]:
        for cp in surf.edge_cps(edge, 0):
            for k in range(3):
                model.constraints.fix(3 * cp + k, u_exact[3 * cp + k])
    u = model.solve_linear()
    assert np.allclose(u, u_exact, atol=1e-10)


def test_pressurised_cylinder_exact():
    R, L, t, E, p = 2.0, 3.0, 0.02, 1000.0, 1.0
    surf = geometry.cylinder(R, L, 0.0, np.pi / 2).k_refine(3, 3)
    w = p * R**2 / (E * t) / (1.0 + t**2 / (12 * R**2))
    P = surf.control_points.reshape(-1, 3, order="F")
    u_exact = (w / R) * np.column_stack([P[:, 0], P[:, 1], np.zeros(len(P))]).ravel()
    # the uniform radial field is exact; the default (p+1)-point Gauss rule leaves a small
    # quadrature error on the rational integrands, an 8-point rule integrates them to 1e-12
    for n_gp, tol in [(None, 1e-4), (8, 1e-10)]:
        model = ShellModel(surf, Material(E, 0.0, t), n_gp=n_gp)
        symmetry(model.constraints, surf, "xi0", 1)  # plane y = 0
        symmetry(model.constraints, surf, "xi1", 0)  # plane x = 0
        symmetry(model.constraints, surf, "eta0", 2)  # plane z = 0
        model.add_pressure(p)
        u = model.solve_linear()
        assert np.abs(u - u_exact).max() < tol * w


def test_cantilever_single_cubic_element():
    L, b, t, E, P = 10.0, 1.0, 0.1, 1000.0, 0.5
    surf = geometry.plate(L, b).k_refine(3, 1)
    model = ShellModel(surf, Material(E, 0.0, t))
    clamp(model.constraints, surf, "xi0", 2)
    model.add_edge_load("xi1", [0.0, 0.0, -P / b])
    u = model.solve_linear()
    I = b * t**3 / 12
    w_tip = P * L**3 / (3 * E * I)
    assert np.isclose(-model.displacement_at(u, 1.0, 0.5)[2], w_tip, rtol=1e-10)
    assert np.isclose(-model.displacement_at(u, 0.5, 0.2)[2], P * (3 * L * (L / 2) ** 2 - (L / 2) ** 3) / (6 * E * I), rtol=1e-10)


def test_gravity_total_load():
    R, L = 25.0, 50.0
    surf = geometry.cylinder(R, L / 2, np.radians(50), np.radians(90)).k_refine(3, 4)
    model = ShellModel(surf, Material(4.32e8, 0.0, 0.25))
    model.add_body_load([0.0, -90.0, 0.0])
    assert np.isclose(model.f_ext[1::3].sum(), -90.0 * R * (L / 2) * np.radians(40))
    assert np.allclose(model.f_ext[0::3], 0.0) and np.allclose(model.f_ext[2::3], 0.0)


def _pinched(surf, load_param, F, sym_z):
    model = ShellModel(surf, Material(3.0e6, 0.3, 3.0))
    c = model.constraints
    rigid_diaphragm(c, surf, "eta0")
    if sym_z:
        symmetry(c, surf, "eta1", 2)
    else:
        rigid_diaphragm(c, surf, "eta1")
    symmetry(c, surf, "xi0", 1)
    symmetry(c, surf, "xi1", 0)
    model.add_point_load(load_param, [-F, 0.0, 0.0])
    u = model.solve_linear()
    return -model.displacement_at(u, *load_param)[0]


def test_symmetry_eighth_vs_quarter_pinched_cylinder():
    # the point load makes the solution non-smooth at z = L/2; the 1/8 model (symmetry edge) and
    # the 1/4 model (C^{p-1} line through the load) therefore differ at the discretisation level
    R, L = 300.0, 600.0
    p, n = 4, 12
    eighth = geometry.cylinder(R, L / 2, 0.0, np.pi / 2).k_refine(p, n)
    quarter = geometry.cylinder(R, L, 0.0, np.pi / 2).k_refine(p, n, 2 * n)
    u8 = _pinched(eighth, (0.0, 1.0), 0.25, True)
    u4 = _pinched(quarter, (0.0, 0.5), 0.5, False)
    assert np.isclose(u8, u4, rtol=0.03)
    assert np.isclose(u8, 1.8264e-5, rtol=0.03)


def _scordelis(surf, sym_z):
    model = ShellModel(surf, Material(4.32e8, 0.0, 0.25))
    c = model.constraints
    rigid_diaphragm(c, surf, "eta0")
    if sym_z:
        symmetry(c, surf, "eta1", 2)
    else:
        rigid_diaphragm(c, surf, "eta1")
    symmetry(c, surf, "xi1", 0)  # crown, plane x = 0
    model.add_body_load([0.0, -90.0, 0.0])
    return model, model.solve_linear()


def test_symmetry_quarter_vs_half_scordelis_lo():
    # smooth solution: the symmetry-plane model must reproduce the unsymmetrised model closely
    R, L = 25.0, 50.0
    p, n = 4, 8
    quarter = geometry.cylinder(R, L / 2, np.radians(50), np.radians(90)).k_refine(p, n)
    half = geometry.cylinder(R, L, np.radians(50), np.radians(90)).k_refine(p, n, 2 * n)
    mq, uq = _scordelis(quarter, True)
    mh, uh = _scordelis(half, False)
    wq = -mq.displacement_at(uq, 0.0, 1.0)[1]
    wh = -mh.displacement_at(uh, 0.0, 0.5)[1]
    assert np.isclose(wq, wh, rtol=1e-4)
    assert np.isclose(wq, 0.3006, rtol=2e-3)
