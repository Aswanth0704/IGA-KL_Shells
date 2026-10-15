import numpy as np

from iga_kl import geometry
from iga_kl.element import reference_data
from iga_kl.kinematics import first_variations
from iga_kl.material import Material, material_matrix
from iga_kl.mesh import Mesh


def _setup(surface, mat=Material(1.0, 0.3, 0.1)):
    mesh = Mesh(surface)
    ref = reference_data(mesh, mat)
    Bm, Bb, *_ = first_variations(mesh.dR, mesh.ddR, ref.st)
    return mesh, ref, Bm, Bb


def _apply(mesh, B, u):
    u_e = mesh.element_values(u).reshape(mesh.n_el, -1)
    return np.einsum("egir,er->egi", B, u_e)


def test_material_matrix_reduces_to_plane_stress():
    E, nu = 210.0, 0.3
    D = material_matrix(np.eye(2)[None], E, nu)[0]
    Dref = E / (1 - nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    assert np.allclose(D, Dref)


def test_reference_curvatures():
    R = 3.0
    mesh, ref, _, _ = _setup(geometry.cylinder(R, 5.0, 0.2, 1.4).k_refine(3, 3))
    # mixed curvature tensor b^a_b = G^{ac} B_{cb}: eigenvalues {1/R, 0} (sign convention aside)
    B = np.zeros(ref.st.gab.shape)
    B[..., 0, 0], B[..., 1, 1] = ref.st.b[..., 0], ref.st.b[..., 1]
    B[..., 0, 1] = B[..., 1, 0] = ref.st.b[..., 2]
    mixed = ref.Gcon @ B
    ev = np.sort(np.abs(np.linalg.eigvals(mixed)), axis=-1)
    assert np.allclose(ev[..., 0], 0.0, atol=1e-10)
    assert np.allclose(ev[..., 1], 1.0 / R)
    mesh, ref, _, _ = _setup(geometry.hemisphere_quarter(R).k_refine(3, 3))
    # sphere with outward normal: b_ab = -G_ab / R
    assert np.allclose(ref.st.b[..., 0], -ref.st.gab[..., 0, 0] / R)
    assert np.allclose(ref.st.b[..., 1], -ref.st.gab[..., 1, 1] / R)
    assert np.allclose(ref.st.b[..., 2], -ref.st.gab[..., 0, 1] / R)


def test_membrane_strain_of_affine_field():
    rng = np.random.default_rng(3)
    A = rng.random((3, 3))
    surf = geometry.hemisphere_quarter(2.0).k_refine(3, 2)
    mesh, ref, Bm, Bb = _setup(surf)
    P = surf.control_points.reshape(-1, 3, order="F")
    u = (P @ A.T).ravel()
    eps = _apply(mesh, Bm, u)
    G1, G2 = ref.st.g1, ref.st.g2
    AG1, AG2 = G1 @ A.T, G2 @ A.T
    e11 = np.einsum("egk,egk->eg", G1, AG1)
    e22 = np.einsum("egk,egk->eg", G2, AG2)
    e12 = np.einsum("egk,egk->eg", G1, AG2) + np.einsum("egk,egk->eg", G2, AG1)
    assert np.allclose(eps[..., 0], e11)
    assert np.allclose(eps[..., 1], e22)
    assert np.allclose(eps[..., 2], e12)


def test_rigid_motion_produces_no_strain():
    rng = np.random.default_rng(4)
    c, om = rng.random(3), rng.random(3)
    for surf in [geometry.cylinder(2.0, 3.0, 0.0, 1.2).k_refine(3, 2), geometry.hemisphere_quarter(2.0).k_refine(2, 3)]:
        mesh, ref, Bm, Bb = _setup(surf)
        P = surf.control_points.reshape(-1, 3, order="F")
        u = (c[None, :] + np.cross(om[None, :], P)).ravel()
        assert np.allclose(_apply(mesh, Bm, u), 0.0, atol=1e-12)
        assert np.allclose(_apply(mesh, Bb, u), 0.0, atol=1e-12)


def test_plate_bending_curvature():
    # w = x^2 / 2 on the unit plate with p = 2: Bezier ordinates (0, 0, 1/2) along xi -> kappa_11 = -1
    surf = geometry.plate(1.0, 1.0).k_refine(2, 1)
    mesh, ref, Bm, Bb = _setup(surf)
    u = np.zeros((surf.n, surf.m, 3))
    u[2, :, 2] = 0.5
    kap = _apply(mesh, Bb, u.reshape(-1, order="F").reshape(-1))
    u_vec = np.zeros(3 * surf.n_cp)
    for i in range(surf.n):
        for j in range(surf.m):
            u_vec[3 * surf.cp_index(i, j) + 2] = u[i, j, 2]
    kap = _apply(mesh, Bb, u_vec)
    assert np.allclose(kap[..., 0], -1.0)
    assert np.allclose(kap[..., 1:], 0.0, atol=1e-12)
    assert np.allclose(_apply(mesh, Bm, u_vec), 0.0, atol=1e-12)
