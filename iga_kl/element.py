"""Element tangent stiffness, internal force and strain energy (total Lagrangian)."""

import contextlib
from dataclasses import dataclass

import numpy as np

from .kinematics import SurfaceState, deformed_state, first_variations, geometric_stiffness, surface_state
from .material import material_matrix

try:  # multithreaded OpenBLAS is ~100x slower than single-threaded on the small per-element matmuls
    from threadpoolctl import threadpool_limits

    def _single_thread_blas():
        return threadpool_limits(limits=1, user_api="blas")

except ImportError:  # pragma: no cover

    def _single_thread_blas():
        return contextlib.nullcontext()


@dataclass
class ReferenceData:
    st: SurfaceState
    Gcon: np.ndarray  # (E, G, 2, 2)
    Dm: np.ndarray  # (E, G, 3, 3)  t D
    Db: np.ndarray  # (E, G, 3, 3)  t^3/12 D
    c: np.ndarray  # (E, G) quadrature weight * reference Jacobian


def reference_data(mesh, mat):
    st = surface_state(mesh.dR, mesh.ddR, mesh.X_e)
    Gcon = np.linalg.inv(st.gab)
    D = material_matrix(Gcon, mat.E, mat.nu)
    return ReferenceData(st, Gcon, mat.t * D, mat.t**3 / 12.0 * D, mesh.w * st.j)


def _slice_state(st, sl):
    return SurfaceState(st.g1[sl], st.g2[sl], st.g3[sl], st.j[sl], st.gab[sl], st.gd[sl], st.b[sl])


def element_arrays(mesh, ref, u=None, geometric=True, chunk=256):
    """Element tangent matrices Ke (E, ndof, ndof) and internal forces fe (E, ndof) at state u.

    With u=None the reference configuration is used (linear stiffness, fe = 0).
    """
    E, G, A = mesh.R.shape
    ndof = 3 * A
    Ke = np.zeros((E, ndof, ndof))
    fe = np.zeros((E, ndof))
    u_e = None if u is None else mesh.element_values(u)
    with _single_thread_blas():
        _element_loop(mesh, ref, u_e, geometric, chunk, Ke, fe)
    return Ke, fe


def _element_loop(mesh, ref, u_e, geometric, chunk, Ke, fe):
    E, G, A = mesh.R.shape
    ndof = 3 * A
    for e0 in range(0, E, chunk):
        sl = slice(e0, min(E, e0 + chunk))
        dR, ddR = mesh.dR[sl], mesh.ddR[sl]
        c = ref.c[sl]
        rst = _slice_state(ref.st, sl)
        if u_e is None:
            st = rst
        else:
            st, eps, kap = deformed_state(dR, ddR, rst, u_e[sl])
        Bm, Bb, dgt, dJ, dg3 = first_variations(dR, ddR, st)
        Ec = Bm.shape[0]
        for B, D in ((Bm, ref.Dm[sl]), (Bb, ref.Db[sl])):
            DB = np.einsum("egij,egjs->egis", D, B) * c[..., None, None]
            Ke[sl] += B.transpose(0, 3, 1, 2).reshape(Ec, ndof, G * 3) @ DB.reshape(Ec, G * 3, ndof)
        if u_e is not None:
            n = np.einsum("egij,egj->egi", ref.Dm[sl], eps)
            m = np.einsum("egij,egj->egi", ref.Db[sl], kap)
            fe[sl] = np.einsum("eg,egir,egi->er", c, Bm, n) + np.einsum("eg,egir,egi->er", c, Bb, m)
            if geometric:
                Ke[sl] += geometric_stiffness(dR, ddR, st, dgt, dJ, dg3, n, m, c)


def stress_resultants(mesh, ref, u):
    """Voigt strains and stress resultants (eps, kap, n, m), each (E, G, 3), at the Gauss points."""
    _, eps, kap = deformed_state(mesh.dR, mesh.ddR, ref.st, mesh.element_values(u))
    return eps, kap, np.einsum("egij,egj->egi", ref.Dm, eps), np.einsum("egij,egj->egi", ref.Db, kap)


def strain_energy(mesh, ref, u):
    eps, kap, n, m = stress_resultants(mesh, ref, u)
    return 0.5 * np.sum(ref.c * (np.einsum("egi,egi->eg", eps, n) + np.einsum("egi,egi->eg", kap, m)))
