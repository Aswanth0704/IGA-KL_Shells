"""External load vectors (all conservative; evaluated on the reference configuration)."""

import numpy as np

from . import bspline as bs
from .assembly import assemble_vector


def body_load(mesh, ref, vec):
    """Uniform load per unit (reference) midsurface area."""
    fe = np.einsum("eg,ega,k->eak", ref.c, mesh.R, np.asarray(vec, float)).reshape(mesh.n_el, -1)
    return assemble_vector(mesh.dof_map, fe, mesh.n_dof)


def pressure_load(mesh, ref, p):
    """Normal pressure p along the reference normal G3."""
    fe = np.einsum("eg,ega,egk->eak", ref.c * p, mesh.R, ref.st.g3).reshape(mesh.n_el, -1)
    return assemble_vector(mesh.dof_map, fe, mesh.n_dof)


def point_load(surf, param, F):
    """Consistent nodal loads of a point force F at parametric location (xi, eta).

    At a patch corner this puts F directly on the corner control point (corners are interpolatory).
    """
    f = np.zeros(3 * surf.n_cp)
    ien, R, _, _ = surf.basis(float(param[0]), float(param[1]), 0)
    F = np.asarray(F, float)
    for a, cp in enumerate(ien):
        f[3 * cp : 3 * cp + 3] += R[a] * F
    return f


def edge_line_load(surf, edge, q, n_gp=None):
    """Consistent nodal loads of a uniform line load q (per unit length) along a patch edge."""
    q = np.asarray(q, float)
    f = np.zeros(3 * surf.n_cp)
    along_eta = edge in ("xi0", "xi1")
    fixed = 0.0 if edge in ("xi0", "eta0") else 1.0
    spans = surf.spans()[1 if along_eta else 0]
    deg = surf.q if along_eta else surf.p
    g, w = bs.gauss_legendre(deg + 1 if n_gp is None else n_gp)
    for a0, a1 in spans:
        for gk, wk in zip(g, w):
            s = 0.5 * (a1 - a0) * gk + 0.5 * (a0 + a1)
            xi, eta = (fixed, s) if along_eta else (s, fixed)
            ien, R, dR, _ = surf.basis(xi, eta)
            P = surf.control_points.reshape(-1, 3, order="F")[ien]
            tangent = dR[:, 1 if along_eta else 0] @ P
            ds = np.linalg.norm(tangent) * 0.5 * (a1 - a0) * wk
            for a, cp in enumerate(ien):
                f[3 * cp : 3 * cp + 3] += R[a] * ds * q
    return f
