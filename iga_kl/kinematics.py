"""Kirchhoff-Love shell kinematics at Gauss points (Kiendl et al. 2009, Sec. 3).

All routines are vectorised over elements (E) and Gauss points (G); A = n_en local functions,
ndof = 3A local DOFs with r = 3a + k.

Curvature change convention: kappa_ab = B_ab - b_ab with b_ab = g_{a,b} . g_3 (Eq. 27).
Voigt vectors: strain [e11, e22, 2 e12], curvature [k11, k22, 2 k12], stress [n11, n22, n12].
"""

from dataclasses import dataclass

import numpy as np

_I3 = np.eye(3)
_LC = np.zeros((3, 3, 3))
_LC[0, 1, 2] = _LC[1, 2, 0] = _LC[2, 0, 1] = 1.0
_LC[0, 2, 1] = _LC[2, 1, 0] = _LC[1, 0, 2] = -1.0


def skew_dot(v):
    """S(v)[..., k, l] = (e_k x e_l) . v."""
    return np.einsum("klm,...m->...kl", _LC, v)


@dataclass
class SurfaceState:
    g1: np.ndarray  # (E, G, 3) covariant base vectors
    g2: np.ndarray
    g3: np.ndarray  # unit normal
    j: np.ndarray  # (E, G) |g1 x g2|
    gab: np.ndarray  # (E, G, 2, 2) covariant metric
    gd: np.ndarray  # (E, G, 3, 3) second derivatives x_{,ab}, rows [11, 22, 12]
    b: np.ndarray  # (E, G, 3) curvature b_ab = x_{,ab} . g3, rows [11, 22, 12]


def surface_state(dR, ddR, x_e):
    g1 = np.einsum("ega,eak->egk", dR[..., 0], x_e)
    g2 = np.einsum("ega,eak->egk", dR[..., 1], x_e)
    gd = np.einsum("egac,eak->egck", ddR, x_e)
    gt = np.cross(g1, g2)
    j = np.linalg.norm(gt, axis=-1)
    g3 = gt / j[..., None]
    gab = np.empty(g1.shape[:2] + (2, 2))
    gab[..., 0, 0] = np.einsum("egk,egk->eg", g1, g1)
    gab[..., 1, 1] = np.einsum("egk,egk->eg", g2, g2)
    gab[..., 0, 1] = gab[..., 1, 0] = np.einsum("egk,egk->eg", g1, g2)
    b = np.einsum("egck,egk->egc", gd, g3)
    return SurfaceState(g1, g2, g3, j, gab, gd, b)


def deformed_state(dR, ddR, rst, u_e):
    """Current-configuration state and Voigt strain measures for element displacements u_e (E, A, 3).

    Strains are evaluated from the displacement gradients rather than as differences of metrics
    (e.g. eps_ab = 1/2 (G_a.u_,b + G_b.u_,a + u_,a.u_,b)) so that no cancellation occurs when the
    deformation is tiny compared with the geometry (bending-dominated thin shells).
    """
    du1 = np.einsum("ega,eak->egk", dR[..., 0], u_e)
    du2 = np.einsum("ega,eak->egk", dR[..., 1], u_e)
    dud = np.einsum("egac,eak->egck", ddR, u_e)
    G1, G2, G3, J = rst.g1, rst.g2, rst.g3, rst.j
    g1, g2, gd = G1 + du1, G2 + du2, rst.gd + dud
    # g1 x g2 = J G3 + Delta,  Delta free of cancellation
    Delta = np.cross(G1, du2) + np.cross(du1, G2) + np.cross(du1, du2)
    gt = J[..., None] * G3 + Delta
    j = np.linalg.norm(gt, axis=-1)
    g3 = gt / j[..., None]
    gab = np.empty(g1.shape[:2] + (2, 2))
    gab[..., 0, 0] = np.einsum("egk,egk->eg", g1, g1)
    gab[..., 1, 1] = np.einsum("egk,egk->eg", g2, g2)
    gab[..., 0, 1] = gab[..., 1, 0] = np.einsum("egk,egk->eg", g1, g2)
    b = np.einsum("egck,egk->egc", gd, g3)
    st = SurfaceState(g1, g2, g3, j, gab, gd, b)
    dot = lambda a, c: np.einsum("egk,egk->eg", a, c)
    eps = np.stack(
        [
            dot(G1, du1) + 0.5 * dot(du1, du1),
            dot(G2, du2) + 0.5 * dot(du2, du2),
            dot(G1, du2) + dot(G2, du1) + dot(du1, du2),
        ],
        axis=-1,
    )
    # d3 = g3 - G3 without cancellation: Delta/j - G3 (j - J)/j, (j - J) = (2 J G3.Delta + Delta.Delta)/(j + J)
    jmJ = (2.0 * J * dot(G3, Delta) + dot(Delta, Delta)) / (j + J)
    d3 = Delta / j[..., None] - G3 * (jmJ / j)[..., None]
    # kappa_ab = B_ab - b_ab = -(X_,ab . d3 + u_,ab . g3)
    kap = -(np.einsum("egck,egk->egc", rst.gd, d3) + np.einsum("egck,egk->egc", dud, g3))
    kap[..., 2] *= 2.0
    return st, eps, kap


def first_variations(dR, ddR, st):
    """B-matrices (E, G, 3, ndof) and the auxiliary normal variations.

    Returns Bm, Bb, dgt (E,G,ndof,3) = d(g1 x g2)/du_r, dJ (E,G,ndof) = dj/du_r, dg3 (E,G,ndof,3).
    """
    E, G, A = dR.shape[:3]
    ndof = 3 * A
    dN1, dN2 = dR[..., 0], dR[..., 1]
    # membrane: rows e11, e22, 2 e12
    Bm = np.stack(
        [
            np.einsum("ega,egk->egak", dN1, st.g1).reshape(E, G, ndof),
            np.einsum("ega,egk->egak", dN2, st.g2).reshape(E, G, ndof),
            (np.einsum("ega,egk->egak", dN1, st.g2) + np.einsum("ega,egk->egak", dN2, st.g1)).reshape(E, G, ndof),
        ],
        axis=2,
    )
    # variation of g~ = g1 x g2:  dgt[r,:] = N_a,1 (e_k x g2) + N_a,2 (g1 x e_k)
    C2 = np.cross(_I3[None, None, :, :], st.g2[:, :, None, :])  # (E,G,3,3) row k = e_k x g2
    C1 = np.cross(st.g1[:, :, None, :], _I3[None, None, :, :])  # row k = g1 x e_k
    dgt = (dN1[..., None, None] * C2[:, :, None] + dN2[..., None, None] * C1[:, :, None]).reshape(E, G, ndof, 3)
    dJ = np.einsum("egrk,egk->egr", dgt, st.g3)
    dg3 = (dgt - dJ[..., None] * st.g3[:, :, None, :]) / st.j[..., None, None]
    # bending: dkappa_c = -[ N_a,c g3_k + dgt . (x_{,c} - b_c g3) / j ]
    rows = []
    for c in range(3):
        v = (st.gd[:, :, c, :] - st.b[:, :, c, None] * st.g3) / st.j[..., None]
        rows.append(-(np.einsum("ega,egk->egak", ddR[..., c], st.g3).reshape(E, G, ndof) + np.einsum("egrk,egk->egr", dgt, v)))
    rows[2] = 2.0 * rows[2]
    Bb = np.stack(rows, axis=2)
    return Bm, Bb, dgt, dJ, dg3


def geometric_stiffness(dR, ddR, st, dgt, dJ, dg3, n, m, c):
    """Integrated geometric stiffness (E, ndof, ndof): n : d2eps + m : d2kappa, weights c = w*J_ref.

    n, m: (E, G, 3) Voigt stress resultants [11, 22, 12].
    """
    E, G, A = dR.shape[:3]
    ndof = 3 * A
    dN1, dN2 = dR[..., 0], dR[..., 1]
    # membrane part
    Nn = (
        n[..., 0, None, None] * dN1[..., :, None] * dN1[..., None, :]
        + n[..., 1, None, None] * dN2[..., :, None] * dN2[..., None, :]
        + n[..., 2, None, None] * (dN1[..., :, None] * dN2[..., None, :] + dN2[..., :, None] * dN1[..., None, :])
    )
    Nnw = np.einsum("eg,egab->eab", c, Nn)
    K = np.einsum("eab,kl->eakbl", Nnw, _I3).reshape(E, ndof, ndof)
    # bending part
    W = dN1[..., :, None] * dN2[..., None, :] - dN2[..., :, None] * dN1[..., None, :]  # (E,G,A,A)
    Mbar = m[..., 0, None] * st.gd[:, :, 0] + m[..., 1, None] * st.gd[:, :, 1] + 2.0 * m[..., 2, None] * st.gd[:, :, 2]
    mN = m[..., 0, None] * ddR[..., 0] + m[..., 1, None] * ddR[..., 1] + 2.0 * m[..., 2, None] * ddR[..., 2]
    Aterm = np.einsum("eg,ega,egsk->eaks", c, mN, dg3, optimize=True).reshape(E, ndof, ndof)
    cj = c / st.j
    Mg = np.einsum("egk,egk->eg", Mbar, st.g3)
    dg3M = np.einsum("egsk,egk->egs", dg3, Mbar)
    t1 = np.einsum("egab,egkl->eakbl", W * cj[..., None, None], skew_dot(Mbar), optimize=True).reshape(E, ndof, ndof)
    t2 = np.einsum("egs,egr->esr", dg3M * cj[..., None], dJ, optimize=True)
    cm = cj * Mg
    t4 = np.einsum("egab,egkl->eakbl", W * cm[..., None, None], skew_dot(st.g3), optimize=True).reshape(E, ndof, ndof)
    t4 += np.einsum("egrk,egsk->ers", dgt * cm[..., None, None], dg3, optimize=True)
    Cm = t1 - t2 - t2.transpose(0, 2, 1) - t4
    K -= Aterm + Aterm.transpose(0, 2, 1) + Cm
    return K
