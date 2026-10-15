"""Element/quadrature data precomputed once for a NURBS surface."""

import numpy as np

from . import bspline as bs
from .nurbs import rational_basis_from_1d


class Mesh:
    """Knot spans -> elements; Gauss points with basis values/derivatives stored per element.

    Attributes (E = n_el, G = n_gp, A = n_en)
    ----------
    IEN     : (E, A) global control-point index of each local function
    dof_map : (E, 3A) global DOF of local DOF r = 3a + k
    R       : (E, G, A) basis values
    dR      : (E, G, A, 2) first parametric derivatives
    ddR     : (E, G, A, 3) second derivatives [xi xi, eta eta, xi eta]
    w       : (E, G) Gauss weight times parametric span factors (dxi/2)(deta/2)
    gp      : (E, G, 2) parametric coordinates of the Gauss points
    X_e     : (E, A, 3) Cartesian control points of each element
    """

    def __init__(self, surface, n_gp=None):
        self.surface = surface
        p, q = surface.p, surface.q
        ngx = p + 1 if n_gp is None else n_gp
        ngy = q + 1 if n_gp is None else n_gp
        gx, wx = bs.gauss_legendre(ngx)
        gy, wy = bs.gauss_legendre(ngy)
        spans_xi, spans_eta = surface.spans()
        self.n_el_xi, self.n_el_eta = len(spans_xi), len(spans_eta)
        E = self.n_el_xi * self.n_el_eta
        A = surface.n_en
        G = ngx * ngy
        self.n_el, self.n_gp, self.n_en = E, G, A
        self.IEN = np.zeros((E, A), dtype=int)
        self.R = np.zeros((E, G, A))
        self.dR = np.zeros((E, G, A, 2))
        self.ddR = np.zeros((E, G, A, 3))
        self.w = np.zeros((E, G))
        self.gp = np.zeros((E, G, 2))
        P = surface.control_points.reshape(-1, 3, order="F")
        for ey, (v0, v1) in enumerate(spans_eta):
            j = bs.find_span(surface.m, q, 0.5 * (v0 + v1), surface.V)
            etas = 0.5 * (v1 - v0) * gy + 0.5 * (v0 + v1)
            Md = [np.zeros((3, q + 1)) for _ in range(ngy)]
            for b, eta in enumerate(etas):
                d = bs.basis_funs_ders(j, eta, q, surface.V, 2)
                Md[b][: d.shape[0]] = d
            for ex, (u0, u1) in enumerate(spans_xi):
                i = bs.find_span(surface.n, p, 0.5 * (u0 + u1), surface.U)
                xis = 0.5 * (u1 - u0) * gx + 0.5 * (u0 + u1)
                Nd = [np.zeros((3, p + 1)) for _ in range(ngx)]
                for a, xi in enumerate(xis):
                    d = bs.basis_funs_ders(i, xi, p, surface.U, 2)
                    Nd[a][: d.shape[0]] = d
                e = ex + self.n_el_xi * ey
                self.IEN[e] = surface.local_indices(i, j)
                wloc = surface.Pw[i - p : i + 1, j - q : j + 1, 3]
                for b in range(ngy):
                    for a in range(ngx):
                        g = a + ngx * b
                        R, dR, ddR = rational_basis_from_1d(Nd[a], Md[b], wloc)
                        self.R[e, g] = R
                        self.dR[e, g] = dR
                        self.ddR[e, g] = ddR
                        self.w[e, g] = wx[a] * wy[b] * 0.25 * (u1 - u0) * (v1 - v0)
                        self.gp[e, g] = (xis[a], etas[b])
        self.X_e = P[self.IEN]
        self.dof_map = (3 * self.IEN[:, :, None] + np.arange(3)[None, None, :]).reshape(E, 3 * A)
        self.n_dof = 3 * surface.n_cp

    def element_values(self, u):
        """Gather a global DOF vector (n_dof,) into element arrays (E, A, 3)."""
        return np.asarray(u).reshape(-1, 3)[self.IEN]
