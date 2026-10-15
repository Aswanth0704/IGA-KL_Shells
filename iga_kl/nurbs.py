"""Tensor-product NURBS surface with rational basis derivatives up to second order.

Conventions
-----------
* Control net ``Pw[i, j, :] = (w x, w y, w z, w)`` (homogeneous), ``i`` along xi (degree p, knots U),
  ``j`` along eta (degree q, knots V).
* Global control-point index ``cp_index(i, j) = i + n * j``.
* Local basis ordering inside a knot span: ``a = aa + (p + 1) * bb`` (aa along xi, bb along eta).
* Second-derivative columns are ordered ``[xi xi, eta eta, xi eta]``.
"""

import numpy as np

from . import bspline as bs


def rational_basis_from_1d(Nd, Md, wloc):
    """Combine 1D basis derivatives into rational surface basis (values, 1st, 2nd derivatives).

    Nd : (3, p+1) values/derivatives of the nonzero xi-functions
    Md : (3, q+1) same along eta
    wloc : (p+1, q+1) weights of the local control points
    Returns R (n_en,), dR (n_en, 2), ddR (n_en, 3) with local index a = aa + (p+1)*bb.
    """
    # A_* = N M w (and derivatives), shape (p+1, q+1) -> flatten in Fortran order so aa is fastest
    def prod(kx, ky):
        return (np.outer(Nd[kx], Md[ky]) * wloc).ravel(order="F")

    A = prod(0, 0)
    Ax, Ay = prod(1, 0), prod(0, 1)
    Axx, Ayy, Axy = prod(2, 0), prod(0, 2), prod(1, 1)
    W = A.sum()
    Wx, Wy = Ax.sum(), Ay.sum()
    Wxx, Wyy, Wxy = Axx.sum(), Ayy.sum(), Axy.sum()
    R = A / W
    Rx = (Ax - R * Wx) / W
    Ry = (Ay - R * Wy) / W
    Rxx = (Axx - 2.0 * Rx * Wx - R * Wxx) / W
    Ryy = (Ayy - 2.0 * Ry * Wy - R * Wyy) / W
    Rxy = (Axy - Rx * Wy - Ry * Wx - R * Wxy) / W
    return R, np.stack([Rx, Ry], axis=1), np.stack([Rxx, Ryy, Rxy], axis=1)


class NurbsSurface:
    def __init__(self, U, V, p, q, Pw):
        self.U = np.asarray(U, dtype=float)
        self.V = np.asarray(V, dtype=float)
        self.p = int(p)
        self.q = int(q)
        self.Pw = np.asarray(Pw, dtype=float)
        assert self.Pw.ndim == 3 and self.Pw.shape[2] == 4
        assert self.Pw.shape[0] == bs.n_basis(self.U, self.p)
        assert self.Pw.shape[1] == bs.n_basis(self.V, self.q)

    # ------------------------------------------------------------------ properties
    @property
    def n(self):
        return self.Pw.shape[0]

    @property
    def m(self):
        return self.Pw.shape[1]

    @property
    def n_cp(self):
        return self.n * self.m

    @property
    def weights(self):
        return self.Pw[:, :, 3]

    @property
    def control_points(self):
        """Cartesian control points, shape (n, m, 3)."""
        return self.Pw[:, :, :3] / self.Pw[:, :, 3:4]

    @property
    def n_en(self):
        return (self.p + 1) * (self.q + 1)

    def cp_index(self, i, j):
        return i + self.n * j

    def copy(self):
        return NurbsSurface(self.U.copy(), self.V.copy(), self.p, self.q, self.Pw.copy())

    # ------------------------------------------------------------------ evaluation
    def spans(self):
        return bs.unique_spans(self.U), bs.unique_spans(self.V)

    def local_indices(self, span_i, span_j):
        """Global control-point indices of the functions nonzero on span (span_i, span_j)."""
        ii = np.arange(span_i - self.p, span_i + 1)
        jj = np.arange(span_j - self.q, span_j + 1)
        return (ii[:, None] + self.n * jj[None, :]).ravel(order="F")

    def basis(self, xi, eta, nders=2):
        """Rational basis at (xi, eta): (ien, R, dR, ddR)."""
        i = bs.find_span(self.n, self.p, xi, self.U)
        j = bs.find_span(self.m, self.q, eta, self.V)
        Nd = np.zeros((3, self.p + 1))
        Md = np.zeros((3, self.q + 1))
        nd = bs.basis_funs_ders(i, xi, self.p, self.U, 2)
        md = bs.basis_funs_ders(j, eta, self.q, self.V, 2)
        Nd[: nd.shape[0]] = nd
        Md[: md.shape[0]] = md
        wloc = self.Pw[i - self.p : i + 1, j - self.q : j + 1, 3]
        R, dR, ddR = rational_basis_from_1d(Nd, Md, wloc)
        return self.local_indices(i, j), R, dR, ddR

    def eval(self, xi, eta, nders=2):
        """Surface point and derivatives: returns (S, dS (2,3), ddS (3,3)) with rows [xx, yy, xy]."""
        ien, R, dR, ddR = self.basis(xi, eta, nders)
        P = self.control_points.reshape(-1, 3, order="F")[ien]
        S = R @ P
        dS = dR.T @ P
        ddS = ddR.T @ P
        return S, dS, ddS

    def point(self, xi, eta):
        return self.eval(xi, eta)[0]

    def displacement_at(self, u, xi, eta):
        """Interpolate a nodal field u (n_dof,) or (n_cp, 3) at (xi, eta)."""
        u = np.asarray(u).reshape(self.n_cp, -1)
        ien, R, _, _ = self.basis(xi, eta, 0)
        return R @ u[ien]

    # ------------------------------------------------------------------ topology helpers
    def edge_cps(self, edge, row=0):
        """Global indices of the control points in the `row`-th row parallel to an edge.

        edge in {'xi0', 'xi1', 'eta0', 'eta1'}; row=0 is the edge itself, row=1 the next one.
        """
        if edge == "xi0":
            return self.cp_index(row, np.arange(self.m))
        if edge == "xi1":
            return self.cp_index(self.n - 1 - row, np.arange(self.m))
        if edge == "eta0":
            return self.cp_index(np.arange(self.n), row)
        if edge == "eta1":
            return self.cp_index(np.arange(self.n), self.m - 1 - row)
        raise ValueError(edge)

    def corner_cp(self, corner):
        """Global index of a corner: corner = (xi_end, eta_end) with entries 0 or 1."""
        ci, cj = corner
        return self.cp_index(0 if ci == 0 else self.n - 1, 0 if cj == 0 else self.m - 1)

    def corner_param(self, corner):
        return (float(corner[0]), float(corner[1]))

    # ------------------------------------------------------------------ refinement
    def degree_elevate(self, direction, t):
        """Bezier degree elevation by t along 'xi' or 'eta' (requires a single knot span)."""
        if t <= 0:
            return self.copy()
        if direction == "xi":
            assert len(bs.unique_spans(self.U)) == 1, "degree elevation implemented for single span"
            new = np.stack([bs.bezier_degree_elevate(self.Pw[:, j, :], t) for j in range(self.m)], axis=1)
            U = np.concatenate([np.zeros(self.p + t + 1), np.ones(self.p + t + 1)])
            return NurbsSurface(U, self.V, self.p + t, self.q, new)
        if direction == "eta":
            assert len(bs.unique_spans(self.V)) == 1, "degree elevation implemented for single span"
            new = np.stack([bs.bezier_degree_elevate(self.Pw[i, :, :], t) for i in range(self.n)], axis=0)
            V = np.concatenate([np.zeros(self.q + t + 1), np.ones(self.q + t + 1)])
            return NurbsSurface(self.U, V, self.p, self.q + t, new)
        raise ValueError(direction)

    def insert_knots(self, direction, knots):
        knots = list(knots)
        if not knots:
            return self.copy()
        if direction == "xi":
            rows = []
            for j in range(self.m):
                U, Q = bs.refine_knots(self.U, self.p, self.Pw[:, j, :], knots)
                rows.append(Q)
            return NurbsSurface(U, self.V, self.p, self.q, np.stack(rows, axis=1))
        if direction == "eta":
            cols = []
            for i in range(self.n):
                V, Q = bs.refine_knots(self.V, self.q, self.Pw[i, :, :], knots)
                cols.append(Q)
            return NurbsSurface(self.U, V, self.p, self.q, np.stack(cols, axis=0))
        raise ValueError(direction)

    def k_refine(self, p_target, n_el, n_el_eta=None):
        """Elevate both directions to degree p_target, then insert uniform knots (C^{p-1})."""
        n_el_eta = n_el if n_el_eta is None else n_el_eta
        s = self.degree_elevate("xi", p_target - self.p).degree_elevate("eta", p_target - self.q)
        new_xi = [u for u in np.linspace(0, 1, n_el + 1)[1:-1] if not np.any(np.isclose(s.U, u))]
        new_eta = [v for v in np.linspace(0, 1, n_el_eta + 1)[1:-1] if not np.any(np.isclose(s.V, v))]
        return s.insert_knots("xi", new_xi).insert_knots("eta", new_eta)
