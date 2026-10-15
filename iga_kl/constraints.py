"""Homogeneous/inhomogeneous Dirichlet constraints and equality couplings via a transformation
matrix T:  u = T u_hat + u_p.  Built with union-find so chained and redundant constraints are fine.
"""

import numpy as np
import scipy.sparse as sp


class Constraints:
    def __init__(self, n_dof):
        self.n_dof = n_dof
        self._parent = np.arange(n_dof)
        self._value = {}  # root -> prescribed value

    # ------------------------------------------------------------------ union-find
    def _find(self, i):
        while self._parent[i] != i:
            self._parent[i] = self._parent[self._parent[i]]
            i = self._parent[i]
        return i

    def _set_value(self, root, value):
        old = self._value.get(root)
        if old is not None and not np.isclose(old, value):
            raise ValueError(f"conflicting prescribed values {old} and {value}")
        self._value[root] = value

    # ------------------------------------------------------------------ public API
    def fix(self, dofs, value=0.0):
        for d in np.atleast_1d(dofs):
            self._set_value(self._find(int(d)), float(value))

    def couple(self, a, b):
        """Enforce u[a] == u[b]."""
        ra, rb = self._find(int(a)), self._find(int(b))
        if ra == rb:
            return
        va, vb = self._value.pop(ra, None), self._value.pop(rb, None)
        self._parent[ra] = rb
        for v in (va, vb):
            if v is not None:
                self._set_value(rb, v)

    def fix_cps(self, cps, comps=(0, 1, 2), value=0.0):
        for k in comps:
            self.fix(3 * np.asarray(cps) + k, value)

    def couple_cps(self, cps_a, cps_b, comps=(0, 1, 2)):
        for a, b in zip(np.atleast_1d(cps_a), np.atleast_1d(cps_b)):
            for k in comps:
                self.couple(3 * a + k, 3 * b + k)

    def build(self):
        roots = np.array([self._find(i) for i in range(self.n_dof)])
        free_roots = sorted(set(roots.tolist()) - set(self._value.keys()))
        col = {r: c for c, r in enumerate(free_roots)}
        u_p = np.zeros(self.n_dof)
        rows, cols = [], []
        for i, r in enumerate(roots):
            if r in self._value:
                u_p[i] = self._value[r]
            else:
                rows.append(i)
                cols.append(col[r])
        T = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(self.n_dof, len(free_roots)))
        return T, u_p


# ---------------------------------------------------------------------- shell BC helpers
def fix_edge(cons, surf, edge, comps=(0, 1, 2), rows=(0,)):
    for r in rows:
        cons.fix_cps(surf.edge_cps(edge, r), comps)


def rigid_diaphragm(cons, surf, edge, comps=(0, 1)):
    """Diaphragm in the plane spanned by `comps` (default x-y plane, cylinder axis z)."""
    cons.fix_cps(surf.edge_cps(edge, 0), comps)


def clamp(cons, surf, edge, normal_comp):
    """Clamped edge (paper Sec. 4.1): first row fully fixed, second row fixed normal to the surface."""
    cons.fix_cps(surf.edge_cps(edge, 0), (0, 1, 2))
    cons.fix_cps(surf.edge_cps(edge, 1), (normal_comp,))


def symmetry(cons, surf, edge, normal_axis):
    """Symmetry about the coordinate plane with normal `normal_axis` (0, 1 or 2).

    Row 1: u.n = 0. Row 2: the two in-plane components equal those of the row-1 partner, so the
    transverse tangent keeps its in-plane components -> no rotation about the edge.
    Exact when the surface meets the plane at right angles and weights are separable.
    """
    r0, r1 = surf.edge_cps(edge, 0), surf.edge_cps(edge, 1)
    cons.fix_cps(r0, (normal_axis,))
    cons.couple_cps(r1, r0, tuple(k for k in range(3) if k != normal_axis))


def couple_pole(cons, surf, edge):
    """Tie all DOFs of a degenerate (collapsed) edge together."""
    cps = surf.edge_cps(edge, 0)
    cons.couple_cps(cps[1:], np.full(len(cps) - 1, cps[0]))
