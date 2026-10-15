"""Convenience wrapper tying surface, mesh, material, loads and constraints together."""

import numpy as np

from . import loads as ld
from .assembly import assemble_matrix, assemble_vector
from .constraints import Constraints
from .element import element_arrays, reference_data, strain_energy
from .mesh import Mesh
from .solver import newton, solve_linear


class ShellModel:
    def __init__(self, surface, material, n_gp=None):
        self.surface = surface
        self.material = material
        self.mesh = Mesh(surface, n_gp)
        self.ref = reference_data(self.mesh, material)
        self.n_dof = self.mesh.n_dof
        self.constraints = Constraints(self.n_dof)
        self.f_ext = np.zeros(self.n_dof)
        self._T = None

    # ------------------------------------------------------------------ loads
    def add_body_load(self, vec):
        self.f_ext += ld.body_load(self.mesh, self.ref, vec)

    def add_pressure(self, p):
        self.f_ext += ld.pressure_load(self.mesh, self.ref, p)

    def add_point_load(self, param, F):
        """Point force at parametric location (xi, eta); corners are (0|1, 0|1)."""
        self.f_ext += ld.point_load(self.surface, param, F)

    def add_edge_load(self, edge, q):
        self.f_ext += ld.edge_line_load(self.surface, edge, q)

    # ------------------------------------------------------------------ constraints
    @property
    def T(self):
        if self._T is None:
            self._T, self._u_p = self.constraints.build()
        return self._T

    @property
    def u_p(self):
        self.T
        return self._u_p

    # ------------------------------------------------------------------ assembly / solve
    def tangent(self, u=None, geometric=True):
        Ke, fe = element_arrays(self.mesh, self.ref, u, geometric)
        K = assemble_matrix(self.mesh.dof_map, Ke, self.n_dof)
        f_int = assemble_vector(self.mesh.dof_map, fe, self.n_dof)
        return K, f_int

    def stiffness(self):
        return self.tangent(None)[0]

    def solve_linear(self):
        K = self.stiffness()
        return solve_linear(K, self.f_ext, self.T, self.u_p)

    def solve_nonlinear(self, n_steps=1, tol=1e-8, max_iter=25, verbose=False, u0=None):
        return newton(self, n_steps, tol, max_iter, verbose, u0)

    def energy(self, u):
        return strain_energy(self.mesh, self.ref, u)

    def displacement_at(self, u, xi, eta):
        return self.surface.displacement_at(u, xi, eta)

    def corner_displacement(self, u, corner):
        cp = self.surface.corner_cp(corner)
        return np.asarray(u)[3 * cp : 3 * cp + 3]
