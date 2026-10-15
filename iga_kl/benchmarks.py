"""Shell obstacle course (Belytschko et al. 1985) as set up in Kiendl et al. 2009, Sec. 4.4 / Fig. 10.

Each builder returns a `Benchmark` whose `solve()` gives the quantity of interest:
  * Scordelis-Lo roof      : vertical displacement at the midpoint of the free edge   (0.3006 / ref 0.3024)
  * Pinched cylinder       : radial displacement under the point load               (1.8264e-5 / ref 1.8248e-5)
  * Hemispherical shell    : radial displacement at the loaded points                (0.0924)
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import geometry
from .constraints import couple_pole, rigid_diaphragm, symmetry
from .material import Material
from .model import ShellModel

REFERENCE = {
    "scordelis_lo": {"paper": 0.3006, "belytschko": 0.3024},
    "pinched_cylinder": {"paper": 1.8264e-5, "belytschko": 1.8248e-5},
    "hemisphere": {"paper": 0.0924, "belytschko": 0.0924},
}

TITLES = {
    "scordelis_lo": "Scordelis-Lo roof",
    "pinched_cylinder": "Pinched cylinder with diaphragms",
    "hemisphere": "Hemispherical shell",
}


@dataclass
class Benchmark:
    name: str
    model: ShellModel
    quantity: Callable[[np.ndarray], float]
    u: np.ndarray = field(default=None, repr=False)

    def solve(self, nonlinear=False, **kw):
        if nonlinear:
            self.u, _ = self.model.solve_nonlinear(**kw)
        else:
            self.u = self.model.solve_linear()
        return self.quantity(self.u)

    @property
    def n_cp_side(self):
        return self.model.surface.n

    @property
    def n_dof(self):
        return self.model.T.shape[1]


def scordelis_lo(p=3, n_el=8, **model_kw):
    """Quarter roof: xi = arc from 50 deg (free edge) to 90 deg (crown), eta = z in [0, L/2]."""
    L, R, t, E, nu, load = 50.0, 25.0, 0.25, 4.32e8, 0.0, 90.0
    surf = geometry.cylinder(R, L / 2, np.radians(50.0), np.radians(90.0)).k_refine(p, n_el)
    model = ShellModel(surf, Material(E, nu, t), **model_kw)
    c = model.constraints
    rigid_diaphragm(c, surf, "eta0", comps=(0, 1))  # z = 0: u_x = u_y = 0
    symmetry(c, surf, "eta1", 2)  # z = L/2
    symmetry(c, surf, "xi1", 0)  # crown, plane x = 0
    model.add_body_load([0.0, -load, 0.0])
    return Benchmark("scordelis_lo", model, lambda u: -model.corner_displacement(u, (0, 1))[1])


def pinched_cylinder(p=3, n_el=8, **model_kw):
    """One eighth: xi = arc 0..90 deg, eta = z in [0, L/2]; load F/4 at (R, 0, L/2)."""
    L, R, t, E, nu, F = 600.0, 300.0, 3.0, 3.0e6, 0.3, 1.0
    surf = geometry.cylinder(R, L / 2, 0.0, np.pi / 2).k_refine(p, n_el)
    model = ShellModel(surf, Material(E, nu, t), **model_kw)
    c = model.constraints
    rigid_diaphragm(c, surf, "eta0", comps=(0, 1))
    symmetry(c, surf, "eta1", 2)  # z = L/2
    symmetry(c, surf, "xi0", 1)  # plane y = 0
    symmetry(c, surf, "xi1", 0)  # plane x = 0
    model.add_point_load((0.0, 1.0), [-F / 4, 0.0, 0.0])
    return Benchmark("pinched_cylinder", model, lambda u: -model.corner_displacement(u, (0, 1))[0])


def hemisphere(p=3, n_el=8, pole_coupling=False, **model_kw):
    """One quarter: xi = azimuth 0..90 deg, eta = equator (free) .. pole; loads F/2 at (R,0,0), (0,R,0)."""
    R, t, E, nu, F = 10.0, 0.04, 6.825e7, 0.3, 2.0
    surf = geometry.hemisphere_quarter(R).k_refine(p, n_el)
    model = ShellModel(surf, Material(E, nu, t), **model_kw)
    c = model.constraints
    symmetry(c, surf, "xi0", 1)  # plane y = 0
    symmetry(c, surf, "xi1", 0)  # plane x = 0
    if pole_coupling:
        couple_pole(c, surf, "eta1")
    model.add_point_load((0.0, 0.0), [F / 2, 0.0, 0.0])
    model.add_point_load((1.0, 0.0), [0.0, -F / 2, 0.0])
    return Benchmark("hemisphere", model, lambda u: model.corner_displacement(u, (0, 0))[0])


BUILDERS = {"scordelis_lo": scordelis_lo, "pinched_cylinder": pinched_cylinder, "hemisphere": hemisphere}
