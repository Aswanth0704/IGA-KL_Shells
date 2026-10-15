"""St. Venant-Kirchhoff plane-stress material in curvilinear (contravariant) form."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Material:
    E: float
    nu: float
    t: float


def material_matrix(Gcon, E, nu):
    """Voigt matrix D (..., 3, 3) such that n^{ab} = t * D @ [e11, e22, 2 e12].

    C^{abcd} = lam G^ab G^cd + mu (G^ac G^bd + G^ad G^bc), lam = E nu / (1 - nu^2), mu = E / (2 (1 + nu)).
    Gcon: (..., 2, 2) contravariant metric of the reference surface.
    """
    lam = E * nu / (1.0 - nu**2)
    mu = E / (2.0 * (1.0 + nu))
    g11 = Gcon[..., 0, 0]
    g22 = Gcon[..., 1, 1]
    g12 = Gcon[..., 0, 1]
    D = np.zeros(Gcon.shape[:-2] + (3, 3))
    D[..., 0, 0] = (lam + 2 * mu) * g11**2
    D[..., 1, 1] = (lam + 2 * mu) * g22**2
    D[..., 0, 1] = D[..., 1, 0] = lam * g11 * g22 + 2 * mu * g12**2
    D[..., 0, 2] = D[..., 2, 0] = (lam + 2 * mu) * g11 * g12
    D[..., 1, 2] = D[..., 2, 1] = (lam + 2 * mu) * g22 * g12
    D[..., 2, 2] = lam * g12**2 + mu * (g11 * g22 + g12**2)
    return D
