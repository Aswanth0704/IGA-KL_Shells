"""Exact NURBS geometries for the benchmark problems (all single-span, coarse nets)."""

import numpy as np

from .nurbs import NurbsSurface


def arc_xy(R, phi0, phi1):
    """Quadratic rational circular arc in the x-y plane, angles in radians (|phi1-phi0| < pi).

    Returns homogeneous control points (3, 4).
    """
    d = phi1 - phi0
    assert 0 < abs(d) < np.pi
    pm = 0.5 * (phi0 + phi1)
    w1 = np.cos(0.5 * d)
    P = np.array(
        [
            [R * np.cos(phi0), R * np.sin(phi0), 0.0],
            [R / w1 * np.cos(pm), R / w1 * np.sin(pm), 0.0],
            [R * np.cos(phi1), R * np.sin(phi1), 0.0],
        ]
    )
    w = np.array([1.0, w1, 1.0])
    return np.column_stack([P * w[:, None], w])


def cylinder(R, L, phi0, phi1):
    """Cylinder segment: xi = circular arc (degree 2) from phi0 to phi1, eta = z in [0, L] (degree 1)."""
    arc = arc_xy(R, phi0, phi1)
    Pw = np.zeros((3, 2, 4))
    for j, z in enumerate([0.0, L]):
        Pw[:, j, :] = arc
        Pw[:, j, 2] = z * arc[:, 3]
    U = np.array([0, 0, 0, 1, 1, 1.0])
    V = np.array([0, 0, 1, 1.0])
    return NurbsSurface(U, V, 2, 1, Pw)


def hemisphere_quarter(R):
    """One quarter of a hemisphere as a surface of revolution.

    xi: azimuth 0 -> 90 deg (x-axis to y-axis); eta: meridian from the equator (z=0) to the pole
    (0,0,R). The pole row (eta = 1) is degenerate: all its control points coincide.
    """
    c = np.sqrt(0.5)
    az = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])  # azimuth arc directions (unweighted)
    w_az = np.array([1.0, c, 1.0])
    rz = np.array([[R, 0.0], [R, R], [0.0, R]])  # meridian (r, z) control points
    w_me = np.array([1.0, c, 1.0])
    Pw = np.zeros((3, 3, 4))
    for i in range(3):
        for j in range(3):
            w = w_az[i] * w_me[j]
            P = np.array([rz[j, 0] * az[i, 0], rz[j, 0] * az[i, 1], rz[j, 1]])
            Pw[i, j, :3] = w * P
            Pw[i, j, 3] = w
    U = np.array([0, 0, 0, 1, 1, 1.0])
    return NurbsSurface(U, U.copy(), 2, 2, Pw)


def plate(Lx, Ly, z=0.0):
    """Flat bilinear plate [0,Lx] x [0,Ly] in the x-y plane."""
    Pw = np.zeros((2, 2, 4))
    for i, x in enumerate([0.0, Lx]):
        for j, y in enumerate([0.0, Ly]):
            Pw[i, j] = [x, y, z, 1.0]
    U = np.array([0, 0, 1, 1.0])
    return NurbsSurface(U, U.copy(), 1, 1, Pw)
