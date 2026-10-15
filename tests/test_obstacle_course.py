"""Regression against the paper's obstacle-course results (Kiendl et al. 2009, Figs. 11-13)."""

import numpy as np
import pytest

from iga_kl.benchmarks import REFERENCE, hemisphere, pinched_cylinder, scordelis_lo

# values read off the paper's convergence plots for p = 2 (n_el -> value); coarse-mesh readings
# are approximate, hence the loose tolerance
FIG_P2 = {
    "scordelis_lo": {2: 0.077, 4: 0.247, 8: 0.296, 16: 0.300},
    "pinched_cylinder": {4: 1.5e-6, 8: 8.8e-6, 16: 1.61e-5, 32: 1.78e-5},
    "hemisphere": {8: 0.019, 16: 0.074, 32: 0.0905},
}
BUILD = {"scordelis_lo": scordelis_lo, "pinched_cylinder": pinched_cylinder, "hemisphere": hemisphere}


@pytest.mark.parametrize("name", list(FIG_P2))
def test_p2_convergence_curve_matches_paper(name):
    for n_el, expected in FIG_P2[name].items():
        val = BUILD[name](2, n_el).solve()
        assert np.isclose(val, expected, rtol=0.15), (n_el, val, expected)


@pytest.mark.parametrize("name", list(FIG_P2))
def test_converged_values(name):
    val = BUILD[name](4, 32).solve()
    assert np.isclose(val, REFERENCE[name]["paper"], rtol=5e-3), (val, REFERENCE[name])


def test_hemisphere_pole_coupling_and_symmetry_of_loads():
    for pole in (False, True):
        bm = hemisphere(4, 16, pole_coupling=pole)
        ux = bm.solve()
        uy = -bm.model.corner_displacement(bm.u, (1, 0))[1]
        assert np.isclose(ux, uy, rtol=1e-8)
        assert np.isclose(ux, 0.0924, rtol=1e-2)


def test_quartic_converges_by_second_refinement():
    # paper: "for quartic and quintic shape functions already the second refinement step yields
    # the converged solution"
    assert np.isclose(scordelis_lo(4, 4).solve(), 0.3006, rtol=2e-3)
    assert np.isclose(pinched_cylinder(5, 8).solve(), 1.8264e-5, rtol=3e-2)
    assert np.isclose(hemisphere(4, 8).solve(), 0.0924, rtol=6e-2)
