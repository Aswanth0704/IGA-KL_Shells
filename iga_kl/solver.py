"""Linear solve and Newton-Raphson in the reduced (constrained) space."""

import numpy as np
import scipy.sparse.linalg as spla


def reduce_system(K, f, T, u_p=None):
    rhs = f if u_p is None else f - K @ u_p
    return (T.T @ K @ T).tocsc(), T.T @ rhs


def solve_linear(K, f, T, u_p=None):
    Kh, fh = reduce_system(K, f, T, u_p)
    uh = spla.spsolve(Kh, fh)
    u = T @ uh
    return u if u_p is None else u + u_p


def newton(model, n_steps=1, tol=1e-8, max_iter=25, verbose=False, u0=None):
    """Load-stepped Newton-Raphson. `model` provides T, u_p, f_ext and tangent(u) -> (K, f_int).

    Convergence per load step: |r| <= tol |lambda f_ext| (residual), or the energy increment
    |du . r| <= tol^2 |du_0 . r_0| (catches round-off stagnation of the residual).
    Returns the displacement and a list of per-step residual-norm histories.
    """
    T, u_p, f_ext = model.T, model.u_p, model.f_ext
    u = u_p.copy() if u0 is None else u0.copy()
    history = []
    for step in range(1, n_steps + 1):
        lam = step / n_steps
        fref = np.linalg.norm(T.T @ (lam * f_ext))
        res_hist = []
        e0 = None
        for it in range(max_iter + 1):
            K, f_int = model.tangent(u)
            r = T.T @ (lam * f_ext - f_int)
            rn = np.linalg.norm(r)
            res_hist.append(rn)
            if verbose:
                print(f"  step {step:3d} iter {it:2d}  |r| = {rn:.3e}")
            if rn <= tol * max(fref, 1e-300):
                break
            if it == max_iter:
                raise RuntimeError(f"Newton did not converge in step {step}: |r| = {rn:.3e}")
            du = spla.spsolve((T.T @ K @ T).tocsc(), r)
            u = u + T @ du
            e = abs(du @ r)
            e0 = e if e0 is None else e0
            if e <= tol**2 * e0:
                break
        history.append(res_hist)
    return u, history
