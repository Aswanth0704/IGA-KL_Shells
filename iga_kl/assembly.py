"""Sparse assembly of element arrays."""

import numpy as np
import scipy.sparse as sp


def assemble_matrix(dof_map, Ke, n_dof):
    E, ndof = dof_map.shape
    rows = np.repeat(dof_map, ndof, axis=1).ravel()
    cols = np.tile(dof_map, (1, ndof)).ravel()
    K = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(n_dof, n_dof))
    return K.tocsr()


def assemble_vector(dof_map, fe, n_dof):
    return np.bincount(dof_map.ravel(), weights=fe.ravel(), minlength=n_dof)
