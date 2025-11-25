from __future__ import annotations
import numpy as np
import sympy as sp


def exact_identities_from_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Given row vectors v_i in R^d (vectors.shape == (N, d)),
    find linear identities of the form sum_i alpha_i v_i = 0
    using SymPy's exact nullspace.
    """
    M = sp.Matrix(vectors).T          # columns = vectors
    nullspace = M.nullspace()        # list of column vectors
    return np.vstack(
        [np.array(v, dtype=float).reshape(1, -1) for v in nullspace]
    )


def gen_operational_identities(
    states: np.ndarray,
    effects: np.ndarray,
    transformations: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Linear operational identities among states, effects, and transformations.
    atol kept only for API compatibility; not used.
    """
    T_flat = transformations.reshape(transformations.shape[0], -1)

    return {
        "states":          exact_identities_from_vectors(states),
        "effects":         exact_identities_from_vectors(effects),
        "transformations": exact_identities_from_vectors(T_flat),
    }
