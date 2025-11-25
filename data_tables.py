from __future__ import annotations
import numpy as np
from utils import unique_rows, is_identity

def lump_transformations_into_states(
    states: np.ndarray,
    transformations: np.ndarray,
) -> np.ndarray:
    """
    Build the lumped preparation set: { T_t ω_s }_(s,t).
    Inputs:
      states: (S,d)
      transformations: (T,d,d)
    Returns:
      lumped_states: (S'_unique, d)
    """

    # ω'_(s,t) = T_t ω_s
    transformed = np.einsum('tdp,sp->tsd', transformations, states)       # (T,S,d)
    lumped_states = transformed.transpose(1, 0, 2).reshape(-1, states.shape[1])

    return unique_rows(lumped_states)


def lump_transformations_into_effects(
    transformations: np.ndarray,
    effects: np.ndarray,
) -> np.ndarray:
    """
    Build the lumped effect set: { e_k ∘ T_t }_(k,t).
    Inputs:
      transformations: (T,d,d)
      effects: (K,d) as row-vectors in V*
    Returns:
      lumped_effects: (K'_unique, d)
    """

    # e'_(k,t) = e_k @ T_t   (left action of row vector)
    k_t_p = np.einsum('kd,tdp->ktp', effects, transformations)           # (K,T,d)
    lumped_effects = k_t_p.reshape(-1, effects.shape[1])

    return unique_rows(lumped_effects)



def compute_data_table(
    states: np.ndarray,
    transformations: np.ndarray,
    effects: np.ndarray,
) -> np.ndarray:
    """
    PTM probabilities: p(k|s,t) = e_k · [T_t(ω_s)]  -> shape (K,S,T).
    If there's a single identity transformation,
    returns the PM table p(k|s) = e_k · ω_s with shape (K,S).
    """

    if is_identity(transformations):
        return np.einsum('kd,sd->ks', effects, states)               # PM

    return np.einsum('kd,tdp,sp->kst', effects, transformations, states)  # PTM
