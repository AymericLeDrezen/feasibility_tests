import numpy as np


# --- build the example scenario -------------------------------------------------

# Each object is expressed in the affine Bloch embedding of the GPT formalism.
# The GPT probability rule p = e · (T ω) = ½(1+m.r) then reproduces Tr[E T(ρ)].

def state(rx: float, ry: float, rz: float) -> np.ndarray:
    """
    Construct a GPT state vector ω = (1, r_x, r_y, r_z).

    Parameters
    ----------
    rx, ry, rz : float
        Components of the Bloch vector representing the qubit state.

    Returns
    -------
    np.ndarray, shape (4,)
        Affine embedding of the state suitable for GPT calculations.
    """
    return np.array([1.0, rx, ry, rz], dtype=float)


def effect(mx: float, my: float, mz: float) -> np.ndarray:
    """
    Construct a GPT effect vector e = (½, ½ m_x, ½ m_y, ½ m_z).

    Parameters
    ----------
    mx, my, mz : float
        Components of the Bloch vector representing the measurement direction.

    Returns
    -------
    np.ndarray, shape (4,)
        Affine embedding of the effect.
    """
    return 0.5 * np.array([1.0, mx, my, mz], dtype=float)


def T_affine(R: np.ndarray) -> np.ndarray:
    """
    Embed a 3x3 Bloch rotation R as a 4x4 affine transformation
    acting on the GPT state/effect vectors.

    The result T satisfies:
        T.dot(1, r_x, r_y, r_z)ᵀ = (1, (R · r)_x, (R · r)_y, (R · r)_z)ᵀ.

    Parameters
    ----------
    R : np.ndarray, shape (3,3)
        Rotation matrix on the Bloch sphere.

    Returns
    -------
    np.ndarray, shape (4,4)
        Affine representation used in PTM computations.
    """

    T = np.eye(4)
    T[1:, 1:] = R
    return T

# 6 states/effects
dirs = [(+1,0,0), (-1,0,0), (0,+1,0), (0,-1,0), (0,0,+1), (0,0,-1)]
states  = np.vstack([state(*r)  for r in dirs])   # (6,4)
effects = np.vstack([effect(*m) for m in dirs])   # (6,4)

identity = np.eye(4)[None]


# 4 transforms: I, Z, S, S^{-1}
R_I  = np.eye(3)
R_Z  = np.diag([-1,-1,1])
R_S  = np.array([[0,1,0],[-1,0,0],[0,0,1]], float)
R_Si = np.array([[0,-1,0],[1,0,0],[0,0,1]], float)
transformations = np.stack([T_affine(R) for R in [R_I, R_Z, R_S, R_Si]])  # (4,4,4)