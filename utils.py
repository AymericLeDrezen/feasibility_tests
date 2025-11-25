import numpy as np


def unique_rows(a: np.ndarray) -> np.ndarray:
    
    return np.unique(a, axis=0)


def is_identity(transformations: np.ndarray) -> bool:
    return (
        transformations.shape[0] == 1
        and np.allclose(transformations[0], np.eye(transformations.shape[1]))
    )
