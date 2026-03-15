import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    Accepts scalars, lists, and numpy arrays. Returns a NumPy array of floats.
    Numerically stable: uses exp(-abs(x)) to avoid overflow.
    """
    x = np.asarray(x, dtype=float)
    # safe computation: s = exp(-|x|) never overflows; choose branch analytically
    s = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + s), s / (1.0 + s))
    pass