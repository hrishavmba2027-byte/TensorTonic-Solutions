import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    x_arr = np.array(x, dtype=float)
    
    # Numerically stable sigmoid handling both positive and negative values
    res = np.where(
        x_arr >= 0, 
        1 / (1 + np.exp(-x_arr)), 
        np.exp(x_arr) / (1 + np.exp(x_arr))
    )
    
    # Return a scalar float if input was scalar
    if isinstance(x, (int, float)):
        return float(res)
    return res