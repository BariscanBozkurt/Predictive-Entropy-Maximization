import numpy as np

def min_max_normalize(data, axis=None, epsilon=1e-12):
    """
    Min-max normalization to [0, 1].
    
    Args:
        data (ndarray): Input data array.
        axis (int or None): Axis along which to normalize. 
                            None for global, 0 for columns, 1 for rows.
        epsilon (float): Small constant to avoid division by zero.
    """
    d_min = np.amin(data, axis=axis, keepdims=True)
    d_max = np.amax(data, axis=axis, keepdims=True)
    
    # Vectorized normalization
    return (data - d_min) / (d_max - d_min + epsilon)

def batch_outer_product(A, B):
    """
    Computes a batch of outer products efficiently.
    If A is (N,) and B is (M,), returns (N, M).
    If A is (Batch, N) and B is (Batch, M), returns (Batch, N, M).
    """
    if A.ndim == 1 and B.ndim == 1:
        return np.outer(A, B)
    
    # Vectorized batch outer product using Einstein summation
    return np.einsum('bi,bj->bij', A, B)