import numpy as np
import pywt

def transform_to_wavelet(X, wavelet='db4', level=1):
    """
    Transforms a mixture matrix X (n_mixtures, T) into Wavelet coefficients.
    Returns a flattened coefficient matrix and the bookkeeping 'slices' for reconstruction.
    """
    n_mixtures, T = X.shape
    coef_list = []
    
    # We need to store the shapes for the inverse transform later
    coeff_slices = None 
    
    for i in range(n_mixtures):
        # Perform DWT
        coeffs = pywt.wavedec(X[i, :], wavelet, level=level)
        # Flatten all levels (Approx and Details) into one long vector
        flat_coeffs, coeff_slices = pywt.coeffs_to_array(coeffs)
        coef_list.append(flat_coeffs)
        
    return np.array(coef_list).astype(np.float64), coeff_slices

def reconstruct_from_wavelet(Y_coeffs, coeff_slices, wavelet='db4'):
    """
    Transforms separated coefficients Y (n_sources, T_coeffs) back to time domain.
    """
    n_sources = Y_coeffs.shape[0]
    reconstructed_signals = []
    
    for i in range(n_sources):
        # Reshape the flat array back into the Wavelet coefficient structure
        coeffs_structured = pywt.array_to_coeffs(Y_coeffs[i, :], coeff_slices, output_format='wavedec')
        # Perform IDWT
        sig = pywt.waverec(coeffs_structured, wavelet)
        reconstructed_signals.append(sig)
        
    return np.array(reconstructed_signals).astype(np.float64)