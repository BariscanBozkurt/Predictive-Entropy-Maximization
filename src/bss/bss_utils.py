import numpy as np
from numba import njit
from scipy import linalg
from scipy.stats import chi2, dirichlet, invgamma, t

def addWGN(signal, SNR):
    """
    Adding white Gaussian Noise to the input signal
    signal              : Input signal, numpy array of shape (number of sources, number of samples)
                          If your signal is a 1D numpy array of shape (number of samples, ), then reshape it
                          by signal.reshape(1,-1) before giving it as input to this function
    SNR                 : Desired input signal to noise ratio
    print_resulting_SNR : If you want to print the numerically calculated SNR, pass it as True

    Returns
    ============================
    signal_noisy        : Output signal which is the sum of input signal and additive noise
    """
    sigpow = np.mean(signal**2, axis=1)
    noisepow = 10 ** (-SNR / 10) * sigpow
    noise = np.sqrt(noisepow)[:, np.newaxis] * np.random.randn(
        signal.shape[0], signal.shape[1]
    )
    signal_noisy = signal + noise
    return signal_noisy

def generate_uncorrelated_uniform_sources(n_sources=5, 
                                          size_sources=500000,
                                          min_val = -1, max_val = 1):
    """
    Generates uncorrelated uniform sources

    required libraries:
    import numpy as np
    """
    S = np.random.uniform(min_val, max_val, size=(n_sources, size_sources))
    return S


def generate_uniform_points_in_simplex(NumberofSources, NumberofSamples, gain=1):
    S = np.random.exponential(scale=1.0, size=(NumberofSources, NumberofSamples))
    S = gain * (S / np.sum(S, axis=0))
    return S


def generate_correlated_copula_sources(
    rho=0.0, df=4, n_sources=5, size_sources=500000, decreasing_correlation=True
):
    """
    rho     : correlation parameter
    df      : degrees for freedom

    required libraries:
    from scipy.stats import invgamma, chi2, t
    from scipy import linalg
    import numpy as np
    """
    if decreasing_correlation:
        first_row = np.array([rho**j for j in range(n_sources)])
        calib_correl_matrix = linalg.toeplitz(first_row, first_row)
    else:
        calib_correl_matrix = (
            np.eye(n_sources) * (1 - rho) + np.ones((n_sources, n_sources)) * rho
        )

    mu = np.zeros(len(calib_correl_matrix))
    s = chi2.rvs(df, size=size_sources)[:, np.newaxis]
    Z = np.random.multivariate_normal(mu, calib_correl_matrix, size_sources)
    X = np.sqrt(df / s) * Z  # chi-square method
    S = t.cdf(X, df).T
    return S

def ProjectRowstoL1NormBall(H):
    Hshape = H.shape
    lr = np.tile(
        np.reshape((1 / np.linspace(1, Hshape[1], Hshape[1])), (1, Hshape[1])),
        (Hshape[0], 1),
    )
    u = -np.sort(-np.abs(H), axis=1)
    sv = np.cumsum(u, axis=1)
    q = np.where(
        u > ((sv - 1) * lr),
        np.tile(
            np.reshape((np.linspace(1, Hshape[1], Hshape[1]) - 1), (1, Hshape[1])),
            (Hshape[0], 1),
        ),
        np.zeros((Hshape[0], Hshape[1])),
    )
    rho = np.max(q, axis=1)
    rho = rho.astype(int)
    lindex = np.linspace(1, Hshape[0], Hshape[0]) - 1
    lindex = lindex.astype(int)
    theta = np.maximum(
        0, np.reshape((sv[tuple([lindex, rho])] - 1) / (rho + 1), (Hshape[0], 1))
    )
    ww = np.abs(H) - theta
    H = np.sign(H) * (ww > 0) * ww
    return H

def ProjectColstoSimplex(v, z=1):
    """v array of shape (n_features, n_samples)."""
    p, n = v.shape
    u = np.sort(v, axis=0)[::-1, ...]
    pi = np.cumsum(u, axis=0) - z
    ind = (np.arange(p) + 1).reshape(-1, 1)
    mask = (u - pi / ind) > 0
    rho = p - 1 - np.argmax(mask[::-1, ...], axis=0)
    theta = pi[tuple([rho, np.arange(n)])] / (rho + 1)
    w = np.maximum(v - theta, 0)
    return w