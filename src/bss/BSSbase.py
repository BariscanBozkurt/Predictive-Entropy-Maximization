import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from IPython.display import Latex, Math, clear_output, display

class BSSBaseClass:
    
    def whiten_input(self, X, n_components=None, return_prewhitening_matrix=False):
        """
        X.shape[0] = Number of sources
        X.shape[1] = Number of samples for each signal
        """
        x_dim = X.shape[0]
        if n_components is None:
            n_components = x_dim
        s_dim = n_components

        N = X.shape[1]
        # Mean of the mixtures
        mX = np.mean(X, axis=1).reshape((x_dim, 1))
        # Covariance of Mixtures
        Rxx = np.dot(X, X.T) / N - np.dot(mX, mX.T)
        # Eigenvalue Decomposition
        d, V = np.linalg.eig(Rxx)
        D = np.diag(d)
        # Sorting indexis for eigenvalues from large to small
        ie = np.argsort(-d)
        # Inverse square root of eigenvalues
        ddinv = 1 / np.sqrt(d[ie[:s_dim]])
        # Pre-whitening matrix
        Wpre = np.dot(np.diag(ddinv), V[:, ie[:s_dim]].T)  # *np.sqrt(12)
        # Whitened mixtures
        H = np.dot(Wpre, X)
        if return_prewhitening_matrix:
            return H, Wpre
        else:
            return H

    @staticmethod
    @njit
    def ProjectOntoLInfty(X):
        return X * (X >= -1.0) * (X <= 1.0) + (X > 1.0) * 1.0 - 1.0 * (X < -1.0)

    @staticmethod
    @njit
    def ProjectOntoNNLInfty(X):
        return X * (X >= 0.0) * (X <= 1.0) + (X > 1.0) * 1.0  # -0.0*(X<0.0)

    def ProjectRowstoL1NormBall(self, H):
        Hshape = H.shape
        # lr=np.ones((Hshape[0],1))@np.reshape((1/np.linspace(1,Hshape[1],Hshape[1])),(1,Hshape[1]))
        lr = np.tile(
            np.reshape((1 / np.linspace(1, Hshape[1], Hshape[1])), (1, Hshape[1])),
            (Hshape[0], 1),
        )
        # Hnorm1=np.reshape(np.sum(np.abs(self.H),axis=1),(Hshape[0],1))

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

    def ProjectColstoSimplex(self, v, z=1):
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

    def ProjectRowstoUnitSimplex(self, H):
        return self.ProjectColstoSimplex(H.T).T

    def ProjectRowstoNNL1NormBall(self, H):
        H = self.ProjectRowstoL1NormBall(H)
        return H * (H > 0)
        
    def ProjectRowstoBoundaryRectangle(self, H, BoundMin, BoundMax):
        Hshape = H.shape
        a0 = 1 - 2 * (np.sum(H, axis=0) < 0) * (BoundMin == 0)
        AA0 = np.diag(np.reshape(a0, (-1,)))
        H = np.dot(H, AA0)
        BoundMaxlist = np.dot(np.ones((Hshape[0], 1)), BoundMax)
        BoundMinlist = np.dot(np.ones((Hshape[0], 1)), BoundMin)
        CheckMin = 1.0 * (H > BoundMinlist)
        a = 1 - 2.0 * (np.sum(CheckMin, axis=0) == 0) * (BoundMin == 0)
        AA = np.diag(np.reshape(a, (-1,)))
        H = np.dot(H, AA)
        CheckMax = 1.0 * (H < BoundMaxlist)
        CheckMin = 1.0 * (H > BoundMinlist)
        H = (
            H * CheckMax * CheckMin
            + (1 - CheckMin) * BoundMinlist
            + (1 - CheckMax) * BoundMaxlist
        )
        return H

    def ProjectColumnsOntoPracticalPolytope(
        self, x, signed_dims, nn_dims, sparse_dims_list, max_number_of_iterations=1
    ):
        dim = len(signed_dims) + len(nn_dims)
        BoundMax = np.ones((dim, 1)).T
        BoundMin = -np.ones((dim, 1)).T
        BoundMin[:, nn_dims] = 0
        x_projected = x.copy()
        x_projected[signed_dims, :] = np.clip(x_projected[signed_dims, :], -1, 1)
        x_projected[nn_dims, :] = np.clip(x_projected[nn_dims, :], 0, 1)
        for kk in range(max_number_of_iterations):
            for j in range(len(sparse_dims_list)):
                x_projected = self.ProjectRowstoBoundaryRectangle(
                    x_projected.T, BoundMin, BoundMax
                ).T
                x_projected[sparse_dims_list[j], :] = self.ProjectRowstoL1NormBall(
                    x_projected[sparse_dims_list[j], :].T
                ).T
        return x_projected

    def outer_prod_broadcasting(self, A, B):
        """Broadcasting trick."""
        return A[..., None] * B[:, None]
    
    #### Debugging functions if the ground truth is available, we can compute the SNR and SINR for each component and store them in a list for later analysis. This is useful for tracking the performance of the algorithm over iterations.
    def find_permutation_between_source_and_estimation(self, S, Y):
        """
        Identify the best 1-to-1 mapping between estimated outputs and ground truth sources.
        
        BSS algorithms suffer from permutation ambiguity. This function uses the Pearson 
        correlation absolute values to find which output corresponds to which source.

        Args:
            S (ndarray): Original source matrix (n_sources, n_samples).
            Y (ndarray): Estimated source matrix (n_sources, n_samples).

        Returns:
            ndarray: Indices of Y that best match the order of S.
        """
        # TODO: Maybe we can also consider using the covariance instead of correlation for this matching, especially if the signals are not zero-mean.
        # For now, we will use correlation as it is more commonly used for this purpose.
        # Calculate pairwise correlation using broadcasting trick
        # We normalize by the norms to get the correlation coefficient
        correlation_numerator = np.abs(self.outer_prod_broadcasting(Y.T, S.T).sum(axis=0))
        normalization = (np.linalg.norm(S, axis=1) * np.linalg.norm(Y, axis=1))
        
        # Find the index of the estimate with the highest correlation for each source
        perm = np.argmax(correlation_numerator / normalization, axis=0)
        return perm

    def signed_and_permutation_corrected_sources(self, S, Y):
        """
        Correct for both permutation and sign ambiguity in the estimated sources.
        
        In ICA/BSS, the output Y_i could be -S_j. This function reorders Y and flips
        the signs so that the output matches the ground truth S as closely as possible.

        Args:
            S (ndarray): Ground truth source matrix (n_sources, n_samples).
            Y (ndarray): Uncorrected estimation matrix (n_sources, n_samples).

        Returns:
            ndarray: Reordered and sign-flipped estimations.
        """
        # 1. Find the permutation (mapping)
        perm = self.find_permutation_between_source_and_estimation(S, Y)
        
        # 2. Determine the sign by looking at the dot product of the matched pairs
        # If the dot product is negative, we need to flip the sign (multiply by -1)
        matched_Y = Y[perm, :]
        signs = np.sign((matched_Y * S).sum(axis=1))
        
        return signs[:, np.newaxis] * matched_Y
    
    @staticmethod
    @njit(parallel=True)
    def ComputeSNR(S_original, S_noisy):
        """
        Calculate the Signal-to-Noise Ratio (SNR) for each source channel.
        
        Args:
            S_original (ndarray): Ground truth source matrix of shape (n_sources, n_samples).
            S_noisy (ndarray): Estimated or noisy source matrix of shape (n_sources, n_samples).
            
        Returns:
            ndarray: SNR values in decibels (dB) for each channel.
        """
        # Residual noise calculation
        N_hat = S_original - S_noisy
        
        # Power calculation per channel (sum of squares)
        N_P = (N_hat**2).sum(axis=1)
        S_P = (S_original**2).sum(axis=1)
        
        # Standard SNR formula in log scale
        snr = 10 * np.log10(S_P / N_P)
        return snr
    
    @staticmethod
    @njit
    def ComputeSINR(S, Y):
        """
        Calculate the Signal-to-Interference-plus-Noise Ratio (SINR) in dB.
        
        This function performs internal Z-score normalization to handle scale ambiguity,
        corrects for permutation/sign via correlation, and computes the ratio of 
        signal power to reconstruction error.

        Args:
            S (ndarray): Ground truth source matrix.
            Y (ndarray): Estimated source matrix.

        Returns:
            float: The aggregate SINR value in dB.
        """
        N_Sources = S.shape[0]
        
        # Numba-friendly normalization: avoid keepdims which is not fully supported
        S_normalized = np.zeros_like(S)
        Y_normalized = np.zeros_like(Y)
        
        for i in range(N_Sources):
            # Subtract mean and divide by std (Z-score) for both S and Y
            s_i = S[i, :]
            S_normalized[i, :] = (s_i - np.mean(s_i)) / (np.std(s_i) + 1e-9)
            
            y_i = Y[i, :]
            Y_normalized[i, :] = (y_i - np.mean(y_i)) / (np.std(y_i) + 1e-9)
            
        # Compute the correlation matrix between all pairs
        corr = np.dot(Y_normalized, S_normalized.T) 
        
        # Solve permutation ambiguity: find which Y index matches which S index
        perm = np.zeros(N_Sources, dtype=np.int64)
        for i in range(N_Sources):
            perm[i] = np.argmax(np.abs(corr[:, i]))
            
        # Solve sign ambiguity: check if the matched signals are inverted
        signs = np.zeros((N_Sources, 1))
        for i in range(N_Sources):
            signs[i, 0] = np.sign(corr[perm[i], i])
        
        # Create the corrected estimate (aligned with S)
        Y_corrected = signs * Y[perm, :]
        
        # Calculate the error (Interference + Noise)
        E = Y_corrected - S
        
        # Calculate final SINR using Frobenius norms
        MSE = np.linalg.norm(E) ** 2
        SigPow = np.linalg.norm(S) ** 2
        
        sinr_val = 10 * np.log10(SigPow / (MSE + 1e-9))
        return sinr_val
    
    def plot_for_debug(self, SIR_list, SNR_list, debug_iteration_point, YforPlot):
        plt.clf()
        plt.subplot(2, 2, 1)
        plt.plot(np.array(SIR_list), linewidth=5)
        plt.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize=45)
        plt.ylabel("SIR (dB)", fontsize=45)
        plt.title("SIR Behaviour", fontsize=45)
        plt.grid()
        plt.xticks(fontsize=45)
        plt.yticks(fontsize=45)

        plt.subplot(2, 2, 2)
        plt.plot(np.array(SNR_list), linewidth=5)
        plt.grid()
        plt.title("Component SNR Check", fontsize=45)
        plt.ylabel("SNR (dB)", fontsize=45)
        plt.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize=45)
        plt.xticks(fontsize=45)
        plt.yticks(fontsize=45)

        plt.subplot(2, 2, 3)
        plt.plot(np.array(self.SV_list), linewidth=5)
        plt.grid()
        plt.title(
            "Singular Value Check, Overall Matrix Rank: "
            + str(np.sum(self.SV_list[-1] > 0)),
            fontsize=45,
        )
        plt.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize=45)
        plt.xticks(fontsize=45)
        plt.yticks(fontsize=45)

        plt.subplot(2, 2, 4)
        plt.plot(YforPlot, linewidth=5)
        plt.title("Y last 25", fontsize=45)
        plt.grid()
        plt.xticks(fontsize=45)
        plt.yticks(fontsize=45)

        clear_output(wait=True)
        display(plt.gcf())