import numpy as np
from numba import njit
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
from bss.PredictiveDecorrBSS import PredictiveDecorrBSS


class PredictiveDecorrGeneralPolytope(PredictiveDecorrBSS):
    """
    Predictive Entropy Maximization for general (practical) polytope source domains.

    Extends the base predictive-coding BSS network to source domains of the form

        { s : s_i in [-1, 1] for i in signed_dims,
              s_i in [ 0, 1] for i in nn_dims,
              || s_{J_k} ||_1 <= 1 for each relative-sparse group J_k }.

    Only the fast neural-dynamics projection changes relative to the base class: after the
    same predictive + lateral-decorrelation gradient step, the activity is projected onto the
    polytope by (i) a per-group soft-threshold that enforces each L1 (relative-sparse) ball via
    an adaptive shared threshold, and (ii) box clipping of the signed / nonnegative coordinates.
    This mirrors CorInfoMax's ``run_neural_dynamics_general_polytope`` but with the predictive
    entropy-maximization gradient (running covariance ``C_y`` and mean ``mu_y``).
    """

    def __init__(
        self,
        n_sources,
        signed_dims,
        nn_dims,
        sparse_dims_list,
        epsilon=1e-3,
        lambda_lateral=0.99,
        gamma_predictive=100.0,
        lr_W=0.05,
        neural_lr_start=0.1,
        neural_lr_stop=1e-10,
        stlambda_lr=0.5,
        neural_dynamics_iterations=500,
        neural_OUTPUT_COMP_TOL=1e-6,
        lr_W_rule="constant",
        lr_W_decay_divider=5000,
        neural_lr_rule="divide_by_loop_index",
        neural_lr_decay_divider=200,
        W=None,
        C_y=None,
        mu_y=None,
        Sgt=None,
        debug_iteration_point=25000,
        plot_debug_during_training=False,
    ):
        super().__init__(
            n_sources=n_sources,
            presumed_domain="antisparse",  # placeholder; fit() uses the polytope dynamics below
            epsilon=epsilon,
            lambda_lateral=lambda_lateral,
            gamma_predictive=gamma_predictive,
            lr_W=lr_W,
            neural_lr_start=neural_lr_start,
            neural_lr_stop=neural_lr_stop,
            stlambda_lr=stlambda_lr,
            neural_dynamics_iterations=neural_dynamics_iterations,
            neural_OUTPUT_COMP_TOL=neural_OUTPUT_COMP_TOL,
            lr_W_rule=lr_W_rule,
            lr_W_decay_divider=lr_W_decay_divider,
            neural_lr_rule=neural_lr_rule,
            neural_lr_decay_divider=neural_lr_decay_divider,
            W=W,
            C_y=C_y,
            mu_y=mu_y,
            Sgt=Sgt,
            debug_iteration_point=debug_iteration_point,
            plot_debug_during_training=plot_debug_during_training,
        )
        # Store the polytope description and build fast Boolean masks for the njit dynamics.
        self.signed_dims = np.asarray(signed_dims)
        self.nn_dims = np.asarray(nn_dims)
        self.sparse_dims_list = [np.asarray(g) for g in sparse_dims_list]

        d = n_sources
        self.signed_mask = np.zeros(d, dtype=np.int64)
        if self.signed_dims.size > 0:
            self.signed_mask[self.signed_dims] = 1
        self.nn_mask = np.zeros(d, dtype=np.int64)
        if self.nn_dims.size > 0:
            self.nn_mask[self.nn_dims] = 1
        self.sparse_mask = np.zeros((len(self.sparse_dims_list), d), dtype=np.int64)
        for g, grp in enumerate(self.sparse_dims_list):
            self.sparse_mask[g, grp] = 1

    @staticmethod
    @njit
    def run_neural_dynamics_general_polytope(
        x, y,
        W, C_y, mu_y,
        gamma_predictive, epsilon,
        signed_mask, nn_mask, sparse_mask,
        stlambd_lr,
        neural_dynamics_iterations,
        neural_lr_start, neural_lr_stop,
        lr_rule, lr_decay_divider,
        neural_OUTPUT_COMP_TOL,
    ):
        d = y.shape[0]
        G = sparse_mask.shape[0]
        STLAMBD = np.zeros(G)

        yke = np.dot(W, x)
        D_y = np.diag(C_y).copy()
        O_y = C_y - np.diag(np.diag(C_y))
        D_reg = D_y + epsilon

        for j in range(neural_dynamics_iterations):
            if lr_rule == "constant":
                lr_y = neural_lr_start
            elif lr_rule == "divide_by_loop_index":
                lr_y = max(neural_lr_start / (j + 1), neural_lr_stop)
            elif lr_rule == "divide_by_slow_loop_index":
                lr_y = max(neural_lr_start / (j * lr_decay_divider + 1), neural_lr_stop)
            else:
                lr_y = neural_lr_start

            y_old = y.copy()

            # Predictive-coding + lateral-decorrelation gradient (predictive entropy max).
            error = y - yke
            y_bar = y - mu_y
            lateral = (np.dot(O_y, y_bar / D_reg) - y_bar) / D_reg
            grady = gamma_predictive * error + lateral
            y = y - lr_y * grady

            # Project onto each relative-sparse L1 ball via an adaptive shared soft threshold.
            for g in range(G):
                l1 = 0.0
                for k in range(d):
                    if sparse_mask[g, k] == 1:
                        if signed_mask[k] == 1:
                            a = abs(y[k]) - STLAMBD[g]
                            if a > 0.0:
                                y[k] = a if y[k] > 0.0 else -a
                            else:
                                y[k] = 0.0
                        elif nn_mask[k] == 1:
                            a = y[k] - STLAMBD[g]
                            y[k] = a if a > 0.0 else 0.0
                        l1 += abs(y[k])
                STLAMBD[g] = max(STLAMBD[g] + stlambd_lr * (l1 - 1.0), 0.0)

            # Box constraints on the signed / nonnegative coordinates.
            for k in range(d):
                if signed_mask[k] == 1:
                    if y[k] > 1.0:
                        y[k] = 1.0
                    elif y[k] < -1.0:
                        y[k] = -1.0
                elif nn_mask[k] == 1:
                    if y[k] > 1.0:
                        y[k] = 1.0
                    elif y[k] < 0.0:
                        y[k] = 0.0

            denom = np.linalg.norm(y)
            if denom < 1e-12:
                denom = 1e-12
            if np.linalg.norm(y - y_old) < neural_OUTPUT_COMP_TOL * denom:
                break
        return y

    def fit(self, X, n_epochs=1, shuffle_samples=False):
        n_mixtures, n_samples = X.shape
        if self.plot_debug_during_training:
            plt.figure(figsize=(45, 30), dpi=80)
        if self.W is None:
            self.W = np.eye(self.n_sources, n_mixtures) + np.random.randn(self.n_sources, n_mixtures) * 0.01

        Sgt_zm = None
        if self.ground_truth_available:
            Sgt_zm = self.Sgt - self.Sgt.mean(axis=1).reshape(-1, 1)

        for epoch in range(n_epochs):
            idx = np.random.permutation(n_samples) if shuffle_samples else np.arange(n_samples)
            for i_sample in tqdm(range(n_samples)):
                if self.ground_truth_available and i_sample % self.debug_iteration_point == 0:
                    Y_ = self.W @ X
                    Y_ = Y_ - Y_.mean(axis=1).reshape(-1, 1)
                    Y_ = self.signed_and_permutation_corrected_sources(Sgt_zm, Y_)
                    coef_ = ((Y_ * Sgt_zm).sum(axis=1) / (Y_ * Y_).sum(axis=1)).reshape(-1, 1)
                    Y_ = coef_ * Y_
                    self.component_SNR_history.append(self.ComputeSNR(Sgt_zm, Y_))
                    self.SINR_history.append(self.ComputeSINR(Sgt_zm, Y_))
                    self.SV_list.append(np.linalg.svd(self.W, compute_uv=False))
                    if self.plot_debug_during_training:
                        self.plot_for_debug(
                            self.SINR_history, self.component_SNR_history,
                            self.debug_iteration_point, Y_[:, idx[i_sample - 25 : i_sample]].T,
                        )

                x_current = np.ascontiguousarray(X[:, idx[i_sample]])
                y = np.zeros(self.n_sources)
                y = self.run_neural_dynamics_general_polytope(
                    x_current, y,
                    self.W, self.C_y, self.mu_y,
                    self.gamma_predictive, self.epsilon,
                    self.signed_mask, self.nn_mask, self.sparse_mask,
                    self.stlambda_lr,
                    self.neural_dynamics_iterations,
                    self.neural_lr_start, self.neural_lr_stop,
                    self.neural_lr_rule, self.neural_lr_decay_divider,
                    self.neural_OUTPUT_COMP_TOL,
                )

                error = y - self.W @ x_current
                if self.lr_W_rule == "constant":
                    lr_W = self.lr_W
                elif self.lr_W_rule == "divide_by_log_index":
                    lr_W = max(self.lr_W / (1 + np.log((epoch * n_samples + i_sample) / self.lr_W_decay_divider + 2)), 1e-8)
                elif self.lr_W_rule == "divide_by_index":
                    lr_W = max(self.lr_W / ((epoch * n_samples + i_sample) / self.lr_W_decay_divider + 1), 1e-8)
                else:
                    lr_W = self.lr_W
                self.W += lr_W * np.outer(error, x_current)

                self.mu_y = self.lambda_lateral * self.mu_y + (1 - self.lambda_lateral) * y
                y_bar = y - self.mu_y
                self.C_y = self.lambda_lateral * self.C_y + (1 - self.lambda_lateral) * np.outer(y_bar, y_bar)

    def predict(self, X):
        return self.W @ X

    def fit_predict(self, X, n_epochs=1, shuffle_samples=False):
        self.fit(X, n_epochs=n_epochs, shuffle_samples=shuffle_samples)
        return self.predict(X)
