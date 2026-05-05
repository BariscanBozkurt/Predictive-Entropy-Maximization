import numpy as np
from numba import njit
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output, display

import sys
sys.path.append("..")

from bss.PredictiveDecorrBSS import PredictiveDecorrBSS


class PredictiveEntropyFaceNMF(PredictiveDecorrBSS):
    """
    Face-image variant of Predictive Entropy Maximization for a nonnegative
    matrix-factorization-style setup.

    Main changes relative to the generic PEM class:
    1. Default latent domain is nonnegative sparse ("nnsparse").
    2. Feedforward weights W are initialized nonnegative.
    3. After each W update, W is projected back to the nonnegative orthant.
    4. The plotting utility is adapted to 64x64 face images.
    """

    def __init__(
        self,
        n_sources,
        presumed_domain="nnsparse",
        epsilon=1e-5,
        lambda_lateral=0.99,
        gamma_predictive=25.0,
        lr_W=0.01,
        neural_lr_start=0.1,
        neural_lr_stop=0.001,
        stlambda_lr=0.1,
        neural_dynamics_iterations=150,
        neural_OUTPUT_COMP_TOL=1e-7,
        lr_W_rule="constant",
        lr_W_decay_divider=5000,
        neural_lr_rule="divide_by_loop_index",
        neural_lr_decay_divider=200,
        W=None,
        C_y=None,
        mu_y=None,
        Sgt=None,
        debug_iteration_point=100,
        plot_debug_during_training=False,
        image_shape=(64, 64),
        project_W_nonnegative=True,
        normalize_W_rows=False,
        seed=None,
    ):
        self.image_shape = image_shape
        self.project_W_nonnegative = project_W_nonnegative
        self.normalize_W_rows = normalize_W_rows
        self.rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

        super().__init__(
            n_sources=n_sources,
            presumed_domain=presumed_domain,
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

        # Face-specific default covariance initialization:
        # keep lateral regularization mild initially.
        if C_y is None:
            self.C_y = 0.05 * np.eye(n_sources)

        # If user provided W, enforce nonnegativity now.
        if self.W is not None and self.project_W_nonnegative:
            self.W = np.maximum(self.W, 0.0)
            if self.normalize_W_rows:
                self.W = self._normalize_rows(self.W)

    @staticmethod
    def _normalize_rows(W):
        row_norms = np.linalg.norm(W, axis=1, keepdims=True)
        row_norms = np.maximum(row_norms, 1e-12)
        return W / row_norms

    @staticmethod
    @njit
    def run_neural_dynamics_nnsparse(
        x, y,
        W, C_y, mu_y,
        gamma_predictive,
        epsilon,
        neural_dynamics_iterations,
        neural_lr_start,
        neural_lr_stop,
        stlambd_lr=0.1,
        lr_rule="divide_by_loop_index",
        lr_decay_divider=200,
        neural_OUTPUT_COMP_TOL=1e-7,
    ):
        """
        Fast inference dynamics for the nonnegative sparse domain.

        This is the face-NMF-style default:
        - nonnegative latent activities
        - adaptive shared threshold (lambda_L-like)
        - epsilon-regularized variance normalization
        """
        STLAMBD = 0.0
        yke = np.dot(W, x)

        D_y = np.diag(C_y)
        O_y = C_y - np.diag(D_y)
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

            error = y - yke
            y_bar = y - mu_y
            lateral = (np.dot(O_y, y_bar / D_reg) - y_bar) / D_reg
            grady = gamma_predictive * error + lateral

            a = y - lr_y * grady

            # Thresholded nonnegative update
            y = np.maximum(a - STLAMBD, 0.0)

            # Mild hard cap for numerical stability
            y = np.minimum(y, 5.0)

            # Shared threshold update to encourage l1-type sparsity
            dval = np.sum(y) - 1.0
            STLAMBD = max(STLAMBD + stlambd_lr * dval, 0.0)

            denom = np.linalg.norm(y)
            if denom < 1e-12:
                denom = 1e-12

            if np.linalg.norm(y - y_old) < neural_OUTPUT_COMP_TOL * denom:
                break

        return y

    @staticmethod
    @njit
    def run_neural_dynamics_nnantisparse(
        x, y,
        W, C_y, mu_y,
        gamma_predictive,
        epsilon,
        neural_dynamics_iterations,
        neural_lr_start,
        neural_lr_stop,
        stlambd_lr=0.0,
        lr_rule="divide_by_loop_index",
        lr_decay_divider=200,
        neural_OUTPUT_COMP_TOL=1e-7,
    ):
        """
        Optional nonnegative bounded variant. Useful if you want bounded latent
        activities instead of sparse nonnegative activities.
        """
        yke = np.dot(W, x)

        D_y = np.diag(C_y)
        O_y = C_y - np.diag(D_y)
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

            error = y - yke
            y_bar = y - mu_y
            lateral = (np.dot(O_y, y_bar / D_reg) - y_bar) / D_reg
            grady = gamma_predictive * error + lateral

            y = y - lr_y * grady
            y = np.clip(y, 0.0, 1.0)

            denom = np.linalg.norm(y)
            if denom < 1e-12:
                denom = 1e-12

            if np.linalg.norm(y - y_old) < neural_OUTPUT_COMP_TOL * denom:
                break

        return y

    def plot_face_filters(
        self,
        W=None,
        title="Learned Face Filters",
        normalize_each=True,
        save_path=None,
    ):
        """
        Plot rows of W as 64x64 face-shaped filters / basis images.
        """
        if W is None:
            W = self.W

        n_filters = W.shape[0]
        img_h, img_w = self.image_shape

        n_cols = int(np.ceil(np.sqrt(n_filters)))
        n_rows = int(np.ceil(n_filters / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))
        fig.suptitle(title, fontsize=20, fontweight="bold", y=0.96)

        if n_rows == 1 and n_cols == 1:
            axes_flat = np.array([axes])
        else:
            axes_flat = np.asarray(axes).flatten()

        for i in range(n_filters):
            rf = W[i, :].reshape(img_h, img_w)

            if normalize_each:
                rf_min = rf.min()
                rf_max = rf.max()
                if rf_max > rf_min:
                    rf_to_show = (rf - rf_min) / (rf_max - rf_min)
                else:
                    rf_to_show = np.zeros_like(rf)
            else:
                rf_to_show = rf

            ax = axes_flat[i]
            ax.imshow(rf_to_show, cmap="gray", interpolation="nearest")
            ax.axis("off")

        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].axis("off")

        plt.subplots_adjust(
            wspace=0.05, hspace=0.05,
            left=0.02, right=0.98,
            bottom=0.02, top=0.92,
        )

        if save_path is not None:
            plt.savefig(save_path, format="pdf", bbox_inches="tight")

        clear_output(wait=True)
        display(plt.gcf())
        plt.close()

    def fit(self, X, n_epochs=1, shuffle_samples=True):
        n_mixtures, n_samples = X.shape

        if shuffle_samples:
            idx = self.rng.permutation(n_samples)
        else:
            idx = np.arange(n_samples)

        if self.W is None:
            # Nonnegative initialization for NMF-style setup
            self.W = 0.01 * self.rng.rand(self.n_sources, n_mixtures)
            if self.normalize_W_rows:
                self.W = self._normalize_rows(self.W)

        for epoch in range(n_epochs):
            for i_sample in tqdm(range(n_samples)):
                if self.plot_debug_during_training and (i_sample % self.debug_iteration_point == 0):
                    self.plot_face_filters(
                        self.W,
                        title=f"Learned Face Filters at Epoch {epoch}, Sample {i_sample}",
                    )

                x_current = np.ascontiguousarray(X[:, idx[i_sample]])
                y = np.zeros(self.n_sources)

                y = self.run_neural_dynamics(
                    x_current, y,
                    self.W, self.C_y, self.mu_y,
                    self.gamma_predictive,
                    self.epsilon,
                    self.neural_dynamics_iterations,
                    self.neural_lr_start,
                    self.neural_lr_stop,
                    stlambd_lr=self.stlambda_lr,
                    lr_rule=self.neural_lr_rule,
                    lr_decay_divider=self.neural_lr_decay_divider,
                    neural_OUTPUT_COMP_TOL=self.neural_OUTPUT_COMP_TOL,
                )

                error = y - self.W @ x_current

                if self.lr_W_rule == "constant":
                    lr_W = self.lr_W
                elif self.lr_W_rule == "divide_by_log_index":
                    lr_W = max(
                        self.lr_W / (
                            1.0 + np.log((epoch * n_samples + i_sample) / self.lr_W_decay_divider + 2.0)
                        ),
                        1e-8,
                    )
                elif self.lr_W_rule == "divide_by_index":
                    lr_W = max(
                        self.lr_W / ((epoch * n_samples + i_sample) / self.lr_W_decay_divider + 1.0),
                        1e-8,
                    )
                else:
                    lr_W = self.lr_W

                # Predictive Hebbian update
                self.W += lr_W * np.outer(error, x_current)

                # Project W onto the nonnegative orthant for the NMF-style setup
                if self.project_W_nonnegative:
                    self.W = np.maximum(self.W, 0.0)

                if self.normalize_W_rows:
                    self.W = self._normalize_rows(self.W)

                # Update running output statistics
                self.mu_y = self.lambda_lateral * self.mu_y + (1.0 - self.lambda_lateral) * y
                y_bar = y - self.mu_y
                self.C_y = self.lambda_lateral * self.C_y + (1.0 - self.lambda_lateral) * np.outer(y_bar, y_bar)

    def predict(self, X):
        return self.W @ X

    def fit_predict(self, X, n_epochs=1, shuffle_samples=True):
        self.fit(X, n_epochs=n_epochs, shuffle_samples=shuffle_samples)
        return self.predict(X)