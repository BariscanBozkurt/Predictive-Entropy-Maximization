"""
Predictive Entropy Maximization (PEM) network for learning face features.

Self-contained variant of the PEM network specialized to nonnegative face-feature
learning (an NMF-style setup). It is intentionally INDEPENDENT of the rest of the
code base: it does not import or modify any of the officially-submitted modules, so
the face experiment can evolve on its own.

The network learns a feedforward weight matrix W online, sample by sample:
  1. Fast neural dynamics infer a nonnegative bounded activity y in [0, 1]^k that
     balances a predictive drive gamma * ||y - W x||^2 against an adaptive lateral
     decorrelation term (a Taylor surrogate of the log-det / correlative-entropy
     objective, using running output statistics C_y, mu_y).
  2. A local error-driven Hebbian rule updates W:  W += lr_W * (y - W x) x^T,
     projected onto the nonnegative orthant.

The lateral decorrelation is what makes units specialize onto distinct facial
features (eyes, mouth, eyebrows, ...) instead of redundant holistic templates.
"""
import numpy as np
from numba import njit
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output, display


class PredictiveEntropyFaceNMF:
    def __init__(
        self,
        n_sources,
        presumed_domain="nnantisparse",
        epsilon=1e-3,
        lambda_lateral=0.99,
        gamma_predictive=30.0,
        lr_W=3e-3,
        neural_lr_start=0.1,
        neural_lr_stop=1e-3,
        stlambda_lr=0.1,
        neural_dynamics_iterations=200,
        neural_OUTPUT_COMP_TOL=1e-7,
        lr_W_rule="divide_by_index",
        lr_W_decay_divider=50000,
        neural_lr_rule="divide_by_loop_index",
        neural_lr_decay_divider=200,
        W=None,
        C_y=None,
        mu_y=None,
        image_shape=(32, 32),
        project_W_nonnegative=True,
        normalize_W_rows=False,
        debug_iteration_point=4000,
        plot_debug_during_training=False,
        seed=None,
    ):
        self.n_sources = n_sources
        self.epsilon = epsilon
        self.lambda_lateral = lambda_lateral
        self.gamma_predictive = gamma_predictive
        self.lr_W = lr_W
        self.neural_lr_start = neural_lr_start
        self.neural_lr_stop = neural_lr_stop
        self.stlambda_lr = stlambda_lr
        self.neural_dynamics_iterations = neural_dynamics_iterations
        self.neural_OUTPUT_COMP_TOL = neural_OUTPUT_COMP_TOL
        self.lr_W_rule = lr_W_rule
        self.lr_W_decay_divider = lr_W_decay_divider
        self.neural_lr_rule = neural_lr_rule
        self.neural_lr_decay_divider = neural_lr_decay_divider

        self.image_shape = image_shape
        self.project_W_nonnegative = project_W_nonnegative
        self.normalize_W_rows = normalize_W_rows
        self.debug_iteration_point = debug_iteration_point
        self.plot_debug_during_training = plot_debug_during_training
        self.rng = np.random.RandomState(seed)

        # Select the fast inference nonlinearity for the assumed source domain.
        if presumed_domain == "nnantisparse":       # nonnegative L-infinity ball, y in [0, 1]
            self.run_neural_dynamics = self.run_neural_dynamics_nnantisparse
        elif presumed_domain == "nnsparse":          # nonnegative L1 ball (shared soft threshold)
            self.run_neural_dynamics = self.run_neural_dynamics_nnsparse
        else:
            raise ValueError(f"presumed_domain '{presumed_domain}' not supported by this class.")
        self.presumed_domain = presumed_domain

        self.W = W
        self.C_y = 0.05 * np.eye(n_sources) if C_y is None else C_y
        self.mu_y = np.zeros(n_sources) if mu_y is None else mu_y
        if self.W is not None and self.project_W_nonnegative:
            self.W = np.maximum(self.W, 0.0)
            if self.normalize_W_rows:
                self.W = self._normalize_rows(self.W)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _normalize_rows(W):
        row_norms = np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)
        return W / row_norms

    # ---------------------------------------------------- fast neural dynamics
    @staticmethod
    @njit
    def run_neural_dynamics_nnantisparse(
        x, y, W, C_y, mu_y, gamma_predictive, epsilon,
        neural_dynamics_iterations, neural_lr_start, neural_lr_stop,
        stlambd_lr=0.0, lr_rule="divide_by_loop_index",
        lr_decay_divider=200, neural_OUTPUT_COMP_TOL=1e-7,
    ):
        """Nonnegative bounded activity y in [0, 1]^k (domain B_{max,+})."""
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

    @staticmethod
    @njit
    def run_neural_dynamics_nnsparse(
        x, y, W, C_y, mu_y, gamma_predictive, epsilon,
        neural_dynamics_iterations, neural_lr_start, neural_lr_stop,
        stlambd_lr=0.1, lr_rule="divide_by_loop_index",
        lr_decay_divider=200, neural_OUTPUT_COMP_TOL=1e-7,
    ):
        """Nonnegative sparse activity on the L1 ball (shared soft threshold / lambda_L)."""
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
            y = np.maximum(a - STLAMBD, 0.0)
            y = np.minimum(y, 10.0)
            dval = np.sum(y) - 1.0
            STLAMBD = max(STLAMBD + stlambd_lr * dval, 0.0)
            denom = np.linalg.norm(y)
            if denom < 1e-12:
                denom = 1e-12
            if np.linalg.norm(y - y_old) < neural_OUTPUT_COMP_TOL * denom:
                break
        return y

    # ------------------------------------------------------- inference / readout
    def infer_codes(self, X):
        """Run the settled fast dynamics for every column of X (features frozen)."""
        Y = np.zeros((self.n_sources, X.shape[1]))
        for i in range(X.shape[1]):
            Y[:, i] = self.run_neural_dynamics(
                np.ascontiguousarray(X[:, i]), np.zeros(self.n_sources),
                self.W, self.C_y, self.mu_y, self.gamma_predictive, self.epsilon,
                self.neural_dynamics_iterations, self.neural_lr_start, self.neural_lr_stop,
                stlambd_lr=self.stlambda_lr, lr_rule=self.neural_lr_rule,
                lr_decay_divider=self.neural_lr_decay_divider,
                neural_OUTPUT_COMP_TOL=self.neural_OUTPUT_COMP_TOL,
            )
        return Y

    # ------------------------------------------------------------- live plotting
    def plot_face_filters(self, W=None, title="Learned face features",
                          sort_rows=None, ncols=None):
        """Show rows of W as image-shaped filters (used live during training)."""
        if W is None:
            W = self.W
        if sort_rows is not None:
            W = W[sort_rows]
        n = W.shape[0]
        h, w = self.image_shape
        ncols = ncols or int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols, nrows))
        fig.suptitle(title, fontsize=14, fontweight="bold")
        axes_flat = np.atleast_1d(axes).ravel()
        for a in axes_flat:
            a.axis("off")
        for i in range(n):
            rf = W[i].reshape(h, w)
            if rf.max() > rf.min():
                rf = (rf - rf.min()) / (rf.max() - rf.min())
            axes_flat[i].imshow(rf, cmap="gray", interpolation="nearest")
        plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0.02, right=0.98,
                            bottom=0.02, top=0.92)
        clear_output(wait=True)
        display(plt.gcf())
        plt.close()

    # -------------------------------------------------------------------- fit
    def fit(self, X, n_epochs=1, shuffle_samples=True):
        n_mixtures, n_samples = X.shape
        if self.W is None:
            # Data-independent random nonnegative init, scaled so mean(W x) ~ 0.5.
            scale = 2 * 0.5 / (max(X.mean(), 1e-9) * n_mixtures)
            self.W = self.rng.rand(self.n_sources, n_mixtures) * scale
            if self.normalize_W_rows:
                self.W = self._normalize_rows(self.W)

        for epoch in range(n_epochs):
            idx = self.rng.permutation(n_samples) if shuffle_samples else np.arange(n_samples)
            for i_sample in tqdm(range(n_samples)):
                global_step = epoch * n_samples + i_sample
                if self.plot_debug_during_training and (global_step % self.debug_iteration_point == 0):
                    self.plot_face_filters(
                        self.W,
                        title=f"Learned features  (epoch {epoch}, step {global_step})",
                    )

                x_current = np.ascontiguousarray(X[:, idx[i_sample]])
                y = self.run_neural_dynamics(
                    x_current, np.zeros(self.n_sources),
                    self.W, self.C_y, self.mu_y, self.gamma_predictive, self.epsilon,
                    self.neural_dynamics_iterations, self.neural_lr_start, self.neural_lr_stop,
                    stlambd_lr=self.stlambda_lr, lr_rule=self.neural_lr_rule,
                    lr_decay_divider=self.neural_lr_decay_divider,
                    neural_OUTPUT_COMP_TOL=self.neural_OUTPUT_COMP_TOL,
                )

                # Local error-driven feedforward (Hebbian) update.
                error = y - self.W @ x_current
                if self.lr_W_rule == "constant":
                    lr_W = self.lr_W
                elif self.lr_W_rule == "divide_by_log_index":
                    lr_W = max(self.lr_W / (1 + np.log(global_step / self.lr_W_decay_divider + 2)), 1e-8)
                elif self.lr_W_rule == "divide_by_index":
                    lr_W = max(self.lr_W / (global_step / self.lr_W_decay_divider + 1), 1e-8)
                else:
                    lr_W = self.lr_W
                self.W += lr_W * np.outer(error, x_current)
                if self.project_W_nonnegative:
                    self.W = np.maximum(self.W, 0.0)
                if self.normalize_W_rows:
                    self.W = self._normalize_rows(self.W)

                # Update running output statistics used by the lateral term.
                self.mu_y = self.lambda_lateral * self.mu_y + (1 - self.lambda_lateral) * y
                y_bar = y - self.mu_y
                self.C_y = self.lambda_lateral * self.C_y + (1 - self.lambda_lateral) * np.outer(y_bar, y_bar)

    def predict(self, X):
        return self.W @ X

    def fit_predict(self, X, n_epochs=1, shuffle_samples=True):
        self.fit(X, n_epochs=n_epochs, shuffle_samples=shuffle_samples)
        return self.predict(X)
