import numpy as np
from scipy.special import logsumexp

from .gaussian import log_gaussian_pdf


class GMM:
    """
    Gaussian Mixture Model fitted using the
    Expectation-Maximization algorithm.

    Parameters
    ----------
    C : int, default=3
        Number of Gaussian components.

    seed : int or None, default=None
        Random seed used for parameter initialization.
    """

    def __init__(self, C=3, seed=None):
        if not isinstance(C, (int, np.integer)) or isinstance(
            C, (bool, np.bool_)
        ) or C < 1:
            raise ValueError("C must be a positive integer.")

        self.C = C
        self.seed = seed

        self.pi = None
        self.mu = None
        self.sigma = None

        self.log_likelihood = None
        self.log_likelihood_history = []

        self.is_fit = False

    def _initialize_parameters(self, X):
        """
        Initialize mixture weights, means,
        and covariance matrices.
        """

        N, d = X.shape

        rng = np.random.default_rng(self.seed)

        # Equal initial mixture weights
        self.pi = np.ones(self.C) / self.C

        # Choose C observations as initial means
        indices = rng.choice(N, self.C, replace=False)
        self.mu = X[indices].copy()

        # Initialize each component with the global covariance
        sigma_global = np.atleast_2d(
            np.cov(X, rowvar=False)
        )

        self.sigma = np.repeat(
            sigma_global[np.newaxis, :, :],
            self.C,
            axis=0,
        )

        self.log_likelihood = None
        self.log_likelihood_history = []
        self.is_fit = False

    def _e_step(self, X):
        """
        Calculate responsibilities.
        """

        N = X.shape[0]
        Q = np.empty((N, self.C))

        for n, x_n in enumerate(X):
            component_log_probs = np.empty(self.C)

            for c in range(self.C):
                component_log_probs[c] = (
                    np.log(self.pi[c])
                    + log_gaussian_pdf(
                        x_n,
                        self.mu[c],
                        self.sigma[c],
                    )
                )

            log_p_x = logsumexp(component_log_probs)

            Q[n] = np.exp(
                component_log_probs - log_p_x
            )

        return Q

    def _m_step(self, X, Q):
        """
        Update mixture weights, means,
        and covariance matrices.
        """

        N, d = X.shape

        # Effective number of observations assigned to each component
        N_c = Q.sum(axis=0)

        # Update mixing coefficients
        self.pi = N_c / N

        for c in range(self.C):
            if N_c[c] <= 0:
                continue

            # Update mean
            self.mu[c] = (
                Q[:, c] @ X
            ) / N_c[c]

            # Update covariance using the updated mean
            diff = X - self.mu[c]

            self.sigma[c] = (
                (Q[:, c, None] * diff).T @ diff
            ) / N_c[c]

            # Matrix regularization
            self.sigma[c] += 1e-6 * np.eye(d)

    def _log_likelihood(self, X):
        """
        Calculate the observed-data log-likelihood.
        """

        total = 0.0

        for x_n in X:
            component_log_probs = np.empty(self.C)

            for c in range(self.C):
                component_log_probs[c] = (
                    np.log(self.pi[c])
                    + log_gaussian_pdf(
                        x_n,
                        self.mu[c],
                        self.sigma[c],
                    )
                )

            total += logsumexp(component_log_probs)

        return total

    def fit(
        self,
        X,
        max_iter=100,
        tol=1e-3,
        verbose=False,
    ):
        """
        Fit the GMM using the EM algorithm.

        Returns
        -------
        self
            Fitted GMM.
        """

        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("X must have shape (N, d).")

        N, d = X.shape

        if N == 0 or d == 0:
            raise ValueError("X must not be empty.")

        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values.")

        if N < self.C:
            raise ValueError(
                "Number of observations must be at least "
                "the number of components."
            )

        if max_iter < 1:
            raise ValueError("max_iter must be at least 1.")

        if tol < 0:
            raise ValueError("tol must be non-negative.")

        # Step 1: Initialize parameters
        self._initialize_parameters(X)

        # Evaluate the initial observed-data log-likelihood
        log_likelihood = self._log_likelihood(X)
        self.log_likelihood_history.append(log_likelihood)

        if verbose:
            print(
                f"Initial log-likelihood: {log_likelihood:.6f}"
            )

        for iteration in range(max_iter):

            Q = self._e_step(X)

            self._m_step(X, Q)

            new_log_likelihood = self._log_likelihood(X)

            self.log_likelihood_history.append(
                new_log_likelihood
            )

            improvement = new_log_likelihood - log_likelihood

            if verbose:
                print(
                    f"Iteration {iteration + 1}: "
                    f"log-likelihood = {new_log_likelihood:.6f}"
                )

            # Check convergence
            converged = (
                abs(improvement)
                < tol * (1.0 + abs(log_likelihood))
            )

            log_likelihood = new_log_likelihood

            if converged:
                if verbose:
                    print("Converged.")
                break

        self.log_likelihood = log_likelihood
        self.is_fit = True

        return self

    def score(self, X):
        """
        Compute mean log-likelihood per observation.

        Parameters
        ----------
        X : ndarray, shape (N, d)
            Input observations.

        Returns
        -------
        float
            Mean log-likelihood per observation.
        """

        if not self.is_fit:
            raise RuntimeError(
                "GMM must be fitted before scoring."
            )

        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("X must have shape (N, d).")

        if X.shape[0] == 0:
            raise ValueError("X must not be empty.")

        if X.shape[1] != self.mu.shape[1]:
            raise ValueError(
                "Feature dimension does not match fitted model."
            )

        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values.")

        return self._log_likelihood(X) / X.shape[0]
