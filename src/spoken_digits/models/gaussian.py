import numpy as np


def log_gaussian_pdf(x, mean, cov, eps=1e-6):
    """
    Compute the log probability density of x under a multivariate Gaussian.

    Parameters
    ----------
    x : ndarray, shape (d,)
        Observation.
    mean : ndarray, shape (d,)
        Gaussian mean.
    cov : ndarray, shape (d, d)
        Gaussian covariance matrix.
    eps : float, default=1e-6
        Diagonal regularization added to the covariance matrix.

    Returns
    -------
    float
        Log probability density.
    """
    x = np.asarray(x, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)

    d = mean.shape[0]

    # Numerical regularization
    cov = cov + eps * np.eye(d)

    sign, logdet = np.linalg.slogdet(cov)

    if sign <= 0:
        raise np.linalg.LinAlgError(
            "Covariance matrix is not positive definite."
        )

    # Mahalanobis distance:
    diff = x - mean
    mahalanobis = diff @ np.linalg.solve(cov, diff)

    return -0.5 * (
        d * np.log(2 * np.pi)
        + logdet
        + mahalanobis
    )