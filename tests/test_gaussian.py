import numpy as np
from scipy.stats import multivariate_normal

from spoken_digits.models.gaussian import log_gaussian_pdf


def test_log_gaussian_pdf_matches_scipy():
    x = np.array([1.0, 2.0])
    mean = np.array([0.5, 1.5])
    cov = np.array([
        [2.0, 0.3],
        [0.3, 1.0],
    ])

    # Your function adds eps * I to the covariance.
    eps = 1e-6
    expected = multivariate_normal.logpdf(
        x,
        mean=mean,
        cov=cov + eps * np.eye(2),
    )

    actual = log_gaussian_pdf(x, mean, cov, eps=eps)

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-10,
        atol=1e-10,
    )