import numpy as np


def compute_delta(features, N=2):
    """Compute temporal delta coefficients.

    Parameters
    ----------
    features : np.ndarray
        2D array with shape (n_frames, n_coefficients).

    N : int
        Number of frames used on each side for the regression.

    Returns
    -------
    np.ndarray
        Delta features with the same shape as `features`.
    """
    n_frames, n_coeffs = features.shape
    delta = np.zeros_like(features)

    denominator = 2 * sum(n**2 for n in range(1, N + 1))

    # Repeat edge frames when computing deltas at boundaries.
    padded = np.pad(
        features,
        ((N, N), (0, 0)),
        mode="edge",
    )

    for t in range(n_frames):
        for n in range(1, N + 1):
            delta[t] += n * (
                padded[t + N + n]
                - padded[t + N - n]
            )

        delta[t] /= denominator

    return delta