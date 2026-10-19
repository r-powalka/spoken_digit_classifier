import numpy as np



def dct(frame, orthonormal=True):
    """Compute the DCT-II of a 1D input vector.

    Parameters
    ----------
    frame : np.ndarray
        1D array of log Mel filterbank energies for one frame.

    orthonormal : bool
        If True, apply orthonormal DCT scaling.

    Returns
    -------
    np.ndarray
        1D array of DCT coefficients with the same length as `frame`.
    """
    N = len(frame)
    out = np.zeros_like(frame)

    # DCT-II
    for k in range(N):
        for n, x_n in enumerate(frame):
            out[k] += x_n * np.cos(
                np.pi * k * (2 * n + 1) / (2 * N)
            )

        # Apply DCT scaling
        scale = (
            np.sqrt(1 / (4 * N))
            if k == 0
            else np.sqrt(1 / (2 * N))
        )

        out[k] *= 2 * scale if orthonormal else 2

    return out