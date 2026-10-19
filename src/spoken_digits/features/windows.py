import numpy as np


def hamming(length):
    """Generate a periodic Hamming window."""
    if length <= 0:
        raise ValueError("length must be positive.")

    n = np.arange(length)

    return (
        0.54
        - 0.46 * np.cos(2 * np.pi * n / length)
    )
