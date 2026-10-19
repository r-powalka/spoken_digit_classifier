import numpy as np


def normalize(waveform):
    """Apply peak normalization."""
    peak = np.max(np.abs(waveform))

    if peak == 0:
        return waveform.copy()

    return waveform / peak


def preemphasis(waveform, alpha=0.95):
    """Apply first-order pre-emphasis filter."""
    return np.convolve(
        waveform,
        np.array([1.0, -alpha]),
        mode="same",
    )

def preprocess_audio(waveform, alpha=0.95):
    """
    Apply waveform preprocessing (peak normalization
    and preemhasis).

    Parameters
    ----------
    waveform : ndarray
        Input audio waveform.

    alpha : float, default=0.95
        Pre-emphasis coefficient.

    Returns
    -------
    ndarray
        Preprocessed waveform.
    """
    waveform = np.asarray(waveform, dtype=np.float64)

    waveform = normalize(waveform)

    waveform = preemphasis(
        waveform,
        alpha=alpha,
    )

    return waveform
