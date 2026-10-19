import numpy as np

from .framing import frame_signal
from .windows import hamming


def stft(x, window_length, hop_length, nfft=1024, center=True):
    """Compute the short-time Fourier transform.

    Parameters
    ----------
    x : np.ndarray
        1D input signal.

    window_length : int
        Length of the analysis window in samples.

    hop_length : int
        Distance between consecutive frames in samples.

    nfft : int
        FFT size.

    center : bool
        Pad the signal so frames are centered around their
        original positions.

    Returns
    -------
    np.ndarray
        Complex STFT with shape
        (n_frames, nfft // 2 + 1).
    """

    if x.ndim != 1:
        raise ValueError("x must be a 1D array.")

    if window_length > nfft:
        raise ValueError(
            "window_length cannot be larger than nfft."
        )

    if center:
        pad = nfft // 2
        x = np.pad(
            x,
            (pad, pad),
            mode="constant",
        )

    # Create nfft-length frames
    frames = frame_signal(
        x,
        frame_length=nfft,
        hop_length=hop_length,
    )

    # Create centered analysis window
    window = np.zeros(nfft)

    w = hamming(window_length)

    start = (nfft - window_length) // 2

    window[start:start + window_length] = w

    frames = frames * window[None, :]

    return np.fft.rfft(
        frames,
        n=nfft,
        axis=1,
    )
