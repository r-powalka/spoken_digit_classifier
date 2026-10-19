import numpy as np
import librosa

from src.spoken_digits.features.stft import stft

def test_stft_matches_librosa():
    """The manual STFT implementation should match librosa.stft numerically."""

    # Synthetic test signal
    fs = 8000
    duration = 1

    t = np.arange(int(fs * duration)) / fs

    x = (
        0.6 * np.sin(2 * np.pi * 440 * t)
        + 0.4 * np.sin(2 * np.pi * 880 * t)
    )

    # STFT parameters
    nfft = 1024
    window_length = int(0.025 * fs)
    hop_length = int(0.015 * fs)

    # Our implementation
    X_my = stft(
        x,
        window_length=window_length,
        hop_length=hop_length,
        nfft=nfft,
        center=True,
    )

    # Reference implementation
    X_librosa = librosa.stft(
        y=x,
        n_fft=nfft,
        hop_length=hop_length,
        win_length=window_length,
        window="hamming",
        center=True,
    )

    # librosa returns (frequency, time)
    # our implementation returns (time, frequency)
    X_librosa = X_librosa.T

    # Same shape
    assert X_my.shape == X_librosa.shape

    # Numerical equivalence
    np.testing.assert_allclose(
        X_my,
        X_librosa,
        rtol=1e-10,
        atol=1e-10,
    )