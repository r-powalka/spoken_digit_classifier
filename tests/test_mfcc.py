import numpy as np
import librosa
from scipy.fft import dct as scipy_dct

from spoken_digits.features import mfcc


def test_mfcc_matches_reference():
    """Compare MFCC coefficients against a librosa-based reference (with scipy DCT)."""
    fs = 8000
    duration = 1

    t = np.arange(int(fs * duration)) / fs
    x = (
        0.6 * np.sin(2 * np.pi * 440 * t)
        + 0.4 * np.sin(2 * np.pi * 880 * t)
    )

    nfft = 1024
    window_length = round(0.025 * fs)
    hop_length = round(0.015 * fs)
    n_mfcc = 13
    n_filters = 20

    my_mfcc = mfcc(
        x,
        fs=fs,
        window_length=window_length,
        hop_length=hop_length,
        center=True,
        n_filters=n_filters,
        normalize=False,
        replace_intercept=False,
        include_deltas=False,
        nfft=nfft,
    )

    mel_spectrogram = librosa.feature.melspectrogram(
        y=x,
        sr=fs,
        n_fft=nfft,
        hop_length=hop_length,
        win_length=window_length,
        window="hamming",
        center=True,
        n_mels=n_filters,
        htk=True,
        norm=None,
        power=2,
    )

    log_mel = librosa.power_to_db(
        mel_spectrogram,
        ref=1.0,
        top_db=None,
    )

    reference_mfcc = scipy_dct(
        log_mel.T,
        type=2,
        axis=1,
        norm="ortho",
    )[:, :13]

    diff = my_mfcc - reference_mfcc

    max_abs_error = np.max(np.abs(diff))

    print(f"Maximum absolute MFCC error: {max_abs_error:.10e}")

    assert my_mfcc.shape == reference_mfcc.shape
    assert my_mfcc.shape[1] == n_mfcc
    assert np.isfinite(my_mfcc).all()

    np.testing.assert_allclose(
        my_mfcc,
        reference_mfcc,
        rtol=1e-6,
        atol=1e-6,
    )