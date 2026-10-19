import numpy as np

from .stft import stft
from .mel import mel_filterbank
from .dct import dct
from .deltas import compute_delta
from spoken_digits.preprocessing import energy_vad


def cepstral_lifter(mfccs, D):
    """Apply sinusoidal cepstral liftering.

    Parameters
    ----------
    mfccs : np.ndarray
        Matrix of computed MFCCs.
    
    D : int
        Lifter coefficient.

    Returns
    -------
    np.ndarray
        Matrix of computed MFCCs with
        applied lifter.

    """
    if D == 0:
        return mfccs

    n = np.arange(mfccs.shape[1])

    return mfccs * (
        1 + (D / 2) * np.sin(np.pi * n / D)
    )


def mel_spectrogram(
    x,
    fs,
    window_length,
    hop_length,
    nfft=1024,
    n_filters=20,
    mean_normalize=True,
    center=True,
    use_vad=True,
):
    """Compute Mel filterbank energies from an audio signal.

    Parameters
    ----------
    x : np.ndarray
        1D audio signal.

    fs : int or float
        Sampling frequency in Hz.

    window_length : int
        STFT window length in samples.

    hop_length : int
        Number of samples between consecutive frames.

    nfft : int
        FFT size.

    n_filters : int
        Number of Mel filters.

    mean_normalize : bool
        Subtract the mean filterbank energy from each filter.

    center : bool
        Whether to center STFT frames using zero-padding.

    use_vad : bool
        Whether to remove non-speech frames using energy_vad.

    Returns
    -------
    filter_energies : np.ndarray
        Mel filterbank energies with shape
        (n_frames, n_filters).

    frame_energies : np.ndarray
        Total energy of each retained frame.
    """
    eps = np.finfo(float).eps

    # Compute power spectrum.
    X = stft(
        x,
        window_length=window_length,
        hop_length=hop_length,
        nfft=nfft,
        center=center,
    )

    power_spectrum = np.abs(X) ** 2

    # Remove frames classified as non-speech.
    if use_vad:
        power_spectrum = power_spectrum[
            energy_vad(power_spectrum)
        ]

    # Compute total energy for each frame.
    frame_energies = np.sum(
        power_spectrum,
        axis=1,
    )

    frame_energies[
        frame_energies == 0
    ] = eps

    # Create Mel filterbank.
    fbank = mel_filterbank(
        nfft,
        n_filters=n_filters,
        fs=fs,
    )

    # Apply Mel filters to the power spectrum.
    filter_energies = power_spectrum @ fbank.T

    if mean_normalize:
        filter_energies -= np.mean(
            filter_energies,
            axis=0,
        )

    filter_energies[
        filter_energies == 0
    ] = eps

    return filter_energies, frame_energies


def mfcc(
    x,
    fs,
    n_mfccs=13,
    center=True,
    n_filters=20,
    normalize=True,
    lifter_coef=0,
    hop_length=None,
    window_length=None,
    replace_intercept=True,
    include_deltas=True,
    use_vad=True,
    nfft=1024,
):
    """Compute MFCC features from an audio signal.

    Parameters
    ----------
    x : np.ndarray
        1D audio signal.

    fs : int or float
        Sampling frequency in Hz.

    n_mfccs : int
        Number of MFCC coefficients to retain.

    center : bool
        Whether to center STFT frames.

    n_filters : int
        Number of Mel filters.

    normalize : bool
        Subtract the mean from each MFCC coefficient.

    lifter_coef : float
        Cepstral liftering coefficient.

    hop_length : int or None
        STFT hop length in samples.

    window_length : int or None
        STFT window length in samples.

    replace_intercept : bool
        Replace the first MFCC coefficient with log frame energy.

    include_deltas : bool
        Append delta and delta-delta coefficients.

    use_vad : bool
        Remove non-speech frames before computing MFCCs.

    nfft : int
        FFT size.

    Returns
    -------
    np.ndarray
        MFCC feature matrix.

        If include_deltas=True:
            shape = (n_frames, 3 * n_mfccs)

        Otherwise:
            shape = (n_frames, n_mfccs)
    """
    # Default frame configuration.
    if window_length is None:
        window_length = int(0.025 * fs)

    if hop_length is None:
        hop_length = int(0.015 * fs)

    filter_energies, frame_energies = mel_spectrogram(
        x=x,
        fs=fs,
        window_length=window_length,
        hop_length=hop_length,
        nfft=nfft,
        n_filters=n_filters,
        mean_normalize=False,
        center=center,
        use_vad=use_vad,
    )

    # Convert Mel energies to log scale.
    log_energies = 10 * np.log10(
        filter_energies
    )

    # Apply DCT and keep the first n_mfccs coefficients.
    mfccs = np.array([
        dct(frame)
        for frame in log_energies
    ])[:, :n_mfccs]

    # Apply cepstral liftering.
    mfccs = cepstral_lifter(
        mfccs,
        D=lifter_coef,
    )

    # Cepstral mean normalization.
    if normalize:
        mfccs -= np.mean(
            mfccs,
            axis=0,
        )

    # Replace the 0th coefficient with log frame energy.
    if replace_intercept:
        mfccs[:, 0] = np.log(
            frame_energies
        )

    if not include_deltas:
        return mfccs

    # First-order temporal derivatives.
    delta = compute_delta(
        mfccs,
        N=2,
    )

    # Second-order temporal derivatives.
    delta_delta = compute_delta(
        delta,
        N=2,
    )

    # [MFCC | delta | delta-delta]
    return np.concatenate(
        [mfccs, delta, delta_delta],
        axis=1,
    )