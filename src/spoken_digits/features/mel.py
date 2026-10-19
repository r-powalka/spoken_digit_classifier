import numpy as np


def hz_to_mel(f):
    """Convert frequency from Hertz to the Mel scale (HTK convention).

    Parameters
    ----------
    f : float or np.ndarray
        Frequency in Hz.

    Returns
    -------
    float or np.ndarray
        Frequency on the Mel scale.
    """
    return 2595 * np.log10(1 + f / 700)


def mel_to_hz(m):
    """Convert frequency from the Mel scale to Hertz (HTK convention).

    Parameters
    ----------
    m : float or np.ndarray
        Frequency on the Mel scale.

    Returns
    -------
    float or np.ndarray
        Frequency in Hz.
    """
    return 700 * (10 ** (m / 2595) - 1)


def mel_filterbank(
    nfft,
    n_filters=20,
    fs=8000,
    min_freq=0,
    max_freq=None,
    normalize=False,
):
    """Construct a triangular Mel filterbank.

    The filterbank contains `n_filters` triangular filters
    distributed uniformly on the Mel frequency scale.

    Parameters
    ----------
    nfft : int
        FFT size. The resulting filterbank has
        nfft // 2 + 1 frequency bins.

    n_filters : int
        Number of triangular Mel filters.

    fs : float
        Sampling frequency in Hz.

    min_freq : float
        Minimum frequency of the filterbank in Hz.

    max_freq : float or None
        Maximum frequency of the filterbank in Hz.
        Defaults to the Nyquist frequency (fs / 2).

    normalize : bool
        If True, apply filter-energy normalization.

    Returns
    -------
    np.ndarray
        Mel filterbank with shape:

            (n_filters, nfft // 2 + 1)
    """

    # By default, the upper frequency is the Nyquist frequency.
    if max_freq is None:
        max_freq = fs / 2

    # Frequencies corresponding to the FFT bins.
    hz_bins = np.linspace(
        0,
        fs / 2,
        1 + nfft // 2,
        endpoint=True,
    )

    # Convert the frequency limits from Hz to Mel.
    min_mel = hz_to_mel(min_freq)
    max_mel = hz_to_mel(max_freq)

    # Create n_filters + 2 Mel-spaced points.
    mel_bins = mel_to_hz(
        np.linspace(
            min_mel,
            max_mel,
            n_filters + 2,
        )
    )

    # Distance between neighboring filter-center frequencies.
    mel_spacing = np.diff(mel_bins)

    # Allocate the filterbank.
    #
    # Each row is one triangular filter.
    # Each column corresponds to an FFT frequency bin.
    fbank = np.zeros(
        (n_filters, nfft // 2 + 1)
    )

    # Compute the distance between every Mel filter
    # boundary and every FFT frequency bin.
    #
    # Shape:
    #     (n_filters + 2, nfft // 2 + 1)
    ramps = (
        mel_bins.reshape(-1, 1)
        - hz_bins.reshape(1, -1)
    )

    # Construct each triangular filter.
    for i in range(n_filters):

        # Rising side of the triangle.
        left = -ramps[i] / mel_spacing[i]

        # Falling side of the triangle.
        right = ramps[i + 2] / mel_spacing[i + 1]

        # Combine the two sides and clip negative values to zero.
        fbank[i] = np.maximum(
            0,
            np.minimum(left, right),
        )

    # Optional filter-energy normalization.
    if normalize:
        energy_norm = (
            2.0
            / (
                mel_bins[2:n_filters + 2]
                - mel_bins[:n_filters]
            )
        )

        fbank *= energy_norm[:, np.newaxis]

    return fbank