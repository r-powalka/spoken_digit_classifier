import numpy as np


def energy_vad(power_spectrum):
    """Detect speech-active frames from power spectrum.

    Parameters
    ----------
    power_spectrum : np.ndarray
        2D power spectrum with shape
        (n_frames, n_frequency_bins).

    Returns
    -------
    np.ndarray
        Boolean mask with shape (n_frames,).
        True indicates a speech-active frame.
    """
    eps = np.finfo(float).eps

    # Compute total energy of each frame.
    frame_energy = np.sum(
        power_spectrum,
        axis=1,
    )

    # Convert frame energy to decibels.
    frame_energy_db = 10 * np.log10(
        frame_energy + eps
    )

    # Threshold.
    threshold_db = (
        np.mean(frame_energy_db) - 3
    )

    return frame_energy_db > threshold_db