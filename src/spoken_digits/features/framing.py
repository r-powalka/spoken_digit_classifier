from numpy.lib.stride_tricks import as_strided


def frame_signal(x, frame_length, hop_length, writeable=False):
    """Split a 1D signal into overlapping frames.

    Parameters
    ----------
    x : np.ndarray
        One-dimensional audio signal.
    frame_length : int
        Number of samples in each frame.
    hop_length : int
        Number of samples between consecutive frames.
    writeable : bool
        Whether the returned view should be writeable.

    Returns
    -------
    np.ndarray
        Array of shape (n_frames, frame_length).
    """
    assert x.ndim == 1
    assert hop_length >= 1
    assert len(x) >= frame_length

    byte = x.itemsize
    n_frames = (len(x) - frame_length) // hop_length + 1

    return as_strided(x,
                      shape=(n_frames, frame_length),
                      strides=(byte * hop_length, byte),
                      writeable=writeable)