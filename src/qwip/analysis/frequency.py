import numpy as np
from scipy.fft import fft, fftfreq, fftshift


def simple_fft(
    ts: np.ndarray,
    ys: np.ndarray,
    axis: int = -1,
    subtract_mean: bool = True,
) -> tuple[np.array, np.array]:
    """Helper function for computing FFT's.

    Uses scipy.fft.fft to perform an fft, and returns both the frequency domain
    data as well as the sampled frequencies in units of Hz (1/s) assuming
    `ts` is given in seconds. The DC offset can be optionally subtracted before
    computing fourier transform.

    Args:
        ts: The array of time values. The sample spacing should be equal but this
            is not verified.
        ys: The array of time domain.
        subtract_mean: If true, the mean of ys will be subtracted before computing
            the fft.

        Returns:
            A tuple with the sampled frequency array and the frequency domain
            data.
    """
    ys = np.asarray(ys)
    if subtract_mean:
        ys = ys - np.mean(ys, axis=axis, keepdims=True)

    yfs = fft(ys, axis=axis)
    fs = fftfreq(yfs.shape[axis], ts[1] - ts[0])

    return fftshift(fs), fftshift(yfs, axes=axis)


def get_frequency_phase(fs: np.ndarray, yfs: np.ndarray, sgn: int | None = 1):
    """Determines the largest frequency component.

    Args:
        fs: The frequency domain values.
        yfs: The frequency domain data.
        sgn: Determines whether the frequency search should be restricted to a
            subset of the domain. If `sgn` is positive or negative, the search
            will be restricted to positive or negative frequencies. Otherwise
            the entire domain will be used.

    Returns:
        The estimated frequency and the phase at that frequency.
    """
    if sgn:
        mask = sgn * fs > 0
        fs = fs[mask]
        yfs = yfs[mask]

    idx = np.argmax(np.abs(yfs))
    return fs[idx], np.angle(yfs[idx])
