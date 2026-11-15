"""Reuse periodic electronic samples on finer nuclear grids."""

from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray


def resample_periodic(
    samples: ArrayLike, n_points: int, *, max_harmonic: int | None = None,
) -> NDArray[np.float64]:
    """Evaluate a trigonometric interpolant on a uniform grid over one period.

    Supply real samples evenly spaced from a shared origin, with the repeated
    endpoint omitted. The first axis is the grid and any trailing axes are
    independent channels. The output covers the same period and shares the
    origin. For even input length, the aliased Nyquist term is represented by
    a cosine. Set ``max_harmonic`` explicitly to truncate the fitted series.

    Apply this only to sufficiently sampled periodic data. A denser output
    grid improves resolution of the nuclear discretization but cannot recover
    features absent from the electronic samples. Dipoles must already have
    consistent Cartesian axes and electronic signs over the whole loop.
    """
    data = np.asarray(samples)
    if data.ndim == 0 or data.shape[0] < 3 or data.dtype.kind not in "fiu" or not np.isfinite(data).all():
        raise ValueError("samples must contain at least three finite real points along the first axis")
    if isinstance(n_points, (bool, np.bool_)) or not isinstance(n_points, Integral) or n_points < data.shape[0]:
        raise ValueError("n_points must be an integer at least as large as the number of samples")
    sample_count = data.shape[0]
    nyquist = sample_count // 2
    if max_harmonic is None:
        max_harmonic = nyquist
    if (isinstance(max_harmonic, (bool, np.bool_)) or not isinstance(max_harmonic, Integral)
        or not 0 <= max_harmonic <= nyquist):
        raise ValueError("max_harmonic is outside the Fourier modes supported by the samples")
    coefficients = np.fft.rfft(data, axis=0, norm="forward")[:max_harmonic + 1]
    if sample_count % 2 == 0 and max_harmonic == nyquist and n_points != sample_count:
        # In the denser grid, this mode is no longer at Nyquist and irfft
        # accounts for both its positive and negative frequencies.
        coefficients[-1] *= 0.5
    return np.fft.irfft(coefficients, n=n_points, axis=0, norm="forward")
