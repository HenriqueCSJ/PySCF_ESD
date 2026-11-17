"""Phase and resample electronic inputs on a closed rotor coordinate."""

from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .anharmonic_dvr import _positive_scalar, _real_vector, _rotor_domain, _uniform_spacing
from .harmonic import _real_array
from .state_tracking import StateTrackingError


@dataclass(frozen=True)
class PhasedRotorDipole:
    """Signed Cartesian samples on a one-cycle grid, with declared seam parity.

    The samples use a continuously transported electronic gauge starting at
    ``grid[0]`` and omit the repeated endpoint. ``closure_sign`` gives the
    factor acquired by the dipole after one cycle. The sign is the product of
    the two electronic closure signs. A declaration documents how the data
    were prepared because the omitted endpoint cannot establish the parity.
    """

    grid: NDArray[np.float64]
    values_au: NDArray[np.float64]
    closure_sign: int


@dataclass(frozen=True)
class PhasedPeriodicScan:
    """Two periodic energies and a phased transition on one rotor grid.

    The ``*_source_phases`` arrays multiply the input electronic states and
    always refer to the original scan samples. The nuclear boundaries carry
    the same signs as the electronic closure so their products are periodic.
    """

    grid: NDArray[np.float64]
    period: float
    ground_potential_hartree: NDArray[np.float64]
    excited_potential_hartree: NDArray[np.float64]
    transition_dipole: PhasedRotorDipole
    ground_source_phases: NDArray[np.intp]
    excited_source_phases: NDArray[np.intp]
    ground_closure_sign: int
    excited_closure_sign: int

    @property
    def ground_boundary(self) -> Literal["periodic", "antiperiodic"]:
        return "periodic" if self.ground_closure_sign == 1 else "antiperiodic"

    @property
    def excited_boundary(self) -> Literal["periodic", "antiperiodic"]:
        return "periodic" if self.excited_closure_sign == 1 else "antiperiodic"


def _scan_phases(overlaps: ArrayLike, count: int, minimum: float, state: str) -> tuple[NDArray[np.intp], int]:
    values = _real_vector(overlaps, f"{state} overlaps")
    if values.shape != (count,) or np.any(np.abs(values) > 1.0 + 1e-8):
        raise ValueError(f"supply one signed normalized {state} overlap per link, including closure")
    weakest = int(np.argmin(np.abs(values)))
    if abs(values[weakest]) < minimum:
        destination = (weakest + 1) % count
        raise StateTrackingError(
            f"{state} scan link {weakest} to {destination} has insufficient overlap ({abs(values[weakest]):.6g})"
        )
    link_signs = np.where(values < 0, -1, 1)
    phases = np.empty(count, dtype=np.intp)
    phases[0] = 1
    np.cumprod(link_signs[:-1], out=phases[1:])
    return phases, int(phases[-1] * link_signs[-1])


def phase_periodic_scan(
    grid: ArrayLike,
    ground_potential_hartree: ArrayLike,
    excited_potential_hartree: ArrayLike,
    transition_dipole_au: ArrayLike,
    *,
    ground_overlaps: ArrayLike,
    excited_overlaps: ArrayLike,
    period: float = 2 * np.pi,
    minimum_overlap: float = 0.65,
    n_points: int | None = None,
) -> PhasedPeriodicScan:
    """Transport two labeled real electronic states around one complete loop.

    Supply uniform samples without the repeated endpoint. Both energy arrays
    use the same zero. At each angle the raw dipole is a real length-gauge
    Cartesian vector in e·bohr with the phases of those same two raw states.
    Each overlap array contains the signed, normalized overlaps from point i
    to point i+1, with the last element linking the last point back to the
    original first point in its original phase. The caller establishes the
    state identities and isolation, fixed atom labels and Cartesian frame,
    same electronic recipe, and geometric closure before computing the links.
    A negative closure sign preserves the labeled state with an opposite phase.

    If ``n_points`` is given, Fourier interpolate onto that many samples over
    the same period. Both scalar potentials are interpolated periodically.
    Interpolate a dipole with negative parity over a double cycle with an
    opposite sign on the second cycle. Interpolation does not resolve missing
    structure in the sampled electronic data.
    """
    coordinates = _real_vector(grid, "grid")
    count = len(coordinates)
    if count < 3:
        raise ValueError("the rotor scan must contain at least three grid points")
    spacing = _uniform_spacing(coordinates)
    period = _positive_scalar(period, "period")
    _rotor_domain(coordinates, spacing, period)
    minimum = _positive_scalar(minimum_overlap, "minimum_overlap")
    if minimum > 1:
        raise ValueError("minimum_overlap must be at most 1")
    if n_points is None:
        n_points = count
    if isinstance(n_points, (bool, np.bool_)) or not isinstance(n_points, Integral) or n_points < count:
        raise ValueError("n_points must be an integer at least as large as the scan")

    ground = _real_vector(ground_potential_hartree, "ground potential")
    excited = _real_vector(excited_potential_hartree, "excited potential")
    dipoles = _real_array(transition_dipole_au, "transition dipoles")
    if ground.shape != (count,) or excited.shape != (count,) or dipoles.shape != (count, 3):
        raise ValueError("the scan needs two energies and a Cartesian transition dipole per grid point")
    ground_phases, ground_sign = _scan_phases(ground_overlaps, count, minimum, "ground")
    excited_phases, excited_sign = _scan_phases(excited_overlaps, count, minimum, "excited")
    dipole_sign = ground_sign * excited_sign
    dipoles = (ground_phases * excited_phases)[:, None] * dipoles

    if n_points != count:
        coordinates = coordinates[0] + (period / n_points) * np.arange(n_points)
        ground = resample_periodic(ground, n_points)
        excited = resample_periodic(excited, n_points)
        if dipole_sign == -1:
            dipoles = resample_periodic(np.concatenate((dipoles, -dipoles)), 2 * n_points)[:n_points].copy()
        else:
            dipoles = resample_periodic(dipoles, n_points)

    for array in (coordinates, ground, excited, dipoles, ground_phases, excited_phases):
        array.flags.writeable = False
    transition = PhasedRotorDipole(coordinates, dipoles, dipole_sign)
    return PhasedPeriodicScan(
        coordinates, period, ground, excited, transition, ground_phases, excited_phases, ground_sign, excited_sign,
    )


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


def evaluate_rotor_samples(
    source_grid: ArrayLike,
    samples: ArrayLike,
    target_grid: ArrayLike,
    *,
    period: float = 2 * np.pi,
    closure_sign: int = 1,
) -> NDArray[np.float64]:
    """Evaluate real periodic or antiperiodic samples at arbitrary angles.

    ``source_grid`` is uniform over one period without its repeated endpoint.
    The first axis of ``samples`` corresponds to this grid. ``target_grid``
    contains angles in the same coordinate and may start at a different
    origin or cross the source seam. For a translated angular coordinate,
    subtract its offset before calling. The caller supplies the electronic
    phase of a signed observable at the first source sample.

    With ``closure_sign=-1``, a shift by one period reverses the result and a
    shift by two periods leaves it unchanged. The even sampling Nyquist term
    uses the real cosine convention relative to the source origin.
    """
    origin_grid = _real_vector(source_grid, "source rotor grid")
    points = _real_vector(target_grid, "target rotor grid")
    if len(origin_grid) < 3 or not len(points):
        raise ValueError("supply at least three source rotor angles and one target angle")
    period = _positive_scalar(period, "period")
    _rotor_domain(origin_grid, _uniform_spacing(origin_grid), period)
    data = _real_array(samples, "rotor samples")
    if data.ndim == 0 or data.shape[0] != len(origin_grid):
        raise ValueError("the first sample axis must match the source rotor grid")
    if (isinstance(closure_sign, (bool, np.bool_)) or not isinstance(closure_sign, Integral)
            or closure_sign not in (-1, 1)):
        raise ValueError("closure_sign must be +1 or -1")
    if closure_sign < 0:
        data = np.concatenate((data, -data))
        period *= 2
    coefficients = np.fft.rfft(data, axis=0, norm="forward")
    # A real function has a conjugate partner for each positive mode except
    # the unique even-grid Nyquist cosine (and the zero mode).
    paired_stop = -1 if len(data) % 2 == 0 else None
    coefficients[1:paired_stop] *= 2
    angle = 2 * np.pi * np.remainder((points - origin_grid[0]) / period, 1)
    waves = np.exp(1j * np.outer(angle, np.arange(len(coefficients))))
    values = (waves @ coefficients.reshape(len(coefficients), -1)).real
    return values.reshape((len(points), *data.shape[1:]))
