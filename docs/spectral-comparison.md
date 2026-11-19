# Plotting and comparison with experiment

Use the retained transition energies and physical strengths as the starting point. Broadening describes the displayed line profile, energy alignment tests the envelope on a common origin, and normalization changes the plotted scale. Record each operation in the caption and retain the original numerical report. The [spectra gallery](spectra-gallery.md) gives the conditions used for the molecular comparisons.

## Choose the observable

The generic CLI exports oscillator strengths and vacuum Einstein coefficients in the stick CSV. Its `*_profile.csv` contains a photon-weighted density in relative units per cm⁻¹. The thermal rotor assembler exports absolute oscillator-strength density, cross section or photon-rate density in columns with those units in their names. Read the column header before combining files from different interfaces.

For absorption, decide whether the experimental ordinate is absorbance, molar absorptivity, molecular cross section, or an oscillator-strength density. For fluorescence and phosphorescence, decide whether the detector reports photon counts or optical energy. Photon-rate and power spectra differ by one factor of photon energy. Detector response, solvent, temperature, concentration and sample environment also define the comparison.

Use `dipole_strength_density_per_cm_inverse` from the thermal solver when the measured spectrum has been reduced by its photon-energy factor. The solver applies this reduction at the line-measure level. A division of the broadened signal by displayed photon energy generally gives a different profile.

## Use eV consistently

The electronic solver uses Hartree. The exported CLI abscissa uses cm⁻¹. Convert to eV with the constants in the package,

$$
E_{\mathrm{eV}}=27.211386245981\,E_{\mathrm h}
=\frac{\tilde\nu}{C},\qquad
C=\frac{219474.63136314}{27.211386245981}
\simeq8065.5440\ \mathrm{cm}^{-1}\mathrm{eV}^{-1}.
$$

A line's integrated oscillator strength or rate is unchanged by its coordinate label. A density changes through the Jacobian. For a density $I_{\tilde\nu}$ per cm⁻¹, the density per eV is

$$
I_E(E)=C\,I_{\tilde\nu}(CE),\qquad
\int I_E\,dE=\int I_{\tilde\nu}\,d\tilde\nu.
$$

For wavelength $\lambda$ in nm, $E_{\mathrm{eV}}=hc/\lambda$ with $hc\simeq1239.842$ eV nm. A photon-rate density transforms as

$$
I_\lambda(\lambda)=I_E(hc/\lambda)\frac{hc}{\lambda^2}.
$$

This nonlinear transformation can alter relative peak heights and envelope asymmetry. Sort the transformed energy coordinate before integrating or interpolating because increasing wavelength corresponds to decreasing energy.

A measured absorption cross section or molar absorptivity is a response value at each photon energy. Changing that curve's horizontal coordinate preserves its ordinate at the corresponding photon. Apply the Jacobian when transforming a density whose integral represents a fixed oscillator strength, count or rate. Identify which quantity the experimental file contains.

## Gaussian widths

The CLI's `--sigma` is a Gaussian standard deviation in cm⁻¹. The thermal profile interfaces take `fwhm_cm_inverse`. Their relation is

$$
\mathrm{FWHM}=2\sqrt{2\ln2}\,\sigma\simeq2.354820045\,\sigma.
$$

For FWHM 80 cm⁻¹, use `--sigma 33.9728720115`. In eV, 80 cm⁻¹ corresponds to FWHM approximately 0.009919 eV. The CLI default `--sigma 50` corresponds to FWHM approximately 117.741 cm⁻¹ or 0.014598 eV.

```sh
pyscf-esd molecule.xyz --model ah --temperature 298 --sigma 33.9728720115 --output ah-298.esd
```

Each Gaussian has unit area,

$$
g_\sigma(E-E_l)=\frac{\exp[-(E-E_l)^2/(2\sigma^2)]}{\sqrt{2\pi}\sigma},\qquad
I(E)=\sum_l w_l g_\sigma(E-E_l).
$$

$w_l$ is the integrated physical weight of line $l$. Apply the photon-energy factor at the original positive line energy before convolution. The standard exporter retains only the positive photon-energy transitions before it generates the broadened profile. A Gaussian has tails outside its line center, and a finite export grid can omit part of that area. Integrated rates from sticks avoid this plotting-grid dependence.

## Rebroaden existing sticks in eV

The following example reads an existing run and writes an absolute emission density in s⁻¹ eV⁻¹. It starts no electronic calculation and leaves the saved sticks and physical report intact. Set `process = "absorption"` to export oscillator-strength density per eV.

```python
import csv
import json
from pathlib import Path

import numpy as np
from pyscf_esd.spectral_properties import HARTREE_TO_EV

folder = Path("run.esd")
process = "emission"
fwhm_ev = 0.01
display_shift_ev = 0.0

diagnostics = json.loads((folder / "diagnostics.json").read_text(encoding="utf-8"))
entry = diagnostics["spectra"][process]
with (folder / entry["sticks"]).open(encoding="utf-8", newline="") as stream:
    rows = list(csv.DictReader(stream))

energy = HARTREE_TO_EV * np.array([float(row["energy_hartree"]) for row in rows])
column = (
    "population_weighted_vacuum_einstein_a_s^-1"
    if process == "emission" else "population_weighted_oscillator_strength"
)
weights = np.array([float(row[column]) for row in rows])
keep = weights > 0
energy, weights = energy[keep], weights[keep]
if energy.size == 0:
    raise ValueError("the retained spectrum has no positive signal")

sigma_ev = fwhm_ev / np.sqrt(8 * np.log(2))
centers = energy + display_shift_ev
axis = np.linspace(centers.min() - 8 * sigma_ev, centers.max() + 8 * sigma_ev, 6001)
density = np.zeros_like(axis)
for center, weight in zip(centers, weights):
    density += weight * np.exp(-0.5 * ((axis - center) / sigma_ev)**2)
density /= np.sqrt(2 * np.pi) * sigma_ev

ordinate = "photon_rate_density_s^-1_per_eV" if process == "emission" else "oscillator_strength_density_per_eV"
np.savetxt(
    folder / f"{process}_display_ev.csv",
    np.column_stack((axis, density)), delimiter=",",
    header=f"photon_energy_eV,{ordinate}", comments="",
)
print("Coverage status", entry["coverage_status"])
print("Integrated retained physical weight", weights.sum())
print("Display shift (eV)", display_shift_ev)
print("Gaussian FWHM (eV)", fwhm_ev)
```

Refine the plotting grid if a very narrow Gaussian spans too few points. The direct summation uses the true line centers. Keeping the absolute density also preserves the information needed for later normalization or integration.

## Energy shifts and normalization

For a comparison shift $\Delta$, use $E_l^{\mathrm{display}}=E_l+\Delta$. Preserve the physical line weights calculated at $E_l$. Recomputing the $E$ or $E^3$ factor at the display energy would change oscillator strengths and rates. A positive shift moves a band to higher photon energy.

Declare the alignment anchor, such as a measured 0–0 origin or an assigned band maximum, and give the shift in eV. A common shift of absorption and emission preserves the predicted peak separation. Independent shifts align the two envelopes separately and remove that separation as a test of the electronic model. Show or retain the native-energy spectra alongside the aligned comparison.

Peak normalization uses $I(E)/\max I$. Area normalization uses $I(E)/\int I(E)\,dE$. Peak normalization compares heights and shapes. Area normalization compares the distribution of integrated intensity. Keep the same normalization rule and spectral window across curves, and specify whether absorption and emission were normalized independently.

## Compare intensities and linewidths

Compare the same solvent, temperature, photon or power convention, energy axis and experimental preprocessing. Use identical declared broadening for direct method comparisons. When linewidths are adjusted for an experimental envelope, report the tested widths and keep the computed transitions available. The width is a presentation or environmental model parameter, and does not correct electronic energies, dipole derivatives or vibrational couplings.

Broadening can merge nearby lines, lower a narrow peak and transfer apparent peak height into a shoulder. A band-area ratio over specified energy intervals is often more stable than the ratio of sampled maxima. Check both when a red band or a narrow vibronic feature appears too strong. Reassess mode displacements, Duschinsky mixing, coherent HT terms, missing electronic transitions and the model's conformer/environment assumptions using the [numerical descriptors](numerical-results.md).

Compare interpolation-based errors over the shared measured range. State the normalization, alignment, linewidth, energy window and sampling used for an RMSE or correlation coefficient. A digitized trace has the resolution and uncertainty of its original plot. Keep the original experimental reference and digitization source with the comparison.

## Information to retain with a figure

- Molecule, charge, multiplicity, electronic state and conformer or ensemble model.
- Functional, basis, backend, solvent treatment and the derivative routes recorded in diagnostics.
- Vibrational model, temperature, Duschinsky and HT settings, and coverage or mesh checks.
- Horizontal units, ordinate definition and photon-count or power convention.
- Line-shape family, all width parameters, any comparison shift and its anchor, and normalization.
- Experimental source, environment and temperature, including any reduction or detector correction.

The [validation guide](validation.md) and [gallery captions](spectra-gallery.md) document those choices for the retained benchmark comparisons.
