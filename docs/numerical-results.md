# Numerical results

## Get the rate and lifetime

Request emission or both directions. This example uses analytic GPU AH at 298 K.

```bash
pyscf-esd molecule.xyz --backend gpu --model ah --process both --temperature 298 --output run.esd
```

The terminal prints a short summary. To get precise values from an existing run, read `run.esd/properties.json`.

```python
import json
from pathlib import Path

report = json.loads(Path("run.esd/properties.json").read_text(encoding="utf-8"))
emission = report["spectra"].get("emission", {})
print("Emission coverage", emission.get("coverage_status", "unavailable"))
if emission.get("coverage_status") == "complete":
    print("k_f (s^-1)", emission["vacuum_radiative_rate_s_inverse"])
    lifetime = emission["vacuum_radiative_lifetime_s"]
    if lifetime is not None:
        print("Radiative lifetime (ns)", 1e9 * lifetime)
```

The reported vacuum rate $k_f=\sum_i p_i\sum_f A_{if}$ is per molecule in the modeled excited electronic state. Here $p_i$ is the initial vibrational population and $A_{if}$ is an Einstein coefficient for emission from level $i$ to level $f$. The radiative lifetime is the reciprocal of that initial population-averaged rate. A zero rate gives JSON `null` for the lifetime. In a mixture this reciprocal can differ from the decay time obtained by fitting a time-dependent fluorescence signal.

The rate uses the vacuum electromagnetic prefactor, including for energies and dipoles calculated with C-PCM. Observable fluorescence lifetimes and yields also depend on nonradiative channels and kinetics. For a single emitting state undergoing first-order decay, $\tau_{\mathrm{obs}}=1/(k_f+k_{\mathrm{nr}})$ and $\Phi_f=k_f\tau_{\mathrm{obs}}$. The rate report supplies the radiative contribution. [IUPAC radiative lifetime](https://goldbook.iupac.org/terms/view/R05064) · [IUPAC fluorescence lifetime](https://goldbook.iupac.org/terms/view/FT07377) · [IUPAC quantum yield](https://goldbook.iupac.org/terms/view/Q04991)

## What the report contains

Paths below start at the root of `properties.json` or `result.physical_properties`. Fields appear when the selected workflow supplies the underlying data.

- **Fluorescence.** `spectra.emission` contains `vacuum_radiative_rate_s_inverse`, `vacuum_radiative_lifetime_s`, and `vacuum_radiant_power_w_per_excited_molecule`. When initial levels can be assigned unambiguously, `emitting_vibrational_states` lists their populations, intrinsic rates and radiative lifetimes.
- **Absorption.** `spectra.absorption.population_weighted_oscillator_strength_sum` is the dimensionless integrated oscillator strength of the retained lines. Both directions report `population_weighted_dipole_strength_sum_au2`.
- **Positions and widths.** Each direction reports `centroid_energy` and `energy_standard_deviation` in Hartree, eV and cm⁻¹. These moments describe the stick distribution. `dominant_lines` gives the five largest weighted lines by default, including the photon energy and available assignments. `origin_00_line.photon_energy` appears when the solver identifies a unique 0–0 stick.
- **Stokes shifts.** `paired.absorption_minus_emission_centroid_energy` and `paired.absorption_minus_emission_strongest_stick_energy` use the sticks. The CLI also reports `gaussian_peak_stokes_shift_cm^-1` when both broadened profiles are complete. The latter depends on the linewidth.
- **Electronic energies.** `electronic_relaxation.electronic_energies_hartree` gives four energies at the two optimized geometries. `electronic_relaxation.energy_differences_cm^-1` gives vertical absorption, vertical emission, the adiabatic electronic gap, excited-state relaxation, and ground-state deformation. The harmonic origin and zero-point correction are `harmonic_reference.harmonic_00_cm^-1` and `harmonic_reference.harmonic_zpe_shift_cm^-1`.
- **Normal modes.** `harmonic_reference.ground_at_excited.modes` and `harmonic_reference.excited_at_ground.modes` give frequencies, displacements, Huang–Rhys factors, quadratic reorganization energies, and HT dipole scales where available. Find the Duschinsky matrix at `harmonic_reference.duschinsky_rows_excited_columns_ground` and the comparison of quadratic and direct deformation energies at `harmonic_cross_minimum_check`.
- **Intensity decomposition.** In each direction, `condon_herzberg_teller` gives the weighted Condon and HT contributions and their signed interference. `final_mode_statistics` gives the mean final occupation and signal fraction involving each harmonic mode or spectator. Combination lines can contribute to multiple modes.

Check `diagnostics.json` for the effective method, state tracking, derivatives and convergence. Each spectrum summary has `coverage_status`, `initial_population_coverage`, and `final_dipole_strength_coverage`. Totals use retained positive-energy sticks with their original populations. An incomplete result can contain a partial rate. Check the rate itself against the vibrational budget when comparing absolute values, since each emission line is weighted by the cube of its photon energy. An unsuccessful run with no usable spectrum may write only `diagnostics.json`.

## Get every line from CSV

`absorption_sticks.csv` and `emission_sticks.csv` each contain one row per retained transition. Partial files carry `_partial` before `_sticks.csv` or `_profile.csv`. The stick columns include:

- `energy_hartree` and `wavenumber_cm^-1`. Multiply Hartree by 27.211386245981 to get eV.
- `population`, `dipole_strength_au^2`, and the Cartesian `transition_dipole_*_au` components where available.
- `oscillator_strength` and `population_weighted_oscillator_strength` for absorption.
- `vacuum_einstein_a_s^-1` for the intrinsic emission line rate, `population_weighted_vacuum_einstein_a_s^-1` for its contribution to $k_f$, and `population_weighted_vacuum_radiant_power_w` for its power contribution.
- `initial_assignment_quanta` and `final_assignment_quanta` or the DVR and spectator columns. `assignment_scheme` identifies the convention. VCI labels can describe the dominant harmonic components of mixed states.
- `condon_strength_au2`, `ht_strength_au2`, and `condon_ht_interference_au2` where the solver provides a decomposition.

Sum the population-weighted Einstein coefficients over the emission CSV to recover the total rate. To get one initial level's rate, sum its intrinsic Einstein coefficients. The standard Gaussian profile CSVs contain relative photon-weighted densities per cm⁻¹. `--sigma 0` still produces sticks and their numerical report. Integrated rates and oscillator strengths come from the sticks and are independent of display broadening.

## Get the values from Python

`esd.run`, `esd.run_vibronic` and `esd.run_prepared_ah` return a result with `spectra`, `physical_properties`, and `diagnostics`. With an existing PySCF `Mole`:

```python
from pyscf import esd

result = esd.run(mol, backend="gpu", model="ah", process="both", temperature_kelvin=298)
report = result.physical_properties
emission = report["spectra"]["emission"]
k_f = emission["vacuum_radiative_rate_s_inverse"]
tau_rad_s = emission["vacuum_radiative_lifetime_s"]
f_total = report["spectra"]["absorption"]["population_weighted_oscillator_strength_sum"]

lines = esd.spectral_line_properties(result.spectra["emission"], "emission")
each_line_a = lines.einstein_a_s_inverse
each_line_contribution = lines.population_weighted_einstein_a_s_inverse
```

For low-level harmonic or anharmonic sticks, use `esd.summarize_spectral_properties({"emission": emission_sticks})`. Coverage is `unassessed` unless the caller supplies an assessed status. The in-memory report includes the stick properties. The CLI derives its Gaussian peak summaries when it generates the profiles.

The [phosphorescence guide](phosphorescence.md) documents the separate prepared spin–orbit result object, including electronic rates for individual triplet sublevels and their thermal average.

## Thermal profiles and prepared rotors

For `thermal_harmonic_spectrum` or `thermal_separable_spectrum`, use `positive_photon_weighted_strength`. It is the integral of the positive-photon-energy signal weighted by $E$ for absorption and $E^3$ for emission, with $E$ in Hartree. Multiply by $2/3$ for total oscillator strength or by $4\alpha^3/(3t_{\mathrm{au}})$ for the vacuum radiative rate. The same factors convert the returned profile density to physical units.

```python
def thermal_rates(emission_profile):
    if emission_profile.process != "emission":
        raise ValueError("supply an emission profile")
    alpha = 7.2973525643e-3
    atomic_time_s = 2.4188843265864e-17
    k_f = 4 * alpha**3 / (3 * atomic_time_s) * emission_profile.positive_photon_weighted_strength
    return k_f, (1 / k_f if k_f > 0 else None)
```

The [prepared rotor assembler](../examples/assemble_separable_rotor.py) already applies these factors. In `spectra_summary.json`, `emission.strength_sum` is $k_f$ in s⁻¹ and `absorption.strength_sum` is the integrated oscillator strength. Take the reciprocal of a positive emission rate for the lifetime. Its CSV files give absolute photon-rate density for emission and oscillator-strength density and cross section for absorption. `profile_area` checks the integral on the output grid. Use the native-energy numerical report for absolute rates when plots have been shifted or peak-normalized.
