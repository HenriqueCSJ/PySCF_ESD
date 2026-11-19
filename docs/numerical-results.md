# Numerical results

PySCF ESD writes assigned transitions, radiative observables, relaxation energies and convergence information. Read `diagnostics.json` first, then `properties.json` and the CSV filenames listed in the diagnostics. The [comparison guide](spectral-comparison.md) covers plotting, broadening and energy alignment.

## Read an existing calculation

This reader inspects a completed or partially completed run without starting an electronic calculation.

```python
import json
from pathlib import Path

folder = Path("run.esd")
diagnostics = json.loads((folder / "diagnostics.json").read_text(encoding="utf-8"))
print("Run status", diagnostics["status"])
for process, files in diagnostics.get("spectra", {}).items():
    print(process, files["coverage_status"], files["sticks"])

properties_path = folder / diagnostics.get("physical_properties", "properties.json")
if properties_path.exists():
    report = json.loads(properties_path.read_text(encoding="utf-8"))
    for process, spectrum in report["spectra"].items():
        for name, moment in spectrum.get("positive_photon_moments", {}).items():
            print(process, name, moment["status"], moment.get("coverage_lower_bound"))
```

A run can stop before a usable spectrum exists and write only `diagnostics.json`. Its failure information explains which stage stopped. An incomplete result can retain assigned lines and their physical sums. The convergence status determines whether those sums have reached the requested coverage.

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
rate_status = emission.get("positive_photon_moments", {}).get("M3", {}).get("status", "unavailable")
print("Radiative rate convergence", rate_status)
if rate_status == "complete":
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

Check `diagnostics.json` for the effective method, state tracking, derivatives and convergence. Each spectrum summary reports coverage for the initial population and final dipole strength, plus separate `positive_photon_moments` checks for absorption strength (`M1`) or emission rate and power (`M3`, `M4`). Each moment sums population times squared dipole times the corresponding power of positive photon energy in Hartree. The harmonic checks report the retained sum, upper bounds for omitted initial and final states, and a lower bound on the fraction captured. The `status` can be `complete`, `incomplete`, or `unassessed`. A reported rate always comes from the retained sticks and their original populations. Interpret its reciprocal using the `M3` status. An unsuccessful run with no usable spectrum may write only `diagnostics.json`.

## Get every line from CSV

`absorption_sticks.csv` and `emission_sticks.csv` each contain one row per retained transition. Partial files carry `_partial` before `_sticks.csv` or `_profile.csv`. The stick columns include:

- `energy_hartree` and `wavenumber_cm^-1`. Multiply Hartree by 27.211386245981 to get eV.
- `population`, `dipole_strength_au^2`, and the Cartesian `transition_dipole_*_au` components where available.
- `oscillator_strength` and `population_weighted_oscillator_strength` for absorption.
- `vacuum_einstein_a_s^-1` for the intrinsic emission line rate, `population_weighted_vacuum_einstein_a_s^-1` for its contribution to $k_f$, and `population_weighted_vacuum_radiant_power_w` for its power contribution.
- `initial_assignment_quanta` and `final_assignment_quanta` or the DVR and spectator columns. The JSON report's `assignment_scheme` identifies the convention. VCI labels can describe the dominant harmonic components of mixed states.
- `condon_strength_au2`, `ht_strength_au2`, and `condon_ht_interference_au2` where the solver provides a decomposition.

Sum the population-weighted Einstein coefficients over the emission CSV to recover the total rate. To get one initial level's rate, sum its intrinsic Einstein coefficients. The standard Gaussian profile CSVs contain relative photon-weighted densities per cm⁻¹. `--sigma 0` still produces sticks and their numerical report. Integrated rates and oscillator strengths come from the sticks and are independent of display broadening.

The `population` column repeats the initial-state population on each line from that state. Sum each distinct initial population once when measuring represented population. Use the population-weighted rate columns directly when measuring the total emission rate.

```python
import csv
import json
import math
from pathlib import Path

folder = Path("run.esd")
diagnostics = json.loads((folder / "diagnostics.json").read_text(encoding="utf-8"))
report = json.loads((folder / "properties.json").read_text(encoding="utf-8"))
filename = diagnostics["spectra"]["emission"]["sticks"]
with (folder / filename).open(encoding="utf-8", newline="") as stream:
    rows = list(csv.DictReader(stream))
k_r = math.fsum(float(row["population_weighted_vacuum_einstein_a_s^-1"]) for row in rows)
expected = report["spectra"]["emission"]["vacuum_radiative_rate_s_inverse"]
assert math.isclose(k_r, expected, rel_tol=1e-12, abs_tol=1e-15)
print("Retained radiative rate (s^-1)", k_r)
```

The diagnostics supply partial filenames when appropriate. Empty CSV cells represent quantities outside the selected line representation. JSON `null` represents an unavailable or undefined value.

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

For prepared AH phosphorescence, `result.physical_properties` gives the vacuum radiative rates and lifetimes, per-sublevel 0–0 origins, dominant lines, and the Condon and Herzberg–Teller terms when present. The [phosphorescence guide](phosphorescence.md) shows how to obtain both the vertical electronic rates and the vibronic results.

## Thermal profiles and prepared rotors

For `thermal_harmonic_spectrum` or `thermal_separable_spectrum`, use `positive_photon_weighted_strength`. It is the integral of the positive-photon-energy signal weighted by $E$ for absorption and $E^3$ for emission, with $E$ in Hartree. Multiply by $2/3$ for total oscillator strength or by $4\alpha^3/(3t_{\mathrm{au}})$ for the vacuum radiative rate. The same factors convert the returned profile density to physical units.

If the experimental spectrum has already been reduced by its photon-energy factor, call `thermal_harmonic_spectrum(..., include_dipole_strength_density=True)` and plot `dipole_strength_density_per_cm_inverse`. This profile uses the same Gaussian width and omits the photon-energy factor before broadening. Use `positive_photon_weighted_strength` for the physical rate.

```python
def thermal_rates(emission_profile):
    if emission_profile.process != "emission":
        raise ValueError("supply an emission profile")
    alpha = 7.2973525643e-3
    atomic_time_s = 2.4188843265864e-17
    k_f = 4 * alpha**3 / (3 * atomic_time_s) * emission_profile.positive_photon_weighted_strength
    return k_f, (1 / k_f if k_f > 0 else None)
```

For the equal-population spin–orbit example, each `profiles` entry is one polarization and already includes the triplet population weights in its amplitudes. Compute `rates_xyz = [thermal_rates(p)[0] for p in profiles]` and `k_r = sum(rates_xyz)`. The total radiative lifetime is `1 / k_r` for a positive rate. If Cartesian z is perpendicular to the molecular plane, `rates_xyz[2] / k_r` gives the perpendicular fraction. The [77 K benchmark data](phosphorescence.md#dielectric-and-linewidth-controls) include the native-energy profile for each component.

The [prepared rotor assembler](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/examples/assemble_separable_rotor.py) already applies these factors. In `spectra_summary.json`, `emission.strength_sum` is $k_f$ in s⁻¹ and `absorption.strength_sum` is the integrated oscillator strength. Take the reciprocal of a positive emission rate for the lifetime. Its CSV files give absolute photon-rate density for emission and oscillator-strength density and cross section for absorption. `profile_area` checks the integral on the output grid. Use the native-energy numerical report for absolute rates when plots have been shifted or peak-normalized.

## Physical definitions and units

For a positive photon energy $E_{if}$ in Hartree, initial vibrational population $p_i$ and Cartesian transition amplitudes $\mu_{if,c}$ in atomic units, the line dipole strength is

$$
S_{if}=\sum_{c=x,y,z}|\mu_{if,c}|^2.
$$

The oscillator strength, vacuum Einstein coefficient and population-weighted radiant power are

$$
f_{if}=\frac{2}{3}E_{if}S_{if},\qquad
A_{if}=\frac{4\alpha^3}{3t_{\mathrm{au}}}E_{if}^3S_{if},\qquad
P_{if}=p_i A_{if}E_{if}E_{\mathrm h}.
$$

Here $\alpha=7.2973525643\times10^{-3}$ is the fine-structure constant, $t_{\mathrm{au}}=2.4188843265864\times10^{-17}$ s is the atomic unit of time and $E_{\mathrm h}=4.3597447222060\times10^{-18}$ J is the Hartree energy. $f_{if}$ is dimensionless, $A_{if}$ is in s⁻¹ and $P_{if}$ is in watts per excited molecule. The code sums the three Cartesian components for the modeled molecular transition. Additional degeneracy factors belong to the chosen state/population model. [Electric-dipole line-strength relations](https://physics.nist.gov/Pubs/AtSpec/node17.html)

The unscaled moments stored by the spectral checks are

$$
M_n=\sum_{i,f\,;\,E_{if}>0}p_i E_{if}^nS_{if}.
$$

Thus $(2/3)M_1$ gives integrated thermal oscillator strength, $[4\alpha^3/(3t_{\mathrm{au}})]M_3$ gives the vacuum rate, and an additional Hartree-to-joule factor converts the corresponding $M_4$ into radiant power.

The harmonic `positive_photon_moments` entries contain `retained`, `omitted_initial_upper_bound`, `omitted_final_upper_bound`, `coverage_lower_bound` and `status`. For retained moment $M_n$ and omitted upper bounds $B_{n,i}$ and $B_{n,f}$,

$$
C_n=\frac{M_n}{M_n+B_{n,i}+B_{n,f}}.
$$

Exact zero moments with zero omission bounds have full coverage. A report is `complete` when the available lower bound reaches the requested target, `incomplete` when it falls below that target, and `unassessed` when no qualified bound or target is available. Rates are calculated from retained lines with the original populations. A basis can capture almost all unweighted dipole strength while still losing an important photon-weighted tail.

## Resolve electronic relaxation and reorganization

Let $R_0$ and $R_1$ be the optimized ground and excited geometries, with total electronic energies $E_0(R)$ and $E_1(R)$. The report uses

$$
\begin{aligned}
E_{\mathrm{abs}}^{\mathrm{vert}}&=E_1(R_0)-E_0(R_0),\\
E_{\mathrm{em}}^{\mathrm{vert}}&=E_1(R_1)-E_0(R_1),\\
\Delta E_{\mathrm{ad}}&=E_1(R_1)-E_0(R_0),\\
\lambda_1&=E_1(R_0)-E_1(R_1),\\
\lambda_0&=E_0(R_1)-E_0(R_0).
\end{aligned}
$$

The vertical gap difference is $\lambda_0+\lambda_1$. The harmonic origin additionally includes half the difference of the two sums of vibrational quanta. The direct deformation terms and local quadratic reorganization descriptors are both available for assessing the harmonic approximation.

Read `electronic_relaxation.energy_differences_hartree` for `vertical_absorption`, `vertical_emission`, `adiabatic_electronic_gap`, `excited_relaxation_from_ground`, `ground_deformation_at_excited` and `vertical_absorption_minus_emission`. The corresponding `energy_differences_cm^-1` has the same keys. Multiply a Hartree difference by `27.211386245981` for eV.

For each mode, the report's displacement-based Huang–Rhys factor and quadratic reorganization energy use $S_k=\omega_k K_k^2/2$ and $\lambda_k=\omega_k S_k$ in atomic units. Duschinsky mixing and HT amplitudes determine the complete spectral intensities. Mode fractions are weighted by oscillator strength for absorption or photon rate for emission. Combination bands can contribute to several fractions, whose sum may exceed one.

## Interpret intensity decomposition and spectral moments

The line amplitude is the coherent sum $\boldsymbol\mu_{if}=\boldsymbol\mu_{if}^{\mathrm C}+\boldsymbol\mu_{if}^{\mathrm{HT}}$. Its squared norm is

$$
S_{if}=|\boldsymbol\mu_{if}^{\mathrm C}|^2+
|\boldsymbol\mu_{if}^{\mathrm{HT}}|^2+
2\operatorname{Re}\!\left[(\boldsymbol\mu_{if}^{\mathrm C})^*\cdot
\boldsymbol\mu_{if}^{\mathrm{HT}}\right].
$$

The interference term can be negative. `fractions_of_total_signal` divides these signed photon-weighted contributions by the full coherent signal. The Condon/HT decomposition depends on the expansion origin, which is recorded alongside it.

For line weights $w_{if}=p_i f_{if}$ in absorption or $w_{if}=p_i A_{if}$ in emission, the centroid and standard deviation use

$$
\bar E=\frac{\sum w_{if}E_{if}}{\sum w_{if}},\qquad
\sigma_E=\sqrt{\frac{\sum w_{if}(E_{if}-\bar E)^2}{\sum w_{if}}}.
$$

These describe retained sticks. The `gaussian_profiles` entries additionally record the sampled maximum and a local quadratic estimate of its position. A Gaussian profile maximum, strongest stick, centroid and 0–0 origin are different observables. Compare like quantities when reporting a Stokes shift or an electronic-structure error.
