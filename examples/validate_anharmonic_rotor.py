"""Check a two-surface hindered rotor against its exact ground transition."""

import numpy as np

from pyscf_esd import anharmonic_spectrum, resample_periodic, solve_dvr


def main():
    electronic_samples = 13
    rotor_points = 96
    theta = 2 * np.pi * np.arange(electronic_samples) / electronic_samples
    grid = 2 * np.pi * np.arange(rotor_points) / rotor_points
    inertia = 20_000.0  # electron mass * bohr**2
    barrier_shape = 2.0
    expected_gap = 0.12  # Hartree

    def potential(shape, origin):
        # For this potential the exact lowest state is exp(shape*cos(theta)).
        return origin + (shape**2 * np.sin(theta)**2 - shape * np.cos(theta)) / (2 * inertia)

    def levels(sampled_potential):
        return solve_dvr(grid, resample_periodic(sampled_potential, rotor_points),
                         boundary="periodic", n_levels=1, kinetic_coefficient=1 / inertia)

    ground = levels(potential(barrier_shape, 0.0))
    excited = levels(potential(-barrier_shape, expected_gap))
    sampled_mu = np.column_stack((np.ones_like(theta), np.cos(theta), np.sin(theta)))
    mu = resample_periodic(sampled_mu, rotor_points)
    absorption = anharmonic_spectrum(ground, excited, transition_dipole_au=mu)
    emission = anharmonic_spectrum(excited, ground, transition_dipole_au=mu, process="emission")
    # The product of the opposing ground-state envelopes is constant. The
    # cos/sin dipole components integrate to zero; the constant component
    # survives with overlap 1/I₀(2*barrier_shape).
    expected_strength = 1 / np.i0(2 * barrier_shape)**2
    for spectrum in (absorption, emission):
        np.testing.assert_allclose(spectrum.photon_energy_hartree, [expected_gap], atol=1e-12, rtol=0)
        np.testing.assert_allclose(spectrum.dipole_strength_au2, [expected_strength], atol=1e-12, rtol=0)
    print(f"0–0 photon energy (Hartree)  {absorption.photon_energy_hartree[0]:.12f}")
    print(f"0–0 dipole strength (au²)   {absorption.dipole_strength_au2[0]:.12f}")
    print(f"Analytic dipole strength    {expected_strength:.12f}")
    print(f"Samples per surface {electronic_samples}, nuclear grid points {rotor_points}")


if __name__ == "__main__":
    main()
