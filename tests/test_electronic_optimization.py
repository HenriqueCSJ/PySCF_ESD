"""Analytic synthetic surfaces for the tracked Cartesian optimizer."""

from collections import OrderedDict
from types import SimpleNamespace
import unittest

import numpy as np

from pyscf_esd.cpu import ElectronicConvergenceError, UnstableReferenceError
from pyscf_esd.electronic_workflow import (
    _ElectronicSession, _TrackedPoint, _atomic_hessian, _cartesian_frame, _check_ht_zero_point, _estimate_curvature,
    _estimate_dipole, _find_minima, _optimize,
)
from pyscf_esd.state_tracking import StateTrackingError, track_state


def _tracked_surface(potential):
    def evaluate(coordinates, previous=None):
        energy, gradient = potential(coordinates)

        def excited_gradient(root):
            if root != 1:
                raise AssertionError("analytic gradient requested for the wrong root")
            return gradient

        point = SimpleNamespace(
            ground_energy_hartree=energy - 0.2,
            excitation_energies_hartree=np.array([0.1, 0.2]),
            excited_gradient=excited_gradient,
        )
        return _TrackedPoint(coordinates, point, 1, -1)

    return evaluate


def _diatomic(distance):
    return np.array([[0., 0., -distance / 2], [0., 0., distance / 2]])


def _stretch(coordinates):
    bond = coordinates[1] - coordinates[0]
    length = np.linalg.norm(bond)
    return length, np.stack((-bond, bond)) / length


class ElectronicOptimizationTests(unittest.TestCase):
    def test_zero_point_dipole_check_rejects_saturated_intensity_borrowing(self):
        coordinates = _diatomic(2.)
        modes = np.array([[[0., 0., -.1], [0., 0., .1]]])
        frequency = np.array([.0024])
        derivative = np.array([[.05, 0., 0.]])
        center_dipole = np.array([0., .01, 0.])
        center = SimpleNamespace(
            coordinates=coordinates, root=0, dipole=center_dipole,
            point=SimpleNamespace(excitation_energies_hartree=np.array([.17, .171])),
        )
        calls = []

        def displaced(geometry, anchor):
            self.assertIs(anchor, center)
            normal = np.vdot(geometry - coordinates, modes[0]) / np.vdot(modes[0], modes[0])
            calls.append(normal)
            change = derivative[0] * normal * saturation
            return SimpleNamespace(dipole=center_dipole + change)

        session = SimpleNamespace(isolated_point=displaced)
        saturation = 1.
        self.assertEqual(_check_ht_zero_point(session, center, modes, frequency, derivative)["status"], "passed")
        np.testing.assert_allclose(calls, np.array([-1., 1.]) / np.sqrt(2 * frequency))
        saturation = .6
        with self.assertRaisesRegex(ElectronicConvergenceError, "linear HT fails"):
            _check_ht_zero_point(session, center, modes, frequency, derivative)
        center.point.excitation_energies_hartree = np.array([.17, .19])
        calls.clear()
        self.assertEqual(
            _check_ht_zero_point(session, center, modes, frequency, derivative)["status"], "not_triggered"
        )
        self.assertFalse(calls)

    def test_ground_hessian_seed_reduces_steps_on_same_surface(self):
        stiffness = 6.

        def harmonic(coordinates):
            length, axis = _stretch(coordinates)
            shift = length - 1.75
            return 0.5 * stiffness * shift**2, stiffness * shift * axis

        evaluate = _tracked_surface(harmonic)
        initial = evaluate(_diatomic(2.))
        _, axis = _stretch(initial.coordinates)
        ground_hessian = _atomic_hessian(stiffness * np.outer(axis.ravel(), axis.ravel()))
        seeded, stats = _optimize(evaluate, initial, np.ones(2), "S1", initial_hessian=ground_hessian)
        unseeded, baseline = _optimize(evaluate, initial, np.ones(2), "S1")

        self.assertEqual(stats["bfgs_iterations"], 1)
        self.assertEqual(stats["optimization_trial_evaluations"], 1)
        self.assertGreater(baseline["bfgs_iterations"], 1)
        np.testing.assert_allclose(seeded.coordinates, unseeded.coordinates, atol=1e-9)
        self.assertEqual((seeded.root, seeded.phase), (1, -1))

    def test_failed_quasi_newton_track_restarts_from_accepted_point(self):
        geometry = np.array([[-0.8, .05, 0.], [.8, .05, 0.], [.15, 1.2, 0.]])
        pairs = [(0, 1), (1, 2), (0, 2)]
        targets = [np.linalg.norm(geometry[j] - geometry[i]) * scale
                   for (i, j), scale in zip(pairs, [.97, 1.05, .99])]

        def valence_potential(coordinates):
            energy = 0.
            gradient = np.zeros_like(coordinates)
            for (first, second), target in zip(pairs, targets):
                length, axis = _stretch(coordinates[[first, second]])
                shift = length - target
                energy += 0.5 * shift**2
                gradient[[first, second]] += shift * axis
            return energy, gradient

        surface = _tracked_surface(valence_potential)
        initial = surface(geometry)
        gradient = initial.total_gradient().ravel()
        steepest = -gradient / np.linalg.norm(gradient)
        frame = _cartesian_frame(geometry)
        curvature = _atomic_hessian((frame * [.2, 3., 8.]) @ frame.T)
        first_accepted = []

        def evaluate(coordinates, previous):
            if previous is initial:
                step = (coordinates - geometry).ravel()
                if step @ steepest < .99 * np.linalg.norm(step):
                    raise StateTrackingError("test state is ambiguous along the quasi-Newton direction")
                first_accepted.append(coordinates)
            return surface(coordinates, previous)

        final, stats = _optimize(evaluate, initial, np.ones(3), "S1", initial_hessian=curvature)

        self.assertEqual(len(first_accepted), 1)
        self.assertGreater(stats["optimization_electronic_rejections"], 0)
        self.assertEqual(stats["optimization_steepest_descent_restarts"], 1)
        self.assertLess(final.energy, 1e-10)

    def test_unresolved_tracking_keeps_the_failure_and_gradient(self):
        def harmonic(coordinates):
            length, axis = _stretch(coordinates)
            shift = length - 1.75
            return 0.5 * shift**2, shift * axis

        initial = _tracked_surface(harmonic)(_diatomic(2.))

        def evaluate(coordinates, previous):
            self.assertIs(previous, initial)
            raise StateTrackingError("the target root is ambiguous")

        with self.assertRaisesRegex(ElectronicConvergenceError, r"0\.25 Hartree/bohr.*target root is ambiguous"):
            _optimize(evaluate, initial, np.ones(2), "S1")

    def test_terminal_accepted_step_is_checked_for_convergence(self):
        def rounded_linear(coordinates):
            length, axis = _stretch(coordinates)
            shift = length - 1.
            width = 0.001
            slope = 0.01
            if abs(shift) <= width:
                return 0.5 * slope * shift**2 / width, slope * shift / width * axis
            return slope * (abs(shift) - width / 2), slope * np.sign(shift) * axis

        evaluate = _tracked_surface(rounded_linear)
        initial = evaluate(_diatomic(3.4))
        final, stats = _optimize(evaluate, initial, np.ones(2), "S1")

        self.assertEqual(stats["bfgs_iterations"], 120)
        np.testing.assert_allclose(final.coordinates, _diatomic(1.), atol=1e-10)

    def test_stationary_reference_is_qualified_before_curvature(self):
        evaluate = _tracked_surface(lambda coordinates: (0., np.zeros_like(coordinates)))
        initial = evaluate(_diatomic(2.))

        def unstable(point):
            self.assertIs(point, initial)
            raise UnstableReferenceError("restricted_to_unrestricted", "stationary reference")

        def curvature(point):
            self.fail("curvature evaluated on an unstable stationary reference")

        with self.assertRaises(UnstableReferenceError) as error:
            _find_minima(
                evaluate, curvature, initial, np.ones(2), "S1", qualify_stationary=unstable,
            )
        self.assertEqual(error.exception.channel, "restricted_to_unrestricted")

    def test_spectrum_probes_cannot_use_another_adiabatic_root(self):
        # The lower root is stationary with negative curvature here. Farther
        # probes can follow the same character onto the upper root and fit a
        # positive curvature with small errors on both gradient and energy.
        center_shift = -6e-5
        coupling = abs(center_shift) * np.sqrt(1e-5)
        stiffness = 0.1
        linear_slope = center_shift / np.hypot(center_shift, coupling)

        def electronic(coordinates):
            length, axis = _stretch(coordinates)
            shift = length - 2
            offset = shift - center_shift
            common_energy = 1 + linear_slope * offset + 0.5 * stiffness * offset**2
            energies, vectors = np.linalg.eigh(
                [[common_energy + shift, coupling], [coupling, common_energy - shift]]
            )

            def gradient(root):
                root_slope = (2 * root - 1) * shift / np.hypot(shift, coupling)
                return (linear_slope + stiffness * offset + root_slope) * axis

            return SimpleNamespace(
                ground_energy_hartree=0., excitation_energies_hartree=energies,
                amplitudes=vectors.T[:, None, :], excited_gradient=gradient,
                transition_dipoles_au=np.array([[0., 0., 0.1], [0., 0., 0.4]]),
            )

        coordinates = _diatomic(2 + center_shift)
        center = _TrackedPoint(coordinates, electronic(coordinates), 0, 1)
        session = _ElectronicSession.__new__(_ElectronicSession)
        session.tracked_cache = OrderedDict()
        session.anchors = [center]
        session.initial_coordinates = coordinates
        session.root_dimension = session.root_search_limit = 2
        session.tracked_cache_hits = 0
        session.overlaps, session.gaps, session.visited_roots = [], [], set()
        session._calculate = lambda geometry, count, seed: electronic(geometry)

        def match(previous, candidate):
            return track_state(
                previous.phase * previous.point.amplitudes[previous.root],
                candidate.amplitudes, np.eye(1), np.eye(2),
                candidate_energies_hartree=candidate.excitation_energies_hartree,
                minimum_gap_hartree=1e-4,
            )

        session._match = match
        self.assertLess(np.linalg.norm(center.total_gradient()), 1e-10)
        with self.assertRaisesRegex(ElectronicConvergenceError, "energy rank"):
            _estimate_curvature(session, center)
        with self.assertRaisesRegex(ElectronicConvergenceError, "energy rank"):
            _estimate_dipole(session, center, _stretch(coordinates)[1][None, :, :])
        with self.assertRaisesRegex(StateTrackingError, "energy rank"):
            session.evaluate(_diatomic(2.02))
