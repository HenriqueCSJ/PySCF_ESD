"""Synthetic Hilbert-space fixtures for the physical-overlap calibration contract."""

from dataclasses import replace
import unittest

import numpy as np

from pyscf_esd.prepared_overlap_qualification import (
    PhysicalRootOverlap,
    RawRootBasis,
    qualify_prepared_overlap,
)
from pyscf_esd.prepared_state_tracking import track_prepared_states


class PreparedOverlapQualificationTests(unittest.TestCase):
    def setUp(self):
        self.source = RawRootBasis("geometry A", "raw A", (11, 3))
        self.destination = RawRootBasis("geometry B", "raw B", (8, 7, 4))
        self.vectors = {
            self.source: np.eye(3)[:, :2],
            self.destination: np.array([[-0.8, 0.36, 0.48], [0.0, 0.8, -0.6], [0.6, 0.48, 0.64]]),
        }
        self.reference = np.array([[-4 / 5, 9 / 25, 12 / 25], [0, 4 / 5, -3 / 5]])
        self.probes = {
            "forward": self.evaluate(self.source, self.destination),
            "reverse": self.evaluate(self.destination, self.source),
            "source_self": self.evaluate(self.source, self.source),
            "destination_self": self.evaluate(self.destination, self.destination),
            "source_flip": self.evaluate(self.source, self.destination, flip_source=11),
            "destination_flip": self.evaluate(self.source, self.destination, flip_destination=7),
            "independent_forward": PhysicalRootOverlap(
                self.source, self.destination, self.reference, "direct analytic projections", False,
            ),
        }

    def evaluate(self, source, destination, *, flip_source=None, flip_destination=None):
        left = self.vectors[source].copy()
        right = self.vectors[destination].copy()
        if flip_source is not None:
            left[:, source.root_order.index(flip_source)] *= -1
        if flip_destination is not None:
            right[:, destination.root_order.index(flip_destination)] *= -1
        return PhysicalRootOverlap(source, destination, left.T @ right, "synthetic contraction", False,
                                   flip_source, flip_destination)

    def qualify(self, **overrides):
        return qualify_prepared_overlap(self.source, self.destination, **(self.probes | overrides))

    def test_qualified_raw_overlap_tracks_root_order_and_sign(self):
        result = self.qualify()
        np.testing.assert_allclose(result, self.reference)
        self.assertFalse(result.flags.writeable)
        path = track_prepared_states([[-1.0, -0.7], [-1.2, -0.65, -0.5]], [result], initial_indices=(0, 1))
        np.testing.assert_array_equal(path.root_indices, [[0, 1], [0, 1]])
        np.testing.assert_array_equal(path.phases, [[1, 1], [-1, 1]])
        self.probes["forward"].matrix[0, 0] = 0.9
        np.testing.assert_allclose(result, self.reference)

    def test_rejects_wrong_axes_raw_checkpoint_or_matrix(self):
        forward = self.probes["forward"]
        mutations = (
            ("root order", replace(forward, source=replace(self.source, root_order=(3, 11)))),
            ("checkpoint identities", replace(forward, destination=replace(self.destination, checkpoint_id="other B"))),
            ("root counts", replace(forward, matrix=self.reference.T)),
            ("finite", replace(forward, matrix=np.full((2, 3), np.nan))),
            ("real rank-2", replace(forward, matrix=self.reference.astype(complex))),
            ("orthonormal states", replace(forward, matrix=np.full((2, 3), 0.7))),
            ("without phase transport", replace(forward, phase_transport_applied=True)),
            ("unexpected source root flip", replace(forward, flipped_source_root=11)),
        )
        for message, probe in mutations:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.qualify(forward=probe)

    def test_requires_cross_geometry_self_and_swapped_evaluations(self):
        for destination in (replace(self.destination, geometry_id=self.source.geometry_id),
                            replace(self.destination, checkpoint_id=self.source.checkpoint_id)):
            with self.subTest(destination=destination), self.assertRaisesRegex(ValueError, "distinct geometries"):
                qualify_prepared_overlap(self.source, destination, **self.probes)

        for name, wrong in (("source_self", np.diag([-1.0, 1.0])), ("destination_self", np.eye(3) * 0.9)):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "self overlap must be the identity"):
                self.qualify(**{name: replace(self.probes[name], matrix=wrong)})
        reverse = self.probes["reverse"]
        wrong = reverse.matrix.copy()
        wrong[0, :] *= -1
        with self.assertRaisesRegex(ValueError, "transpose"):
            self.qualify(reverse=replace(reverse, matrix=wrong))

    def test_rejects_non_discriminating_probes(self):
        square = RawRootBasis("geometry C", "raw C", (7, 4))
        self.vectors[square] = self.vectors[self.source].copy()
        with self.assertRaisesRegex(ValueError, "distinguishes the two orientations"):
            qualify_prepared_overlap(
                self.source, square,
                forward=self.evaluate(self.source, square), reverse=self.evaluate(square, self.source),
                source_self=self.probes["source_self"], destination_self=self.evaluate(square, square),
                source_flip=self.evaluate(self.source, square, flip_source=11),
                destination_flip=self.evaluate(self.source, square, flip_destination=4),
                independent_forward=PhysicalRootOverlap(self.source, square, np.eye(2), "analytic oracle", False),
            )

        zero_row = self.reference.copy()
        zero_row[0, :] = 0.0
        with self.assertRaisesRegex(ValueError, "detectable root flips"):
            self.qualify(forward=replace(self.probes["forward"], matrix=zero_row),
                         reverse=replace(self.probes["reverse"], matrix=zero_row.T))

    def test_detects_ignored_gauge_flip_or_sign_change_on_wrong_axis(self):
        source_flip = self.probes["source_flip"]
        destination_flip = self.probes["destination_flip"]
        with self.assertRaisesRegex(ValueError, "flip only the specified row"):
            self.qualify(source_flip=replace(source_flip, matrix=self.reference))
        with self.assertRaisesRegex(ValueError, "flip only the specified column"):
            self.qualify(destination_flip=replace(destination_flip, matrix=source_flip.matrix))
        with self.assertRaisesRegex(ValueError, "one explicit source root flip"):
            self.qualify(source_flip=replace(source_flip, flipped_source_root=True))

    def test_independent_oracle_detects_shared_cross_geometry_rescaling(self):
        biased = {
            name: replace(self.probes[name], matrix=self.probes[name].matrix * 0.8)
            for name in ("forward", "reverse", "source_flip", "destination_flip")
        }
        with self.assertRaisesRegex(ValueError, "disagrees with the independent physical forward overlap"):
            self.qualify(**biased)
        oracle = self.probes["independent_forward"]
        with self.assertRaisesRegex(ValueError, "from an independent evaluator"):
            self.qualify(independent_forward=replace(oracle, evaluator=self.probes["forward"].evaluator))
        with self.assertRaisesRegex(ValueError, "same candidate evaluator"):
            self.qualify(source_flip=replace(self.probes["source_flip"], evaluator=oracle.evaluator))
        with self.assertRaisesRegex(ValueError, "checkpoint identities"):
            self.qualify(independent_forward=replace(oracle, destination=replace(self.destination, geometry_id="C")))


if __name__ == "__main__":
    unittest.main()
