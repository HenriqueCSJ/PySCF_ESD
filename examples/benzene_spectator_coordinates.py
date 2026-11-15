"""Build symmetry-adapted benzene spectator cuts from the archived analytic S0 Hessian."""

import argparse
import json
from pathlib import Path

import numpy as np

from validate_benzene_3d import CM_PER_HARTREE


AMU_TO_ELECTRON_MASS = 1822.888486209
WILSON_BY_ASCENDING_FREQUENCY = {
    "A1g": (2,), "A2g": (3,), "A2u": (11,),
    "B2g": (4, 5), "B1u": (12, 13), "B2u": (15, 14),
    "E1g": (10,), "E1u": (18, 19, 20), "E2g": (9, 7), "E2u": (16, 17),
}


def operations(geometry):
    """Return real Cartesian D6h representations in the archived atom order."""
    def operation(matrix):
        moved = geometry @ matrix.T
        distance = np.linalg.norm(moved[:, None, :] - geometry[None, :, :], axis=2)
        mapped = distance.argmin(axis=1)
        if len(set(mapped)) != 12 or distance[np.arange(12), mapped].max() > 1e-7:
            raise ValueError("reference does not have the archived D6h geometry")
        representation = np.zeros((36, 36))
        for source, target in enumerate(mapped):
            representation[3*target:3*target+3, 3*source:3*source+3] = matrix
        return representation

    result = {}
    for turn in range(6):
        angle = turn * np.pi / 3
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        for vertical in range(2):
            for horizontal in range(2):
                mirror = np.diag([1, (-1)**vertical, (-1)**horizontal])
                result[turn, vertical, horizontal] = operation(rotation @ mirror)
    return result


def projectors(group):
    irreps = {"A1": (0, 1), "A2": (0, -1), "B1": (3, 1), "B2": (3, -1),
              "E1": (1, None), "E2": (2, None)}
    for symbol, (harmonic, mirror_sign) in irreps.items():
        dimension = 2 if mirror_sign is None else 1
        for parity in ("g", "u"):
            # Inversion is C2 followed by reflection through the ring plane.
            horizontal_sign = (-1)**harmonic * (1 if parity == "g" else -1)
            projector = np.zeros((36, 36))
            for (turn, vertical, horizontal), operation in group.items():
                rotation = np.cos(harmonic * turn * np.pi / 3)
                # A vertical reflection is the corresponding C2' times σh.
                vertical_character = ((0 if vertical else 2) if dimension == 2
                                      else (mirror_sign * horizontal_sign)**vertical)
                projector += rotation * vertical_character * horizontal_sign**horizontal * operation
            yield symbol + parity, dimension, dimension * projector / len(group)


def build_coordinates(reference, hessian):
    positions = reference["positions_bohr"]
    masses = reference["masses_amu"]
    weights = np.sqrt(masses.repeat(3))
    group = operations(positions)
    hessian = hessian.transpose(0, 2, 1, 3).reshape(36, 36)
    dynamical = hessian / np.outer(weights, weights) / AMU_TO_ELECTRON_MASS
    average = sum(matrix @ dynamical @ matrix.T for matrix in group.values()) / len(group)
    mirror = group[0, 1, 0]
    inversion = group[3, 0, 1]
    rotation = group[1, 0, 0]
    irreps = {label: (dimension, projector) for label, dimension, projector in projectors(group)}
    archived_active = reference["mode_mass_weighted"].reshape(3, 36)
    breathing = irreps["A1g"][1] @ archived_active[0]
    breathing /= np.linalg.norm(breathing)
    even = archived_active[1:] @ irreps["E2g"][1]
    left, _, right = np.linalg.svd(even, full_matrices=False)
    active = np.vstack((breathing, left @ right))
    e2g_partner = ((rotation @ active[1:].T).T + 0.5 * active[1:]) / (np.sqrt(3) / 2)
    projected = np.vstack((active, e2g_partner))
    np.testing.assert_allclose(projected @ projected.T, np.eye(5), atol=3e-10, rtol=0)
    np.testing.assert_allclose((mirror @ e2g_partner.T).T, -e2g_partner, atol=3e-10, rtol=0)
    centered = positions - np.average(positions, weights=masses, axis=0)
    translation = np.broadcast_to(np.eye(3)[:, None, :], (3, 12, 3)).reshape(3, 36) * weights
    rotational = np.cross(np.eye(3)[:, None, :], centered).reshape(3, 36) * weights
    constraints = np.vstack((translation, rotational, projected))

    rows = []
    studied_partner = {6: (e2g_partner[0], 1), 8: (e2g_partner[1], 2)}
    for number, (vector, active_index) in studied_partner.items():
        # Preserve the archived E2g subspace and its shared coordinate scale.
        frequency = float(reference["frequencies_cm1"][active_index])
        rows.append((number, "E2g", 1, frequency, vector, True))
    for label, (multiplicity, projector) in irreps.items():
        if multiplicity == 2:
            # The odd partner changes sign under a vertical reflection.
            projector = projector @ ((np.eye(36) - mirror) / 2)
        value, vector = np.linalg.eigh(projector)
        irrep_space = vector[:, value > .5]
        _, singular, right = np.linalg.svd(constraints @ irrep_space, full_matrices=True)
        allowed = irrep_space @ right[int(np.count_nonzero(singular > 1e-8)):].T
        if not allowed.shape[1]:
            continue
        energy, vector = np.linalg.eigh(allowed.T @ average @ allowed)
        frequencies = np.sqrt(energy) * CM_PER_HARTREE
        modes = (allowed @ vector).T
        numbers = WILSON_BY_ASCENDING_FREQUENCY.get(label)
        if numbers is None or len(numbers) != len(frequencies) or np.any(energy <= 0):
            raise ValueError(f"unexpected {label} vibration subspace: {frequencies}")
        for number, frequency, mode in zip(numbers, frequencies, modes):
            sign_flip = next((op for op in (mirror, inversion, rotation, *group.values())
                              if np.linalg.norm(op @ mode + mode) < 1e-8), None)
            if (sign_flip is None) != (label == "A1g"):
                raise ValueError(f"unexpected sign symmetry for {label} ν{number}")
            signed_mode = mode * np.sign(mode[np.argmax(np.abs(mode))])
            rows.append((number, label, multiplicity, frequency, signed_mode, sign_flip is not None))

    rows.sort(key=lambda row: row[0])
    numbers, irreps, multiplicities, frequencies, vectors, symmetric = zip(*rows)
    modes = np.asarray(vectors)
    np.testing.assert_allclose(modes @ modes.T, np.eye(len(rows)), atol=3e-10, rtol=0)
    np.testing.assert_allclose(modes @ active.T, 0, atol=3e-10, rtol=0)
    if sum(multiplicities) != 27 or len(set(numbers)) != 19:
        raise ValueError("spectator cuts must represent all 27 omitted vibration coordinates")
    frequencies = np.asarray(frequencies)
    mass_scale = weights * np.sqrt(AMU_TO_ELECTRON_MASS)
    oscillator = modes / mass_scale / np.sqrt(frequencies[:, None] / CM_PER_HARTREE)
    labels = [f"nu{number}" + ("_partner" if number in (6, 8) else "") for number in numbers]
    archive = {
        "mode_vectors_bohr": oscillator.reshape(-1, 12, 3),
        "mode_vectors_mass_weighted": modes.reshape(-1, 12, 3),
        "q1_vector_bohr": (breathing / mass_scale /
                           np.sqrt(reference["frequencies_cm1"][0] / CM_PER_HARTREE)).reshape(12, 3),
        "archived_active_defect_norms": np.linalg.norm(active - archived_active, axis=1),
        "frequencies_cm1": frequencies, "multiplicities": multiplicities,
        "sign_symmetric": symmetric, "labels": labels, "irreps": irreps,
    }
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("center", type=Path, help="archived optimized geometry, masses and active modes")
    parser.add_argument("hessian", type=Path, help="archived analytic S0 Hessian")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with np.load(args.center) as reference:
        coordinates = build_coordinates(reference, np.load(args.hessian))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **coordinates)
    keys = ("labels", "irreps", "multiplicities", "frequencies_cm1", "sign_symmetric")
    metadata = [{key: np.asarray(coordinates[key])[i].item() for key in keys}
                for i in range(len(coordinates["labels"]))]
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
