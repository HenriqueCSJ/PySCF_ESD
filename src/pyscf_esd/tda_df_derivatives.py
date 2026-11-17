"""Nuclear directional derivatives of molecular Coulomb-metric density fitting."""

from contextlib import ExitStack
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from numpy.typing import NDArray


_Array = NDArray[np.float64]
_JK = tuple[_Array | None, _Array | None]


@dataclass(frozen=True)
class DFJKDerivatives:
    """Fixed-density J and K matrices for each derivative order."""

    value: _JK
    first_x: _JK
    first_y: _JK | None
    mixed_xy: _JK | None


@dataclass(frozen=True)
class DFBilinearDerivatives:
    """Fixed-density Coulomb and exchange contractions for each derivative order."""

    value: tuple[float | None, float | None]
    first_x: tuple[float | None, float | None]
    first_y: tuple[float | None, float | None] | None
    mixed_xy: tuple[float | None, float | None] | None


def _real_array(value, name):
    array = np.asarray(value)
    if np.iscomplexobj(array) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite and real")
    return array.astype(float, copy=False)


def _ao_owners(mol):
    return np.repeat(mol._bas[:, 0], np.diff(mol.ao_loc_nr()))


def _owner_blocks(owners):
    boundaries = np.r_[0, np.flatnonzero(owners[1:] != owners[:-1]) + 1, owners.size]
    return [(int(owners[a]), slice(a, b)) for a, b in zip(boundaries[:-1], boundaries[1:])]


def _density_pair_weights(dm):
    i, j = np.tril_indices(dm.shape[-1])
    packed = dm[..., i, j] + dm[..., j, i]
    packed[..., i == j] *= 0.5
    return packed


def _lower_half_diagonal(matrix):
    lower = np.tril(matrix)
    lower[np.diag_indices_from(lower)] *= 0.5
    return lower


def _symmetric(matrix):
    return (matrix + matrix.T) * 0.5


class MolecularDFDerivatives:
    """Reuse the base DF integrals at one molecular geometry.

    Supply the native CPU PySCF DF object for the reference. The fit is
    reconstructed from its orbital and auxiliary basis in the full Coulomb
    metric. Cholesky-truncated or custom transformed fits require a matching
    differentiable representation. This implementation returns unscaled J
    and K for real AO densities, including nonsymmetric transition densities.
    """

    def __init__(self, with_df):
        from pyscf.df.addons import make_auxmol
        from pyscf.df.df import DF
        from scipy.linalg import cholesky

        if type(with_df) is not DF:
            raise TypeError("a native CPU PySCF molecular DF object is required")
        mol = with_df.mol
        auxmol = with_df.auxmol
        if auxmol is None:
            auxmol = make_auxmol(mol, with_df.auxbasis)
        if mol.cart != auxmol.cart or mol.natm != auxmol.natm or not np.array_equal(
            mol.atom_coords(unit="Bohr"), auxmol.atom_coords(unit="Bohr"),
        ):
            raise ValueError("the DF auxiliary basis must use the reference atoms and AO representation")
        if mol.omega != 0 or auxmol.omega != 0:
            raise ValueError("a full-range Coulomb metric is required")

        metric = auxmol.intor("int2c2e", hermi=1)
        try:
            self._lower = cholesky(metric, lower=True)
        except np.linalg.LinAlgError as error:
            raise ValueError("the full DF Coulomb metric must be positive definite") from error

        self._mol = mol
        self._auxmol = auxmol
        self._ao_owners = _ao_owners(mol)
        self._aux_owners = _ao_owners(auxmol)
        self._packed_indices = np.tril_indices(mol.nao)
        self._memory_bytes = int(with_df.max_memory * 1e6)
        # Limit the largest derivative integral block using the reference's
        # memory budget. The scalar packed DF factors are retained separately.
        self._aux_block_size = max(1, min(80, int(self._memory_bytes * 0.12 / (9 * 8 * mol.nao**2))))

    @cached_property
    def _basis_environment(self):
        from pyscf.gto import conc_env

        mol, auxmol = self._mol, self._auxmol
        return conc_env(mol._atm, mol._bas, mol._env, auxmol._atm, auxmol._bas, auxmol._env)

    @cached_property
    def _shell_blocks(self):
        from pyscf.ao2mo.outcore import balance_partition

        loc = self._auxmol.ao_loc_nr()
        return [
            ((0, self._mol.nbas, 0, self._mol.nbas, self._mol.nbas + sh0, self._mol.nbas + sh1),
             slice(loc[sh0], loc[sh1]))
            for sh0, sh1, _ in balance_partition(loc, self._aux_block_size)
        ]

    @cached_property
    def _three_center_optimizers(self):
        from pyscf.gto.moleintor import make_cintopt

        return {
            name: make_cintopt(*self._basis_environment, self._mol._add_suffix(name))
            for name in ("int3c2e", "int3c2e_ip1", "int3c2e_ip2", "int3c2e_ipip1", "int3c2e_ipvip1")
        }

    def _int3c(self, name, shells, aosym="s1"):
        from pyscf.gto.moleintor import getints

        return getints(
            self._mol._add_suffix(name), *self._basis_environment, shls_slice=shells,
            aosym=aosym, cintopt=self._three_center_optimizers[name],
        )

    @cached_property
    def _whitened_integrals(self):
        raw = np.empty((self._auxmol.nao, self._packed_indices[0].size))
        for shells, rows in self._shell_blocks:
            raw[rows] = self._int3c("int3c2e", shells, "s2ij").T
        return self._whiten(raw)

    @cached_property
    def _inverse_integrals(self):
        whitened = self._whitened_integrals
        inverse = np.empty_like(whitened)
        for first in range(0, inverse.shape[1], 1024):
            columns = slice(first, first + 1024)
            inverse[:, columns] = self._inverse_projected(whitened[:, columns])
        return inverse

    def _whiten(self, raw, corrections=()):
        from scipy.linalg import solve_triangular

        # Keep the triangular-solve buffer and the matrix products bounded
        # independently of the number of packed AO pairs.
        for first in range(0, raw.shape[1], 1024):
            columns = slice(first, first + 1024)
            raw[:, columns] = solve_triangular(self._lower, raw[:, columns], lower=True, check_finite=False)
            for factor, integral in corrections:
                raw[:, columns] -= factor @ integral[:, columns]
        return raw

    @cached_property
    def _metric_ip(self):
        return self._auxmol.intor("int2c2e_ip1")

    @cached_property
    def _metric_ipip(self):
        auxiliary = self._auxmol
        return auxiliary.intor("int2c2e_ipip1").reshape(3, 3, auxiliary.nao, auxiliary.nao)

    def _whiten_metric(self, matrix):
        from scipy.linalg import solve_triangular

        left = solve_triangular(self._lower, matrix, lower=True, check_finite=False)
        both = solve_triangular(self._lower, left.T, lower=True, check_finite=False)
        return _symmetric(both)

    def _metric_derivatives(self, x, y):
        aux_x = x[self._aux_owners]
        dx = aux_x[:, None] - aux_x[None, :]
        dm_x = -np.einsum("cpq,pqc->pq", self._metric_ip, dx)
        px = _lower_half_diagonal(self._whiten_metric(dm_x))
        if y is None:
            return px, None, None
        aux_y = y[self._aux_owners]
        dy = aux_y[:, None] - aux_y[None, :]
        dm_y = -np.einsum("cpq,pqc->pq", self._metric_ip, dy)
        dm_xy = np.einsum("cdpq,pqc,pqd->pq", self._metric_ipip, dx, dy)
        py = _lower_half_diagonal(self._whiten_metric(dm_y))
        product = px @ py.T
        pxy = _lower_half_diagonal(self._whiten_metric(dm_xy) - product - product.T)
        return px, py, pxy

    def _raw_integral_derivatives(self, x, y):
        shape = self._whitened_integrals.shape
        raw_x = np.empty(shape)
        raw_y = raw_xy = None
        if y is not None:
            raw_y, raw_xy = np.empty(shape), np.empty(shape)
        i, j = self._packed_indices
        ao_x = x[self._ao_owners]
        ao_y = None if y is None else y[self._ao_owners]
        nao = self._mol.nao
        for shells, rows in self._shell_blocks:
            centers = self._aux_owners[rows]
            # Translation invariance keeps the auxiliary center fixed by using
            # each AO center's displacement relative to that auxiliary center.
            dx = ao_x[:, None] - x[centers][None, :]
            first = self._int3c("int3c2e_ip1", shells)
            bx = -np.einsum("cijp,ipc->ijp", first, dx)
            raw_x[rows] = (bx[i, j] + bx[j, i]).T
            if y is None:
                continue
            dy = ao_y[:, None] - y[centers][None, :]
            by = -np.einsum("cijp,ipc->ijp", first, dy)
            raw_y[rows] = (by[i, j] + by[j, i]).T
            del first

            same_ao = self._int3c("int3c2e_ipip1", shells).reshape(3, 3, nao, nao, -1)
            bxy = np.einsum("cdijp,ipc,ipd->ijp", same_ao, dx, dy)
            del same_ao
            different_ao = self._int3c("int3c2e_ipvip1", shells).reshape(3, 3, nao, nao, -1)
            bxy += np.einsum("cdijp,ipc,jpd->ijp", different_ao, dx, dy)
            del different_ao
            raw_xy[rows] = (bxy[i, j] + bxy[j, i]).T
        return raw_x, raw_y, raw_xy

    def directional(self, direction_x, direction_y=None):
        """Prepare a reusable DF derivative jet for real (natm, 3) directions in Bohr."""
        x = _real_array(direction_x, "direction_x")
        y = None if direction_y is None else _real_array(direction_y, "direction_y")
        if x.shape != (self._mol.natm, 3) or (y is not None and y.shape != x.shape):
            raise ValueError("nuclear directions must have shape (natm, 3)")

        displaced = np.flatnonzero(np.any(x, axis=1))
        if y is None and displaced.size == 1:
            atom = displaced[0]
            return self._single_atom_derivative(atom, x[atom])

        # For M=LLᵀ and W=L⁻¹B, differentiate both triangular equations
        # M=LLᵀ and LW=B. Each derivative of L is normalized by L⁻¹.
        base = self._whitened_integrals
        px, py, pxy = self._metric_derivatives(x, y)
        raw_x, raw_y, raw_xy = self._raw_integral_derivatives(x, y)
        wx = self._whiten(raw_x, ((px, base),))
        if y is None:
            return DFDirectionalDerivatives(self._mol.nao, self._aux_block_size, (base, wx))
        wy = self._whiten(raw_y, ((py, base),))
        wxy = self._whiten(raw_xy, ((px, wy), (py, wx), (pxy, base)))
        return DFDirectionalDerivatives(self._mol.nao, self._aux_block_size, (base, wx, wy, wxy))

    def reference_jk(self, density, *, with_j=True, with_k=True):
        """Apply the reconstructed same-geometry DF fit to an AO density or batch."""
        return _apply_jk_products(
            density, self._mol.nao, self._aux_block_size, (self._whitened_integrals,),
            (((0, 0),),), with_j, with_k, use_low_rank=True,
        )[0]

    def _single_atom_derivative(self, atom, displacement):
        # With T=M⁻¹B and M_x a symmetric commutator between the auxiliary
        # atom projector and the antisymmetric int2c2e_ip1, d(Bᵀ M⁻¹ B)
        # is the symmetrized product of T and (B_x + projector * int2c2e_ip1 * T).
        # This avoids whitening the full derivative for every Cartesian axis.
        inverse = self._inverse_integrals
        derivative = np.zeros_like(inverse)
        mol = self._mol
        i, j = self._packed_indices
        ao_shell_start, ao_shell_end, ao_start, ao_end = mol.aoslice_by_atom()[atom]
        at_i, at_j = self._ao_owners[i] == atom, self._ao_owners[j] == atom
        if ao_start != ao_end:
            for shells, rows in self._shell_blocks:
                block = self._int3c("int3c2e_ip1", (ao_shell_start, ao_shell_end, *shells[2:]))
                partial = -np.einsum("cijp,c->pij", block, displacement)
                derivative[rows, at_i] += partial[:, i[at_i] - ao_start, j[at_i]]
                derivative[rows, at_j] += partial[:, j[at_j] - ao_start, i[at_j]]
        aux_shell_start, aux_shell_end, aux_start, aux_end = self._auxmol.aoslice_by_atom()[atom]
        if aux_start != aux_end:
            rows = slice(aux_start, aux_end)
            shells = (0, mol.nbas, 0, mol.nbas, mol.nbas + aux_shell_start, mol.nbas + aux_shell_end)
            block = self._int3c("int3c2e_ip2", shells, "s2ij")
            derivative[rows] -= np.einsum("cjp,c->pj", block, displacement)
            ip = np.einsum("cpq,c->pq", self._metric_ip[:, rows], displacement)
            for first in range(0, inverse.shape[1], 1024):
                columns = slice(first, first + 1024)
                derivative[rows, columns] += ip @ inverse[:, columns]
        return DFDirectionalDerivatives(
            mol.nao, self._aux_block_size, (self._whitened_integrals, inverse, derivative),
            _single_atom=True,
        )

    def bilinear_hessian(self, left_density, right_density, *, with_j=True, with_k=True):
        """Contract the fixed-density DF Hessian for all Cartesian pairs at once.

        Return a pair of (3*natm, 3*natm) Hessians for the Coulomb and exchange
        forms, in atom-major Cartesian order and Hartree/Bohr². The density
        matrices are real and held fixed in the moving AO basis. Either result
        may be omitted. Exchange uses the numerical ranks of the densities
        with the standard machine-precision singular-value tolerance.
        """
        left = _real_array(left_density, "left_density")
        right = _real_array(right_density, "right_density")
        nao = self._mol.nao
        if left.shape != (nao, nao) or right.shape != left.shape:
            raise ValueError("both densities must have shape (nao, nao)")
        if not with_j and not with_k:
            raise ValueError("request Coulomb, exchange, or both")

        natm = self._mol.natm
        shape = (3 * natm, 3 * natm)
        if not np.any(left) or not np.any(right):
            return np.zeros(shape) if with_j else None, np.zeros(shape) if with_k else None

        # f = sum(PQ) M^-1[PQ] h(B[P], B[Q]), with symmetric bilinear h.
        # T = M^-1 B and R_x = B_x - M_x T give its mixed derivative
        # 2 sum(P) h(B_xy[P], T[P]) - sum(PQ) M_xy[PQ] h(T[P], T[Q])
        # + 2 sum(PQ) M^-1[PQ] h(R_x[P], R_y[Q]).
        coulomb = _CoulombBilinear(self, left, right) if with_j else None
        exchange = _ExchangeBilinear(self, left, right) if with_k else None
        forms = [form for form in (coulomb, exchange) if form is not None]
        projections = [projection for form in forms for projection in form.projections]
        sizes = [projection.base.shape[1] for projection in projections]
        naux = self._auxmol.nao
        working_bytes = natm * 3 * naux * 8 * sum(sizes)
        with ExitStack() as stack:
            directory = None
            if working_bytes > self._memory_bytes * 0.4:
                directory = Path(stack.enter_context(TemporaryDirectory(prefix="pyscf-esd-df-")))
            for index, (projection, size) in enumerate(zip(projections, sizes)):
                dims = (natm, 3, naux, size)
                if directory is None:
                    projection.residual = np.zeros(dims)
                else:
                    storage = np.memmap(directory / f"residual-{index}.bin", mode="w+", shape=dims, dtype=float)
                    stack.callback(storage._mmap.close)
                    projection.residual = storage
            hessians = self._contract_bilinear_hessians(forms, projections)
        return hessians[0] if with_j else None, hessians[-1] if with_k else None

    def _contract_bilinear_hessians(self, forms, projections):
        natm = self._mol.natm
        nao = self._mol.nao
        ao_blocks = _owner_blocks(self._ao_owners)
        ao_atoms = np.array([owner for owner, _ in ao_blocks])
        ao_starts = [block.start for _, block in ao_blocks]
        hessians = [np.zeros((natm, natm, 3, 3)) for _ in forms]
        for shells, rows in self._shell_blocks:
            auxiliary_blocks = _owner_blocks(self._aux_owners[rows])
            first = self._int3c("int3c2e_ip1", shells)
            for projection in projections:
                projection.first_integrals(first, rows, ao_blocks, auxiliary_blocks)
            del first

            weights = [form.three_center_weights(rows) for form in forms]
            same = self._int3c("int3c2e_ipip1", shells).reshape(3, 3, nao, nao, -1)
            for hessian, weight in zip(hessians, weights):
                for aux_atom, aux_rows in auxiliary_blocks:
                    local = 2 * np.einsum("cdijp,pij->icd", same[..., aux_rows], weight[aux_rows])
                    block = np.add.reduceat(local, ao_starts, axis=0)
                    np.add.at(hessian, (ao_atoms, ao_atoms), block)
                    np.add.at(hessian[:, aux_atom], ao_atoms, -block)
                    np.add.at(hessian[aux_atom], ao_atoms, -block)
                    hessian[aux_atom, aux_atom] += block.sum(axis=0)
            del same

            cross = self._int3c("int3c2e_ipvip1", shells).reshape(3, 3, nao, nao, -1)
            for hessian, weight in zip(hessians, weights):
                for aux_atom, aux_rows in auxiliary_blocks:
                    local = 2 * np.einsum("cdijp,pij->ijcd", cross[..., aux_rows], weight[aux_rows])
                    block = np.add.reduceat(np.add.reduceat(local, ao_starts, axis=0), ao_starts, axis=1)
                    np.add.at(hessian, (ao_atoms[:, None], ao_atoms[None]), block)
                    np.add.at(hessian[:, aux_atom], ao_atoms, -block.sum(axis=1))
                    np.add.at(hessian[aux_atom], ao_atoms, -block.sum(axis=0))
                    hessian[aux_atom, aux_atom] += block.sum(axis=(0, 1))
            del cross

        auxiliary_blocks = _owner_blocks(self._aux_owners)
        for form, hessian in zip(forms, hessians):
            weight = form.metric_weights()
            for atom_a, rows_a in auxiliary_blocks:
                for atom_b, rows_b in auxiliary_blocks:
                    if atom_a == atom_b:
                        continue
                    block = -np.einsum("cdpq,pq->cd", self._metric_ipip[:, :, rows_a, rows_b], weight[rows_a, rows_b])
                    hessian[atom_a, atom_a] += block
                    hessian[atom_b, atom_b] += block
                    hessian[atom_a, atom_b] -= block
                    hessian[atom_b, atom_a] -= block

        for projection in projections:
            self._whiten_projected_residual(projection, auxiliary_blocks)
        results = []
        for form, hessian in zip(forms, hessians):
            matrix = hessian.transpose(0, 2, 1, 3).reshape(3 * natm, 3 * natm)
            results.append(_symmetric(matrix) + 2 * _symmetric(form.residual_gram()))
        return results

    def _inverse_projected(self, whitened):
        from scipy.linalg import solve_triangular

        return solve_triangular(self._lower, whitened, trans="T", lower=True, check_finite=False)

    def _whiten_projected_residual(self, projection, auxiliary_blocks):
        from scipy.linalg import solve_triangular

        t = projection.base
        residual = projection.residual
        step = max(1, min(4096, int(self._memory_bytes * 0.02 / (8 * t.shape[0]))))
        for first in range(0, t.shape[1], step):
            columns = slice(first, first + step)
            t_block = t[:, columns]
            for axis, metric_ip in enumerate(self._metric_ip):
                metric_full = metric_ip @ t_block
                for atom, rows in auxiliary_blocks:
                    current = residual[atom, axis, :, columns]
                    # M_x = -[projector onto atom x, int2c2e_ip1].
                    current[rows] += metric_full[rows]
                    current -= metric_ip[:, rows] @ t_block[rows]
            for atom in range(self._mol.natm):
                for axis in range(3):
                    current = residual[atom, axis, :, columns]
                    current[:] = solve_triangular(self._lower, current, lower=True, check_finite=False)


def _factor_density(matrix):
    if np.array_equal(matrix, matrix.T):
        weights, vectors = np.linalg.eigh(matrix)
        keep = np.abs(weights) > np.finfo(float).eps * matrix.shape[0] * np.max(np.abs(weights))
        vectors = vectors[:, keep]
        return vectors, vectors, weights[keep]
    left, weights, right_t = np.linalg.svd(matrix, full_matrices=False)
    keep = weights > np.finfo(float).eps * matrix.shape[0] * weights[0]
    return left[:, keep], right_t[keep].T, weights[keep]


def _apply_jk_products(density, nao, aux_block_size, integrals, products, with_j, with_k, *, use_low_rank=False):
    from pyscf.lib import unpack_tril

    dm = _real_array(density, "density")
    if dm.ndim not in (2, 3) or dm.shape[-2:] != (nao, nao):
        raise ValueError("density must have shape (nao, nao) or (n_dm, nao, nao)")
    single = dm.ndim == 2
    if single:
        dm = dm[None]
    if not with_j and not with_k:
        raise ValueError("request Coulomb, exchange, or both")

    used = sorted({index for order in products for pair in order for index in pair})
    coulomb = [None] * len(products)
    exchange = [None] * len(products)
    if with_j:
        dm_packed = _density_pair_weights(dm)
        charges = {index: dm_packed @ integrals[index].T for index in used}
        for order, pairs in enumerate(products):
            coulomb[order] = unpack_tril(sum(charges[b] @ integrals[a] for a, b in pairs))
    if with_k:
        exchange = [np.zeros_like(dm) for _ in products]
        factors = [_factor_density(matrix) for matrix in dm] if use_low_rank else [None] * len(dm)
        for first in range(0, integrals[0].shape[0], aux_block_size):
            rows = slice(first, first + aux_block_size)
            block = {index: unpack_tril(integrals[index][rows]) for index in used}
            for i, (density_matrix, factor) in enumerate(zip(dm, factors)):
                if factor is None or 2 * factor[2].size >= nao:
                    lhs = {
                        index: (matrix @ density_matrix).transpose(1, 0, 2).reshape(nao, -1)
                        for index, matrix in block.items()
                    }
                    rhs = {index: matrix.reshape(-1, nao) for index, matrix in block.items()}
                else:
                    bra, ket, weights = factor
                    left = {index: matrix @ bra for index, matrix in block.items()}
                    right = left if bra is ket else {index: matrix @ ket for index, matrix in block.items()}
                    lhs = {index: matrix.transpose(1, 0, 2).reshape(nao, -1) for index, matrix in left.items()}
                    rhs = {
                        index: (matrix * weights).transpose(0, 2, 1).reshape(-1, nao)
                        for index, matrix in right.items()
                    }
                for action, pairs in zip(exchange, products):
                    for a, b in pairs:
                        action[i] += lhs[a] @ rhs[b]
    if single:
        coulomb = [None if matrix is None else matrix[0] for matrix in coulomb]
        exchange = [None if matrix is None else matrix[0] for matrix in exchange]
    return list(zip(coulomb, exchange))


class _CoulombBilinear:
    def __init__(self, fit, left, right):
        self.densities = np.stack((_symmetric(left), _symmetric(right)))
        packed = _density_pair_weights(self.densities)
        self.base = fit._inverse_projected(fit._whitened_integrals @ packed.T)
        self.projections = (self,)
        self.residual = None

    def first_integrals(self, first, rows, ao_blocks, auxiliary_blocks):
        for atom, orbitals in ao_blocks:
            self.residual[atom, :, rows] -= 2 * np.einsum(
                "cijp,mij->cpm", first[:, orbitals], self.densities[:, orbitals], optimize=True,
            )
        _add_auxiliary_motion(self.residual, rows, auxiliary_blocks)

    def three_center_weights(self, rows):
        return np.einsum("pm,mij->pij", self.base[rows, ::-1], self.densities)

    def metric_weights(self):
        return _symmetric(np.outer(self.base[:, 0], self.base[:, 1]))

    def residual_gram(self):
        flattened = self.residual.reshape(-1, self.base.shape[0], 2)
        return flattened[:, :, 0] @ flattened[:, :, 1].T


class _ExchangeProjection:
    def __init__(self, fit, left, right):
        from pyscf.lib import unpack_tril

        self.left, self.right = left, right
        self.pair_shape = left.shape[1], right.shape[1]
        self.triangular = np.tril_indices(left.shape[1]) if left is right else None
        naux = fit._auxmol.nao
        npair = np.prod(self.pair_shape) if self.triangular is None else self.triangular[0].size
        whitened = np.empty((naux, npair))
        for _, rows in fit._shell_blocks:
            block = unpack_tril(fit._whitened_integrals[rows])
            whitened[rows] = self.pack(self.left.T @ (block @ self.right))
        self.base = fit._inverse_projected(whitened)
        self.residual = None

    def pack(self, matrices):
        if self.triangular is None:
            return matrices.reshape(*matrices.shape[:-2], -1)
        i, j = self.triangular
        return matrices[..., i, j]

    def projected_matrices(self, rows):
        from pyscf.lib import unpack_tril

        if self.triangular is not None:
            return unpack_tril(self.base[rows])
        return self.base[rows].reshape(-1, *self.pair_shape)

    def first_integrals(self, first, rows, ao_blocks, auxiliary_blocks):
        # The first-AO and second-AO derivatives are transposes. Project
        # the uncontracted AO index once for the whole primitive block.
        primitive = first.transpose(0, 3, 1, 2)
        to_right = primitive @ self.right
        to_left = to_right if self.left is self.right else primitive @ self.left
        for atom, orbitals in ao_blocks:
            a = self.left[orbitals].T @ to_right[:, :, orbitals]
            b = a if self.left is self.right else self.right[orbitals].T @ to_left[:, :, orbitals]
            self.residual[atom, :, rows] -= self.pack(a + b.swapaxes(-1, -2))
        _add_auxiliary_motion(self.residual, rows, auxiliary_blocks)


def _add_auxiliary_motion(residual, rows, auxiliary_blocks):
    # The positive auxiliary-center term cancels the sum of the two negative
    # AO-center terms for a rigid displacement of all three integral centers.
    summed = residual[:, :, rows].sum(axis=0)
    for atom, local_rows in auxiliary_blocks:
        global_rows = slice(rows.start + local_rows.start, rows.start + local_rows.stop)
        residual[atom, :, global_rows] -= summed[:, local_rows]


class _ExchangeBilinear:
    def __init__(self, fit, left, right):
        left_bra, left_ket, left_weights = _factor_density(left)
        if np.array_equal(left, right):
            right_bra, right_ket, right_weights = left_bra, left_ket, left_weights
        else:
            right_bra, right_ket, right_weights = _factor_density(right)
        self.weights = np.outer(left_weights, right_weights)
        self.bra = _ExchangeProjection(fit, left_bra, right_bra)
        self.ket = self.bra
        if left_bra is not left_ket or right_bra is not right_ket:
            self.ket = _ExchangeProjection(fit, left_ket, right_ket)
        self.projections = (self.bra,) if self.bra is self.ket else (self.bra, self.ket)
        self.pair_weights = self.bra.pack(self.weights)
        if self.bra.triangular is not None:
            i, j = self.bra.triangular
            self.pair_weights[i != j] *= 2

    def three_center_weights(self, rows):
        value = self.bra.left @ (self.ket.projected_matrices(rows) * self.weights) @ self.bra.right.T
        if self.bra is self.ket:
            return value + value.swapaxes(-1, -2)
        other = self.ket.left @ (self.bra.projected_matrices(rows) * self.weights) @ self.ket.right.T
        value += other
        # The Coulomb three-center integrals are symmetric on the two AOs.
        return (value + value.swapaxes(-1, -2)) * 0.5

    def metric_weights(self):
        return _symmetric((self.bra.base * self.pair_weights) @ self.ket.base.T)

    def residual_gram(self):
        ncoord = self.bra.residual.shape[0] * 3
        bra = self.bra.residual.reshape(ncoord, -1)
        ket = self.ket.residual.reshape(ncoord, -1)
        weights = self.pair_weights
        gram = np.zeros((ncoord, ncoord))
        for first in range(0, bra.shape[1], 16384):
            last = min(first + 16384, bra.shape[1])
            weighted = ket[:, first:last] * weights[np.arange(first, last) % weights.size]
            gram += bra[:, first:last] @ weighted.T
        return gram


@dataclass(frozen=True)
class DFDirectionalDerivatives:
    """Molecular DF factors and their directional nuclear derivatives."""

    _nao: int
    _aux_block_size: int
    _integrals: tuple[_Array, ...]
    _single_atom: bool = False

    @property
    def _products(self):
        if self._single_atom:
            return (((0, 0),), ((1, 2), (2, 1)))
        return (
            ((0, 0),),
            ((1, 0), (0, 1)),
            ((2, 0), (0, 2)),
            ((3, 0), (1, 2), (2, 1), (0, 3)),
        )[:len(self._integrals)]

    def jk(self, density, *, with_j=True, with_k=True):
        """Apply J and K and their derivatives to one AO density or a batch."""
        results = _apply_jk_products(
            density, self._nao, self._aux_block_size, self._integrals, self._products, with_j, with_k,
        )
        return DFJKDerivatives(*results, *([None] * (4 - len(results))))

    def first_jk(self, density, *, with_j=True, with_k=True):
        """Apply only J_x and K_x, using low-rank exchange density projections."""
        return _apply_jk_products(
            density, self._nao, self._aux_block_size, self._integrals,
            (self._products[1],), with_j, with_k, use_low_rank=True,
        )[0]

    def bilinear(self, left_density, right_density, *, with_j=True, with_k=True):
        """Contract each J and K order against a fixed left AO density."""
        left = _real_array(left_density, "left_density")
        if left.shape != (self._nao, self._nao) or np.shape(right_density) != left.shape:
            raise ValueError("both densities must have shape (nao, nao)")
        actions = self.jk(right_density, with_j=with_j, with_k=with_k)

        def contract(pair):
            if pair is None:
                return None
            return tuple(None if matrix is None else float(np.vdot(left, matrix)) for matrix in pair)

        return DFBilinearDerivatives(
            contract(actions.value), contract(actions.first_x),
            contract(actions.first_y), contract(actions.mixed_xy),
        )
