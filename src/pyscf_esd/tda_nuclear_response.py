"""Spin-resolved analytic nuclear response of a molecular PySCF RKS/TDA operator."""

from functools import cached_property
from numbers import Integral

import numpy as np

from .cis_integrals import IntegralDerivativeEngine
from .tda_df_derivatives import MolecularDFDerivatives
from .tda_hessian import _real_array, solve_tda_root_response
from .tda_orbital_curvature import OrbitalDerivative, OrbitalResponse, tda_first_action
from .tda_xc_operator import (
    _SECOND_AO,
    _accumulate_first_xc_matrices,
    _add_density_response,
    _first_density_nodes,
)


class _MovingXCFirst:
    """Ground and spin-adapted kernel tangents on the primary CPU grid."""

    def __init__(self, mf, density, xctype, batch):
        self.mf, self.density, self.xctype, self.batch = mf, density, xctype, batch
        self.bounds = mf.mol.aoslice_by_atom()[:, 2:]
        owners = np.repeat(np.arange(mf.mol.natm), np.diff(self.bounds, axis=1).ravel())
        self.ao_map = np.eye(mf.mol.natm)[owners]

    @cached_property
    def _grid(self):
        from pyscf.grad.rks import grids_response_cc

        grids = self.mf.grids
        atom_grids = tuple(grids_response_cc(grids))
        coords = np.concatenate([part[0] for part in atom_grids])
        weights = np.concatenate([part[1] for part in atom_grids])
        # PySCF sorts the primary grid and can append zero-weight padding.
        # Compare its physical nodes with the moving grid in a common order.
        tail = slice(len(coords), None)
        if np.any(grids.weights[tail] != 0):
            raise ValueError("the native RKS quadrature includes unexpected nonzero quadrature weights")
        native_coords, native_weights = grids.coords[:len(coords)], grids.weights[:len(coords)]
        i = np.lexsort(coords.T)
        j = np.lexsort(native_coords.T)
        if (len(i) != len(j) or not np.allclose(coords[i], native_coords[j], atol=1e-11, rtol=0)
                or not np.allclose(weights[i], native_weights[j], atol=1e-10, rtol=1e-10)):
            raise ValueError("the native RKS quadrature does not match the moving atom-centered grid")
        return atom_grids

    def _blocks(self):
        mf, xctype = self.mf, self.xctype
        gga = xctype == "GGA"
        nvar = 4 if gga else 1
        natm = mf.mol.natm
        for atom, (coords, weights, weight1) in enumerate(self._grid):
            for start in range(0, len(coords), self.batch):
                rows = slice(start, start + self.batch)
                width = len(coords[rows])
                ao = mf._numint.eval_ao(mf.mol, coords[rows], deriv=1 + gga).transpose(0, 2, 1)
                ao_grad = np.stack([ao[1:4], *(ao[list(i)] for i in _SECOND_AO if gga)], axis=1)
                grid_owners = np.full(width, atom)
                grid_map = np.arange(natm)[:, None] == grid_owners[None, :]
                yield ao[:nvar], ao_grad, grid_map, grid_owners, weights[rows], weight1[..., rows].reshape(-1, width)

    def _block_count(self):
        return sum((len(coords) + self.batch - 1) // self.batch for coords, _, _ in self._grid)

    def ground_derivatives(self, progress):
        """Fixed-AO-density Vxc tangent for every Cartesian coordinate."""
        mf = self.mf
        result = np.zeros((3 * mf.mol.natm, mf.mol.nao, mf.mol.nao))
        nblocks = self._block_count()
        for index, (ao, ao_grad, grid_map, owners, w, dw) in enumerate(self._blocks()):
            rho, rho1 = _first_density_nodes(ao, ao_grad, self.density, self.ao_map, grid_map, np)
            density = rho if self.xctype == "GGA" else rho[0]
            _, vxc, fxc, _ = mf._numint.eval_xc_eff(mf.xc, density, 2, xctype=self.xctype, spin=0)
            potential1 = np.einsum("ijg,cjg->cig", fxc, rho1, optimize=True) * w + vxc * dw[:, None]
            _accumulate_first_xc_matrices(result, ao, ao_grad, vxc * w, potential1, owners, self.bounds, 8, np)
            progress(index + 1, nblocks)
        return result + result.swapaxes(-1, -2)

    def _kernel_blocks(self, ground_derivatives, singlet):
        mf = self.mf
        for ao, ao_grad, grid_map, owners, w, dw in self._blocks():
            rho, rho1 = _first_density_nodes(ao, ao_grad, self.density, self.ao_map, grid_map, np)
            _add_density_response(rho1, ao, ground_derivatives, 8, np)

            # Native PySCF applies half of f_aa ± f_ab to the transition
            # density. Along the closed-shell RKS path, drho_a = drho_b
            # = drho_total/2, so the kxc contraction gives another half.
            rho_spin = np.stack((rho, rho)) * .5
            density = rho_spin if self.xctype == "GGA" else rho_spin[:, 0]
            _, _, fxc, kxc = mf._numint.eval_xc_eff(mf.xc, density, 3, xctype=self.xctype, spin=1)
            spin_sign = 1 if singlet else -1
            kernel = .5 * (fxc[0, :, 0] + spin_sign * fxc[0, :, 1])
            kxc_sum = kxc[0, :, 0].sum(axis=2) + spin_sign * kxc[0, :, 1].sum(axis=2)
            kernel1 = .25 * np.einsum("ijkg,ckg->cijg", kxc_sum, rho1, optimize=True)
            yield ao, ao_grad, grid_map, owners, w, dw, kernel, kernel1

    def kernel_derivatives(self, probes, ground_derivatives, *, singlet):
        """Derivative of the spin-adapted XC kernel at fixed transition AO density."""
        mf, ncart, nao = self.mf, 3 * self.mf.mol.natm, self.mf.mol.nao
        result = np.zeros((len(probes), ncart, nao, nao))
        probes = (probes + probes.swapaxes(-1, -2)) * .5
        for ao, ao_grad, grid_map, owners, w, dw, kernel, kernel1 in self._kernel_blocks(ground_derivatives, singlet):
            for fixed_probe, accumulator in zip(probes, result):
                t, t1 = _first_density_nodes(ao, ao_grad, fixed_probe, self.ao_map, grid_map, np)
                base = np.einsum("ijg,jg->ig", kernel, t, optimize=True)
                coefficients = np.einsum("cijg,jg->cig", kernel1, t, optimize=True)
                coefficients += np.einsum("ijg,cjg->cig", kernel, t1, optimize=True)
                coefficients = coefficients * w + base * dw[:, None]
                _accumulate_first_xc_matrices(
                    accumulator, ao, ao_grad, base * w, coefficients, owners, self.bounds, 8, np,
                )
        return result + result.swapaxes(-1, -2)

    def kernel_bilinears(self, probes, left, right, ground_derivatives, *, singlet, progress):
        """Contract the spin-adapted XC kernel derivative on two AO densities."""
        result = np.zeros((len(ground_derivatives), len(left)))
        symmetric = (probes + probes.swapaxes(-1, -2)) * .5
        nblocks = self._block_count()
        blocks = self._kernel_blocks(ground_derivatives, singlet)
        for index, (ao, ao_grad, grid_map, _, w, dw, kernel, kernel1) in enumerate(blocks):
            jets = [_first_density_nodes(ao, ao_grad, p, self.ao_map, grid_map, np) for p in symmetric]
            densities, density1 = (np.stack(part) for part in zip(*jets))
            potentials = np.einsum("ijg,pjg->pig", kernel, densities, optimize=True)
            # The kxc term is symmetric in the transition-density slots.
            # Contract both slots before accumulating the directions.
            for pair, (l, r) in enumerate(zip(left, right)):
                result[:, pair] += dw @ np.einsum("ig,ig->g", densities[l], potentials[r])
                result[:, pair] += np.einsum("cig,ig,g->c", density1[l], potentials[r], w, optimize=True)
                result[:, pair] += np.einsum("cig,ig,g->c", density1[r], potentials[l], w, optimize=True)
                result[:, pair] += np.einsum(
                    "cijg,ijg->c", kernel1, densities[l, :, None] * densities[r, None, :] * w, optimize=True,
                )
            progress(index + 1, nblocks)
        return result


class MolecularTDANuclearResponse:
    """Derivatives in a smooth occupied and virtual orbital frame at one geometry.

    The coefficient transform C' = C U has U_oo and U_vv symmetric and
    U + U.T = -C.T S' C. The occupied-virtual blocks follow RKS stationarity.
    Amplitude derivatives returned here use this same frame and retain the
    phase of the supplied native TDA root. All arrays use atom-major Cartesian
    ordering on the leading axis unless explicit directions were supplied.
    """

    def __init__(self, mf, directions, *, cphf_tolerance, max_cycle, xc_batch, progress):
        # Load the CPU response methods on RKS instances.
        from pyscf.scf import _response_functions  # noqa: F401

        self.mf, self.directions = mf, directions
        self.progress, self.max_cycle = progress, max_cycle
        self.core = IntegralDerivativeEngine(mf.mol)
        _, _, self.hybrid = mf._numint.rsh_and_hybrid_coeff(mf.xc, mf.mol.spin)
        self.df = None if not getattr(mf, "with_df", None) else MolecularDFDerivatives(mf.with_df)
        if self.df is None:
            # The conventional integral engine retains int2e_ip1 and builds a
            # four-index derivative. Large references can use the DF path.
            from pyscf import lib

            if 8 * 6 * mf.mol.nao**4 > max(0, mf.max_memory - lib.current_memory()[0]) * 1e6:
                raise ValueError("insufficient memory for conventional ERI derivatives, use a DF RKS reference")

        self.ground = np.asarray(mf.make_rdm1())
        fock = mf.get_fock(dm=self.ground)
        if not np.allclose(mf.get_hcore(), self.core.one_electron.hcore, atol=1e-9, rtol=1e-10):
            raise ValueError("the RKS core Hamiltonian differs from the analytic molecular integrals")
        nocc = np.count_nonzero(mf.mo_occ == 2)
        self._ground_kernel = mf.gen_response(hermi=1)
        self.orbitals = OrbitalResponse(mf.mo_coeff, nocc, fock, self._ground_kernel, atol=cphf_tolerance,
                                        rtol=cphf_tolerance, max_cycle=max_cycle)
        if not np.allclose(self.orbitals.fock_mo, np.diag(mf.mo_energy), atol=1e-6, rtol=1e-8):
            raise ValueError("the native RKS Fock matrix does not reproduce the supplied canonical orbitals")
        xctype = mf._numint._xc_type(mf.xc)
        self.xc = _MovingXCFirst(mf, self.ground, xctype, xc_batch)
        self.shape = self.orbitals.nocc, self.orbitals.virtual.shape[1]
        self._spin_kernels = {}
        self._grid_directions = np.eye(3 * mf.mol.natm).reshape(-1, mf.mol.natm, 3)
        self._orbital_derivatives = self._compute_orbitals(cphf_tolerance)

    def _compute_orbitals(self, tolerance):
        from .tda_soc_response import _solve_response

        fixed_xc = self.xc.ground_derivatives(
            lambda completed, total: self._notify("rks_moving_xc_fock", completed, total),
        )
        relative_core = []
        for index, direction in enumerate(self._grid_directions):
            derivative = self.core.first_one_electron(direction)
            jk = self._fixed_jk(index, self.ground[None], singlet=True)[0]
            fixed_xc[index] += derivative.hcore + jk
            relative_core.append(derivative.overlap)
            self._notify("rks_fixed_fock_derivatives", index + 1, len(self._grid_directions))

        c, o, v = self.orbitals.coefficients, self.orbitals.occupied, self.orbitals.virtual
        nocc = self.shape[0]
        nvir = self.shape[1]
        fock_mo = self.orbitals.fock_mo
        foo, fvv = fock_mo[:nocc, :nocc], fock_mo[nocc:, nocc:]
        diagonal = (fvv.diagonal()[:, None] - foo.diagonal()).ravel()

        def density_response(mo1):
            d = 2 * (c @ mo1.reshape(-1, c.shape[1], nocc)) @ o.T
            return d + d.swapaxes(-1, -2)

        def stationarity(vectors):
            rotation = vectors.reshape(-1, nvir, nocc)
            first_density = 2 * (v @ rotation) @ o.T
            potential = self._ground_kernel(first_density + first_density.swapaxes(-1, -2))
            image = fvv @ rotation - rotation @ foo + v.T @ potential @ o
            return image.reshape(len(vectors), -1)

        response = []
        for start in range(0, len(relative_core), 6):
            rows = slice(start, start + 6)
            sx = c.T @ np.asarray(relative_core[rows]) @ c
            fx = fixed_xc[rows]
            transform = -.5 * sx
            fock_metric = fx + self._ground_kernel(density_response(transform[:, :, :nocc]))
            geometric = fock_mo @ transform
            source = -(c.T @ fock_metric @ c + geometric + geometric.swapaxes(-1, -2))[:, nocc:, :nocc]
            try:
                solution, _ = _solve_response(stationarity, diagonal, source.reshape(len(sx), -1),
                                              tolerance, self.max_cycle)
                rotation = solution.reshape(len(sx), nvir, nocc)
            except ValueError:
                # A stationary unrestricted-in-variation RKS reference can
                # have an indefinite orbital Hessian. GMRES handles it.
                rotation = np.stack([self.orbitals.solve(vectors)[0] for vectors in source])
            transform[:, nocc:, :nocc] += rotation
            transform[:, :nocc, nocc:] -= rotation.swapaxes(-1, -2)
            occupied = transform[:, :, :nocc]
            density = density_response(occupied)
            fock = fx + self._ground_kernel(density)
            fock_ref = c.T @ fock @ c
            geometric = fock_mo @ transform
            first_fock_mo = fock_ref + geometric + geometric.swapaxes(-1, -2)
            residuals = np.linalg.norm(first_fock_mo[:, nocc:, :nocc], axis=(-2, -1))
            for i, residual in enumerate(residuals):
                threshold = tolerance * (1 + np.linalg.norm(source[i]))
                if residual <= threshold:
                    point = OrbitalDerivative(c @ transform[i], density[i], fock[i], first_fock_mo[i], transform[i],
                                              fock_ref[i], float(residual))
                else:
                    point = self.orbitals.first(relative_core[start + i], fx[i], occupied_response=occupied[i])
                response.append(point)
                self._notify("rks_orbital_response", len(response), len(relative_core))
        return tuple(response)

    def _notify(self, stage, completed, total):
        if self.progress is not None:
            self.progress(stage, completed, total)

    def _fixed_jk(self, index, densities, *, singlet):
        if not singlet and not self.hybrid:
            return np.zeros_like(densities)
        direction = self._grid_directions[index]
        if self.df is not None:
            coulomb, exchange = self.df.directional(direction).first_jk(
                densities, with_j=singlet, with_k=self.hybrid != 0,
            )
        else:
            eri = self.core.first(direction).eri
            coulomb = np.einsum("pqrs,nrs->npq", eri, densities, optimize=True) if singlet else None
            exchange = (np.einsum("prqs,nrs->npq", eri, densities, optimize=True) if self.hybrid else None)
        result = np.zeros_like(densities)
        if coulomb is not None:
            result += coulomb
        if exchange is not None:
            result -= .5 * self.hybrid * exchange
        return result

    def _combine(self, cartesian):
        if np.array_equal(self.directions, self._grid_directions):
            return cartesian
        return np.tensordot(self.directions, cartesian.reshape(*self.directions.shape[1:], *cartesian.shape[1:]),
                            axes=((1, 2), (0, 1)))

    @cached_property
    def mo_coeff_derivatives(self):
        """AO coefficient response of shape (direction, ao, mo), per Bohr."""
        return self._combine(np.stack([point.coefficients for point in self._orbital_derivatives]))

    @cached_property
    def fock_mo_derivatives(self):
        """Full co-moving MO Fock derivative of shape (direction, mo, mo)."""
        return self._combine(np.stack([point.fock_mo for point in self._orbital_derivatives]))

    @property
    def orbital_response_residual(self):
        """Largest Cartesian occupied-virtual stationarity residual in Hartree/Bohr."""
        return max(point.residual_norm for point in self._orbital_derivatives)

    def _transition_kernel(self, singlet):
        if singlet not in self._spin_kernels:
            self._spin_kernels[singlet] = self.mf.gen_response(singlet=singlet, hermi=0)
        return self._spin_kernels[singlet]

    def _check_df_reference(self, density):
        if self.df is None:
            return
        probes = np.concatenate((self.ground[None], density[:1]))
        for native, fitted in zip(self.mf.get_jk(dm=probes, hermi=0), self.df.reference_jk(probes)):
            if not np.allclose(native, fitted, atol=2e-9, rtol=2e-9):
                raise ValueError("the analytic Coulomb-metric fit does not reproduce native DF J/K")

    def _amplitudes(self, amplitude, singlet):
        if not isinstance(singlet, (bool, np.bool_)):
            raise ValueError("singlet must be a boolean")
        q = _real_array(amplitude, "TDA amplitude")
        single = q.ndim == 2
        q = q[None] if single else q
        if q.ndim != 3 or q.shape[1:] != self.shape or not len(q):
            raise ValueError("amplitude must have shape (occ, vir) or (nvec, occ, vir)")
        return q, single

    def operator_action(self, amplitude, *, singlet):
        """Apply the unperturbed spin-adapted TDA operator to one vector or a batch."""
        q, single = self._amplitudes(amplitude, singlet)
        o, v = self.orbitals.occupied, self.orbitals.virtual
        density = (v @ q.swapaxes(-1, -2)) @ o.T
        gap = self.mf.mo_energy[self.shape[0]:] - self.mf.mo_energy[:self.shape[0], None]
        potential = self._transition_kernel(singlet)(density)
        action = q * gap + 2 * (o.T @ potential.swapaxes(-1, -2) @ v)
        return action[0] if single else action

    def operator_derivatives(self, amplitude, *, singlet):
        """Apply A' to one vector or a batch, with a leading direction axis."""
        q, single = self._amplitudes(amplitude, singlet)
        o, v = self.orbitals.occupied, self.orbitals.virtual
        density = (v @ q.swapaxes(-1, -2)) @ o.T
        kernel = self._transition_kernel(singlet)
        potential = kernel(density)
        self._check_df_reference(density)
        density1 = np.stack([point.density for point in self._orbital_derivatives])
        xc1 = self.xc.kernel_derivatives(density, density1, singlet=singlet)
        result = []
        for index, point in enumerate(self._orbital_derivatives):
            o1, v1 = np.split(point.coefficients, [self.shape[0]], axis=1)
            transition1 = (v1 @ q.swapaxes(-1, -2)) @ o.T + (v @ q.swapaxes(-1, -2)) @ o1.T
            potential1 = self._fixed_jk(index, density, singlet=singlet) + xc1[:, index] + kernel(transition1)
            result.append(np.stack([tda_first_action(self.orbitals, x, point, p, p1)
                                    for x, p, p1 in zip(q, potential, potential1)]))
            self._notify("tda_singlet_operator_response" if singlet else "tda_triplet_operator_response",
                         index + 1, len(self._grid_directions))
        result = self._combine(np.stack(result))
        return result[:, 0] if single else result

    def operator_derivative(self, direction_index, amplitude, *, singlet):
        """Apply one requested directional derivative of the TDA operator."""
        return self.operator_derivatives(amplitude, singlet=singlet)[direction_index]

    def operator_derivative_bilinears(self, left, right, *, singlet):
        """Return pairwise leftᵀ A' right, in Hartree/Bohr.

        Both inputs take shape (nvec, occ, vir), or (occ, vir) for one pair.
        Output takes shape (direction, nvec), or (direction,) for one pair.
        Repeated amplitudes share AO quadrature and exchange contractions.
        """
        p, single_p = self._amplitudes(left, singlet)
        q, single_q = self._amplitudes(right, singlet)
        if p.shape != q.shape:
            raise ValueError("left and right must supply the same number of TDA amplitudes")
        shape = p.shape[1:]
        unique, indices = np.unique(np.concatenate((p, q)).reshape(2 * len(p), -1), axis=0, return_inverse=True)
        ip, iq = indices[:len(p)], indices[len(p):]
        unique = unique.reshape(-1, *shape)
        o, v = self.orbitals.occupied, self.orbitals.virtual
        density = (v @ unique.swapaxes(-1, -2)) @ o.T
        potentials = self._transition_kernel(singlet)(density)
        self._check_df_reference(density)
        tag = "singlet" if singlet else "triplet"
        density1 = np.stack([point.density for point in self._orbital_derivatives])
        xc1 = self.xc.kernel_bilinears(
            density, ip, iq, density1, singlet=singlet,
            progress=lambda completed, total: self._notify(f"tda_{tag}_xc_bilinears", completed, total),
        )
        result = 2 * xc1
        unique_right, right_inverse = np.unique(iq, return_inverse=True)
        for index, point in enumerate(self._orbital_derivatives):
            o1, v1 = np.split(point.coefficients, [self.shape[0]], axis=1)
            transition1 = (v1 @ unique.swapaxes(-1, -2)) @ o.T + (v @ unique.swapaxes(-1, -2)) @ o1.T
            oo, vv = point.fock_mo[:self.shape[0], :self.shape[0]], point.fock_mo[self.shape[0]:, self.shape[0]:]
            fixed_jk = self._fixed_jk(index, density[unique_right], singlet=singlet)
            result[index] += np.einsum("kia,kia->k", p, q @ vv - oo @ q)
            result[index] += 2 * np.einsum("kij,kij->k", transition1[ip], potentials[iq])
            result[index] += 2 * np.einsum("kij,kij->k", potentials[ip], transition1[iq])
            result[index] += 2 * np.einsum("kij,kij->k", density[ip], fixed_jk[right_inverse])
            self._notify(f"tda_{tag}_operator_bilinears", index + 1, len(self._grid_directions))
        result = self._combine(result)
        return result[:, 0] if single_p and single_q else result

    def root_response(self, td, root_index, *, first_actions=None, root_gap_tolerance=1e-5,
                      response_tolerance=1e-8):
        """Differentiate a solved native TDA eigenpair in the co-moving frame."""
        from pyscf.tdscf.rks import TDA

        if (not isinstance(td, TDA) or td._scf is not self.mf or td.wfnsym is not None
                or not np.all(td.get_frozen_mask())):
            raise ValueError("supply native TDA on this RKS reference without orbital or symmetry masks")
        if (not isinstance(root_index, Integral) or isinstance(root_index, bool) or td.xy is None
                or not 0 <= root_index < len(td.xy)):
            raise ValueError("root_index must select a solved native TDA root")
        energies = _real_array(td.e, "solved TDA energies")
        if (energies.shape != (len(td.xy),) or np.shape(td.converged) != energies.shape
                or not np.all(td.converged) or np.any(energies <= 0) or np.any(np.diff(energies) < 0)):
            raise ValueError("TDA roots must be converged, positive, and sorted")
        e = float(energies[root_index])
        if len(energies) < np.prod(self.shape):
            if root_index == len(energies) - 1:
                raise ValueError("solve a higher TDA root to check target-root isolation")
            if e - td.positive_eig_threshold < root_gap_tolerance:
                raise ValueError("the positive-eigenvalue threshold prevents checking target-root isolation")
        gap = np.min(np.abs(np.delete(energies, root_index) - e), initial=np.inf)
        x, y = td.xy[root_index]
        x = _real_array(x, "native TDA amplitude")
        if x.shape != self.shape or np.any(y != 0) or not np.isclose(np.vdot(x, x), .5, atol=1e-8, rtol=0):
            raise ValueError("the native solution must have unmasked, normalized TDA amplitudes")
        q = x / np.linalg.norm(x)
        if first_actions is None:
            first_actions = self.operator_derivatives(q, singlet=td.singlet)
        else:
            first_actions = _real_array(first_actions, "first TDA derivative actions")
            if first_actions.shape != (len(self.directions), *self.shape):
                raise ValueError("first_actions must supply A' applied to the phase-preserved unit amplitude")
        _, diagonal = td.gen_vind()

        def action(vectors):
            return self.operator_action(vectors, singlet=td.singlet)

        return solve_tda_root_response(
            action, e, q, first_actions, diagonal=diagonal.reshape(self.shape), operator_batch=action,
            root_separation_hartree=gap, root_tolerance=response_tolerance, root_gap_tolerance=root_gap_tolerance,
            response_rtol=response_tolerance, maxiter=self.max_cycle, lowest_root=root_index == 0,
        )


def analytic_tda_nuclear_response(mf, directions=None, *, cphf_tolerance=1e-10, max_cycle=150,
                                  xc_batch=2048, progress=None):
    """Prepare analytic CPU PySCF TDA nuclear derivatives for both spin channels.

    Supply a converged gas-phase all-electron RKS with spherical AOs, a
    global LDA/GGA functional, and its primary atom-centered grid without
    density pruning. Conventional and full-Coulomb-metric density-fitted
    J and K are supported. Directions have shape (ndir, natm, 3) or (natm, 3)
    for one direction. By default all Cartesian derivatives are returned, in
    atom-major order. No geometry or wavefunction is recomputed at a displaced
    geometry.
    """
    from pyscf.dft.rks import RKS

    if not isinstance(mf, RKS) or not mf.converged:
        raise ValueError("converge the native CPU PySCF RKS reference before computing its response")
    mol, ni = mf.mol, mf._numint
    if (mol.spin or mol.cart or mol.has_ecp() or mol.pseudo or mf.do_nlc()
            or getattr(mf, "with_x2c", None) is not None or getattr(mf, "mm_mol", None) is not None
            or getattr(mf, "with_solvent", None) is not None or getattr(mf, "only_dfj", False)):
        raise ValueError("response requires a closed-shell gas-phase all-electron primary RKS reference")
    if (mf.small_rho_cutoff != 0 or getattr(mf, "second_grids", None) not in (None, mf.grids)
            or getattr(mf, "cphf_grids", mf.grids) is not mf.grids):
        raise ValueError("the primary RKS grid must also be used for response without density pruning")
    if ni._xc_type(mf.xc) not in ("LDA", "GGA") or ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)[0] != 0:
        raise ValueError("analytic response supports global LDA and GGA functionals")
    ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
    occupations = _real_array(mf.mo_occ, "RKS occupations")
    nocc = np.count_nonzero(occupations == 2)
    if (not np.array_equal(occupations[:nocc], np.full(nocc, 2)) or np.any(occupations[nocc:] != 0)):
        raise ValueError("RKS occupations must be ordered as closed-shell occupied then virtual")
    if (not np.isfinite(cphf_tolerance) or cphf_tolerance <= 0
            or not isinstance(max_cycle, Integral) or max_cycle <= 0
            or not isinstance(xc_batch, Integral) or xc_batch <= 0):
        raise ValueError("response tolerance, cycle count, and grid batch size must be positive")
    if directions is None:
        directions = np.eye(3 * mol.natm).reshape(-1, mol.natm, 3)
    else:
        directions = _real_array(directions, "nuclear directions")
        if directions.ndim == 2:
            directions = directions[None]
        if directions.ndim != 3 or directions.shape[1:] != (mol.natm, 3) or not len(directions):
            raise ValueError("directions must have shape (ndir, natm, 3) or (natm, 3)")
        directions = directions.copy()
    directions.setflags(write=False)
    return MolecularTDANuclearResponse(mf, directions, cphf_tolerance=cphf_tolerance, max_cycle=max_cycle,
                                        xc_batch=xc_batch, progress=progress)
