"""Native GPU4PySCF spin-flip TDA and analytic nuclear gradients."""

from __future__ import annotations

from math import isfinite, sqrt
from numbers import Integral

import numpy as np

from .cpu import ElectronicConvergenceError
from .gpu import _fp64_host, _require_device_fp64


def solve_spin_flip_tda(
    mean_field, *, nroots: int = 8, tolerance: float = 1e-8,
    collinear_samples: int = 20, max_cycle: int = 100, initial_guess=None,
):
    """Solve and check spin-flip-down response from a converged high-spin UKS reference.

    The eigenvalues are offsets from the auxiliary triplet. Use the difference
    between identified singlets for optical energies. The returned solver uses
    orbital-relaxed spin-flip transition dipoles and oscillator strengths.
    """
    import cupy as cp
    from gpu4pyscf.dft.uks import UKS
    from gpu4pyscf.tdscf.uks import SpinFlipTDA

    if not isinstance(mean_field, UKS) or mean_field.mol.spin != 2:
        raise ValueError("spin-flip-down singlet response requires a GPU UKS triplet reference")
    if not mean_field.converged:
        raise ElectronicConvergenceError("spin-flip UKS reference did not converge")

    _require_device_fp64(mean_field.mo_coeff, cp, "UKS orbitals")
    with mean_field.mo_coeff.device:
        alpha_occ, beta_occ = mean_field.nelec
        dimension = alpha_occ * (mean_field.mo_coeff.shape[-1] - beta_occ)
        if isinstance(nroots, bool) or not isinstance(nroots, Integral) or not 1 <= nroots <= dimension:
            raise ValueError("nroots must be an integer within the spin-flip response dimension")
        if not 0 < tolerance < 1:
            raise ValueError("tolerance must be between zero and one Hartree")
        if not isinstance(collinear_samples, Integral) or collinear_samples < 2:
            raise ValueError("at least two multicollinear samples are required")
        fock = mean_field.get_hcore() + mean_field.get_veff(dm=mean_field.make_rdm1())
        residual = float(cp.linalg.norm(mean_field.get_grad(mean_field.mo_coeff, mean_field.mo_occ, fock)))
        orbital_tolerance = mean_field.conv_tol_grad
        if orbital_tolerance is None:
            orbital_tolerance = sqrt(mean_field.conv_tol)
        if not isfinite(residual) or residual > orbital_tolerance:
            raise ElectronicConvergenceError("spin-flip UKS orbital residual exceeds its tolerance")

        class OrbitalRelaxedSpinFlipTDA(SpinFlipTDA):
            def transition_dipole(self, ref=1, state=None):
                """Return orbital-relaxed length-gauge moments between one-based roots."""
                return _spin_flip_dipoles(self, ref, state)

        response = OrbitalRelaxedSpinFlipTDA(mean_field)
        response.extype = 1
        response.collinear = "mcol"
        response.collinear_samples = int(collinear_samples)
        response.nstates = int(nroots)
        response.conv_tol = tolerance
        response.lindep = min(response.lindep, (0.01 * tolerance) ** 2)
        response.max_cycle = max_cycle
        # Native Davidson orthogonalizes the full set of candidate vectors before
        # retaining a block of nroots. Bound the trial space by the requested roots.
        response.max_memory = 16 * dimension * (12 * nroots) / 1e6
        guess = None if initial_guess is None else cp.array(initial_guess, copy=True)
        response.kernel(x0=guess)

        if len(response.xy) != nroots or not all(response.converged):
            raise ElectronicConvergenceError("native spin-flip TDA did not converge the requested roots")
        amplitudes = cp.stack([x.ravel() for x, _ in response.xy])
        energies = cp.asarray(response.e)
        vind, _ = response.gen_vind()
        residuals = cp.linalg.norm(vind(amplitudes) - energies[:, None] * amplitudes, axis=1)
        if not bool(cp.all(residuals <= tolerance)):
            raise ElectronicConvergenceError("spin-flip TDA root residual exceeds its tolerance")
        gram = amplitudes.conj() @ amplitudes.T
        if not bool(cp.all(cp.abs(gram - cp.eye(nroots)) <= 1e-8)):
            raise ElectronicConvergenceError("spin-flip TDA roots failed the orthonormality check")
    return response


def _spin_flip_dipoles(response, ref, state):
    import cupy as cp

    if isinstance(ref, bool) or not isinstance(ref, Integral) or not 1 <= ref <= len(response.xy):
        raise ValueError("ref must be a one-based spin-flip root")
    states = np.arange(1, len(response.xy) + 1) if state is None else np.atleast_1d(state)
    if states.ndim != 1 or (states.size and (states.dtype.kind not in "iu"
                                           or np.any(states < 1) or np.any(states > len(response.xy)))):
        raise ValueError("state must contain one-based spin-flip roots")
    if np.isscalar(state) and state == ref:
        raise ValueError("the target state must differ from the reference state")
    pairs = [(int(ref) - 1, int(j) - 1) for j in states if j != ref]
    if not pairs:
        return np.empty((0, 3))
    mf = response._scf
    with mf.mo_coeff.device:
        if mf.do_nlc():
            raise NotImplementedError("spin-flip optical response with nonlocal correlation is unavailable")
        mf._numint.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
        position = cp.asarray(response.mol.intor_symmetric("int1e_r", comp=3))
        operator = mf.TDDFT().gen_vind()
        spin_conserving_response = mf.gen_response(hermi=0)
        return np.stack([
            _orbital_relaxed_dipole(response, i, j, position, operator, spin_conserving_response)
            for i, j in pairs
        ])


def _orbital_relaxed_dipole(response, root_i, root_j, position, operator, spin_conserving_response):
    import cupy as cp
    from cupyx.scipy.sparse.linalg import LinearOperator, gmres
    from gpu4pyscf.grad.tduks_sf import _contract_xc_kernel

    if not (response.converged[root_i] and response.converged[root_j]):
        raise ElectronicConvergenceError("both spin-flip roots must be converged for optical properties")
    mf = response._scf
    xi, xj = (response.xy[k][0] for k in (root_i, root_j))
    gap = float(response.e[root_j] - response.e[root_i])
    oa, ob = [c[:, n > 0] for c, n in zip(mf.mo_coeff, mf.mo_occ)]
    va, vb = [c[:, n == 0] for c, n in zip(mf.mo_coeff, mf.mo_occ)]
    dm_i, dm_j = [vb @ x.T @ oa.T for x in (xi, xj)]

    # Idempotency fixes the occupied and virtual blocks of the mixed response.
    blocks = cp.stack((-oa @ (xi @ xj.T) @ oa.T, vb @ (xj.T @ xi) @ vb.T))
    gradient_context = response.Gradients()
    vxc_i, _, _, kxc = _contract_xc_kernel(
        gradient_context, mf.xc, dm_i, with_vxc=False, with_kxc=True, with_nac=True, dmvo_2=dm_j,
    )
    vxc_j, _, _, _ = _contract_xc_kernel(gradient_context, mf.xc, dm_j, with_vxc=False, with_kxc=False)
    omega, alpha, hybrid = mf._numint.rsh_and_hybrid_coeff(mf.xc, mf.mol.spin)

    def transverse_potential(vxc, density):
        potential = vxc[0]
        if hybrid:
            potential = potential - hybrid * mf.get_k(dm=density, hermi=0)
        if omega and alpha != hybrid:
            potential = potential - (alpha - hybrid) * mf.get_k(dm=density, hermi=0, omega=omega)
        return potential

    wi, wj = transverse_potential(vxc_i, dm_i), transverse_potential(vxc_j, dm_j)
    fock = spin_conserving_response(blocks[:, None])[:, 0] + kxc[:, 0]

    def vo_source(potential, transverse, amplitudes):
        # Mixed commutators [δF_i†, δP_j] and [δF_j, δP_i†] drive the orbital response.
        a = oa.T @ potential[0].T @ va + amplitudes @ (vb.T @ transverse @ va)
        b = ob.T @ potential[1].T @ vb - (ob.T @ transverse @ oa) @ amplitudes
        return cp.concatenate((a.ravel(), b.ravel()))

    source = cp.concatenate((-vo_source(fock, wi, xj), vo_source(fock.transpose(0, 2, 1), wj, xi)))
    vind, diagonal = operator

    def action(z):
        return vind(z[None])[0] - gap * z

    dimension = source.size
    system = LinearOperator((dimension, dimension), matvec=action, dtype=cp.float64)
    separation = diagonal - gap
    separation = cp.where(cp.abs(separation) > 1e-8, separation, 1.0)
    preconditioner = LinearOperator((dimension, dimension), matvec=lambda z: z / separation, dtype=cp.float64)
    z, status = gmres(system, source, M=preconditioner, restart=25, maxiter=150, tol=1e-10, atol=1e-12)
    residual = float(cp.linalg.norm(action(z) - source))
    target = max(1e-12, 1e-10 * float(cp.linalg.norm(source)))
    if status or not isfinite(residual) or residual > target:
        raise ElectronicConvergenceError(f"spin-flip optical orbital response failed (residual {residual:.3g})")

    x, y = cp.split(z, 2)
    split = oa.shape[1] * va.shape[1]
    a = (x[:split] + y[:split]).reshape(oa.shape[1], va.shape[1])
    b = (x[split:] + y[split:]).reshape(ob.shape[1], vb.shape[1])
    # Position integrals are symmetric, so the VO and OV blocks contract together.
    density = blocks.sum(axis=0) + va @ a.T @ oa.T + vb @ b.T @ ob.T
    return _fp64_host(cp.einsum("xpq,pq->x", position, density), (3,), "spin-flip transition dipole")


def spin_flip_tda_gradient(response, root_index: int, *, cphf_conv_tol: float = 1e-10):
    """Return the native analytic nuclear gradient for a zero-based SF-TDA root."""
    from gpu4pyscf.df.grad.tduks_sf import Gradients as DensityFittedGradients
    from gpu4pyscf.tdscf.uks import SpinFlipTDA

    if not isinstance(response, SpinFlipTDA):
        raise ValueError("a native GPU spin-flip TDA response is required")
    if isinstance(root_index, bool) or not isinstance(root_index, Integral):
        raise ValueError("root_index must be a zero-based integer")
    if response.xy is None or not 0 <= root_index < len(response.xy) or not response.converged[root_index]:
        raise ValueError("an explicitly selected, converged spin-flip root is required")
    if not 0 < cphf_conv_tol < 1:
        raise ValueError("cphf_conv_tol must be between zero and one")

    class ExchangeOnlyRangeSeparatedGradients(DensityFittedGradients):
        def jk_energies_per_atom(
            self, dm_list, j_factor=None, k_factor=None, omega=None,
            lr_factor=None, sr_factor=None, hermi=0, sum_results=False, verbose=None,
        ):
            # Long-range exchange carries zero Coulomb weight. Density fitting
            # denotes an absent J contribution with None rather than zero factors.
            if omega and j_factor is not None and all(factor == 0 for factor in j_factor):
                j_factor = None
            return super().jk_energies_per_atom(
                dm_list, j_factor, k_factor, omega=omega, lr_factor=lr_factor,
                sr_factor=sr_factor, hermi=hermi, sum_results=sum_results, verbose=verbose,
            )

    with response._scf.mo_coeff.device:
        gradient = response.Gradients()
        if isinstance(gradient, DensityFittedGradients):
            gradient = ExchangeOnlyRangeSeparatedGradients(response)
        gradient.cphf_conv_tol = cphf_conv_tol
        return _fp64_host(
            gradient.kernel(state=int(root_index) + 1), (response.mol.natm, 3), "spin-flip gradient",
        )
