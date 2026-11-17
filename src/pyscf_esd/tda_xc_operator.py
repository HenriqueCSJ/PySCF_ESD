"""Moving atom-centered XC quadrature for molecular singlet TDA response."""

from copy import copy

import numpy as np


# libcint AO derivative ordering for PySCF's deriv=2 and deriv=3 grids.
_SECOND_AO = ((4, 5, 6), (5, 7, 8), (6, 8, 9))
_THIRD_AO = (((10, 11, 12), (11, 13, 14), (12, 14, 15)),
             ((11, 13, 14), (13, 16, 17), (14, 17, 18)),
             ((12, 14, 15), (14, 17, 18), (15, 18, 19)))


def _density_pair(left, right, xp):
    value = (left[..., 0, :, :] * right[..., 0, :, :]).sum(axis=-2, keepdims=True)
    if left.shape[-3] == 1:
        return value
    gradient = (left[..., 1:, :, :] * right[..., :1, :, :]).sum(axis=-2)
    gradient += (left[..., :1, :, :] * right[..., 1:, :, :]).sum(axis=-2)
    return xp.concatenate((value, gradient), axis=-2)


def _weighted_ao(field, coefficients):
    value = field * coefficients[..., :1, None, :]
    if field.shape[-3] > 1:
        value[..., 0, :, :] += (field[..., 1:, :, :] * coefficients[..., 1:, None, :]).sum(axis=-3)
        value[..., 1:, :, :] = field[..., :1, :, :] * coefficients[..., 1:, None, :]
    return value


def _jet_product(operation, left, right):
    terms = []
    for order in range(len(left)):
        summands = [operation(left[i], right[order ^ i]) for i in range(order + 1)
                    if (i & order) == i and left[i] is not None and right[order ^ i] is not None]
        terms.append(sum(summands) if summands else None)
    return tuple(terms)


def _ao_jets(ao, x, y, gga, xp):
    nvar = 4 if gga else 1
    base = ao[:nvar]
    spatial = [ao[1:4]]
    spatial += [ao[list(indices)] for indices in _SECOND_AO] if gga else []
    dx = xp.stack([xp.einsum("aig,aig->ig", field, x) for field in spatial])
    if y is None:
        return base, dx
    dy = xp.stack([xp.einsum("aig,aig->ig", field, y) for field in spatial])
    indices = (_SECOND_AO, *_THIRD_AO) if gga else (_SECOND_AO,)
    dxy = xp.stack([sum(x[i] * y[j] * ao[index[i][j]] for i in range(3) for j in range(3))
                    for index in indices])
    return base, dx, dy, dxy


def _density(ao, matrices, xp):
    def applied(matrix, field):
        return matrix @ field

    return _jet_product(lambda a, b: _density_pair(a, b, xp), ao, _jet_product(applied, matrices, ao))


def _potential(ao, coefficients):
    def project(left, right):
        return sum(l_a @ r_a.T for l_a, r_a in zip(left, right))

    return _jet_product(project, ao, _jet_product(_weighted_ao, ao, coefficients))


def _xc_coefficients(ni, xc, rho, probe, gga, xp):
    """Energy derivatives at the nodes, including the changing SCF density."""
    from .tda_xc_curvature import semilocal_xc_fourth

    xctype = "GGA" if gga else "LDA"
    order = 2 if len(rho) == 2 and probe is None else 3
    evaluated = ni.eval_xc_eff(xc, rho[0] if gga else rho[0][0], order, xctype=xctype, spin=0)
    vxc, fxc = evaluated[1:3]
    einsum = xp.einsum
    vx = einsum("ijg,jg->ig", fxc, rho[1])
    if len(rho) == 2:
        v = vxc, vx
    else:
        kxc = evaluated[3]
        vy = einsum("ijg,jg->ig", fxc, rho[2])
        vxy = einsum("ijkg,jg,kg->ig", kxc, rho[1], rho[2])
        if rho[3] is not None:
            vxy += einsum("ijg,jg->ig", fxc, rho[3])
        v = vxc, vx, vy, vxy
    if probe is None:
        return v, None

    kxc = evaluated[3]
    fx = einsum("ijkg,kg->ijg", kxc, rho[1])
    if len(rho) == 2:
        f = fxc, fx
    else:
        lxc = semilocal_xc_fourth(xc, rho[0], xp=xp)
        fy = einsum("ijkg,kg->ijg", kxc, rho[2])
        fxy = einsum("ijklg,kg,lg->ijg", lxc, rho[1], rho[2])
        if rho[3] is not None:
            fxy += einsum("ijkg,kg->ijg", kxc, rho[3])
        f = fxc, fx, fy, fxy
    return v, _jet_product(lambda tensor, vector: einsum("ijg,jg->ig", tensor, vector), f, probe)


def _first_density_nodes(ao, spatial, matrix, ao_map, grid_map, xp):
    """Compute the density and its relative node/AO motion at fixed matrix."""
    applied = matrix @ ao
    value = _density_pair(ao, applied, xp)
    left = spatial * applied[0]
    if ao.shape[0] > 1:
        left[:, 1:] += spatial[:, :1] * applied[1:]
    # The right AO gives the same contribution for a symmetric density.
    natm, width = grid_map.shape
    center = ao_map.T @ left.transpose(2, 0, 1, 3).reshape(ao.shape[1], -1)
    node = grid_map[:, None, None, :] * left.sum(axis=2)[None]
    derivative = 2 * (node - center.reshape(natm, 3, ao.shape[0], width))
    return value, derivative.reshape(3 * natm, ao.shape[0], width)


def _add_density_response(derivative, ao, matrices, cart_batch, xp):
    """Add rho[D_x] using the symmetry of each AO derivative density."""
    nao, width = ao.shape[1:]
    for first in range(0, len(matrices), cart_batch):
        rows = slice(first, first + cart_batch)
        block = matrices[rows]
        symmetric = (block + block.swapaxes(-1, -2)) * .5
        applied = (symmetric.reshape(-1, nao) @ ao[0]).reshape(-1, nao, width)
        response = xp.einsum("cig,vig->cvg", applied, ao)
        response[:, 1:] *= 2
        derivative[rows] += response


def _accumulate_first_xc_matrices(result, ao, spatial, base, coefficients,
                                  grid_owners, ao_bounds, cart_batch, xp):
    """Add derivatives of the AO projection to a single-sided accumulator."""
    ncart, nao, _ = result.shape
    for first in range(0, ncart, cart_batch):
        rows = slice(first, first + cart_batch)
        # c0 phi_i phi_j contributes half on each side. For a gradient
        # coefficient the two terms are transposes of one another.
        local = coefficients[rows].copy()
        local[:, 0] *= .5
        projected = xp.einsum("cvg,vig->cig", local, ao)
        result[rows] += projected @ ao[0].T

    # Differentiate one AO in V[base]. Its spatial derivative moves with
    # the grid-point owner and against its AO owner. Project each owner's
    # points once, then subtract their sum from each atom's own AO rows.
    weighted = _weighted_ao(ao, base)
    order = xp.argsort(grid_owners)
    counts = xp.bincount(grid_owners, minlength=len(ao_bounds))
    counts = counts if xp is np else xp.asnumpy(counts)
    grid_offsets = np.cumsum(np.r_[0, counts])
    result_by_atom = result.reshape(len(ao_bounds), 3, nao, nao)
    all_points = xp.zeros((3, nao, nao), dtype=result.dtype)
    for atom, (first, stop) in enumerate(zip(grid_offsets[:-1], grid_offsets[1:])):
        if first == stop:
            continue
        points = order[first:stop]
        spatial_group = spatial[..., points].transpose(0, 2, 1, 3).reshape(3, nao, -1)
        weighted_group = weighted[..., points].transpose(1, 0, 2).reshape(nao, -1)
        projection = spatial_group @ weighted_group.T
        result_by_atom[atom] += projection
        all_points += projection
    for atom, (first, stop) in enumerate(ao_bounds):
        result_by_atom[atom, :, first:stop] -= all_points[:, first:stop]


class MovingXCOperator:
    """Analytic local AO potential and Hessian-of-XC actions on the native GPU grid."""

    def __init__(self, mean_field, *, batch=2048):
        import cupy as cp

        self.mf = mean_field
        self.batch = batch
        self.gga = mean_field._numint._xc_type(mean_field.xc) == "GGA"
        grids = mean_field.grids
        owned = grids.atm_idx >= 0
        self.grids = copy(grids)
        for name in ("coords", "weights", "quadrature_weights", "atm_idx"):
            setattr(self.grids, name, getattr(grids, name)[owned])
        self.ao_bounds = mean_field.mol.aoslice_by_atom()[:, 2:]
        counts = np.diff(self.ao_bounds, axis=1).ravel()
        self.ao_owners = cp.asarray(np.repeat(np.arange(mean_field.mol.natm), counts))

    def first_derivatives(self, ground, probe=None, ground_derivatives=None):
        """Return all first Cartesian derivatives of Vxc or Kxc[probe].

        Ground and probe are real AO density matrices. The probe is fixed
        in the moving AO representation. Supplying ground_derivatives of
        shape (3 * natm, nao, nao) adds its first-order density response.
        """
        import cupy as cp
        from gpu4pyscf.dft.numint import eval_ao
        from gpu4pyscf.hessian.rks import get_dweight_dA

        mf, grids = self.mf, self.grids
        mol, ni = mf.mol, mf._numint
        nao, ncart = mol.nao_nr(), 3 * mol.natm
        nvar = 4 if self.gga else 1
        xctype = "GGA" if self.gga else "LDA"
        ground = cp.asarray(ground)
        ground = (ground + ground.T) * .5
        if probe is not None:
            probe = cp.asarray(probe)
            probe = (probe + probe.T) * .5
        if ground_derivatives is not None:
            ground_derivatives = cp.asarray(ground_derivatives)
            if ground_derivatives.shape != (ncart, nao, nao):
                raise ValueError("ground_derivatives must have shape (3 * natm, nao, nao)")
        result = cp.zeros((ncart, nao, nao), dtype=cp.float64)
        ao_map = cp.eye(mol.natm)[self.ao_owners]
        for start in range(0, len(grids.weights), self.batch):
            stop = min(start + self.batch, len(grids.weights))
            rows = slice(start, stop)
            width = stop - start
            cart_batch = max(1, min(16, (48 * 2**20) // (8 * nao * width)))
            owners = grids.atm_idx[rows]
            grid_map = cp.arange(mol.natm)[:, None] == owners[None, :]
            kernel_ao = eval_ao(mol, grids.coords[rows], deriv=1 + int(self.gga),
                                gdftopt=ni.gdftopt, transpose=False)
            ao = kernel_ao[:nvar]
            spatial_components = [kernel_ao[1:4]]
            if self.gga:
                spatial_components.extend(kernel_ao[list(i)] for i in _SECOND_AO)
            spatial = cp.stack(spatial_components, axis=1)
            rho, rho1 = _first_density_nodes(ao, spatial, ground, ao_map, grid_map, cp)
            if ground_derivatives is not None:
                _add_density_response(rho1, ao, ground_derivatives, cart_batch, cp)

            evaluated = ni.eval_xc_eff(mf.xc, rho if self.gga else rho[0],
                                       2 if probe is None else 3, xctype=xctype, spin=0)
            vxc, fxc = evaluated[1:3]
            if probe is None:
                base = vxc
                coefficients = cp.einsum("ijg,cjg->cig", fxc, rho1)
            else:
                t, t1 = _first_density_nodes(ao, spatial, probe, ao_map, grid_map, cp)
                base = cp.einsum("ijg,jg->ig", fxc, t)
                vertex = cp.einsum("ijkg,jg->ikg", evaluated[3], t)
                coefficients = cp.einsum("ijg,cjg->cig", vertex, rho1)
                coefficients += cp.einsum("ijg,cjg->cig", fxc, t1)
            weights = grids.weights[rows]
            dw = get_dweight_dA(mol, grids, (start, stop)).reshape(ncart, width)
            coefficients *= weights
            coefficients += base * dw[:, None, :]
            _accumulate_first_xc_matrices(result, ao, spatial, base * weights, coefficients,
                                          owners, self.ao_bounds, cart_batch, cp)
        for first in range(0, ncart, 16):
            block = result[first:first + 16]
            block += block.swapaxes(-1, -2).copy()
        return cp.asnumpy(result)

    def density_adjoint(self, ground, transition):
        """AO density derivative of the scalar transition XC term 2 f[T,T]."""
        import cupy as cp
        from gpu4pyscf.dft.numint import eval_ao

        mf, grids = self.mf, self.grids
        ground = cp.asarray(ground)
        transition = cp.asarray(transition)
        result = cp.zeros_like(ground)
        xctype = "GGA" if self.gga else "LDA"
        for start in range(0, len(grids.weights), self.batch):
            rows = slice(start, start + self.batch)
            ao = (eval_ao(mf.mol, grids.coords[rows], deriv=int(self.gga),
                          gdftopt=mf._numint.gdftopt, transpose=False),)
            if not self.gga:
                ao = (ao[0].reshape(1, *ground.shape[:1], -1),)
            rho = _density(ao, (ground,), cp)[0]
            t = _density(ao, (transition,), cp)[0]
            evaluated = mf._numint.eval_xc_eff(mf.xc, rho if self.gga else rho[0], 3, xctype=xctype, spin=0)
            local = 2 * cp.einsum("ijkg,jg,kg,g->ig", evaluated[3], t, t, grids.weights[rows])
            result += _potential(ao, (local,))[0]
        return cp.asnumpy(result)

    def bilinear_hessian(self, ground, ground_derivatives, relaxed, transition):
        """Contract the clamped mixed XC terms over every Cartesian direction.

        Computes the Hessian of ``Vxc[R] + 2 fxc[T,T]`` with fixed AO test
        densities R/T. The ground density varies to first order in each
        direction. The caller adds the term linear in its mixed derivative.
        """
        import cupy as cp
        from gpu4pyscf.dft.numint import eval_ao
        from gpu4pyscf.hessian.rks import get_dweight_dA, get_d2weight_dAdB

        from .tda_xc_curvature import semilocal_xc_derivatives

        mf, grids = self.mf, self.grids
        natm, nao = mf.mol.natm, mf.mol.nao_nr()
        ncart, nvar = 3 * natm, 4 if self.gga else 1
        indices = (_SECOND_AO, *_THIRD_AO) if self.gga else (_SECOND_AO,)
        ao_map = cp.eye(natm)[self.ao_owners]
        d1 = cp.asarray(ground_derivatives)[:, None]
        matrices = tuple(cp.asarray((m + m.T) * 0.5) for m in (ground, relaxed, transition))
        result = cp.zeros((ncart, ncart))
        batch = max(1, min(self.batch, (144 * 2**20) // (8 * ncart * nvar * nao)))
        for start in range(0, len(grids.weights), batch):
            stop = min(start + batch, len(grids.weights))
            rows = slice(start, stop)
            width = stop - start
            grid_map = cp.eye(natm)[grids.atm_idx[rows]]
            kernel_ao = eval_ao(mf.mol, grids.coords[rows], deriv=3 if self.gga else 2,
                                gdftopt=mf._numint.gdftopt, transpose=False)
            ao = kernel_ao[:nvar]
            spatial = cp.stack([kernel_ao[1:4]] + [kernel_ao[list(i)] for i in _SECOND_AO] * self.gga)
            displacement = (grid_map.T[:, None, :] - ao_map.T[:, :, None])[:, None, None]
            ao1 = (displacement * spatial.transpose(1, 0, 2, 3)[None]).reshape(ncart, nvar, nao, width)
            applied = tuple(matrix @ ao for matrix in matrices)
            p, r, t = (_density_pair(ao, field, cp) for field in applied)
            p1, r1, t1 = (2 * _density_pair(ao1, field, cp) for field in applied)
            applied_d1 = d1 @ ao
            p1 += _density_pair(ao, applied_d1, cp)

            _, v, f, k, fourth = semilocal_xc_derivatives(mf.xc, p, xp=cp)
            et = cp.einsum("ijg,jg->ig", f, t)
            ep = cp.einsum("ijg,jg->ig", f, r) + 2 * cp.einsum("ijkg,jg,kg->ig", k, t, t)
            e0 = cp.sum(v * r + 2 * et * t, axis=0)
            e1 = cp.sum(v * r1 + ep * p1 + 4 * et * t1, axis=1)
            dw = get_dweight_dA(mf.mol, grids, (start, stop)).reshape(ncart, width)
            cross = dw @ e1.T
            result += cross + cross.T
            ddw = get_d2weight_dAdB(mf.mol, grids, (start, stop))
            result += cp.einsum("abijg,g->aibj", ddw, e0).reshape(ncart, ncart)
            del ddw

            weight = grids.weights[rows]
            pp = cp.einsum("ijkg,kg->ijg", k, r) + 2 * cp.einsum("ijklg,kg,lg->ijg", fourth, t, t)
            tp = 4 * cp.einsum("ijkg,kg->ijg", k, t)

            def nodal_pair(left, vertex, right):
                contracted = cp.einsum("ijg,ajg,g->aig", vertex, right, weight)
                return left.reshape(ncart, -1) @ contracted.reshape(ncart, -1).T

            cross = nodal_pair(r1, f, p1) + nodal_pair(t1, tp, p1)
            result += cross + cross.T + nodal_pair(p1, pp, p1) + 4 * nodal_pair(t1, f, t1)

            for matrix, field, coefficient in zip(matrices, applied, (ep, v, 4 * et)):
                coefficient = coefficient * weight
                right = _weighted_ao(field, coefficient)
                # At a node owned by B and an AO on A, its second
                # derivative acts along the relative displacement B - A.
                for a in range(3):
                    for b in range(3):
                        ao2 = kernel_ao[[index[a][b] for index in indices]]
                        nodal = 2 * (right * ao2).sum(axis=0)
                        by_atom = ao_map.T @ nodal @ grid_map
                        block = cp.diag(by_atom.sum(axis=0) + by_atom.sum(axis=1)) - by_atom - by_atom.T
                        result[a::3, b::3] += block
                weighted_first = _weighted_ao(ao1, coefficient).reshape(ncart, -1)
                result += 2 * weighted_first @ (matrix @ ao1).reshape(ncart, -1).T
                if matrix is matrices[0]:
                    cross = 2 * weighted_first @ applied_d1.reshape(ncart, -1).T
                    result += cross + cross.T
        return cp.asnumpy(result)

    def directional(self, x, y, ground, probe=None, *, scalar_probes=None):
        """Return jets of Vxc and fxc[probe] (orders value, x, y, xy).

        Density jets are AO matrices with None for absent coefficients. In
        scalar mode integrate directly against the two corresponding AO test
        densities. All displacements are nuclear and use Bohr.
        """
        import cupy as cp
        from gpu4pyscf.dft.numint import eval_ao
        from gpu4pyscf.hessian.rks import get_dweight_dA, get_d2weight_dAdB

        grids, mol, ni = self.grids, self.mf.mol, self.mf._numint
        x = cp.asarray(x)
        y = None if y is None else cp.asarray(y)
        ground = tuple(None if m is None else cp.asarray(m) for m in ground)
        if probe is not None:
            probe = tuple(None if m is None else cp.asarray(m) for m in probe)
        if scalar_probes is not None:
            scalar_probes = tuple(None if p is None else tuple(None if m is None else cp.asarray(m) for m in p)
                                  for p in scalar_probes)
        matrices = [None, None]
        derivative = int(self.gga) + (1 if y is None else 2)
        for start in range(0, len(grids.weights), self.batch):
            stop = min(start + self.batch, len(grids.weights))
            rows = slice(start, stop)
            owner = grids.atm_idx[rows]
            dx = (x[owner, None] - x[None, self.ao_owners]).transpose(2, 1, 0)
            dy = None if y is None else (y[owner, None] - y[None, self.ao_owners]).transpose(2, 1, 0)
            ao = eval_ao(mol, grids.coords[rows], deriv=derivative, gdftopt=ni.gdftopt, transpose=False)
            ao = _ao_jets(ao, dx, dy, self.gga, cp)
            rho = _density(ao, ground, cp)
            rho_probe = None if probe is None else _density(ao, probe, cp)
            v, response = _xc_coefficients(ni, self.mf.xc, rho, rho_probe, self.gga, cp)
            dw = get_dweight_dA(mol, grids, (start, stop))
            weight = (grids.weights[rows], cp.einsum("aig,ai->g", dw, x))
            if y is not None:
                ddw = get_d2weight_dAdB(mol, grids, (start, stop))
                weight += (cp.einsum("aig,ai->g", dw, y), cp.einsum("abijg,ai,bj->g", ddw, x, y))
                del ddw
            for i, coefficients in enumerate((v, response)):
                if coefficients is None:
                    continue
                weighted = _jet_product(lambda w, c: w * c, weight, coefficients)
                if scalar_probes is None:
                    contribution = _potential(ao, weighted)
                elif scalar_probes[i] is not None:
                    density = _density(ao, scalar_probes[i], cp)
                    contribution = _jet_product(lambda c, d: cp.sum(c * d), weighted, density)
                else:
                    continue
                if matrices[i] is None:
                    matrices[i] = cp.stack(contribution)
                else:
                    matrices[i] += cp.stack(contribution)
        return tuple(None if value is None else tuple(cp.asnumpy(value)) for value in matrices)
