"""Moving-grid contribution to the singlet TD-RKS Lagrangian gradient."""

from __future__ import annotations


def _density_at_nodes(ao, matrix, *, gga, xp):
    """Return the density variables and their spatial derivatives at fixed AO centers."""
    values = ao[0]
    gradients = ao[1:4]
    applied = matrix @ values
    density = xp.einsum("ig,ig->g", values, applied)
    gradient = 2 * xp.einsum("aig,ig->ag", gradients, applied)
    if not gga:
        return density[None, :], gradient[:, None, :]

    # PySCF's ao deriv=2 components are xx, xy, xz, yy, yz, zz.
    second_ao = ao[xp.asarray([[4, 5, 6], [5, 7, 8], [6, 8, 9]])]
    hessian = 2 * (
        xp.einsum("abig,ig->abg", second_ao, applied)
        + xp.einsum("aig,big->abg", gradients, matrix @ gradients)
    )
    return xp.concatenate((density[None, :], gradient)), xp.concatenate(
        (gradient[:, None, :], hessian), axis=1
    )


def _lagrangian_grid_integrand(vxc, fxc, kxc, ground_node, relaxed, relaxed_node,
                               transition, transition_node, xp):
    """Differentiate v[P] + f[T,T] with respect to the spatial grid node."""
    f_transition = xp.einsum("ijg,jg->ig", fxc, transition)
    integrand = xp.einsum("ig,ig->g", vxc, relaxed)
    integrand += xp.einsum("ig,ig->g", f_transition, transition)
    derivative = xp.einsum("ig,aig->ag", vxc, relaxed_node)
    derivative += xp.einsum("ijg,jg,aig->ag", fxc, relaxed, ground_node)
    derivative += 2 * xp.einsum("ig,aig->ag", f_transition, transition_node)
    derivative += xp.einsum("ijkg,jg,kg,aig->ag", kxc, transition, transition, ground_node)
    return integrand, derivative


def moving_xc_grid_correction(mean_field, difference_density, transition_density):
    """Return the quadrature-node and partition-weight gradient in Eh/bohr.

    ``difference_density`` is the native relaxed TD density excluding the
    reference density. ``transition_density`` is the spin-summed singlet
    density D + D.T, where D = C_v X.T C_o.T for GPU4PySCF's amplitude X.
    The native TDA gradient already differentiates the AO centers. This
    contribution keeps those centers fixed and differentiates the numerical
    integration rule, using the unchanged grid topology at the solved point.
    The common ground-state force cancels in a polarized TDA gradient.
    """
    import cupy as cp
    from gpu4pyscf.dft.numint import eval_ao
    from gpu4pyscf.hessian.rks import get_dweight_dA

    mf = mean_field
    mol, grids, ni = mf.mol, mf.grids, mf._numint
    xctype = ni._xc_type(mf.xc)
    if xctype not in ("LDA", "GGA") or mf.do_nlc():
        raise NotImplementedError("analytic TDA quadrature response requires LDA or GGA without NLC")
    for name in ("coords", "weights", "quadrature_weights", "atm_idx"):
        if getattr(grids, name, None) is None:
            raise ValueError("TDA quadrature response requires a built native atom-centered grid")
    active = grids.atm_idx >= 0
    count = int(cp.count_nonzero(active))
    if (not count or bool(cp.any(grids.atm_idx >= mol.natm))
            or bool(cp.any(grids.weights[~active] != 0))
            or bool(cp.any(grids.quadrature_weights[~active] != 0))):
        raise ValueError("invalid atom indices or alignment padding on the native TD grid")
    if count != grids.coords.shape[0]:
        # Sorting can interleave alignment points with atom-owned points.
        # The native partition derivative expects valid atom indices.
        from copy import copy

        grids = copy(grids)
        for name in ("coords", "weights", "quadrature_weights", "atm_idx"):
            setattr(grids, name, getattr(grids, name)[active])

    ground = cp.asarray(mf.make_rdm1())
    relaxed = cp.asarray(difference_density)
    transition = cp.asarray(transition_density)
    matrices = (ground, (relaxed + relaxed.T) * .5, (transition + transition.T) * .5)
    if any(m.shape != (mol.nao_nr(), mol.nao_nr()) or m.dtype != cp.float64
           or not bool(cp.isfinite(m).all()) for m in matrices):
        raise ValueError("TDA quadrature response requires three finite real FP64 AO densities")
    ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
    gga = xctype == "GGA"
    if ni.gdftopt is None:
        ni.build(mol, grids.coords)
    response = cp.zeros((mol.natm, 3), dtype=cp.float64)
    batch = 4096
    for start in range(0, count, batch):
        stop = min(count, start + batch)
        ao = eval_ao(mol, grids.coords[start:stop], deriv=2 if gga else 1,
                     gdftopt=ni.gdftopt, transpose=False)
        (rho0, node0), (rho_p, node_p), (rho_t, node_t) = (
            _density_at_nodes(ao, density, gga=gga, xp=cp) for density in matrices
        )
        _, vxc, fxc, kxc = ni.eval_xc_eff(mf.xc, rho0 if gga else rho0[0],
                                          3, xctype=xctype, spin=0)[:4]
        integrand, node = _lagrangian_grid_integrand(
            vxc, fxc, kxc, node0, rho_p, node_p, rho_t, node_t, cp
        )
        weight_response = get_dweight_dA(mol, grids, (start, stop))
        response += cp.einsum("aig,g->ai", weight_response, integrand)
        cp.add.at(response, grids.atm_idx[start:stop], node.T * grids.weights[start:stop, None])
    if not bool(cp.isfinite(response).all()):
        raise ValueError("TDA quadrature response produced a nonfinite gradient")
    return cp.asnumpy(response)
