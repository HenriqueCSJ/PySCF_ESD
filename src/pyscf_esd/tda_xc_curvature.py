"""Fourth semilocal XC derivatives for restricted LDA and GGA TDA."""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
from functools import lru_cache
from math import factorial
import os
from pathlib import Path

import numpy as np


_LXC_LIBRARY = "PYSCF_ESD_LIBXC_LXC"
_LIBXC_UNPOLARIZED = 1
_LIBXC_HAVE_LXC = 1 << 4
_LIBXC_LDA_FAMILIES = (1, 128)
_LIBXC_GGA_FAMILIES = (2, 32)
_PBE_EXCHANGE, _PBE_CORRELATION, _PBE0 = 101, 130, 406

# Packed coefficients in PySCF derivative order: 1, rho, sigma, rho^2,
# rho*sigma, sigma^2, etc. Products truncate at total degree four.
_EXPONENTS = tuple((degree - j, j) for degree in range(5) for j in range(degree + 1))
_INDEX = {powers: index for index, powers in enumerate(_EXPONENTS)}
_PRODUCTS = tuple(
    tuple((_INDEX[a, b], _INDEX[i - a, j - b])
          for a in range(i + 1) for b in range(j + 1))
    for i, j in _EXPONENTS
)


class _Taylor2:
    """Vectorized fourth-order Taylor polynomial in two local increments."""

    __array_priority__ = 1000

    def __init__(self, coefficients):
        self.coefficients = coefficients

    @classmethod
    def variable(cls, value, scale, index):
        coefficients = np.zeros((len(_EXPONENTS), value.size))
        coefficients[0] = value
        coefficients[index] = scale
        return cls(coefficients)

    def __add__(self, other):
        if isinstance(other, _Taylor2):
            return _Taylor2(self.coefficients + other.coefficients)
        result = self.coefficients.copy()
        result[0] += other
        return _Taylor2(result)

    __radd__ = __add__

    def __neg__(self):
        return _Taylor2(-self.coefficients)

    def __mul__(self, other):
        if not isinstance(other, _Taylor2):
            return _Taylor2(self.coefficients * other)
        left, right = self.coefficients, other.coefficients
        out = np.empty_like(left)
        for index, products in enumerate(_PRODUCTS):
            i, j = products[0]
            out[index] = left[i] * right[j]
            for i, j in products[1:]:
                out[index] += left[i] * right[j]
        return _Taylor2(out)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        return (self**-1) * other

    def __pow__(self, exponent):
        base = self.coefficients[0]
        increment = self * (1 / base)
        increment.coefficients[0] = 0
        c2 = exponent * (exponent - 1) / 2
        c3 = c2 * (exponent - 2) / 3
        c4 = c3 * (exponent - 3) / 4
        higher = increment * (exponent + increment * (c2 + increment * (c3 + c4 * increment)))
        return (1 + higher) * base**exponent

    def log1p(self):
        base = self.coefficients[0]
        increment = self * (1 / (1 + base))
        increment.coefficients[0] = 0
        higher = increment * (1 + increment * (-.5 + increment * (1/3 + (-.25) * increment)))
        return higher + np.log1p(base)

    def expm1(self):
        base = self.coefficients[0]
        increment = self + (-base)
        higher = increment * (1 + increment * (.5 + increment * (1/6 + increment / 24)))
        return higher * np.exp(base) + np.expm1(base)


def _pbe_energy_density(density, sigma, exchange):
    kf_prefactor = (3 * np.pi**2)**(1/3)
    if exchange:
        cx, kappa, mu = .75 * (3 / np.pi)**(1/3), .804, .2195149727645171
        s_squared = sigma / (4 * kf_prefactor**2 * density**(8/3))
        scaled = (mu / kappa) * s_squared
        return -cx * density**(4/3) * (1 + kappa * scaled / (1 + scaled))

    # Unpolarized PW92 with the constants used in PBE (PW92 modified).
    a, alpha = .0310907, .21370
    rs = (3 / (4 * np.pi))**(1/3) * density**(-1/3)
    root = rs**.5
    poly = root * (7.5957 + root * (3.5876 + root * (1.6382 + .49294 * root)))
    pw92 = -2 * a * (1 + alpha * rs) * (1 / (2 * a * poly)).log1p()
    beta, gamma = .06672455060314922, (1 - np.log(2)) / np.pi**2
    t_squared = (np.pi / (16 * kf_prefactor)) * sigma / density**(7/3)
    scaled = (beta / gamma) * t_squared / (-pw92 / gamma).expm1()

    # The PBE correlation energy per particle is PW92 + H. Combine the
    # two logarithms algebraically to retain its large-gradient limit.
    # Writing t^2 directly also removes cancellation from differentiating
    # sqrt(sigma) in the zero-gradient limit.
    correction = (pw92 / gamma).expm1() / (1 + scaled * (1 + scaled))
    return gamma * density * correction.log1p()


def _packed_pbe(rho, weights):
    """Evaluate derivatives of the unpolarized LibXC PBE formula analytically."""
    density, gradient = rho[0], rho[1:]
    sigma = np.einsum("ig,ig->g", gradient, gradient)
    packed = np.zeros((len(_EXPONENTS), density.size))
    for coefficient, exchange, threshold in zip(weights, (True, False), (1e-15, 1e-12)):
        # LibXC's exchange evaluates two spin channels, each at n/2.
        active = density > 2 * threshold if exchange else density >= threshold
        if not coefficient or not np.any(active):
            continue
        n = density[active]
        # Match native LibXC point screening and its input sigma floor.
        n_scale, sigma_scale = n, n**(8/3)
        n_jet = _Taylor2.variable(n, n_scale, 1)
        sigma_jet = _Taylor2.variable(np.maximum(sigma[active], threshold**(8/3)), sigma_scale, 2)
        energy = _pbe_energy_density(n_jet, sigma_jet, exchange).coefficients
        packed[0, active] += coefficient * energy[0] / n
        for k, (i, j) in enumerate(_EXPONENTS[1:], start=1):
            scale = n_scale**i * sigma_scale**j
            packed[k, active] += coefficient * factorial(i) * factorial(j) * (energy[k] / scale)
    return packed


@lru_cache(maxsize=4)
def _load_libxc(path, expected_version):
    # On glibc, give this LibXC's internal calls precedence over a LibXC
    # already loaded by PySCF. Keep its symbols local to this handle.
    mode = ctypes.RTLD_LOCAL | getattr(os, "RTLD_DEEPBIND", 0)
    library = ctypes.CDLL(str(path), mode=mode)
    pointer = ctypes.c_void_p
    size = ctypes.c_size_t
    signatures = {
        "xc_version_string": (ctypes.c_char_p, ()),
        "xc_func_alloc": (pointer, ()),
        "xc_func_init": (ctypes.c_int, (pointer, ctypes.c_int, ctypes.c_int)),
        "xc_func_get_info": (pointer, (pointer,)),
        "xc_func_info_get_flags": (ctypes.c_int, (pointer,)),
        "xc_func_info_get_family": (ctypes.c_int, (pointer,)),
        "xc_func_end": (None, (pointer,)),
        "xc_func_free": (None, (pointer,)),
        "xc_lda_lxc": (None, (pointer, size, pointer, pointer)),
        "xc_gga_lxc": (None, (pointer, size, pointer, pointer) + (pointer,) * 5),
    }
    for name, (restype, argtypes) in signatures.items():
        function = getattr(library, name)
        function.restype = restype
        function.argtypes = argtypes

    version = library.xc_version_string().decode("ascii")
    if version != expected_version:
        raise ValueError(
            f"the fourth-derivative LibXC library must match PySCF's LibXC "
            f"{expected_version}; found {version} at {path}"
        )
    return library


@contextmanager
def _libxc_component(library, functional_id):
    function = library.xc_func_alloc()
    if not function:
        raise MemoryError("LibXC could not allocate a functional")
    try:
        if library.xc_func_init(function, functional_id, _LIBXC_UNPOLARIZED) != 0:
            raise ValueError(f"LibXC could not initialize functional {functional_id}")
        try:
            info = library.xc_func_get_info(function)
            if not library.xc_func_info_get_flags(info) & _LIBXC_HAVE_LXC:
                raise NotImplementedError(
                    f"LibXC functional {functional_id} has no analytic fourth derivative"
                )
            family = library.xc_func_info_get_family(info)
            if family not in _LIBXC_LDA_FAMILIES + _LIBXC_GGA_FAMILIES:
                raise NotImplementedError(f"unsupported LibXC functional family {family}")
            yield function, family
        finally:
            library.xc_func_end(function)
    finally:
        library.xc_func_free(function)


def _fourth_derivative_backend(xc_code):
    from pyscf.dft import libxc

    xctype = libxc.xc_type(xc_code)
    if xctype not in ("LDA", "GGA") or libxc.is_nlc(xc_code):
        raise NotImplementedError("fourth XC derivatives require restricted LDA or GGA without NLC")
    custom = xc_code in getattr(libxc, "_CUSTOM_FUNC_R", {})
    if not custom:
        hybrid, components = libxc.parse_xc(xc_code)
        weights = dict(components)
        if not hybrid[2] and weights and weights.keys() <= {_PBE_EXCHANGE, _PBE_CORRELATION, _PBE0}:
            pbe0 = weights.get(_PBE0, 0.0)
            pbe_weights = (weights.get(_PBE_EXCHANGE, 0.0) + .75 * pbe0,
                           weights.get(_PBE_CORRELATION, 0.0) + pbe0)
            return xctype, None, (), pbe_weights
    if libxc.test_deriv_order(xc_code, 4):
        return xctype, None, (), None

    libxc.test_deriv_order(xc_code, 3, raise_error=True)
    library_path = os.environ.get(_LXC_LIBRARY)
    if not library_path:
        raise NotImplementedError(
            f"PySCF's LibXC has no fourth derivative for {xc_code}. "
            f"Set {_LXC_LIBRARY} to a shared LibXC {libxc.__version__} library "
            "built with MAXORDER=4."
        )
    if custom:
        raise NotImplementedError("the separate LibXC library requires a built-in XC functional")
    if hybrid[2]:
        raise NotImplementedError("the separate LibXC library does not apply a custom range parameter")
    path = Path(library_path).expanduser().resolve(strict=True)
    return xctype, _load_libxc(path, libxc.__version__), components, None


def require_semilocal_xc_fourth(xc_code):
    """Check that a fourth derivative can be evaluated before a Hessian run."""
    _, library, components, _ = _fourth_derivative_backend(xc_code)
    if library is not None:
        for functional_id, _ in components:
            with _libxc_component(library, functional_id):
                pass


def _packed_libxc_fourth(library, components, rho, xctype):
    """Return LibXC's unique raw partials in rho and sigma ordering."""
    ngrids = rho.shape[-1]
    gga = xctype == "GGA"
    output = np.zeros((5 if gga else 1, ngrids))
    scratch = np.empty_like(output)
    if gga:
        sigma = np.einsum("ig,ig->g", rho[1:], rho[1:])
    for functional_id, coefficient in components:
        scratch.fill(0.0)
        with _libxc_component(library, functional_id) as (function, family):
            if family in _LIBXC_LDA_FAMILIES:
                library.xc_lda_lxc(function, ngrids, rho[0].ctypes.data, scratch[0].ctypes.data)
            elif gga:
                library.xc_gga_lxc(
                    function, ngrids, rho[0].ctypes.data, sigma.ctypes.data,
                    *(row.ctypes.data for row in scratch),
                )
            else:
                raise ValueError("LibXC GGA component requires Cartesian gradient inputs")
        output += coefficient * scratch
    return output


def _packed_fourth(xc_code, rho, xp):
    from pyscf.dft import libxc

    xctype, library, components, pbe_weights = _fourth_derivative_backend(xc_code)
    rho = np.asarray(rho if xp is np else xp.asnumpy(rho))
    if rho.ndim == 1 and xctype == "LDA":
        rho = rho[None, :]
    nvar = 4 if xctype == "GGA" else 1
    if rho.ndim != 2 or rho.shape[0] != nvar or np.iscomplexobj(rho) or not np.isfinite(rho).all():
        raise ValueError(f"fourth XC derivatives require finite real density shaped ({nvar}, ngrid)")
    rho = np.ascontiguousarray(rho, dtype=np.float64)
    if rho.shape[-1] == 0:
        packed = np.empty((15 if xctype == "GGA" else 5, 0))
    elif pbe_weights is not None:
        packed = _packed_pbe(rho, pbe_weights)
    elif library is None:
        packed = libxc.eval_xc1(xc_code, rho, spin=0, deriv=4)
    else:
        # The Cartesian transform also needs the raw second and third partials
        # because sigma = grad(rho).grad(rho) has nonzero second derivatives.
        lower = libxc.eval_xc1(xc_code, rho, spin=0, deriv=3)
        fourth = _packed_libxc_fourth(library, components, rho, xctype)
        packed = np.concatenate((lower, fourth))
    return rho, packed, xctype


def _cartesian_xc(rho, packed, xctype, order, xp):
    from pyscf.dft import xc_deriv

    if rho.shape[-1] == 0:
        return xp.empty((rho.shape[0],) * order + (0,), dtype=xp.float64)
    result = xc_deriv.transform_xc(rho, packed, xctype, spin=0, order=order)
    if not np.isfinite(result).all():
        raise ValueError(f"nonfinite Cartesian XC derivatives at order {order}")
    return xp.asarray(result)


def semilocal_xc_fourth(xc_code, rho, *, xp=np):
    """Return the Cartesian fourth derivative of the semilocal XC energy density.

    Supply total density on one grid block, shaped (1, ngrid) for LDA or
    (4, ngrid) for GGA, in PySCF's order (rho, dx, dy, dz). LDA also accepts a
    one-dimensional array. The output is (nvar, nvar, nvar, nvar, ngrid).
    With CuPy as xp, XC is evaluated in FP64 on the CPU and returned to the GPU.

    For PBE and PBE0, the exact unpolarized PBE expression is differentiated
    before forming the Cartesian tensor, which regularizes evaluation at
    zero gradient. Other functionals use native PySCF LibXC when available.
    PYSCF_ESD_LIBXC_LXC can specify a same-version LibXC shared library
    compiled with MAXORDER=4 for built-in semilocal functional components.
    """
    rho, packed, xctype = _packed_fourth(xc_code, rho, xp)
    return _cartesian_xc(rho, packed, xctype, 4, xp)


def semilocal_xc_derivatives(xc_code, rho, *, xp=np):
    """Return (exc, vxc, fxc, kxc, lxc) from the same Cartesian XC evaluation."""
    rho, packed, xctype = _packed_fourth(xc_code, rho, xp)
    return tuple(_cartesian_xc(rho, packed, xctype, order, xp) for order in range(5))
