import jax.numpy as jnp
from jaxoplanet.experimental.starry.ylm import Ylm

from jaxodi.doppler_surface import DopplerSurface
from jaxodi.utils import enforce_bounds
from jaxodi.doppler_forward import (
    get_kT,
    dot_design_matrix_fixed_map_into,
    get_flux_from_dotconv,
)

from jax.experimental import checkify


def light_curve(surface: DopplerSurface):
    ydeg = surface.y.ell_max
    vsini = surface.veq * jnp.sin(surface.inc)

    # check bounds at runtime
    err, vsini = enforce_bounds(vsini, 0.0, surface.wav.vsini_max)
    checkify.check_error(err)

    kT = get_kT(
        surface.wav.xamp, vsini, ydeg, 0, surface.wav.nk, surface.inc, surface.theta
    )
    spec0_int = surface.spec0.spec0_int
    nc = surface.spec0.nc
    _y = surface.y.todense()[:, jnp.newaxis]

    flux = dot_design_matrix_fixed_map_into(
        kT,
        _y,
        nc,
        surface.wav.nw0_int,
        surface.nt,
        surface.wav.nk,
        surface.wav.nw_int,
        spec0_int,
    )

    lc_int = get_flux_from_dotconv(flux, surface.nt, surface.wav.nw_int)

    # Interpolate to the `wav` grid
    lc = jnp.dot(lc_int, surface.wav._Si2eTr)

    return lc
