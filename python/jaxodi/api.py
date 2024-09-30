import jax.numpy as jnp
from jaxoplanet.experimental.starry.ylm import Ylm

from jaxodi.doppler_surface import DopplerWav, DopplerSpec, DopplerSurface
from jaxodi.utils import enforce_bounds
from jaxodi.doppler_forward import (
    get_kT,
    dot_design_matrix_fixed_map_into,
    get_flux_from_dotconv,
)

from jax.experimental import checkify


def light_curve(y: Ylm, wav: DopplerWav, spec0: DopplerSpec, surface: DopplerSurface):
    ydeg = y.ell_max
    vsini = surface.veq * jnp.sin(surface.inc)

    # check bounds at runtime
    err, vsini = enforce_bounds(vsini, 0.0, wav.vsini_max)
    checkify.check_error(err)

    kT = get_kT(wav.xamp, vsini, ydeg, 0, wav.nk, surface.inc, surface.theta)
    spec0_int = spec0.spec0_int
    nc = spec0.nc
    _y = y.todense()[:, jnp.newaxis]

    flux = dot_design_matrix_fixed_map_into(
        kT, _y, nc, wav.nw0_int, surface.nt, wav.nk, wav.nw_int, spec0_int
    )

    lc_int = get_flux_from_dotconv(flux, surface.nt, wav.nw_int)

    # Interpolate to the `wav` grid
    lc = jnp.dot(lc_int, wav._Si2eTr)

    return lc
