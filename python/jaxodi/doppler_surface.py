from functools import partial

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
import jax
import jax.numpy as jnp
import equinox as eqx

from jaxoplanet.experimental.starry.ylm import Ylm
from jaxoplanet.types import Array

from jaxodi.doppler_forward import (
    get_kT,
    dot_design_matrix_fixed_map_into,
    get_flux_from_dotconv,
)

from jaxodi.utils import enforce_bounds


class DopplerWav(eqx.Module):
    wav: Array = eqx.field(static=True)
    vsini_max: Array = eqx.field(static=True)
    nw_int: int = eqx.field(static=True)
    nw0_int: int = eqx.field(static=True)
    nk: int = eqx.field(static=True)
    xamp: Array

    # constants
    clight = 299792458.0  # m/s

    # default value
    default_vsini_max = 1.0e5  # m/s
    default_wav = jnp.linspace(642.75, 643.25, 200)  # FeI 6430

    def __init__(self, wav: Array | None = None, vsini_max: Array | None = None):
        if wav is None:
            wav = self.default_wav

        if vsini_max is None:
            vsini_max = self.default_vsini_max

        self.wav = wav
        self.vsini_max = vsini_max
        self.nw_int = self._nw_int
        self.nw0_int = self._nw0_int
        self.nk = self._nk
        self.xamp = self._xamp

    @property
    def nw(self):
        """The number of wavelength bins"""
        return len(self.wav)

    @property
    def _nw_int(self):
        """The number of wavelength bins in the internal grid"""
        nw = len(self.wav)
        if (nw % 2) == 0:
            nw += 1
        return nw

    @property
    def wav1(self):
        """minimum wavelength"""
        wav1 = jnp.min(self.wav)
        return wav1

    @property
    def wav2(self):
        """maximum wavelength"""
        wav2 = jnp.max(self.wav)
        return wav2

    @property
    def wavr(self):
        """Reference wavelength"""
        wavr = jnp.exp(
            0.5 * (jnp.log(self.wav1) + jnp.log(self.wav2))
        )  # reference wavelength
        return wavr

    @property
    def log_wav_int(self):
        log_wav_int = jnp.linspace(
            jnp.log(self.wav1 / self.wavr), jnp.log(self.wav2 / self.wavr), self.nw_int
        )
        return log_wav_int

    @property
    def wav_int(self):
        wav_int = self.wavr * jnp.exp(self.log_wav_int)
        return wav_int

    @property
    def dlam(self):
        """bin width in log scale"""
        dlam = self.log_wav_int[1] - self.log_wav_int[0]
        return dlam

    @property
    def hw(self):
        """number of bins of half-wavelength of the kernel"""
        betasini_max = self.vsini_max / self.clight
        hw = jnp.array(
            jnp.ceil(
                0.5
                * jnp.abs(jnp.log((1 + betasini_max) / (1 - betasini_max)))
                / self.dlam
            ),
            dtype="int32",
        )
        return hw

    @property
    def log_wav0_int(self):
        x = jnp.arange(0, self.hw + 1) * self.dlam
        pad_l = self.log_wav_int[0] - self.hw * self.dlam + x[:-1]
        pad_r = self.log_wav_int[-1] + x[1:]
        log_wav0_int = jnp.concatenate([pad_l, self.log_wav_int, pad_r])
        return log_wav0_int

    @property
    def wav0_int(self):
        wav0_int = self.wavr * jnp.exp(self.log_wav0_int)
        return wav0_int

    @property
    def _nw0_int(self):
        return len(self.wav0_int)

    @property
    def wav0(self):
        delta_wav = jnp.median(jnp.diff(jnp.sort(self.wav)))
        pad_l = jnp.arange(self.wav1, self.wav0_int[0] - delta_wav, -delta_wav)
        pad_l = pad_l[::-1][:-1]
        pad_r = jnp.arange(self.wav2, self.wav0_int[-1] + delta_wav, delta_wav)
        pad_r = pad_r[1:]
        wav0 = jnp.concatenate([pad_l, self.wav, pad_r])
        return wav0

    @property
    def nw0(self):
        return len(self.wav0)

    @property
    def _nk(self):
        nk = 2 * self.hw + 1
        return int(nk)

    @property
    def lam_kernel(self):
        lam_kernel = self.log_wav0_int[
            self.nw0_int // 2 - self.hw : self.nw0_int // 2 + self.hw + 1
        ]
        return lam_kernel

    @property
    def _xamp(self):
        xamp = (
            self.clight
            * (jnp.exp(-2 * self.lam_kernel) - 1)
            / (jnp.exp(-2 * self.lam_kernel) + 1)
        )
        return xamp

    def _get_spline_operator(self, input_grid, output_grid):
        S = np.zeros((len(output_grid), len(input_grid)))
        for n in range(len(input_grid)):
            y = np.zeros_like(input_grid)
            y[n] = 1.0
            S[:, n] = Spline(input_grid, y, k=1)(output_grid)
        return S

    # todo: double check interpolation calculations
    @property
    def S0(self):
        return self._get_spline_operator(self.wav0_int, self.wav0)

    @property
    def ST0(self):
        return self.S0.T


class DopplerSpec(eqx.Module):
    nc: Array = eqx.field(static=True)
    spec0_int: Array = eqx.field(converter=jnp.asarray)

    def __init__(self, wav: DopplerWav, spec0, nc=1, interp=True):
        self.nc = nc
        self.spec0_int = self._spec0_int(spec0, wav, nc, interp)

    def _spec0_int(self, spec0, wav, nc, interp):
        # todo: implement 2d spectrum with nc > 1
        assert len(spec0) == wav.nw0
        spec0_cast = spec0[np.newaxis, :]
        if interp:
            spec0_int = np.dot(spec0_cast, wav.S0)
        else:
            spec0_int = spec0_cast
        return spec0_int


class DopplerSurface(eqx.Module):
    theta: Array
    nt: Array = eqx.field(static=True)
    inc: Array
    obl: Array
    veq: Array

    def __init__(self, theta, inc=np.pi / 2, obl=0, veq=0.0):
        self.theta = theta
        self.nt = len(theta)
        self.inc = inc
        self.obl = obl
        self.veq = veq

    def __call__(self, y: Ylm, wav: DopplerWav, spec0: DopplerSpec):
        ydeg = y.ell_max
        vsini = self.veq * jnp.sin(self.inc)
        err, vsini = enforce_bounds(vsini, 0.0, wav.vsini_max)
        kT = get_kT(wav.xamp, vsini, ydeg, 0, wav.nk, self.inc, self.theta)
        spec0_int = spec0.spec0_int
        nc = spec0.nc
        flux = dot_design_matrix_fixed_map_into(
            kT, y.todense(), nc, wav.nw0_int, self.nt, wav.nk, wav.nw_int, spec0_int
        )
        res = get_flux_from_dotconv(flux, self.nt, wav.nw_int)
        return res
