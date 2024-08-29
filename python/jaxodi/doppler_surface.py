from functools import partial

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
import jax
import jax.numpy as jnp
import equinox as eqx

from jaxoplanet.experimental.starry.ylm import Ylm
from jaxoplanet.types import Array

from jaxodi.doppler_forward import (
    get_x,
    get_rT,
    get_kT0,
    get_kT,
    dot_design_matrix_fixed_map_into,
    get_flux_from_dotconv,
)


class DopplerWav(eqx.Module):
    wav: Array
    vsini_max: Array

    # constants
    clight = 299792458.0  # m/s

    # default value
    default_vsini_max = 1.0e5  # m/s
    default_wav = np.linspace(642.75, 643.25, 200)  # FeI 6430

    def __init__(self, wav: Array | None = None, vsini_max: Array | None = None):
        if wav is None:
            wav = self.default_wav

        if vsini_max is None:
            vsini_max = self.default_vsini_max

        self.wav = wav
        self.vsini_max = vsini_max

    @property
    def nw(self):
        """The number of wavelength bins"""
        return len(self.wav)

    @property
    def nw_int(self):
        """The number of wavelength bins in the internal grid"""
        nw = len(self.wav)
        if (nw % 2) == 0:
            nw += 1
        return nw

    @property
    def wav1(self):
        """minimum wavelength"""
        wav1 = np.min(self.wav)
        return wav1

    @property
    def wav2(self):
        """maximum wavelength"""
        wav2 = np.max(self.wav)
        return wav2

    @property
    def wavr(self):
        """Reference wavelength"""
        wavr = np.exp(
            0.5 * (np.log(self.wav1) + np.log(self.wav2))
        )  # reference wavelength
        return wavr

    @property
    def log_wav_int(self):
        log_wav_int = np.linspace(
            np.log(self.wav1 / self.wavr), np.log(self.wav2 / self.wavr), self.nw_int
        )
        return log_wav_int

    @property
    def wav_int(self):
        wav_int = self.wavr * np.exp(self.log_wav_int)
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
        hw = np.array(
            np.ceil(
                0.5
                * np.abs(np.log((1 + betasini_max) / (1 - betasini_max)))
                / self.dlam
            ),
            dtype="int32",
        )
        return hw

    @property
    def log_wav0_int(self):
        x = np.arange(0, self.hw + 1) * self.dlam
        pad_l = self.log_wav_int[0] - self.hw * self.dlam + x[:-1]
        pad_r = self.log_wav_int[-1] + x[1:]
        log_wav0_int = np.concatenate([pad_l, self.log_wav_int, pad_r])
        return log_wav0_int

    @property
    def wav0_int(self):
        wav0_int = self.wavr * np.exp(self.log_wav0_int)
        return wav0_int

    @property
    def nw0_int(self):
        return len(self.wav0_int)

    @property
    def wav0(self):
        delta_wav = np.median(np.diff(np.sort(self.wav)))
        pad_l = np.arange(self.wav1, self.wav0_int[0] - delta_wav, -delta_wav)
        pad_l = pad_l[::-1][:-1]
        pad_r = np.arange(self.wav2, self.wav0_int[-1] + delta_wav, delta_wav)
        pad_r = pad_r[1:]
        wav0 = np.concatenate([pad_l, self.wav, pad_r])
        return wav0

    @property
    def nw0(self):
        return len(self.wav0)

    @property
    def nk(self):
        nk = 2 * self.hw + 1
        return nk

    @property
    def lam_kernel(self):
        lam_kernel = self.log_wav0_int[
            self.nw0_int // 2 - self.hw : self.nw0_int // 2 + self.hw + 1
        ]
        return lam_kernel

    @property
    def xamp(self):
        xamp = (
            self.clight
            * (np.exp(-2 * self.lam_kernel) - 1)
            / (np.exp(-2 * self.lam_kernel) + 1)
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

    def get_spec0_int(self, spec0):
        assert len(spec0) == self.nw0
        spec0_cast = spec0.reshape((1, self.nw0))
        spec0_int = np.dot(spec0_cast, self.S0)
        return spec0_int


# class DopplerSpec(eqx.Module):
#     nc: Array
#     wav: DopplerWav
#     spec0_int: Array

#     def __init__(self, wav: DopplerWav, nc)


class DopplerSurface(eqx.Module):
    theta: Array
    nt: Array
    inc: Array
    obl: Array

    def __init__(self, theta, inc=np.pi / 2, obl=0):
        self.theta = theta
        self.nt = len(theta)
        self.inc = inc
        self.obl = obl

    def __call__(self, y: Ylm, wav: DopplerWav, spec0: Array):
        ydeg = y.ell_max
        kT = get_kT(wav.xamp, wav.vsini_max, ydeg, 0, wav.nk, self.inc, self.theta)
        spec0_int = wav.get_spec0_int(spec0)
        nc = 1
        flux = dot_design_matrix_fixed_map_into(
            kT, y.todense(), nc, wav.nw0_int, self.nt, wav.nk, wav.nw_int, spec0_int
        )
        res = get_flux_from_dotconv(flux, self.nt, wav.nw_int)
        return res
