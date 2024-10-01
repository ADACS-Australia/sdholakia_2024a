from functools import partial

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from scipy.sparse import block_diag as sparse_block_diag

import jax
import jax.numpy as jnp
import equinox as eqx

from jaxoplanet.experimental.starry.ylm import Ylm
from jaxoplanet.types import Array

from jaxodi.utils import unit_radian

jax.config.update("jax_enable_x64", True)


class DopplerWav(eqx.Module):
    """
    Equinox class contains wavelength grids.

    Parameters
    ----------
    eqx : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    wav: Array = eqx.field(static=True)
    vsini_max: Array = eqx.field(static=True)
    nw_int: int = eqx.field(static=True)
    nw0_int: int = eqx.field(static=True)
    nk: int = eqx.field(static=True)
    xamp: Array
    interp_order: int = eqx.field(static=True)
    interp_tol: float = eqx.field(static=True)

    # constants
    clight = 299792458.0  # m/s

    # default value
    default_vsini_max = 1.0e5  # m/s
    default_wav = jnp.linspace(642.75, 643.25, 200)  # FeI 6430
    default_interp_order = 1
    default_interp_tol = 1.0e-12

    def __init__(
        self,
        wav: Array | None = None,
        vsini_max: Array | None = None,
        interp_order: Array | None = None,
        interp_tol: Array | None = None,
    ):

        if wav is None:
            wav = self.default_wav

        if vsini_max is None:
            vsini_max = self.default_vsini_max

        if interp_order is None:
            interp_order = self.default_interp_order

        if interp_tol is None:
            interp_tol = self.default_interp_tol

        # check args
        assert (
            interp_order >= 1 and interp_order <= 5
        ), "Keyword ``interp_order`` must be in the range [1, 5]."

        self.wav = wav
        self.vsini_max = vsini_max
        self.nw_int = self._nw_int
        self.nw0_int = self._nw0_int
        self.nk = self._nk
        self.xamp = self._xamp
        self.interp_order = interp_order
        self.interp_tol = interp_tol

    @property
    def nw(self):
        """Length of the user-facing flux wavelength grid :py:attr:`wav`."""
        return len(self.wav)

    @property
    def _nw_int(self):
        """Length of the *internal* flux wavelength grid :py:attr:`wav_int`."""
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
        """The *internal* model wavelength grid."""
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
        """The *internal* rest frame spectrum wavelength grid."""
        wav0_int = self.wavr * jnp.exp(self.log_wav0_int)
        return wav0_int

    @property
    def _nw0_int(self):
        """Length of the *internal* rest frame spectrum wavelength grid
        :py :attr:`wav0_int`.
        """
        return len(self.wav0_int)

    @property
    def wav0(self):
        """The rest-frame wavelength grid."""
        delta_wav = jnp.median(jnp.diff(jnp.sort(self.wav)))
        pad_l = jnp.arange(self.wav1, self.wav0_int[0] - delta_wav, -delta_wav)
        pad_l = pad_l[::-1][:-1]
        pad_r = jnp.arange(self.wav2, self.wav0_int[-1] + delta_wav, delta_wav)
        pad_r = pad_r[1:]
        wav0 = jnp.concatenate([pad_l, self.wav, pad_r])
        return wav0

    @property
    def nw0(self):
        """Length of the user-facing rest frame spectrum wavelength grid
        :py:attr:`wav0`.
        """
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
            S[:, n] = Spline(input_grid, y, k=self.interp_order)(output_grid)
        return S

    @property
    def _S0i2e(self):
        """Interpolate from `wav0_int` to `wav0`"""
        S = self._get_spline_operator(self.wav0_int, self.wav0)
        S[np.abs(S) < self.interp_tol] = 0
        return jnp.array(S)

    @property
    def _S0i2eTr(self):
        """Transpose of the interpolation matrix of `wav0_int` to `wav0`"""
        return self._S0i2e.T

    @property
    def _S0e2i(self):
        """Interpolate from `wav0` to `wav0_int`"""
        S = self._get_spline_operator(self.wav0, self.wav0_int)
        S[np.abs(S) < self.interp_tol] = 0
        return jnp.array(S)

    @property
    def _S0e2iTr(self):
        """Transpose of the interpolation matrix of `wav0` to `wav0_int`"""
        return self._S0e2i.T

    @property
    def _Si2e(self):
        """Interpolate from `wav_int` to `wav`"""
        S = self._get_spline_operator(self.wav_int, self.wav)
        S[np.abs(S) < self.interp_tol] = 0
        return S

    @property
    def _Si2eTr(self):
        """Transpose of interpolation matrix of `wav_int` to `wav`"""
        return self._Si2e.T

    @property
    def _Se2i(self):
        """Interpolate from `wav` to `wav_int`"""
        S = self._get_spline_operator(self.wav, self.wav_int)
        S[np.abs(S) < self.interp_tol] = 0
        return jnp.array(S)

    def get_Si2eBlk(self, nt):
        csr_arr = sparse_block_diag([self._Si2e for n in range(nt)], format="csr")
        return jnp.array(csr_arr.toarray())

    def get_Si2eTrBlk(self, nt):
        csr_arr = sparse_block_diag([self._Si2eTr for n in range(nt)], format="csr")
        return jnp.array(csr_arr.toarray())


class DopplerSpec(eqx.Module):
    nc: int = eqx.field(static=True)
    spec0_int: Array = eqx.field(converter=jnp.asarray)

    def __init__(self, wav: DopplerWav, spec0, nc=1):
        self.nc = nc
        self.spec0_int = self._spec0_int(spec0, wav, nc)

    def _spec0_int(self, spec0, wav, nc):
        # todo: implement 2d spectrum with nc > 1
        assert len(spec0) == wav.nw0
        spec0_cast = spec0[np.newaxis, :]
        # todo: add option for no interpolation
        spec0_int = np.dot(spec0_cast, wav._S0e2iTr)

        return spec0_int

    # todo: add the spec0 property (DopplerMap.spectrum in starry)
    @property
    def spec0(self):
        """The rest frame spectrum."""
        pass


class DopplerSurface(eqx.Module):
    y: Ylm
    wav: DopplerWav
    spec0: DopplerSpec
    theta: Array = eqx.field(converter=unit_radian)
    nt: Array = eqx.field(static=True)
    inc: Array = eqx.field(converter=jnp.asarray)
    obl: Array = eqx.field(converter=jnp.asarray)
    veq: Array = eqx.field(converter=jnp.asarray)

    def __init__(
        self,
        y: Ylm,
        wav: DopplerWav,
        spec0: DopplerSpec,
        theta: Array,
        inc=0.5 * jnp.pi,
        obl=0.0,
        veq=0.0,
    ):
        self.y = y
        self.wav = wav
        self.spec0 = spec0
        self.theta = theta
        self.nt = len(theta)
        self.inc = inc
        self.obl = obl
        self.veq = veq

    @property
    def _Si2eBlk(self):
        return self.wav.get_Si2eBlk(self.nt)

    @property
    def _Si2eTrBlk(self):
        return self.wav.get_Si2eTrBlk(self.nt)
