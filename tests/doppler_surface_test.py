import pytest

import jax
import jax.numpy as jnp
import numpy as np
import starry

from jaxodi.doppler_surface import DopplerWav, DopplerSpec, DopplerSurface

from jaxoplanet.test_utils import assert_allclose

jax.config.update("jax_enable_x64", True)

ydeg = 10
veq = 60000.0  # m/s
nt = 16
N = (ydeg + 1) ** 2
y_ = np.random.rand(N)
y_[0] = 1.0
theta = np.append([-180], np.linspace(-90, 90, nt - 1))


@pytest.fixture
def starry_doppler_map():
    map_ = starry.DopplerMap(ydeg, veq=veq, vsini_max=veq, inc=90, nt=nt, lazy=False)
    map_[:, :] = y_
    map_.spectrum = 1.0 - 0.75 * np.exp(-0.5 * (map_.wav0 - 643.0) ** 2 / 0.0085**2)
    return map_


@pytest.fixture
def doppler_wav():
    return DopplerWav(vsini_max=veq)


@pytest.fixture
def doppler_spec(doppler_wav):
    spec0 = 1.0 - 0.75 * np.exp(-0.5 * (doppler_wav.wav0 - 643.0) ** 2 / 0.0085**2)
    return DopplerSpec(doppler_wav, spec0)


@pytest.fixture
def doppler_surface():
    return DopplerSurface(theta, veq=veq)


class TestDopplerWav:
    @pytest.fixture(autouse=True)
    def init(self, starry_doppler_map, doppler_wav):
        self.smap = starry_doppler_map
        self.jwav = doppler_wav

    def test_nw(self):
        assert self.jwav.nw == self.smap.nw

    def test_nw0(self):
        assert self.jwav.nw0 == self.smap.nw0

    def test_nw_int(self):
        assert self.jwav.nw_int == self.smap.nw_

    def test_nw0_int(self):
        assert self.jwav.nw0_int == self.smap.nw0_

    def test_nk(self):
        assert self.jwav.nk == self.smap.ops.nk

    def test_wav(self):
        assert np.allclose(self.jwav.wav, self.smap.wav)

    def test_wav0(self):
        assert np.allclose(self.jwav.wav0, self.smap.wav0)

    def test_wav_int(self):
        assert np.array_equal(self.jwav.wav_int, self.smap.wav_)

    def test_log_wav0_int(self):
        assert np.allclose(self.jwav.log_wav0_int, self.smap._log_wav0_int)

    def test_wav0_int(self):
        assert np.array_equal(self.jwav.wav0_int, self.smap.wav0_)

    def test_wavr(self):
        assert np.array_equal(self.jwav.wavr, self.smap._wavr)

    def test_xamp(self):
        assert np.array_equal(self.jwav.xamp, self.smap.ops.xamp)

    def test_S0i2e(self):
        assert jnp.allclose(self.jwav._S0i2e, self.smap._S0i2e.todense())

    def test_S0i2eTr(self):
        assert jnp.allclose(self.jwav._S0i2eTr, self.smap._S0i2eTr.todense())

    def test_S0e2i(self):
        assert jnp.allclose(self.jwav._S0e2i, self.smap._S0e2i.todense())

    def test_S0e2iTr(self):
        assert jnp.allclose(self.jwav._S0e2iTr, self.smap._S0e2iTr.todense())

    def test_Si2eTr(self):
        assert jnp.allclose(self.jwav._Si2eTr, self.smap._Si2eTr.todense())

    def test_Se2i(self):
        assert np.allclose(self.jwav._Se2i, self.smap._Se2i.todense())


class TestDopplerSpec:
    @pytest.fixture(autouse=True)
    def init(self, starry_doppler_map, doppler_spec):
        self.smap = starry_doppler_map
        self.jspec = doppler_spec

    def test_spec0_int(self):
        assert np.allclose(self.jspec.spec0_int, self.smap._spectrum)


class TestDopplerSurface:
    @pytest.fixture(autouse=True)
    def init(self, starry_doppler_map, doppler_surface):
        self.smap = starry_doppler_map
        self.jsurface = doppler_surface

    def test_theta(self):
        assert np.allclose(self.jsurface.theta, self.smap._get_default_theta(theta))
