import pytest
import jax
import jax.numpy as jnp
import numpy as np
import starry
from jaxodi.doppler_forward import (
    dot_design_matrix_fixed_map_into,
    get_flux_from_dotconv,
    get_kT,
    get_kT0,
    get_rT,
    get_x,
)

from jaxodi.doppler_surface import DopplerWav, DopplerSpec, DopplerSurface

# todo: maybe we should write our own assert function
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


class TestForwardCore:
    @pytest.fixture(autouse=True)
    def init(self, starry_doppler_map, doppler_wav, doppler_spec, doppler_surface):
        self.smap = starry_doppler_map
        self.jwav = doppler_wav
        self.jspec = doppler_spec
        self.jsurface = doppler_surface

    @pytest.fixture
    def x(self, vsini=60000.0):
        return self.smap.ops.get_x(vsini)

    @pytest.fixture
    def theta_(self):
        theta_ = self.smap._get_default_theta(theta)
        return theta_

    @pytest.fixture
    def kT(self, theta_):
        kT = self.smap.ops.get_kT(self.smap._inc, theta_, self.smap._veq, self.smap._u)
        return kT

    @pytest.fixture
    def spec_matrix(self):
        spec_ = self.smap._spectrum
        matrix = np.reshape(spec_, (-1,))
        return matrix

    @pytest.fixture
    def dot_design_matrix(self, theta_, spec_matrix):
        res = self.smap.ops.dot_design_matrix_fixed_map_into(
            self.smap._inc,
            theta_,
            self.smap._veq,
            self.smap._u,
            self.smap._y,
            spec_matrix,
        )
        return res

    @pytest.mark.parametrize("vsini", [50000.0, 60000.0])
    def test_x(self, vsini):
        # starry x
        starry_x = self.smap.ops.get_x(vsini)
        # calculated x
        xamp = self.jwav.xamp
        cal_x = get_x(xamp, vsini)
        assert_allclose(starry_x, cal_x)

    def test_rT(self, x):
        # starry rT
        starry_rT = self.smap.ops.get_rT(x)
        # calculated rT
        rT = get_rT(x, ydeg, 0, self.jwav.nk)
        assert_allclose(starry_rT, rT)

    def test_kT0(self, x):
        # get rT
        rT = self.smap.ops.get_rT(x)
        # starry kT0
        starry_kT0 = self.smap.ops.get_kT0(rT)
        # calculated kT0
        kT0 = get_kT0(rT, ydeg)
        # np.testing.assert_allclose(starry_kT0, kT0, rtol=5e-4)
        assert_allclose(starry_kT0, kT0)

    def test_kT(self, kT):
        # calculated kT
        vsini = self.jsurface.veq * jnp.sin(self.jsurface.inc)
        calc_kT = get_kT(
            self.jwav.xamp,
            vsini,
            ydeg,
            0,
            self.jwav.nk,
            self.jsurface.inc,
            self.jsurface.theta,
        )
        assert_allclose(kT, calc_kT)

    def test_dot_design_matrix_fixed_map_into(self, kT, dot_design_matrix):
        # calculated result
        res = dot_design_matrix_fixed_map_into(
            kT,
            self.smap._y,
            self.jspec.nc,
            self.jwav.nw0_int,
            self.jsurface.nt,
            self.jwav.nk,
            self.jwav.nw_int,
            self.jspec.spec0_int,
        )
        assert_allclose(dot_design_matrix, res)

    def test_flux_from_dotconv(self, theta_, dot_design_matrix):
        # starry flux
        starry_flux = self.smap.ops.get_flux_from_dotconv(
            self.smap._inc,
            theta_,
            self.smap._veq,
            self.smap._u,
            self.smap._y,
            self.smap._spectrum,
        )

        # calculated flux
        flux = get_flux_from_dotconv(
            dot_design_matrix, self.jsurface.nt, self.jwav.nw_int
        )
        assert_allclose(starry_flux, flux)
