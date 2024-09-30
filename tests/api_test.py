import pytest
import jax
import numpy as np
import starry
from jaxoplanet.experimental.starry.ylm import Ylm

from jaxodi.doppler_surface import DopplerWav, DopplerSpec, DopplerSurface
from jaxodi.api import light_curve

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


def test_light_curve(starry_doppler_map, doppler_wav, doppler_spec, doppler_surface):
    # starry light curve
    lc_starry = starry_doppler_map.flux(theta, normalize=False)

    # calculated light curve
    y_j = Ylm.from_dense(y_)
    lc_j = light_curve(y_j, doppler_wav, doppler_spec, doppler_surface)

    assert_allclose(lc_starry, lc_j)
