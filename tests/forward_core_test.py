import jax
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

# todo: maybe we should write our own assert function
from jaxoplanet.test_utils import assert_allclose

jax.config.update("jax_enable_x64", True)

ydeg = 10
veq = 60000.0  # m/s
nt = 16
theta = np.append([-180], np.linspace(-90, 90, nt - 1))
N = (ydeg + 1) ** 2
map_ = starry.DopplerMap(ydeg, veq=veq, vsini_max=veq, inc=90, nt=nt, lazy=False)
y_ = np.random.rand(N)
y_[0] = 1.0
map_[:, :] = y_

# todo: write function to get the following constants
xamp = map_.ops.xamp
vsini = 60000.0
nk = 105
theta_ = map_._get_default_theta(theta)


def test_compare_starry_x():
    starry_x = map_.ops.get_x(vsini)
    x = get_x(xamp, vsini)
    # np.testing.assert_allclose(starry_x, x)
    assert_allclose(starry_x, x)


def test_compare_starry_rT():
    # get x
    x = map_.ops.get_x(vsini)
    starry_rT = map_.ops.get_rT(x)
    rT = get_rT(x, ydeg, 0, nk)
    assert_allclose(starry_rT, rT)


def test_compare_starry_kT0():
    x = map_.ops.get_x(vsini)
    rT = map_.ops.get_rT(x)
    starry_kT0 = map_.ops.get_kT0(rT)
    kT0 = get_kT0(rT, ydeg)
    # np.testing.assert_allclose(starry_kT0, kT0, rtol=5e-4)
    assert_allclose(starry_kT0, kT0)


def test_compare_starry_kT():
    starry_kT = map_.ops.get_kT(map_._inc, theta_, map_._veq, map_._u)
    kT = get_kT(xamp, vsini, ydeg, 0, nk, map_._inc, theta_)
    assert_allclose(starry_kT, kT)


def test_compare_starry_dot_dmfmi():
    spec_ = map_._spectrum
    matrix = np.reshape(spec_, (-1,))
    starry_res = map_.ops.dot_design_matrix_fixed_map_into(
        map_._inc, theta_, map_._veq, map_._u, map_._y, matrix
    )
    # get kT as input
    kT = map_.ops.get_kT(map_._inc, theta_, map_._veq, map_._u)
    res = dot_design_matrix_fixed_map_into(
        kT, map_._y, 1, map_.nw0_, nt, nk, map_.nw_, matrix
    )
    assert_allclose(starry_res, res)


def test_compare_starry_flux_dotconv():
    starry_flux = map_.ops.get_flux_from_dotconv(
        map_._inc, theta_, map_._veq, map_._u, map_._y, map_._spectrum
    )
    # get flux (before reshape) as input
    matrix = np.reshape(map_._spectrum, (-1,))
    flux_input = map_.ops.dot_design_matrix_fixed_map_into(
        map_._inc, theta_, map_._veq, map_._u, map_._y, matrix
    )
    flux = get_flux_from_dotconv(flux_input, nt, map_.nw_)
    assert_allclose(starry_flux, flux)
