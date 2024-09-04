"""
Doppler inference tests.
"""

import pytest

import jax
import jax.numpy as jnp
import numpy as np
import scipy

from jaxoplanet.types import Array
from jaxoplanet.test_utils import assert_allclose

import starry
from starry._core.math import greedy_linalg

from jaxodi.doppler_inference import (
    get_kT,
    get_D_fixed_spectrum,
    sparse_dot,
    design_matrix,
    _get_S,
    cho_solve,
    map_solve,
    process_inputs,
    solve_for_map_linear,
    get_default_theta,
    solve_bilinear,
    solve
)

from jaxodi.doppler_forward import get_kT

jax.config.update("jax_enable_x64", True)

# import importlib.resources
# import jaxodi

# TEST_DATA_S = str(importlib.resources.files(jaxodi).joinpath("tests", "map_solve_S_input.npy"))
# TEST_DATA_FLUX = str(importlib.resources.files(jaxodi).joinpath("tests", "map_solve_flux_input.npy"))
# TEST_DATA_CHO_C = str(importlib.resources.files(jaxodi).joinpath("tests", "map_solve_cho_C_input.npy"))
# TEST_DATA_MU = str(importlib.resources.files(jaxodi).joinpath("tests", "map_solve_mu_input.npy"))
# TEST_DATA_INVL = str(importlib.resources.files(jaxodi).joinpath("tests", "map_solve_invL_input.npy"))

# @pytest.fixture(autouse=True)
# def saved_input_data():

#     with open(TEST_DATA_S, "rb") as f:
#         S = jnp.load(f)
#     with open(TEST_DATA_FLUX, "rb") as f:
#         flux = jnp.load(f)
#     with open(TEST_DATA_CHO_C, "rb") as f:
#         cho_C = jnp.load(f)
#     with open(TEST_DATA_MU, "rb") as f:
#         mu = jnp.load(f)
#     with open(TEST_DATA_INVL, "rb") as f:
#         invL = jnp.load(f)

#     return (S, flux, cho_C, mu, invL)

import os
cwd = os.getcwd()

@pytest.fixture(autouse=True)
def saved_input_data():

    with open(f"{cwd}/tests/map_solve_S_input.npy", "rb") as f:
        S = jnp.load(f)
    with open(f"{cwd}/tests/map_solve_flux_input.npy", "rb") as f:
        flux = jnp.load(f)
    with open(f"{cwd}/tests/map_solve_cho_C_input.npy", "rb") as f:
        cho_C = jnp.load(f)
    with open(f"{cwd}/tests/map_solve_mu_input.npy", "rb") as f:
        mu = jnp.load(f)
    with open(f"{cwd}/tests/map_solve_invL_input.npy", "rb") as f:
        invL = jnp.load(f)

    return (S, flux, cho_C, mu, invL)


# Set the seed.
np.random.seed(0)

# Set arguments for instantiating a Doppler map.
kwargs = dict(
    ydeg=15,
    udeg=0,
    nc=1,
    veq=60000,
    inc=40,
    vsini_max=50000,
    nt=16,
    wav=np.linspace(642.85, 643.15, 70),
    wav0=np.linspace(642.74, 643.26, 300),
)

# Instantiate the Doppler map.
map = starry.DopplerMap(lazy=False, **kwargs)

# No limb darkening at this stage.
# get_kT doesn't yet have limb darkening implemented.

# Rest frame spectrum.
spectrum = (
    1.0
    - 0.85 * np.exp(-0.5 * (map.wav0 - 643.0) ** 2 / 0.0085 ** 2)
    - 0.40 * np.exp(-0.5 * (map.wav0 - 642.97) ** 2 / 0.0085 ** 2)
    - 0.20 * np.exp(-0.5 * (map.wav0 - 643.1) ** 2 / 0.0085 ** 2)
)

# Load the component maps.
map.load(spectra=spectrum, smoothing=0.075)

# Get rotational phases.
theta = np.linspace(-180, 180, map.nt, endpoint=False)

# Generate unnormalized data. Scale the error so it's
# the same magnitude relative to the baseline as the
# error in the normalized dataset so we can more easily
# compare the inference results in both cases.
flux_err = 2e-4
flux = map.flux(theta=theta, normalize=False)
flux_err = flux_err * np.median(flux)
flux += flux_err * np.random.randn(*flux.shape)

# Set test data.
vsini_max = 50000
nt = 16
xamp = map.ops.xamp # (253,)
nwp = map.ops.nwp # 603
vsini = map.vsini # 38567.256581192356
ydeg = map.ydeg # 15
udeg = map.udeg # 0
nk = map.ops.nk # 253
nw = 351 # != map.nw = 70

_angle_factor = np.pi / 180
fix_spectrum = True
normalized = False
_interp = True
baseline_var = 0
_S0e2i = jnp.array(map._S0e2i.toarray())
_Si2eBlk = jnp.array(map._Si2eBlk.toarray())


def test_get_D_fixed_spectrum():

    # Get calculated output.
    calc_doppler_spectrum = get_D_fixed_spectrum(
        xamp, vsini, ydeg, udeg, nk, map._inc, theta, map._spectrum, map.nc, nwp, map.nt, map.Ny, nw
    )

    # Get expected output.
    exp_doppler_spectrum = map.ops.get_D_fixed_spectrum(map._inc, theta, map._veq, map._u, map._spectrum)
    
    # Compare calculated and expected results.
    assert_allclose(calc_doppler_spectrum, exp_doppler_spectrum)


# 29/7 -> pass unjitted, fail jitted
def test_sparse_dot():

    sparse_matrix = jnp.where(jax.random.uniform(jax.random.PRNGKey(0), (100, 100)) > 0.95, 1.0, 0.0)
    sparse_csr_matrix = sparse_csr_matrix = scipy.sparse.csr_matrix(np.array(sparse_matrix))
    dense_matrix = jax.random.normal(jax.random.PRNGKey(1), (100, 50))

    # Get calculated output.
    calc_D = sparse_dot(sparse_matrix, dense_matrix)

    # Get expected output.
    exp_D = map._math.sparse_dot(sparse_csr_matrix, dense_matrix)
    
    # Compare calculated and expected results.
    assert_allclose(calc_D, exp_D)


def test_design_matrix():

    # _interp = True
    theta_ = theta / _angle_factor

    # Get calculated output.
    calc_dm = design_matrix(
        theta_, _angle_factor, xamp, vsini, ydeg, udeg, nk, map._inc,
        map._spectrum, map.nc, nwp, map.nt, map.Ny, nw, _interp, _Si2eBlk, fix_spectrum=True
    )

    # Get expected output.
    exp_dm = map.design_matrix(theta_, fix_spectrum=True)
    
    # Compare calculated and expected results.
    assert_allclose(calc_dm, exp_dm)


# # Not sure how to call starry's equivalent function
# def test_get_S():

#     # Get calculated output.
#     calc_S = _get_S(
#         theta, _angle_factor, xamp, vsini, ydeg, udeg, nk, map._inc, map._spectrum,
#         map.nc, nwp, nt, map.Ny, nw, _interp, _Si2eBlk, fix_spectrum,
#     )

#     # Get expected output.
#     map._solver.theta = theta * _angle_factor
#     exp_S = map._solver._get_S()

#     # Compare calculated and expected results.
#     assert_allclose(calc_S, exp_S)


# 22/7 -> pass
def test_get_default_theta():

    # Get calculated outputs.
    calc_theta = get_default_theta(theta, _angle_factor)
    
    # Get expected outputs.
    exp_theta = map._get_default_theta(theta)

    # Compare calculated and expected results.
    assert_allclose(calc_theta, exp_theta)


# 22/7 -> pass
def test_process_inputs():

    # Get calculated outputs.
    calc_processed = process_inputs(
        flux, map.nt, map.nw, map.nc, map.Ny, map.nw0, map.nw0_, _S0e2i, normalized=False, flux_err=flux_err,
    )

    # Get expected outputs.
    map._solver.reset()
    map._solver.process_inputs(
        flux, normalized=False, flux_err=flux_err,
    )

    # Compare calculated and expected results.
    assert_allclose(map._solver.flux, calc_processed['flux'])
    assert_allclose(map._solver.flux_err, calc_processed['flux_err'])
    assert_allclose(map._solver.spatial_mean, calc_processed['spatial_mean'])
    assert_allclose(map._solver.spatial_cov, calc_processed['spatial_cov'])
    assert_allclose(map._solver.spatial_inv_cov, calc_processed['spatial_inv_cov'])
    assert_allclose(map._solver.spectral_mean, calc_processed['spectral_mean'])
    assert_allclose(map._solver.spectral_cov, calc_processed['spectral_cov'])
    assert_allclose(map._solver.spectral_inv_cov, calc_processed['spectral_inv_cov'])
    assert_allclose(map._solver.T, calc_processed['T'])
    assert_allclose(map._solver.linear, calc_processed['linear'])


# 22/7 -> pass
def test_map_solve(saved_input_data):

    # Load the inputs.
    S, flux, cho_C, mu, invL = saved_input_data

    # Get calculated outputs.
    calc_mean, calc_cho_ycov = map_solve(S, flux, cho_C, mu, invL)

    # Get expected outputs.
    exp_mean, exp_cho_ycov = greedy_linalg.solve(S, flux, cho_C, mu, invL)

    # Compare calculated and expected results.
    assert_allclose(calc_mean, exp_mean)
    assert_allclose(calc_cho_ycov, exp_cho_ycov)


# 22/7 -> pass
def test_solve_for_map_linear():

    T = 1
    theta_ = theta * _angle_factor

    # Process the inputs.
    processed_inputs = process_inputs(
        flux, map.nt, map.nw, map.nc, map.Ny, map.nw0, map.nw0_, _S0e2i,
        normalized=False, flux_err=flux_err,
    )
    # flux = processed_inputs["flux"] # doesn't change in this case
    spatial_mean = processed_inputs["spatial_mean"]
    spatial_inv_cov = processed_inputs["spatial_inv_cov"]
    flux_err_ = processed_inputs["flux_err"]

    # Get calculated outputs.
    calc_y, calc_cho_ycov = solve_for_map_linear(
        spatial_mean, spatial_inv_cov, flux_err_, map.nt, nw, map.nw, T, flux,
        theta_, _angle_factor, xamp, vsini, ydeg, udeg, nk, map._inc, map._spectrum,
        map.nc, nwp, map.Ny, _interp, _Si2eBlk, fix_spectrum,
    )

    # Get expected outputs.
    map._solver.theta = theta * _angle_factor
    map._solver.process_inputs(flux, normalized=False, fix_spectrum=True, flux_err=flux_err_)
    map._solver.spectrum_ = map.spectrum_
    map._solver.solve_for_map_linear()

    exp_y = map._solver.y
    exp_cho_ycov = map._solver.cho_ycov

    # Compare calculated and expected results.
    assert_allclose(calc_y, exp_y)
    assert_allclose(calc_cho_ycov, exp_cho_ycov)


# 22/7 -> pass
def test_solve_bilinear():

    theta_ = theta * _angle_factor

    # Get calculated outputs.
    calc_y, calc_cho_ycov = solve_bilinear(
        flux, nt, nw, map.nw, map.nc, map.Ny, map.nw0, map.nw0_, _S0e2i, flux_err, normalized,
        theta_, _angle_factor, xamp, vsini, ydeg, udeg, nk, map._inc, map._spectrum,
        nwp, _interp, _Si2eBlk, fix_spectrum,
    )

    # Get expected outputs.
    metadata = map._solver.solve_bilinear(
        flux, theta_, map.y, map.spectrum_, map._veq, map._inc, map._u,
        fix_spectrum=True,
        flux_err=flux_err,
        normalized=False,
    )
    exp_y = metadata["y"]
    exp_cho_ycov = metadata["cho_ycov"]

    # Compare calculated and expected results.
    assert_allclose(calc_y, exp_y)
    assert_allclose(calc_cho_ycov, exp_cho_ycov)


# 22/7 -> pass
def test_solve():

    # Get calculated outputs.
    calc_y, calc_cho_ycov = solve(
        flux, map.nt, nw, map.nw, map.nc, map.Ny, map.nw0, map.nw0_, _S0e2i, flux_err, normalized,
        theta, _angle_factor, xamp, vsini, ydeg, udeg, nk, map._inc, map._spectrum, nwp, _interp, _Si2eBlk, fix_spectrum,
    )

    # Get expected outputs.
    exp_solution = map.solve(
        flux, theta=theta, normalized=False, fix_spectrum=True, flux_err=flux_err, quiet="false"
    )

    # Compare calculated and expected results.
    assert_allclose(calc_y, exp_solution["y"])
    assert_allclose(calc_cho_ycov, exp_solution["cho_ycov"])
