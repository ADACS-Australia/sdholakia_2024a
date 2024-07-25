"""
Tests for Figure 7's map.solve() and dependencies.
"""

import pytest

import jax
import jax.numpy as jnp
import numpy as np
import scipy

from jaxoplanet.types import Array
from jaxoplanet.test_utils import assert_allclose

# To generate variables ---
# import paparazzi
import starry
from starry._core.math import greedy_linalg

from jaxodi.paparazzi.src.scripts.utils.generate import generate_data

from jaxodi.fig7_functions import (
    cho_solve,
    map_solve,
    process_inputs,
    solve_for_map_linear,
    get_default_theta,
    solve_bilinear,
    solve
)

jax.config.update("jax_enable_x64", True)


@pytest.fixture(autouse=True)
def saved_input_data():

    with open("map_solve_S_input.npy", "rb") as f:
        S = jnp.load(f)
    with open("map_solve_flux_input.npy", "rb") as f:
        flux = jnp.load(f)
    with open("map_solve_cho_C_input.npy", "rb") as f:
        cho_C = jnp.load(f)
    with open("map_solve_mu_input.npy", "rb") as f:
        mu = jnp.load(f)
    with open("map_solve_invL_input.npy", "rb") as f:
        invL = jnp.load(f)

    return (S, flux, cho_C, mu, invL)


@pytest.fixture(autouse=True)
def saved_output_data():

    with open("map_solve_mean_output.npy", "rb") as f:
        exp_mean = np.load(f)
    with open("map_solve_cho_ycov_output.npy", "rb") as f:
        exp_cho_ycov = np.load(f)
    
    return (exp_mean, exp_cho_ycov)


@pytest.fixture(autouse=True)
def generated_data():

    data = generate_data()

    # truth
    # y_true = data["truths"]["y"]
    # spectrum_true = data["truths"]["spectrum"]

    # data
    theta = data["data"]["theta"] # np.linspace(-180, 180, nt=16, endpoint=False)
    flux = data["data"]["flux0"]
    flux_err = data["data"]["flux0_err"]

    # kwargs
    wav = data["kwargs"]["wav"] # array (70,1)
    wav0 = data["kwargs"]["wav0"]

    # Instantiate the map
    map = starry.DopplerMap(lazy=False, **data["kwargs"])
    map.spectrum = data["truths"]["spectrum"]
    for n in range(map.udeg):
        map[1 + n] = data["props"]["u"][n]
    
    # Provided values
    # ydeg = 15
    # udeg = 2

    # veq = 60000
    # inc = 40

    vsini_max = 50000

    # nc = 1
    # nt = 16

    # Ny = 256
    # Nu = 3
    # N = 324
    nw = 70
    # nw0 = 300
    # nw0_ = 603

    _angle_factor = np.pi / 180

    fix_spectrum = True
    normalized = False
    baseline_var = 0
    _S0e2i = jnp.array(map._S0e2i.toarray())

    # ------------------------
    # Much code for setting S.
    nw = 351
    vsini_max = 50000
    wav1 = np.min(wav)
    wav2 = np.max(wav)
    wavr = np.exp(0.5 * (np.log(wav1) + np.log(wav2)))
    log_wav = jnp.linspace(np.log(wav1 / wavr), jnp.log(wav2 / wavr), nw)
    wav_int = wavr * np.exp(log_wav)
    interp_tol = 1e-12
    _clight = 299792458.0  # m/s
    dlam = log_wav[1] - log_wav[0]
    betasini_max = vsini_max / _clight
    hw = jnp.array(
        np.ceil(
            0.5
            * jnp.abs(jnp.log((1 + betasini_max) / (1 - betasini_max)))
            / dlam
        ),
        dtype="int32",
    )
    x = jnp.arange(0, hw + 1) * dlam
    pad_l = log_wav[0] - hw * dlam + x[:-1]
    pad_r = log_wav[-1] + x[1:]
    log_wav0_int = jnp.concatenate([pad_l, log_wav, pad_r])
    wav0_int = wavr * jnp.exp(log_wav0_int)
    wav0 = jnp.array(wav0)
    # nw0 = len(wav0)
    # nwp = len(log_wav0_int)

    S = map._get_spline_operator(wav_int, wav)
    S[np.abs(S) < interp_tol] = 0
    S = scipy.sparse.csr_matrix(S)
    S = map._get_spline_operator(wav0_int, wav0)
    S[np.abs(S) < interp_tol] = 0
    S = scipy.sparse.csr_matrix(S)
    S = map._get_spline_operator(wav, wav_int)
    S[np.abs(S) < interp_tol] = 0
    S = scipy.sparse.csr_matrix(S)
    S = map._get_spline_operator(wav0, wav0_int)
    S[np.abs(S) < interp_tol] = 0
    S = scipy.sparse.csr_matrix(S)

    S = jnp.array(S.toarray())
    # ----------------------------

    return (map, theta, _angle_factor, flux, flux_err, wav, wav0, fix_spectrum, normalized, baseline_var, S, _S0e2i)


# 19/7 -> 
def test_get_default_theta_against_data():

    pass


# 22/7 -> pass
def test_get_default_theta_compare_stary(generated_data):

    # Load the inputs.
    map, theta, _angle_factor, flux, flux_err, wav, wav0, fix_spectrum, normalized, baseline_var, S, _S0e2i = generated_data

    # Get calculated outputs.
    calc_theta = get_default_theta(theta, _angle_factor)
    
    # Get expected outputs.
    exp_theta = map._get_default_theta(theta)

    # Compare calculated and expected results.
    assert_allclose(calc_theta, exp_theta)


# 22/7 -> 
def test_process_inputs_against_data():

    pass


# 22/7 -> pass
def test_process_inputs_compare_starry(generated_data):

    # Load the inputs.
    map, theta, _angle_factor, flux, flux_err, wav, wav0, fix_spectrum, normalized, baseline_var, S, _S0e2i = generated_data

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


# TODO: how to get inputs/outputs to compare against?
def test_cho_solve():

    # cho_solve()
    pass


# 22/7 -> pass
def test_map_solve_against_data(saved_input_data, saved_output_data):

    # Load the inputs.
    S, flux, cho_C, mu, invL = saved_input_data

    # Load saved expected outputs.
    exp_mean, exp_cho_ycov = saved_output_data

    # Get calculated outputs.
    calc_mean, calc_cho_ycov = map_solve(S, flux, cho_C, mu, invL)

    # Compare calculated and expected results.
    assert_allclose(calc_mean, exp_mean)
    assert_allclose(calc_cho_ycov, exp_cho_ycov)


# 22/7 -> pass
def test_map_solve_compare_starry(saved_input_data):

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
def test_solve_for_map_linear_against_data(generated_data, saved_output_data):

    # Load the inputs.
    map, theta, _angle_factor, flux, flux_err, wav, wav0, fix_spectrum, normalized, baseline_var, S, _S0e2i = generated_data
    T = 1
    # TODO: S is dummy variable, passed in but not used until my functions are combined with Tiger's.

    # Process the inputs.
    processed_inputs = process_inputs(
        flux, map.nt, map.nw, map.nc, map.Ny, map.nw0, map.nw0_, _S0e2i,
        normalized=False, flux_err=flux_err,
    )
    # flux = processed_inputs["flux"] # doesn't change in this case
    spatial_mean = processed_inputs["spatial_mean"]
    spatial_inv_cov = processed_inputs["spatial_inv_cov"]
    flux_err = processed_inputs["flux_err"]

    # Get calculated outputs.
    calc_y, calc_cho_ycov = solve_for_map_linear(
        spatial_mean, spatial_inv_cov, flux_err, map.nt, map.nw, baseline_var, T, flux, S, map.nc, map.Ny,
    )

    # Load saved expected outputs.
    exp_mean, exp_cho_ycov = saved_output_data
    exp_y = np.transpose(np.reshape(exp_mean, (map.nc, map.Ny)))

    # Compare calculated and expected results.
    assert_allclose(calc_y, exp_y)
    assert_allclose(calc_cho_ycov, exp_cho_ycov)


# 22/7 -> pass
def test_solve_for_map_linear_compare_starry(generated_data):

    # Load the inputs.
    map, theta, _angle_factor, flux, flux_err, wav, wav0, fix_spectrum, normalized, baseline_var, S, _S0e2i = generated_data
    T = 1
    # TODO: S is dummy variable, passed in but not used until my functions are combined with Tiger's.

    # Process the inputs.
    processed_inputs = process_inputs(
        flux, map.nt, map.nw, map.nc, map.Ny, map.nw0, map.nw0_, _S0e2i,
        normalized=False, flux_err=flux_err,
    )
    # flux = processed_inputs["flux"] # doesn't change in this case
    spatial_mean = processed_inputs["spatial_mean"]
    spatial_inv_cov = processed_inputs["spatial_inv_cov"]
    flux_err = processed_inputs["flux_err"]

    # Get calculated outputs.
    calc_y, calc_cho_ycov = solve_for_map_linear(
        spatial_mean, spatial_inv_cov, flux_err, map.nt, map.nw, baseline_var, T, flux, S, map.nc, map.Ny,
    )

    # Get expected outputs.
    map._solver.theta = theta * _angle_factor
    map._solver.process_inputs(flux, normalized=False, fix_spectrum=True, flux_err=flux_err)
    map._solver.spectrum_ = map.spectrum_
    map._solver.solve_for_map_linear()

    exp_y = map._solver.y
    exp_cho_ycov = map._solver.cho_ycov

    # Compare calculated and expected results.
    assert_allclose(calc_y, exp_y)
    assert_allclose(calc_cho_ycov, exp_cho_ycov)


# 22/7 -> pass
def test_solve_bilinear_against_data(generated_data, saved_output_data):

    # Process the inputs.
    map, theta, _angle_factor, flux, flux_err, wav, wav0, fix_spectrum, normalized, baseline_var, S, _S0e2i = generated_data

    # Get calculated outputs.
    calc_y, calc_cho_ycov = solve_bilinear(
        flux, map._nt, map.nw, map.nc, map.Ny, map.nw0, map.nw0_, _S0e2i, flux_err, normalized,
        fix_spectrum, baseline_var, S,
    )

    # Load saved expected outputs.
    exp_mean, exp_cho_ycov = saved_output_data
    exp_y = np.transpose(np.reshape(exp_mean, (map.nc, map.Ny)))

    # Compare calculated and expected results.
    assert_allclose(calc_y, exp_y)
    assert_allclose(calc_cho_ycov, exp_cho_ycov)


# 22/7 -> pass
def test_solve_bilinear_compare_starry(generated_data):

    # Process the inputs.
    map, theta, _angle_factor, flux, flux_err, wav, wav0, fix_spectrum, normalized, baseline_var, S, _S0e2i = generated_data
    theta = theta * _angle_factor

    # Get calculated outputs.
    calc_y, calc_cho_ycov = solve_bilinear(
        flux, map._nt, map.nw, map.nc, map.Ny, map.nw0, map.nw0_, _S0e2i, flux_err, normalized,
        fix_spectrum, baseline_var, S,
    )

    # Get expected outputs.
    metadata = map._solver.solve_bilinear(
        flux, theta, map.y, map.spectrum_, map._veq, map._inc, map._u,
        fix_spectrum=True,
        flux_err=flux_err,
        normalized=False,
    )
    
    exp_y = metadata["y"]
    exp_cho_ycov = metadata["cho_ycov"]

    # Compare calculated and expected results.
    assert_allclose(calc_y, exp_y)
    assert_allclose(calc_cho_ycov, exp_cho_ycov)


def test_solve_against_data():

    pass


# 22/7 -> pass
def test_solve_compare_starry(generated_data):

    # Process the inputs.
    map, theta, _angle_factor, flux, flux_err, wav, wav0, fix_spectrum, normalized, baseline_var, S, _S0e2i = generated_data

    # Get calculated outputs.
    calc_y, calc_cho_ycov = solve(
        flux, map.nt, map.nw, map.nc, map.Ny, map.nw0, map.nw0_, _S0e2i, flux_err, normalized,
        fix_spectrum, baseline_var, S,
        theta, _angle_factor,
    )

    # Get expected outputs.
    exp_solution = map.solve(
        flux, theta=theta, normalized=False, fix_spectrum=True, flux_err=flux_err, quiet="false"
    )

    # Compare calculated and expected results.
    assert_allclose(calc_y, exp_solution["y"])
    assert_allclose(calc_cho_ycov, exp_solution["cho_ycov"])
