"""
Tests for Figure 7's map.solve() and dependencies.

19/7 Update
-----------

test_get_default_theta      |                   pass by call    |
test_process_inputs         |   pass by data                    |
test_map_solve              |                   fail by call    |
test_solve_for_map_linear   |   pass by data    fail by call    |
test_solve_bilinear         |   pass by data                    |
test_solve                  |                   pass by call    |


TODO:

    - load saved data in data()
    - create separate tests for data loads and function calls


"""

import pytest

# import jax
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


# TODO: load saved data.
@pytest.fixture(autouse=True)
def data():

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
    N = 324
    nw = 70
    # nw0 = 300
    # nw0_ = 603

    _angle_factor = np.pi / 180

    # ------------------------
    # Much code for setting S.
    nw = 351
    vsini_max = 50000
    nt = 16
    wav1 = np.min(wav)
    wav2 = np.max(wav)
    wavr = np.exp(0.5 * (np.log(wav1) + np.log(wav2)))
    log_wav = np.linspace(np.log(wav1 / wavr), np.log(wav2 / wavr), nw)
    wav_int = wavr * np.exp(log_wav)
    interp_tol = 1e-12
    _clight = 299792458.0  # m/s
    dlam = log_wav[1] - log_wav[0]
    betasini_max = vsini_max / _clight
    hw = np.array(
        np.ceil(
            0.5
            * np.abs(np.log((1 + betasini_max) / (1 - betasini_max)))
            / dlam
        ),
        dtype="int32",
    )
    x = np.arange(0, hw + 1) * dlam
    pad_l = log_wav[0] - hw * dlam + x[:-1]
    pad_r = log_wav[-1] + x[1:]
    log_wav0_int = np.concatenate([pad_l, log_wav, pad_r])
    wav0_int = wavr * np.exp(log_wav0_int)
    nwp = len(log_wav0_int)
    wav0 = np.array(wav0)
    nw0 = len(wav0)
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
    # ----------------------------
    
    return (map, theta, _angle_factor, flux, flux_err, wav, wav0, S)


# 19/7 -> load data     -> 
#      -> call starry   -> pass
def test_get_default_theta(data):

    # Load the inputs.
    map, theta, _angle_factor, flux, flux_err, wav, wav0, S = data

    # Get calculated outputs.
    calc_theta = get_default_theta(theta, _angle_factor)
    
    # NOTE: To get expected outputs, can either call starry function OR load saved data.
    # Get expected outputs.
    exp_theta = map._get_default_theta(theta)
    # Compare calculated and expected results.
    assert_allclose(calc_theta, exp_theta)


# 19/7 -> load data     -> pass
#      -> call starry   -> 
def test_process_inputs(data):

    # Load the inputs.
    map, theta, _angle_factor, flux, flux_err, wav, wav0, S = data

    # Get calculated outputs.
    calc_processed = process_inputs(
        flux, map.nt, map.nw, map.nc, map.Ny, map.nw0, map.nw0_, map._S0e2i, normalized=False, flux_err=flux_err,
    )

    # NOTE: To get expected outputs, can either call starry function OR load saved data.
    # Get expected outputs.
    map._solver.reset()
    map._solver.process_inputs(
        flux, normalized=False, flux_err=flux_err,
    )

    # TODO: What is this for? Do I still need it?
    # print(f"test_process_inputs::exp_linear: {map._solver.linear}")
    # print(f"test_process_inputs::calc_linear: {calc_processed['linear']}")

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


# TODO: write test
def test_cho_solve(data):

    # cho_solve()
    pass


# 19/7 -> load data     -> 
#      -> call starry   -> fail
def test_map_solve(data):

    # Load the inputs.
    map, theta, _angle_factor, flux, flux_err, wav, wav0, S = data
    with open("map_solve_S_input.npy", "rb") as f:
        S = np.load(f)
    with open("map_solve_flux_input.npy", "rb") as f:
        flux = np.load(f)
    with open("map_solve_cho_C_input.npy", "rb") as f:
        cho_C = np.load(f)
    with open("map_solve_mu_input.npy", "rb") as f:
        mu = np.load(f)
    with open("map_solve_invL_input.npy", "rb") as f:
        invL = np.load(f)
    
    # Convert them to jax arrays?
    S = jnp.array(S)
    flux = jnp.array(flux)
    cho_C = jnp.array(cho_C)
    mu = jnp.array(mu)
    invL = jnp.array(invL)

    # Get calculated outputs.
    calc_mean, calc_cho_ycov = map_solve(S, flux, cho_C, mu, invL)

    # NOTE: To get expected outputs, can either call starry function OR load saved data.

    # Get expected outputs.
    exp_mean, exp_cho_ycov = greedy_linalg.solve(S, flux, cho_C, mu, invL)

    # TODO: Try loading data. Not sure if this code is correct?
    # # Load saved expected outputs.
    # with open("map_solve_mean_output.npy", "rb") as f:
    #     exp_mean = np.load(f)
    # with open("map_solve_cho_ycov_output.npy", "rb") as f:
    #     exp_cho_ycov = np.load(f)

    # Compare calculated and expected results.
    assert_allclose(calc_mean, exp_mean)
    assert_allclose(calc_cho_ycov, exp_cho_ycov)


# 19/7 -> load data     -> pass
#      -> call starry   -> fail
def test_solve_for_map_linear(data):

    # Load the inputs.
    map, theta, _angle_factor, flux, flux_err, wav, wav0, S = data
    T = 1
    baseline_var = 0
    S = "dummy"

    # Process the inputs.
    processed_inputs = process_inputs(
        flux, map.nt, map.nw, map.nc, map.Ny, map.nw0, map.nw0_, map._S0e2i,
        normalized=False, flux_err=flux_err,
    )
    flux = processed_inputs["flux"] # doesn't change in this case
    spatial_mean = processed_inputs["spatial_mean"]
    spatial_inv_cov = processed_inputs["spatial_inv_cov"]
    flux_err = processed_inputs["flux_err"]

    # Get calculated outputs.
    calc_y, calc_cho_ycov = solve_for_map_linear(
        spatial_mean, spatial_inv_cov, flux_err, map.nt, map.nw, baseline_var, T, flux, S, map.nc, map.Ny,
    )

    # NOTE: To get expected outputs, can either call starry function OR load saved data.

    # TODO: Fix spectrum is None error.
    # Get expected outputs.
    # map._solver.theta = theta * _angle_factor
    # map._solver.process_inputs(flux)
    # exp_y, exp_cho_ycov = map._solver.solve_for_map_linear() # --> ERROR: self.spectrum is None when design_matrix() is called

    # Load saved expected outputs.
    with open("map_solve_mean_output.npy", "rb") as f:
        exp_mean = np.load(f)
    with open("map_solve_cho_ycov_output.npy", "rb") as f:
        exp_cho_ycov = np.load(f)
    exp_y = np.transpose(np.reshape(exp_mean, (map.nc, map.Ny)))

    # Compare calculated and expected results.
    assert_allclose(calc_y, exp_y)
    assert_allclose(calc_cho_ycov, exp_cho_ycov)


# 19/7 -> load data     -> pass
#      -> call starry   -> ?
def test_solve_bilinear(data):

    # Process the inputs.
    map, theta, _angle_factor, flux, flux_err, wav, wav0, S = data
    fix_spectrum = True
    baseline_var = 0

    # Get calculated outputs.
    calc_y, calc_cho_ycov = solve_bilinear(
        flux, map._nt, map.nw, map.nc, map.Ny, map.nw0, map.nw0_, map._S0e2i, flux_err,
        fix_spectrum, baseline_var, S,
    )

    # NOTE: To get expected outputs, can either call starry function OR load saved data.
    # Get expected outputs.
    # soln = map._solver.solve_bilinear(
    #             flux, theta, map.y, map.spectrum_, map._veq, map._inc, map._u, **kwargs   # TODO: fix inputs
    #         )
    # Load saved expected outputs.
    with open("map_solve_mean_output.npy", "rb") as f:
        exp_mean = np.load(f)
    with open("map_solve_cho_ycov_output.npy", "rb") as f:
        exp_cho_ycov = np.load(f)
    exp_y = np.transpose(np.reshape(exp_mean, (map.nc, map.Ny)))

    # Compare calculated and expected results.
    assert_allclose(calc_y, exp_y)
    assert_allclose(calc_cho_ycov, exp_cho_ycov)


# 19/7 -> load data     -> 
#      -> call starry   -> pass
def test_solve(data):

    # Process the inputs.
    map, theta, _angle_factor, flux, flux_err, wav, wav0, S = data
    fix_spectrum = True
    baseline_var = 0

    # Get calculated outputs.
    calc_y, calc_cho_ycov = solve(
        flux, map.nt, map.nw, map.nc, map.Ny, map.nw0, map.nw0_, map._S0e2i, flux_err,
        fix_spectrum, baseline_var, S,
        theta, _angle_factor,
    )

    # NOTE: To get expected outputs, can either call starry function OR load saved data.
    # Get expected outputs.
    exp_solution = map.solve(
        flux, theta=theta, normalized=False, fix_spectrum=True, flux_err=flux_err, quiet="false"
    )

    # Compare calculated and expected results.
    assert_allclose(calc_y, exp_solution["y"])
    assert_allclose(calc_cho_ycov, exp_solution["cho_ycov"])
