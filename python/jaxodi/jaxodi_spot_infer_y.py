# -*- coding: utf-8 -*-
"""
Setup for the SPOT problem.

"""
import starry
import numpy as np
import os

from jaxodi.paparazzi.src.scripts.utils.generate import generate_data
from jaxodi.paparazzi.src.scripts.utils.plot import plot_timeseries, plot_maps
from jaxodi.fig7_functions import solve

import jax
import jax.numpy as jnp
import scipy

jax.config.update("jax_enable_x64", True)

# Generate the synthetic dataset
data = generate_data()
y_true = data["truths"]["y"]
spectrum_true = data["truths"]["spectrum"]
theta = data["data"]["theta"]
flux = data["data"]["flux0"]
flux_err = data["data"]["flux0_err"]

# Instantiate the map
map = starry.DopplerMap(lazy=False, **data["kwargs"])
map.spectrum = data["truths"]["spectrum"]
for n in range(map.udeg):
    map[1 + n] = data["props"]["u"][n]

# Solve for the Ylm coeffs
starry_solution = map.solve(
    flux,
    theta=theta,
    normalized=False,
    fix_spectrum=True,
    flux_err=flux_err,
    quiet=os.getenv("CI", "false") == "true",
)

# # Get the inferred map
# starry_y_inferred = map.y

# # Compute the Ylm expansion of the posterior standard deviation field
# P = map.sht_matrix(inverse=True)
# Q = map.sht_matrix()
# L = np.tril(soln["cho_ycov"])
# W = P @ L
# y_uncert = Q @ np.sqrt(np.diag(W @ W.T))

# # Plot the maps
# starry_fig = plot_maps(y_true, starry_y_inferred)
# starry_fig.savefig("paparazzi/src/tex/figures/spot_infer_y_maps_starry.pdf", bbox_inches="tight", dpi=150)

# # Plot the timeseries
# s_fig = plot_timeseries(data, starry_y_inferred, spectrum_true, normalized=False)
# s_fig.savefig("paparazzi/src/tex/figures/spot_infer_y_starry_timeseries.pdf", bbox_inches="tight", dpi=300)


# --------------- jaxodi ---------------
wav = data["kwargs"]["wav"]
wav0 = data["kwargs"]["wav0"]
vsini_max = 50000
nw = 70
_angle_factor = np.pi / 180
fix_spectrum = True
normalized = False
baseline_var = 0
_S0e2i = jnp.array(map._S0e2i.toarray())
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

# Get calculated outputs.
jaxodi_y_inferred, jaxodi_cho_ycov = solve(
    flux, map.nt, map.nw, map.nc, map.Ny, map.nw0, map.nw0_, _S0e2i, flux_err, normalized,
    fix_spectrum, baseline_var, S,
    theta, _angle_factor,
)

# Plot the maps
jaxodi_fig = plot_maps(y_true, jnp.squeeze(jaxodi_y_inferred))
jaxodi_fig.savefig("paparazzi/src/tex/figures/spot_infer_y_maps_jaxodi.pdf", bbox_inches="tight", dpi=150)
j_fig = plot_timeseries(data, jnp.squeeze(jaxodi_y_inferred), spectrum_true, normalized=False)
j_fig.savefig("paparazzi/src/tex/figures/spot_infer_y_jaxodi_timeseries.pdf", bbox_inches="tight", dpi=300)
