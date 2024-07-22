"""
Functions for Figure 7's map.solve()
"""

from typing import Tuple
# from collections.abc import Callable
# from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import scipy

from jaxoplanet.types import Array

# To generate variables ---
# import paparazzi
import starry

from jaxodi.paparazzi.src.scripts.utils.generate import generate_data


# Functions
#
#   - cho_solve
#   - map_solve
#   - process_inputs
#   - solve_for_map_linear
#   - get_default_theta
#   - solve_bilinear
#   - solve


# @jax.jit
def cho_solve(A: Array, b: Array) -> Array:
    b_ = jax.scipy.linalg.solve_triangular(A, b, lower=True)
    return jax.scipy.linalg.solve_triangular(jnp.transpose(A), b_, lower=False)


# @jax.jit
def map_solve(
    X: Array,
    flux: Array,
    cho_C: float | Array,
    mu: Array,
    LInv: float | Array
) -> Tuple[Array, Array]:
    """
    Compute the maximum a posteriori (MAP) prediction for the
    spherical harmonic coefficients of a map given a flux timeseries.

    Args:
        X (matrix): The flux design matrix.
        flux (array): The flux timeseries.
        cho_C (scalar/vector/matrix): The lower cholesky factorization
            of the data covariance.
        mu (array): The prior mean of the spherical harmonic coefficients.
        LInv (scalar/vector/matrix): The inverse prior covariance of the
            spherical harmonic coefficients.

    Returns:
        The vector of spherical harmonic coefficients corresponding to the
        MAP solution and the Cholesky factorization of the corresponding
        covariance matrix.

    """
    # Compute C^-1 . X
    if cho_C.ndim == 0:
        CInvX = X / cho_C**2
    elif cho_C.ndim == 1:
        CInvX = jnp.dot(jnp.diag(1 / cho_C**2), X)
    else:
        CInvX = cho_solve(cho_C, X)
    print(f"CInvX: {CInvX}")
    # breakpoint()
    # Compute W = X^T . C^-1 . X + L^-1
    W = jnp.dot(jnp.transpose(X), CInvX)
    print(f"W: {W}")
    # If LInv is a scalar or a 1-dimensional array, increment the
    # diagonal elements of W with the values from LInv.
    if LInv.ndim == 0 or LInv.ndim == 1:
        W = W.at[jnp.diag_indices_from(W)].set(W[jnp.diag_indices_from(W)] + LInv)
        LInvmu = mu * LInv
    # If LInv is a matrix, directly add LInv to W.
    else:
        W += LInv
        LInvmu = jnp.dot(LInv, mu)
    print(f"W: {W}")
    # with open("solve_midpoint_W.npy", "rb") as f:
    #     exp_W = np.load(f)
    # print(f"exp_W: {cho_W}")
    print(f"LInvmu: {LInvmu}")

    # Compute the max like y and its covariance matrix
    cho_W = jax.scipy.linalg.cholesky(W, lower=True)
    # print(f"cho_W: {cho_W}")
    # with open("solve_midpoint_cho_W.npy", "rb") as f:
    #     exp_cho_W = np.load(f)
    # print(f"exp_cho_W: {cho_W}")
    M = cho_solve(cho_W, jnp.transpose(CInvX))
    # print(f"M: {M}")
    yhat = jnp.dot(M, flux) + cho_solve(cho_W, LInvmu)
    # print(f"yhat: {yhat}")
    ycov = cho_solve(cho_W, jnp.eye(cho_W.shape[0]))
    # print(f"ycov: {ycov}")
    cho_ycov = jax.scipy.linalg.cholesky(ycov, lower=True)
    # print(f"cho_ycov: {cho_ycov}")

    return yhat, cho_ycov


def process_inputs(
        flux: Array,
        nt: int,
        nw: int,
        nc: int,
        Ny: int,
        nw0: int,
        nw0_: int,
        S0e2i: Array,
        flux_err: float=None,
        normalized: bool=True,
        baseline=None,
        spatial_mean=None,
        spatial_cov=None,
        spectral_mean=None,
        spectral_cov=None,
        logT0=None,
        logTf=None,
        nlogT=None,
    ):

    # Process defaults
    if flux_err is None:
        flux_err = 1e-4
    if spatial_mean is None:
        spatial_mean = jnp.zeros(Ny)
        spatial_mean = spatial_mean.at[0].set(1.0)
    if spatial_cov is None:
        spatial_cov = 1e-4
    if spectral_mean is None:
        spectral_mean = 1.0
    if spectral_cov is None:
        spectral_cov = 1e-3
    if logT0 is None:
        logT0 = 2
    if logTf is None:
        logTf = 0
    if nlogT is None:
        nlogT = 50
    else:
        nlogT = max(1, nlogT)
    
    # Flux must be a matrix (nt, nw)
    # if nt == 1:
    # else:
    assert jnp.array_equal(
        jnp.shape(flux), jnp.array([nt, nw])
    ), "Invalid shape for `flux`."

    # Flux error may be a scalar, a vector, or a matrix (nt, nw)
    flux_err = jnp.array(flux_err)
    # if flux_err.ndim == 0:
    #     flux_err = flux_err
    # else:

    # Spatial mean may be a scalar, a vector (Ny), or a list of those
    # Reshape it to a matrix of shape (Ny, nc)
    if type(spatial_mean) not in (list, tuple):
        # Use the same mean for all components
        spatial_mean = [spatial_mean for n in range(nc)]
    # else:
    for n in range(nc):
        spatial_mean[n] = jnp.array(spatial_mean[n])
        assert spatial_mean[n].ndim < 2
        spatial_mean[n] = jnp.reshape(
            spatial_mean[n] * jnp.ones(Ny), (-1, 1)
        )
    spatial_mean = jnp.concatenate(spatial_mean, axis=-1)

    # Spatial cov may be a scalar, a vector, a matrix (Ny, Ny),
    # or a list of those. Invert it and reshape to a matrix of
    # shape (Ny, nc) (inverse variances) or a tensor of shape
    # (Ny, Ny, nc) (nc separate inverse covariance matrices)
    if type(spatial_cov) not in (list, tuple):
        # Use the same covariance for all components
        spatial_cov = [spatial_cov for n in range(nc)]
    # else:
    spatial_inv_cov = [None for n in range(nc)]
    ndim = jnp.array(spatial_cov[0]).ndim
    for n in range(nc):
        spatial_cov[n] = jnp.array(spatial_cov[n])
        assert spatial_cov[n].ndim == ndim
        if spatial_cov[n].ndim < 2:
            spatial_inv_cov[n] = jnp.reshape(
                jnp.ones(Ny) / spatial_cov[n], (-1, 1)
            )
            spatial_cov[n] = jnp.reshape(
                jnp.ones(Ny) * spatial_cov[n], (-1, 1)
            )
        # else:
    
    # Tensor of nc (inverse) variance vectors or covariance matrices
    spatial_cov = jnp.concatenate(spatial_cov, axis=-1)
    spatial_inv_cov = jnp.concatenate(spatial_inv_cov, axis=-1)

    # Baseline must be a vector (nt,)
    # if baseline is not None:
    # else:

    # Spectral mean must be a scalar, a vector (nw0), or a list of those
    # Interpolate it to the internal grid (nw0_) and reshape to (nc, nw0_)
    if type(spectral_mean) not in (list, tuple):
        # Use the same mean for all components
        spectral_mean = [spectral_mean for n in range(nc)]
    # else:
    for n in range(nc):
        spectral_mean[n] = jnp.array(spectral_mean[n])
        assert spectral_mean[n].ndim < 2
        spectral_mean[n] = jnp.reshape(
            spectral_mean[n] * jnp.ones(nw0), (-1, 1)
        )
        spectral_mean[n] = S0e2i.dot(spectral_mean[n]).T
    spectral_mean = jnp.concatenate(spectral_mean, axis=0)

    # Spectral cov may be a scalar, a vector, a matrix (nw0, nw0),
    # or a list of those. Interpolate it to the internal grid,
    # then invert it and reshape to a matrix of
    # shape (nc, nw0_) (inverse variances) or a tensor of shape
    # (nc, nw0_, nw0_) (nc separate inverse covariance matrices)
    if type(spectral_cov) not in (list, tuple):
        # Use the same covariance for all components
        spectral_cov = [spectral_cov for n in range(nc)]
    # else:
    spectral_inv_cov = [None for n in range(nc)]
    ndim = jnp.array(spectral_cov[0]).ndim
    for n in range(nc):
        spectral_cov[n] = jnp.array(spectral_cov[n])
        assert spectral_cov[n].ndim == ndim
        if spectral_cov[n].ndim < 2:
            if spectral_cov[n].ndim == 0:
                cov = jnp.ones(nw0_) * spectral_cov[n]
            else:
                cov = S0e2i.dot(spectral_cov[n])
            inv = 1.0 / cov
            spectral_inv_cov[n] = jnp.reshape(inv, (1, -1))
            spectral_cov[n] = jnp.reshape(cov, (1, -1))
        # else:

    # Tensor of nc (inverse) variance vectors or covariance matrices
    spectral_cov = jnp.concatenate(spectral_cov, axis=0)
    spectral_inv_cov = jnp.concatenate(spectral_inv_cov, axis=0)

    # Spectral guess must be a scalar, a vector (nw0), or a list of those
    # Interpolate it to the internal grid (nw0_) and reshape to (nc, nw0_)
    # if spectral_guess is not None:
    # else:

    # Tempering schedule
    if nlogT == 1:
        T = jnp.array([10 ** logTf])
    elif logT0 == logTf:
        T = logTf * jnp.ones(nlogT)
    else:
        T = jnp.logspace(logT0, logTf, nlogT)

    # Are we lucky enough to do a purely linear solve for the map?
    linear = (not normalized) or (baseline is not None)

    processed_inputs = {}

    processed_inputs["flux"] = flux
    processed_inputs["flux_err"] = flux_err
    processed_inputs["spatial_mean"] = spatial_mean
    processed_inputs["spatial_cov"] = spatial_cov
    processed_inputs["spatial_inv_cov"] = spatial_inv_cov
    processed_inputs["spectral_mean"] = spectral_mean
    processed_inputs["spectral_cov"] = spectral_cov
    processed_inputs["spectral_inv_cov"] = spectral_inv_cov
    processed_inputs["T"] = T
    processed_inputs["linear"] = linear

    return processed_inputs


def get_kT(inc, theta, veq, u):
    """
    Get the kernels at an array of angular phases `theta`.
    """

    # enforce_bounds()
    # get_x()
    # get_rT()
    # get_kT0()
    # F() #?
    # right_project()

    # many theano operations

    pass


def get_D_fixed_spectrum(self, inc, theta, veq, u, spectrum):
    """
    Return the Doppler matrix for a fixed spectrum.

    This routine is heavily optimized, and can often be the fastest
    way to compute the flux!

    """
    # # Get the convolution kernels
    # kT = self.get_kT(inc, theta, veq, u)

    # # The dot product is just a 2d convolution!
    # product = tt.nnet.conv2d(
    #     tt.reshape(spectrum, (self.nc, 1, 1, self.nwp)),
    #     tt.reshape(kT, (self.nt * self.Ny, 1, 1, self.nk)),
    #     border_mode="valid",
    #     filter_flip=False,
    #     input_shape=(self.nc, 1, 1, self.nwp),
    #     filter_shape=(self.nt * self.Ny, 1, 1, self.nk),
    # )
    # product = tt.reshape(product, (self.nc, self.nt, self.Ny, self.nw))
    # product = tt.swapaxes(product, 1, 2)
    # product = tt.reshape(product, (self.Ny * self.nc, self.nt * self.nw))
    # product = tt.transpose(product)
    # return product

    pass


def sparse_dot(A, B):
    """
    Performs matrix multiplication, optimising computation time by utilising sparse matrices.
    """

    # if scipy.sparse.issparse(A):
    #     return A.dot(B)
    # elif scipy.sparse.issparse(B):
    #     return (B.T.dot(A.T)).T
    # else:
    #     raise ValueError("At least one input must be sparse.")
    
    # Guess at jax versions:
    if jax.issparse(A):
        return A.dot(B)
    elif jax.issparse(B):
        return (B.T.dot(A.T)).T
    else:
        raise ValueError("At least one input must be sparse.")


def design_matrix(theta, spectrum_, fix_spectrum=True, fix_map=False):
    """
    Return the Doppler imaging design matrix.

    This matrix dots into the spectral map to yield the model for the
    observed spectral timeseries (the ``flux``).
    """
    # theta = _get_default_theta(theta) # this is just undoing what get_S() did!

    # # Compute the Doppler operator
    # if fix_spectrum:
        
    #     # Fixed spectrum (dense)
    #     D = get_D_fixed_spectrum(_inc, theta, _veq, _u, _spectrum)
    #  # Interpolate to the output grid
    # if _interp:
    #     D = map._math.sparse_dot(_Si2eBlk, D)

    # return D

    pass


# S = <603x300 sparse matrix with 1206 stored elements>
def _get_S(theta, _angle_factor, fix_spectrum, spectrum_):

    theta = theta / _angle_factor
    dm = design_matrix(theta=theta, fix_spectrum=True)

    return dm


def solve_for_map_linear(
        spatial_mean: Array,
        spatial_inv_cov: Array,
        flux_err: float,
        nt: int,
        nw: int,
        baseline_var: int,
        T,
        flux: Array,
        S: Array,
        nc: int,
        Ny: int,
    ) -> tuple[Array, Array]:
    """
    Solve for `y` linearly, given a baseline or unnormalized data.
    """
    # Reshape the priors
    mu = jnp.reshape(jnp.transpose(spatial_mean), (-1))
    if spatial_inv_cov.ndim == 2:
        invL = jnp.reshape(jnp.transpose(spatial_inv_cov), (-1))

    # Ensure the flux error is a vector
    if flux_err.ndim == 0:
        flux_err = flux_err * jnp.ones((nt, nw))

    # Factorised data covariance
    if baseline_var == 0:
        cho_C = jnp.reshape(jnp.sqrt(T) * flux_err, (-1,))

    # Unroll the data into a vector
    flux = jnp.reshape(flux, (-1,))

    # Get S
    # S = _get_S()                                          # TODO: _get_S()

    # load from data instead of _get_S()
    with open("map_solve_S_input.npy", "rb") as f:
        S = jnp.load(f)
    
    # Solve the L2 problem
    # mean, cho_ycov = map_solve(S, flux, cho_C, mu, invL)  # TODO: map_solve()

    # load from data instead of map_solve()
    with open("map_solve_mean_output.npy", "rb") as f:
        mean = jnp.load(f)
    with open("map_solve_cho_ycov_output.npy", "rb") as f:
        cho_ycov = jnp.load(f)

    y = jnp.transpose(jnp.reshape(mean, (nc, Ny)))

    return y, cho_ycov


def get_default_theta(
        theta: Array,
        _angle_factor: float,
    ) -> Array:

    return theta * _angle_factor


def solve_bilinear(
        # pass to process_inputs()
        flux: Array,
        nt: int,
        nw: int,
        nc: int,
        Ny: int,
        nw0: int,
        nw0_: int,
        S0e2i: Array,
        flux_err: float,
        # pass to solve_for_map_linear()
        fix_spectrum: bool,
        baseline_var: int,
        S: Array,
        # not being passed in yet - all for S?
        # theta, _angle_factor, y, spectrum_, veq, inc, u, normalized,
    ) -> tuple[Array, Array]:

    # reset() - if have a class with self attributes.

    processed_inputs = process_inputs(
        flux, nt, nw, nc, Ny, nw0, nw0_, S0e2i,
        normalized=False, flux_err=flux_err,
    )

    linear = processed_inputs["linear"]
    flux = processed_inputs["flux"] # doesn't change in this case
    spatial_mean = processed_inputs["spatial_mean"]
    spatial_inv_cov = processed_inputs["spatial_inv_cov"]
    flux_err = processed_inputs["flux_err"]
    # T = processed_inputs["T"]

    # print(f"------ linear: {linear}")     # TODO: Remove after debugging.

    if fix_spectrum:
        if linear:
            y, cho_ycov = solve_for_map_linear(
                spatial_mean, spatial_inv_cov, flux_err, nt, nw, baseline_var, 1, flux, S, nc, Ny,
            )
    #     else:
    # else:
    
    return y, cho_ycov


def solve(
        flux: Array,
        nt: int,
        nw: int,
        nc: int,
        Ny: int,
        nw0: int,
        nw0_: int,
        S0e2i: Array,
        flux_err: float,
        fix_spectrum: bool,
        baseline_var: int,
        S: Array,
        theta: Array,
        _angle_factor: float,
        solver: str="bilinear",
        # not being passed in yet - all for S?
        # y, spectrum_, veq, inc, u, normalized,
    ) -> tuple[Array, Array]:
    """
    Iteratively solves the bilinear or nonlinear problem for the spatial
    and/or spectral map given a spectral timeseries.
    """

    # Used to calculate S.
    theta = get_default_theta(theta, _angle_factor)

    if solver.lower().startswith("bi"):
        y, cho_ycov = solve_bilinear(
            flux, nt, nw, nc, Ny, nw0, nw0_, S0e2i, flux_err,                   # pass to process_inputs()
            fix_spectrum, baseline_var, S,                                      # pass to solve_for_map_linear()
            # theta, _angle_factor, y, spectrum_, veq, inc, u, normalized,      # not being passed in yet - all for S?
        )
    # else:
    
    return y, cho_ycov
