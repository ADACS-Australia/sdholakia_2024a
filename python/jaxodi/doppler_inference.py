"""
Doppler inference functions.
"""

import jax
import jax.numpy as jnp
import numpy as np
import scipy

from typing import Tuple
from functools import partial
from jaxoplanet.types import Array
from .doppler_forward import get_kT


@jax.jit
def cho_solve(A: Array, b: Array) -> Array:
    """
    Solve the linear system A x = b using the Cholesky decomposition of A.

    Args:
        A (Array): Lower triangular Cholesky factor of the matrix.
        b (Array): Right-hand side of the equation.
    
    Returns:
        Array: Solution to the linear system.
    """
    b_ = jax.scipy.linalg.solve_triangular(A, b, lower=True)
    return jax.scipy.linalg.solve_triangular(jnp.transpose(A), b_, lower=False)


@jax.jit
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
        X (Array): The flux design matrix.
        flux (Array): The flux timeseries.
        cho_C (float | Array): The lower cholesky factorization
            of the data covariance.
        mu (Array): The prior mean of the spherical harmonic coefficients.
        LInv (float | Array): The inverse prior covariance of the
            spherical harmonic coefficients.

    Returns:
        Tuple[Array, Array]: The vector of spherical harmonic coefficients
            corresponding to the MAP solution and the Cholesky factorization
            of the corresponding covariance matrix.
    """
    # Compute C^-1 . X
    if cho_C.ndim == 0:
        CInvX = X / cho_C**2
    elif cho_C.ndim == 1:
        CInvX = jnp.dot(jnp.diag(1 / cho_C**2), X)
    else:
        CInvX = cho_solve(cho_C, X)

    # Compute W = X^T . C^-1 . X + L^-1
    W = jnp.dot(jnp.transpose(X), CInvX)

    # If LInv is a scalar or a 1-dimensional array, increment the
    # diagonal elements of W with the values from LInv.
    if LInv.ndim == 0 or LInv.ndim == 1:
        W = W.at[jnp.diag_indices_from(W)].set(W[jnp.diag_indices_from(W)] + LInv)
        LInvmu = mu * LInv
    # If LInv is a matrix, directly add LInv to W.
    else:
        W += LInv
        LInvmu = jnp.dot(LInv, mu)

    # Compute the max like y and its covariance matrix
    cho_W = jax.scipy.linalg.cholesky(W, lower=True)
    M = cho_solve(cho_W, jnp.transpose(CInvX))
    yhat = jnp.dot(M, flux) + cho_solve(cho_W, LInvmu)
    ycov = cho_solve(cho_W, jnp.eye(cho_W.shape[0]))
    cho_ycov = jax.scipy.linalg.cholesky(ycov, lower=True)

    return yhat, cho_ycov


@partial(jax.jit, static_argnames=("nt", "nw", "nc", "Ny", "nw0", "nw0_"))
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
    """
    Process and normalize inputs for Doppler inference.

    Args:
        flux (Array): The observed spectral timeseries.
        nt (int): Number of time points.
        nw (int): Number of wavelength points.
        nc (int): Number of components.
        Ny (int): Number of spatial points.
        nw0 (int): Initial number of wavelength points.
        nw0_ (int): Final number of wavelength points.
        S0e2i (Array): Transformation matrix.
        flux_err (float, optional): Flux error. Defaults to 1e-4.
        normalized (bool, optional): Whether the flux is normalized. Defaults to True.
        baseline (optional): Baseline value. Defaults to None.
        spatial_mean (optional): Mean of the spatial component. Defaults to None.
        spatial_cov (optional): Covariance of the spatial component. Defaults to None.
        spectral_mean (optional): Mean of the spectral component. Defaults to None.
        spectral_cov (optional): Covariance of the spectral component. Defaults to None.
        logT0 (optional): Logarithmic starting temperature. Defaults to None.
        logTf (optional): Logarithmic final temperature. Defaults to None.
        nlogT (optional): Number of logarithmic temperature steps. Defaults to None.

    Returns:
        dict: Processed input values.
    """

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
    # assert jnp.array_equal(
    #     jnp.shape(flux), jnp.array([nt, nw])
    # ), "Invalid shape for `flux`."

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
    linear = jnp.logical_or(jnp.logical_not(normalized), baseline is not None)

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


@partial(jax.jit, static_argnames=("ydeg", "udeg", "nk", "nt", "nw", "nc", "Ny", "nwp"))
def get_D_fixed_spectrum(
        xamp, vsini, ydeg, udeg, nk, inc, theta, spectrum, nc, nwp, nt, Ny, nw,
    ):
    """
    Return the Doppler convolution matrix for a fixed spectrum.

    Args:
        xamp: Amplitude of the Doppler kernel.
        vsini: Rotational velocity.
        ydeg: Degree of the Y polynomial.
        udeg: Degree of the U polynomial.
        nk: Number of Doppler kernel points.
        inc: Inclination angle.
        theta: Parameters for the Doppler kernel.
        spectrum: Spectral data.
        nc: Number of components.
        nwp: Number of wavelength points in the spectrum.
        nt: Number of time points.
        Ny: Number of spatial points.
        nw: Number of wavelength points in the observed data.

    Returns:
        Array: Convolution matrix.
    """
    # Get the convolution kernels
    kT = get_kT(xamp, vsini, ydeg, udeg, nk, inc, theta)

    # The dot product is just a 2d convolution!
    product = jax.lax.conv_general_dilated(
        jnp.reshape(spectrum, (nc, 1, 1, nwp)),
        jnp.reshape(kT, (nt * Ny, 1, 1, nk)),
        window_strides=(1,1),
        padding="VALID",
        lhs_dilation=None,
        rhs_dilation=None,
    )

    product = jnp.reshape(product, (nc, nt, Ny, nw))
    product = jnp.swapaxes(product, 1, 2)
    product = jnp.reshape(product, (Ny * nc, nt * nw))
    product = jnp.transpose(product)

    return product


# JAX does not store sparse matrices in sparse format
# and has no way to check for sparsity.
# Not currently jittable with is_sparse() returning a bool.
# @jax.jit
def sparse_dot(A, B):
    """
    Performs matrix multiplication, optimising computation time by utilising sparse matrices.

    Args:
        A (Array): First matrix.
        B (Array): Second matrix.

    Returns:
        Array: Result of the matrix multiplication.
    
    Raises:
        ValueError: If neither input matrix is sparse.
    """
    def is_sparse(dense_matrix, threshold=0.9):
        flattened_matrix = dense_matrix.flatten()
        num_zeros = jnp.sum(flattened_matrix == 0)
        total_elements = flattened_matrix.size
        sparsity_ratio = num_zeros / total_elements
        return sparsity_ratio > threshold

    # JAX has no check for sparse matrices.
    if is_sparse(A):
        return A.dot(B)
    elif is_sparse(B):
        return (B.T.dot(A.T)).T
    else:
        raise ValueError("At least one input must be sparse.")


@jax.jit
def get_default_theta(
        theta: Array,
        _angle_factor: float,
    ) -> Array:
    """
    Scale the parameters of the Doppler kernel by the provided angle factor.

    The angle factor is used to convert between different units of angle
    measurement, such as degrees to radians. The value of ``_angle_factor``
    determines how the input ``theta`` is scaled.

    Args:
        theta (Array): Parameters of the Doppler kernel.
        _angle_factor (float): Factor to convert angle units. For example,
            use ``np.pi / 180`` to convert degrees to radians, or
            ``180 / np.pi`` to convert radians to degrees.

    Returns:
        Array: Scaled parameters.
    """
    return theta * _angle_factor


# Can't be jitted until sparse_dot is jitted
@partial(jax.jit, static_argnames=("nt", "nw", "nc", "Ny", "nwp", "_angle_factor", "ydeg", "udeg", "nk"))
def design_matrix(
        theta, _angle_factor, xamp, vsini, ydeg, udeg, nk, inc, spectrum,
        nc, nwp, nt, Ny, nw, _interp, _Si2eBlk,
        fix_spectrum=True, fix_map=False
    ):
    """
    Return the Doppler imaging design matrix.

    Args:
        theta (Array): The angular phase(s) at which to compute the
            design matrix. This must be a vector of size :py:attr:`nt`.
        _angle_factor (float): Scaling factor for the angle.
        xamp: Amplitude of the Doppler kernel.
        vsini (float): Rotational velocity.
        ydeg (int): Degree of the Y polynomial.
        udeg (int): Degree of the U polynomial.
        nk (int): Number of Doppler kernel points.
        inc (float): Inclination angle.
        spectrum (Array): Spectral data.
        nc (int): Number of components.
        nwp (int): Number of wavelength points in the spectrum.
        nt (int): Number of time points.
        Ny (int): Number of spatial points.
        nw (int): Number of wavelength points in the observed data.
        _interp (bool): Whether to interpolate the matrix.
        _Si2eBlk (Array): Transformation matrix for interpolation.
        fix_spectrum (bool, optional): Whether to fix the spectrum. Defaults to True.
        fix_map (bool, optional): Whether to fix the map. Defaults to False.

    Returns:
        Array: The design matrix for the inference problem.
    """
    theta = get_default_theta(theta, _angle_factor) # this is just undoing what get_S() did!

    # Compute the Doppler operator
    # if fix_spectrum:
    D = get_D_fixed_spectrum(
        xamp, vsini, ydeg, udeg, nk, inc, theta, spectrum, nc, nwp, nt, Ny, nw,
    )
    
    # Interpolate to the output grid
    # if _interp:
    # D = sparse_dot(_Si2eBlk, D)
    D = jnp.dot(_Si2eBlk, D)

    return D


@partial(jax.jit, static_argnames=("nt", "nw", "nc", "Ny", "nwp", "_angle_factor", "ydeg", "udeg", "nk", "fix_spectrum"))
def _get_S(
        theta, _angle_factor, xamp, vsini, ydeg, udeg, nk, inc, spectrum,
        nc, nwp, nt, Ny, nw, _interp, _Si2eBlk, fix_spectrum,
    ):
    """
    Compute the design matrix for the Doppler inference, adjusted by the scaling factor.

    Args:
        theta (Array): Doppler kernel parameters.
        _angle_factor (float): Scaling factor for the angle.
        xamp: Amplitude of the Doppler kernel.
        vsini (float): Rotational velocity.
        ydeg (int): Degree of the Y polynomial.
        udeg (int): Degree of the U polynomial.
        nk (int): Number of Doppler kernel points.
        inc (float): Inclination angle.
        spectrum (Array): Spectral data.
        nc (int): Number of components.
        nwp (int): Number of wavelength points in the spectrum.
        nt (int): Number of time points.
        Ny (int): Number of spatial points.
        nw (int): Number of wavelength points in the observed data.
        _interp (bool): Whether to interpolate the matrix.
        _Si2eBlk (Array): Transformation matrix for interpolation.
        fix_spectrum (bool): Whether to fix the spectrum.

    Returns:
        Array: The design matrix adjusted by the scaling factor.
    """
    theta = theta / _angle_factor

    dm = design_matrix(
        theta, _angle_factor, xamp, vsini, ydeg, udeg, nk, inc, spectrum,
        nc, nwp, nt, Ny, nw, _interp, _Si2eBlk, fix_spectrum=fix_spectrum
    )

    return dm


@partial(jax.jit, static_argnames=("nt", "nw", "nw_", "nc", "Ny", "nwp", "_angle_factor", "ydeg", "udeg", "nk", "fix_spectrum"))
def solve_for_map_linear(
        spatial_mean: Array,
        spatial_inv_cov: Array,
        flux_err: float,
        nt: int,
        nw: int,
        nw_: int,
        T,
        flux: Array,
        theta: Array,
        _angle_factor: float,
        xamp,
        vsini: float,
        ydeg: int,
        udeg: int,
        nk: int,
        inc: float,
        spectrum: Array,
        nc: int,
        nwp: int,
        Ny: int,
        _interp: bool,
        _Si2eBlk: Array,
        fix_spectrum: bool,
    ) -> tuple[Array, Array]:
    """
    Solve the Doppler inference problem for a linear model, given a baseline
    of unnormalized data.

    Args:
        spatial_mean (Array): Mean vector of the spatial component.
        spatial_inv_cov (Array): Inverse covariance matrix of the spatial component.
        flux_err (float): Flux error.
        nt (int): Number of time points.
        nw (int): Number of wavelength points.
        nw_ (int): Number of wavelength points for the error model.
        T (Array): Temperature parameter.
        flux (Array): The observed spectral timeseries.
        theta (Array): Doppler kernel parameters.
        _angle_factor (float): Scaling factor for the angle.
        xamp: Amplitude of the Doppler kernel.
        vsini (float): Rotational velocity.
        ydeg (int): Degree of the Y polynomial.
        udeg (int): Degree of the U polynomial.
        nk (int): Number of Doppler kernel points.
        inc (float): Inclination angle.
        spectrum (Array): Spectral data.
        nc (int): Number of components.
        nwp (int): Number of wavelength points in the spectrum.
        Ny (int): Number of spatial points.
        _interp (bool): Whether to interpolate the matrix.
        _Si2eBlk (Array): Transformation matrix for interpolation.
        fix_spectrum (bool): Whether to fix the spectrum.

    Returns:
        Tuple[Array, Array]: Tuple containing the solution vector and its covariance matrix.
    """
    # Reshape the priors
    mu = jnp.reshape(jnp.transpose(spatial_mean), (-1))
    if spatial_inv_cov.ndim == 2:
        invL = jnp.reshape(jnp.transpose(spatial_inv_cov), (-1))

    # Ensure the flux error is a vector
    if flux_err.ndim == 0:
        flux_err = flux_err * jnp.ones((nt, nw_))

    # Factorised data covariance
    # cho_C = jnp.where(
    #     jnp.equal(baseline_var, 0),
    #     jnp.reshape(jnp.sqrt(T) * flux_err, (-1,)),
    #     None
    # )
    cho_C = jnp.reshape(jnp.sqrt(T) * flux_err, (-1,))

    # Unroll the data into a vector
    flux = jnp.reshape(flux, (-1,))

    # Get S
    S = _get_S(
        theta, _angle_factor, xamp, vsini, ydeg, udeg, nk, inc, spectrum,
        nc, nwp, nt, Ny, nw, _interp, _Si2eBlk, fix_spectrum,
    )
    
    # Solve the L2 problem
    mean, cho_ycov = map_solve(S, flux, cho_C, mu, invL)
    y = jnp.transpose(jnp.reshape(mean, (nc, Ny)))

    return y, cho_ycov


@partial(jax.jit, static_argnames=("nt", "nw", "nw_", "nc", "Ny", "nwp", "_angle_factor", "ydeg", "udeg", "nk", "nw0", "nw0_", "fix_spectrum"))
def solve_bilinear(
        # pass to process_inputs()
        flux: Array,
        nt: int,
        nw: int,
        nw_: int,
        nc: int,
        Ny: int,
        nw0: int,
        nw0_: int,
        S0e2i: Array,
        flux_err: float,
        normalized: bool,
        # pass to solve_for_map_linear()
        theta: Array,
        _angle_factor: float,
        xamp,
        vsini: float,
        ydeg: int,
        udeg: int,
        nk: int,
        inc: float,
        spectrum,
        nwp: int,
        _interp: bool,
        _Si2eBlk: Array,
        fix_spectrum: bool,
    ) -> tuple[Array, Array]:
    """
    Solve the Doppler inference problem for the spatial and/or spectral map
    given a spectral timeseries, using a bilinear model approach.

    Args:
        flux (Array): The observed spectral timeseries.
        nt (int): Number of time points.
        nw (int): Number of wavelength points in the observed data.
        nw_ (int): Number of wavelength points in the error model.
        nc (int): Number of spatial components.
        Ny (int): Number of spatial points.
        nw0 (int): Initial number of wavelength points.
        nw0_ (int): Final number of wavelength points.
        S0e2i (Array): Precomputed inverse covariance or related matrix.
        flux_err (float): Error associated with the flux data.
        normalized (bool): Indicates if the flux is normalized.
        theta (Array): Parameters of the Doppler kernel.
        _angle_factor (float): Scaling factor for the angle.
        xamp: Amplitude of the Doppler kernel.
        vsini (float): Rotational velocity.
        ydeg (int): Degree of the Y polynomial.
        udeg (int): Degree of the U polynomial.
        nk (int): Number of Doppler kernel points.
        inc (float): Inclination angle.
        spectrum: Spectral data.
        nwp (int): Number of wavelength points in the spectrum.
        _interp (bool): Indicates whether to interpolate the matrix.
        _Si2eBlk (Array): Transformation matrix for interpolation.
        fix_spectrum (bool): Whether to fix the spectrum in the model.

    Returns:
        Tuple[Array, Array]: The first element is the solution array `y`,
            and the second element is the covariance matrix `cho_ycov`.
    """
    # reset() - if have a class with self attributes.

    processed_inputs = process_inputs(
        flux, nt, nw_, nc, Ny, nw0, nw0_, S0e2i,
        normalized=normalized, flux_err=flux_err,
    )

    linear = processed_inputs["linear"]
    flux = processed_inputs["flux"] # doesn't change in this case
    spatial_mean = processed_inputs["spatial_mean"]
    spatial_inv_cov = processed_inputs["spatial_inv_cov"]
    flux_err = processed_inputs["flux_err"]
    # T = processed_inputs["T"]

    # if fix_spectrum and linear:
    y, cho_ycov = solve_for_map_linear(
        spatial_mean, spatial_inv_cov, flux_err, nt, nw, nw_, 1, flux,
        theta, _angle_factor, xamp, vsini, ydeg, udeg, nk, inc, spectrum,
        nc, nwp, Ny, _interp, _Si2eBlk, fix_spectrum,
    )
    
    return y, cho_ycov


@partial(jax.jit, static_argnames=("nt", "nw", "nw_", "nc", "Ny", "nwp", "_angle_factor", "ydeg", "udeg", "nk", "nw0", "nw0_", "fix_spectrum"))
def solve(
        flux: Array,
        nt: int,
        nw: int,
        nw_: int,
        nc: int,
        Ny: int,
        nw0: int,
        nw0_: int,
        S0e2i: Array,
        flux_err: float,
        normalized: bool,
        theta: Array,
        _angle_factor: float,
        xamp,
        vsini: float,
        ydeg: int,
        udeg: int,
        nk: int,
        inc: float,
        spectrum,
        nwp: int,
        _interp: bool,
        _Si2eBlk: Array,
        fix_spectrum: bool,
        solver: str="bilinear",
    ) -> tuple[Array, Array]:
    """
    Iteratively solves the Doppler inference problem for the spatial
    and/or spectral map given a spectral timeseries.

    Args:
        flux (Array): The observed spectral timeseries.
        nt (int): Number of time points.
        nw (int): Number of wavelength points in the observed data.
        nw_ (int): Number of wavelength points in the error model.
        nc (int): Number of spatial components.
        Ny (int): Number of spatial points.
        nw0 (int): Initial number of wavelength points.
        nw0_ (int): Final number of wavelength points.
        S0e2i (Array): Precomputed inverse covariance or related matrix.
        flux_err (float): The data uncertainty.
        normalized (bool): Whether the ``flux`` dataset is
            continuum-normalized.
        theta (Array): The angular phase(s) at which the spectra
            were observed.
        _angle_factor (float): Scaling factor for the angle.
        xamp: Amplitude of the Doppler kernel.
        vsini (float): Rotational velocity.
        ydeg (int): Degree of the Y polynomial.
        udeg (int): Degree of the U polynomial.
        nk (int): Number of Doppler kernel points.
        inc (float): Inclination angle.
        spectrum: Spectral data.
        nwp (int): Number of wavelength points in the spectrum.
        _interp (bool): Indicates whether to interpolate the matrix.
        _Si2eBlk (Array): Transformation matrix for interpolation.
        fix_spectrum (bool): If True, fixes the spectrum at the
            current value and solves only for the map.
        solver (str, optional): The solver method to use, default is "bilinear".

    Returns:
        Tuple[Array, Array]: The first element is the solution array `y`,
            and the second element is the covariance matrix `cho_ycov`.
    """
    # Used to calculate S.
    theta = get_default_theta(theta, _angle_factor)

    if solver.lower().startswith("bi"):
        y, cho_ycov = solve_bilinear(
            flux, nt, nw, nw_, nc, Ny, nw0, nw0_, S0e2i, flux_err, normalized,  # pass to process_inputs()
            theta, _angle_factor, xamp, vsini, ydeg, udeg, nk, inc, spectrum,   # pass to solve_for_map_linear()
            nwp, _interp, _Si2eBlk, fix_spectrum,
        )
    # else:
    
    return y, cho_ycov
