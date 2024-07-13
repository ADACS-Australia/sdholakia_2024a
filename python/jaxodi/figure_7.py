"""


Figure 7
--------


Infer the stellar map assuming perfect knowledge of:

    - rest frame spectrum s
    - baseline
    - stellar inclination, rotation period & limb-darkening coefficients

    
"""




# when to use np.dot() vs DopplerMap.dot()?
# https://github.com/rodluger/starry/blob/b72dff08588532f96bd072f2f1005e227d8e4ed8/starry/doppler.py#L1357


 

# Stuff I might need
# ------------------

from typing import Tuple
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from jaxoplanet.types import Array

# To generate variables ---
import paparazzi
import starry

from paparazzi.src.scripts.utils.generate import generate_data
# ---




# VARIABLES -----------------------------------------------------------------------------------------------

lazy = False # when reading starry code


# Generate the synthetic dataset ---
data = generate_data()
# ---


y_true = data["truths"]["y"] # array (256,)
spectrum_true = data["truths"]["spectrum"] # array (1,300)
theta = data["data"]["theta"] # array (16,)
flux = data["data"]["flux0"] # array (16,70)
flux_err = data["data"]["flux0_err"] # 0.00013241138706868687


# Instantiate the map ---
map = starry.DopplerMap(lazy=False, **data["kwargs"])
map.spectrum = data["truths"]["spectrum"]
for n in range(map.udeg):
    map[1 + n] = data["props"]["u"][n]
# ---


ydeg = data["kwargs"]['ydeg'] # 15
udeg = data["kwargs"]['udeg'] # 2 (leaves out the 0th deg = -1)
nc = data["kwargs"]['nc'] # 1
veq = data["kwargs"]['veq'] # 60000
inc = data["kwargs"]['inc'] # 40
vsini_max = data["kwargs"]['vsini_max'] # 50000
nt = data["kwargs"]['nt'] # 16
wav = data["kwargs"]['wav'] # array (70,)

baseline = map.baseline # array (16,)

normalised = False
fix_spectrum = True

T = 1

baseline_var = 0

nw = 70
Ny = 256

_angle_factor = np.pi/180 # converts between degrees and radians

interp = True

_Si2eBlk = map._Si2eBlk # array (1120, 5616)
S0e2i = map._S0e2i # array (603, 300)

u = map.u # array (3,) vector of limb darkening coefficients

spectrum = spectrum_true  # array (1,300)
spectrum_

spatial_mean
spatial_inv_cov # (array)
n
S
cho_C
mu
invL
meta["y_lin"]
y = map.y # array (256,)
cho_ycov


# FUNCTIONS ----------------------------------------------------------------------------------------------------------------------

# Functions from 2023B
# --------------------

# cho_solve
#
# https://github.com/ADACS-Australia/bpope_2023b/blob/df765188cd1182694f28da7f4dab28cfdcd76265/src/jaxoplanet/experimental/starry/light_curve/inference.py#L33
#
@jax.jit
def cho_solve(A: Array, b: Array) -> Array:
    b_ = jax.scipy.linalg.solve_triangular(A, b, lower=True)
    return jax.scipy.linalg.solve_triangular(jnp.transpose(A), b_, lower=False)


# map_solve
#
# https://github.com/ADACS-Australia/bpope_2023b/blob/df765188cd1182694f28da7f4dab28cfdcd76265/src/jaxoplanet/experimental/starry/light_curve/inference.py#L192
#
@jax.jit
def map_solve(
    X: Array, flux: Array, cho_C: float | Array, mu: Array, LInv: float | Array
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

# ---





# Tiger's Functions
# -----------------

# get_default_theta
#
# https://github.com/rodluger/starry/blob/b72dff08588532f96bd072f2f1005e227d8e4ed8/starry/doppler.py#L1059

# how function is called in solve()
theta = _get_default_theta(kwargs.pop("theta", None))

# class attributes
_nt
_angle_factor

def _get_default_theta(theta):

    math.cast()
    ops.enforce_shape()

    # if theta is None:
    #     theta = math.cast(
    #         np.linspace(0, 2 * np.pi, nt, endpoint=False)
    #     )
    # else:
    #     theta = (
    #         ops.enforce_shape(
    #             math.cast(theta), np.array([nt])
    #         )
    #         * angle_factor
    #     )

    return theta


# get_kT
#
# https://github.com/rodluger/starry/blob/b72dff08588532f96bd072f2f1005e227d8e4ed8/starry/_core/core.py#L1894

# how function is called in get_D_fixed_spectrum()
get_kT(inc, theta, veq, u)

# class attributes
vsini_max
F # not sure if function or attribute?
A1Inv
A1
nt
Ny
nk

def get_kT(inc, theta, veq, u):
    """
    Get the kernels at an array of angular phases `theta`.
    """

    # many function calls
    enforce_bounds()
    get_x()
    get_rT()
    get_kT0()
    F() #?
    right_project()

    # many theano operations

    pass

# ---










# Functions called by Figure 7's map.solve()
# ------------------------------------------


# process_inputs
#
# https://github.com/rodluger/starry/blob/b72dff08588532f96bd072f2f1005e227d8e4ed8/starry/doppler_solve.py#L222

# how function is called in solve_bilinear()
process_inputs(flux, **kwargs)

# class attributes
Ny
nt
nw
nc
nw0 # maybe the shape of spectral_mean[n] arrays?
S0e2i
nw0_ # maybe a transformed version of nw0 for spectral covariance matrices?

# def process_inputs(spatial_cov, S0e2i, spectral_mean):
def process_inputs(
        flux,
        flux_err=None,
        spatial_mean=None,
        spatial_cov=None,
        spectral_mean=None,
        spectral_cov=None,
        spectral_guess=None,
        spectral_lambda=None,
        spectral_maxiter=None,
        spectral_eps=None,
        spectral_tol=None,
        spectral_method=None,
        normalized=True,
        baseline=None,
        baseline_var=None,
        fix_spectrum=False,
        fix_map=False,
        logT0=None,
        logTf=None,
        nlogT=None,
        quiet=False,
    ):
    """
    Checks shapes and sets defaults.
    """

    # if not spatial_cov[n].ndim < 2:
    # cho_factor()
    # cho_solve()

    # is this dot function Numpy's or starry's?
    # spectral_mean[n] = S0e2i.dot(spectral_mean[n]).T

    pass


# get_D_fixed_spectrum
#
# https://github.com/rodluger/starry/blob/b72dff08588532f96bd072f2f1005e227d8e4ed8/starry/_core/core.py#L1986

# how function is called in design_matrix()
D = get_D_fixed_spectrum(
    _inc, theta, _veq, _u, _spectrum
)

# class attributes
nc
nwp
nt
Ny
nk
nw

def get_D_fixed_spectrum(inc, theta, veq, u, spectrum):
    """
    Return the Doppler matrix for a fixed spectrum.
    """

    kT = get_kT(inc, theta, veq, u) # Tiger task

    # A bunch of theano action here.
    product = 0
    # The dot product is just a 2d convolution!
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

    return product


# sparse_dot
#
# https://github.com/rodluger/starry/blob/b72dff08588532f96bd072f2f1005e227d8e4ed8/starry/_core/math.py#L218

# how function is called in design_matrix()
D = math.sparse_dot(_Si2eBlk, D)

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


# design_matrix
#
# https://github.com/rodluger/starry/blob/b72dff08588532f96bd072f2f1005e227d8e4ed8/starry/doppler.py#L1170

# how function is called in get_S()
map.design_matrix(
    theta=theta/_angle_factor, fix_spectrum=True
)

# class attributes
_inc
_veq
_u
_spectrum
_interp = True
_Si2eBlk

def design_matrix(theta=None, fix_spectrum=None, fix_map=False):
    """
        Return the Doppler imaging design matrix.

        This matrix dots into the spectral map to yield the model for the
        observed spectral timeseries (the ``flux``).

        Note that if this method is used to compute the spectral timeseries,
        the result should be reshaped into a matrix of shape
        (:py:attr:`nt`, :py:attr:`nw`) and optionally divided by the
        :py:meth:`baseline()` to match the return value of :py:meth:`flux()`.
        """
    
    """
        there is much more info in the docstring
        havent read yet
    """

    theta = _get_default_theta(theta) # Tiger task

    # if fix_spectrum:
    D = get_D_fixed_spectrum(_inc, theta, _veq, _u, _spectrum)

    # if _interp:
    D = sparse_dot(_Si2eBlk, D)

    return D



# get_S
#
# S: https://github.com/rodluger/starry/blob/b72dff08588532f96bd072f2f1005e227d8e4ed8/starry/doppler_solve.py#L201
# get_S: https://github.com/rodluger/starry/blob/b72dff08588532f96bd072f2f1005e227d8e4ed8/starry/doppler_solve.py#L124

# class attributes
spectrum_ # map._spectrum = self.spectrum_
theta
_angle_factor
fix_spectrum

def get_S():
    """
    Get design matrix conditioned on the current spectrum.
    """
    return design_matrix(theta/_angle_factor, fix_spectrum=True)


# solve_for_map_linear
#
# https://github.com/rodluger/starry/blob/b72dff08588532f96bd072f2f1005e227d8e4ed8/starry/doppler_solve.py#L467

# how function is called in solve_bilinear()
solve_for_map_linear()

# class attributes
spatial_mean
spatial_inv_cov
nc
flux_err
nt
nw
baseline
flux
S
Ny

def solve_for_map_linear(T=1, baseline_var=0):
    """
    Solve for `y` linearly, given a baseline or unnormalised data.
    """

    # ~60 lines of code that don't use theano

    block_diag()
    # scipy.linalg.block_diag()

    cho_factor()
    # cho_factor == scipy.linalg.cholesky(*args, **kwargs, lower=True)
    # == jax.scipy.linalg.cholesky(_input_, lower=True)

    # can get S via call
    S = get_S()

    #  mean, cho_ycov = greedy_linalg.solve(S, flux, cho_C, mu, invL)
    mean, cho_ycov = map_solve(S, flux, cho_C, mu, invL)

    return mean, cho_ycov


# reset
#
# https://github.com/rodluger/starry/blob/b72dff08588532f96bd072f2f1005e227d8e4ed8/starry/doppler_solve.py#L177

# how function is called in solve_bilinear()
reset()

# class attributes
spectrum_
y
_S
_C
_KT0
meta

def reset():

    spectrum_ = None
    y = None
    _S = None
    _C = None
    _KT0 = None
    meta = {}


# solve_bilinear
#
# https://github.com/rodluger/starry/blob/b72dff08588532f96bd072f2f1005e227d8e4ed8/starry/doppler_solve.py#L737

# how function is called in solve()
solution = solve_bilinear(
    flux, theta, y, spectrum_, veq, inc, u, **kwargs
)

# class attributes
fix_spectrum
linear
meta["y"]
meta["cho_ycov"]
meta["spectrum"]
meta["cho_scov"]

def solve_bilinear(flux, theta, y, spectrum, veq, inc, u, **kwargs):
    """
    Solve the linear problem for the spatial and/or spectral map
    given a spectral timeseries.
    """

    reset()
    process_inputs(flux, **kwargs)

    # if fix_spectrum and linear: # The problem is exactly linear!
    # Solve for the map conditioned on the spectrum
    solve_for_map_linear()

    meta = True # {y, cho_ycov, spectrum, cho_scov}

    return meta


# solve
#
# https://github.com/rodluger/starry/blob/b72dff08588532f96bd072f2f1005e227d8e4ed8/starry/doppler.py#L1773

# how function is called in paper
soln = map.solve(
    flux,
    theta=theta,
    normalized=False,
    fix_spectrum=True,
    flux_err=flux_err,
)

# class attributes
_y = map.y # array (256,)
spectrum_ = data["truths"]["spectrum"] # array (1,300)
_veq = data["kwargs"]['veq'] # 60000
_inc = data["kwargs"]['inc'] # 40
_u = data["props"]["u"] # the vector of limb darkening coefficients

def solve(flux, solver="bilinear", **kwargs):
    """
    Iteratively solves the bilinear or nonlinear problem for the spatial
    and/or spectral map given a spectral timeseries.
    """

    theta = _get_default_theta(kwargs.pop("theta", None))

    # if bilinear
    # if solver.lower().startswith("bi"):
    solution = solve_bilinear(
        flux, theta, y, spectrum, veq, inc, u, **kwargs
    )
    # else linear
    # solution = solve_nonlinear(
    #     flux, theta, y, spectrum, veq, inc, u, **kwargs
    # )

    return solution
