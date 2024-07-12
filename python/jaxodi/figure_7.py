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
udeg = data["kwargs"]['udeg'] # 2
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

angle_factor = np.pi/180 # converts between degrees and radians

interp = True

Si2eBlk = map._Si2eBlk # array (1120, 5616)

u = data["props"]["u"] # the vector of limb darkening coefficients

spectrum = spectrum_true


spatial_mean
spatial_inv_cov # (array)
n
S
cho_C
mu
invL
meta["y_lin"]
y
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
#
def get_default_theta(theta):

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
#
def get_kT(inc, theta, veq, u):
    """
    Get the kernels at an array of angular phases `theta`.
    """
    # # Compute the convolution kernels
    # vsini = self.enforce_bounds(veq * tt.sin(inc), 0.0, self.vsini_max)
    # x = self.get_x(vsini)
    # rT = self.get_rT(x)
    # kT0 = self.get_kT0(rT)

    # # Compute the limb darkening operator
    # if self.udeg > 0:
    #     F = self.F(
    #         tt.as_tensor_variable(u), tt.as_tensor_variable([np.pi])
    #     )
    #     L = ts.dot(ts.dot(self.A1Inv, F), self.A1)
    #     kT0 = tt.dot(tt.transpose(L), kT0)

    # # Compute the kernels at each epoch
    # kT = tt.zeros((self.nt, self.Ny, self.nk))
    # for m in range(self.nt):
    #     kT = tt.set_subtensor(
    #         kT[m],
    #         tt.transpose(
    #             self.right_project(
    #                 tt.transpose(kT0),
    #                 inc,
    #                 tt.as_tensor_variable(0.0),
    #                 theta[m],
    #             )
    #         ),
    #     )
    # return kT

    pass

# ---










# Functions called by Figure 7's map.solve()
# ------------------------------------------

# spatial_cov
S0e2i = map._S0e2i # array (603, 300)
# spectral_mean

# process_inputs
#
# https://github.com/rodluger/starry/blob/b72dff08588532f96bd072f2f1005e227d8e4ed8/starry/doppler_solve.py#L222
#
def process_inputs(spatial_cov, S0e2i, spectral_mean):
    """
    Checks shapes and sets defaults.
    """

    # potentially calls to cho_factor and cho_solve
    # if not spatial_cov[n].ndim < 2: then ^^

    # spectral_mean[n] = S0e2i.dot(spectral_mean[n]).T

    pass


# get_D_fixed_spectrum
#
# https://github.com/rodluger/starry/blob/b72dff08588532f96bd072f2f1005e227d8e4ed8/starry/_core/core.py#L1986
#
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
#
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
    
    # if jax.issparse(A):
    #     return A.dot(B)
    # elif jax.issparse(B):
    #     return (B.T.dot(A.T)).T
    # else:
    #     raise ValueError("At least one input must be sparse.")

    pass


# design_matrix
#
# https://github.com/rodluger/starry/blob/b72dff08588532f96bd072f2f1005e227d8e4ed8/starry/doppler.py#L1170
#
fix_spectrum
inc
veq
u
spectrum
interp
Si2eBlk
#
def design_matrix():
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
        it is overwhelming
        havent read yet
    """

    theta = get_default_theta(theta) # Tiger task

    # if fix_spectrum:
    D = get_D_fixed_spectrum(inc, theta, veq, u, spectrum)

    # if interp:
    D = sparse_dot(Si2eBlk, D)

    return D



# get_S
#
# S: https://github.com/rodluger/starry/blob/b72dff08588532f96bd072f2f1005e227d8e4ed8/starry/doppler_solve.py#L201
# get_S: https://github.com/rodluger/starry/blob/b72dff08588532f96bd072f2f1005e227d8e4ed8/starry/doppler_solve.py#L124
#
theta
angle_factor
fix_spectrum
#
def get_S():
    """
    Get design matrix conditioned on the current spectrum.
    """
    return design_matrix(theta/angle_factor, fix_spectrum=True)



# solve_for_map_linear
#
# https://github.com/rodluger/starry/blob/b72dff08588532f96bd072f2f1005e227d8e4ed8/starry/doppler_solve.py#L467
#
flux
cho_C
mu_
invL
#
def solve_for_map_linear():
    """
    Solve for `y` linearly, given a baseline or unnormalised data.
    """

    # ~100 lines of code that don't use theano

    # scipy.linalg.block_diag()
    # cho_factor == scipy.linalg.cholesky(*args, **kwargs, lower=True)
    # == jax.scipy.linalg.cholesky(_input_, lower=True)

    S = get_S()
    # finishes by calling map_solve (already written)
    mean, cho_ycov = map_solve(S, flux, cho_C, mu, invL)
    # a few more lines

    return mean, cho_ycov


# solve_bilinear
#
# https://github.com/rodluger/starry/blob/b72dff08588532f96bd072f2f1005e227d8e4ed8/starry/doppler_solve.py#L737
#
def solve_bilinear(flux, theta, y, spectrum, veq, inc, u, **kwargs):
    """
    Solve the linear problem for the spatial and/or spectral map
    given a spectral timeseries.
    """

    process_inputs(flux, **kwargs)

    # if fix_spectrum:
    # Solve for the map conditioned on the spectrum
    solve_for_map_linear() # The problem is exactly linear!

    metadata = True # {y, cho_ycov, spectrum, cho_scov}

    return metadata


# solve
#
# https://github.com/rodluger/starry/blob/b72dff08588532f96bd072f2f1005e227d8e4ed8/starry/doppler.py#L1773
#
def solve(flux, theta, y, spectrum, veq, inc, u, **kwargs):
    """
    Iteratively solves the bilinear or nonlinear problem for the spatial
    and/or spectral map given a spectral timeseries.
    """

    if theta is None:
        theta = get_default_theta(None)
    else:
        theta = get_default_theta(theta)

    # if bilinear
    solution = solve_bilinear(
        flux, theta, y, spectrum, veq, inc, u, **kwargs
    )

    # else linear
    # solution = solve_nonlinear(
    #     flux, theta, y, spectrum, veq, inc, u, **kwargs
    # )

    return solution
