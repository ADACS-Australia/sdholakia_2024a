from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxoplanet.experimental.starry.basis import A1

from .rotation import right_project


@jax.jit
def get_x(xamp, vsini):
    return xamp / jnp.maximum(
        1.0, vsini
    )  # todo: check why we need to use maximum function here


@partial(jax.jit, static_argnums=(1, 2, 3))
def get_rT(x, ydeg, udeg, nk):
    deg = ydeg + udeg
    sijk = jnp.zeros((deg + 1, deg + 1, 2, nk))
    # Initial conditions
    r2 = jnp.maximum(1 - x**2, jnp.zeros_like(x))

    # Silly hack to prevent issues with the undefined derivative at x = 1
    # This just computes the square root of r2, zeroing out values very
    # close to zero.
    # todo: replace the following line with custom jvp
    r = jnp.maximum(1 - x**2, jnp.zeros_like(x) + 1e-100) ** 0.5
    # r = jnp.where(r > 1e-49, r, np.zeros_like(r))

    sijk = sijk.at[0, 0, 0].set(2 * r)
    sijk = sijk.at[0, 0, 1].set(0.5 * np.pi * r2)

    # Upward recursion in j
    for j in range(2, deg + 1, 2):
        sijk = sijk.at[0, j, 0].set(((j - 1.0) / (j + 1.0)) * r2 * sijk[0, j - 2, 0])
        sijk = sijk.at[0, j, 1].set(((j - 1.0) / (j + 2.0)) * r2 * sijk[0, j - 2, 1])

    # Upward recursion in i
    for i in range(1, deg + 1):
        sijk = sijk.at[i].set(sijk[i - 1] * x)

    # Full vector
    N = (deg + 1) ** 2
    # s = np.zeros((N, np.shape(x)[0]))
    n = jnp.arange(N)
    LAM = jnp.floor(jnp.sqrt(n))
    DEL = 0.5 * (n - LAM**2)
    i = jnp.floor(LAM - DEL).astype(int)
    j = jnp.floor(DEL).astype(int)
    k = (jnp.ceil(DEL) - jnp.floor(DEL)).astype(int)

    s = sijk[i, j, k]

    return s


@partial(jax.jit, static_argnums=(1,))
def get_kT0(rT, ydeg):
    a1 = A1(ydeg).todense()
    kT0 = jnp.dot(jnp.transpose(a1), rT)
    # Normalize to preserve the unit baseline
    return kT0 / jnp.sum(kT0[0])


@partial(jax.jit, static_argnums=(2, 3, 4))
def get_kT(xamp, vsini, ydeg, udeg, nk, inc, theta):
    x = get_x(xamp, vsini)
    rT = get_rT(x, ydeg, udeg, nk)
    kT0 = get_kT0(rT, ydeg)
    vmap_rp = jax.vmap(right_project, in_axes=(None, None, None, 0, None, None))
    res = vmap_rp(ydeg, inc, 0.0, theta, 0.0, jnp.transpose(kT0))
    kT = jnp.transpose(res, (0, 2, 1))
    return kT


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def dot_design_matrix_fixed_map_into(kT, y, nc, nwp, nt, nk, nw, matrix):
    kTy = jnp.swapaxes(jnp.dot(jnp.transpose(y), kT), 0, 1)

    # Reshape and transpose the input matrix
    input_reshaped = jnp.transpose(matrix).reshape((-1, nc, 1, nwp))

    # Reshape the kernel tensor
    kernel_reshaped = kTy.reshape((nt, nc, 1, nk))

    # Convolution
    result = lax.conv_general_dilated(
        input_reshaped,
        kernel_reshaped,
        window_strides=(1, 1),
        padding="VALID",
        lhs_dilation=None,
        rhs_dilation=None,
        # dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
    result_reshaped = jnp.transpose(jnp.reshape(result, (-1, nt * nw)))
    return result_reshaped


@partial(
    jax.jit,
    static_argnums=(
        1,
        2,
    ),
)
def get_flux_from_dotconv(flux, nt, nw):
    flux_reshaped = jnp.reshape(flux, (nt, nw))
    return flux_reshaped
