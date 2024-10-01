from jax.experimental import checkify
import jax.numpy as jnp
from jaxoplanet.types import Array


@checkify.checkify
def enforce_bounds(x, lower=-jnp.inf, upper=jnp.inf, name="vsini"):
    cond = (x >= lower) & (x <= upper)
    checkify.check(cond, f"{name}: {x} is out of bounds")
    return x


def unit_radian(degree: Array):
    return jnp.asarray(degree * jnp.pi / 180)
