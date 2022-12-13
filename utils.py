from jax import lax
import jax.numpy as jnp

def exists(val):
    return val is not None

def default(val, default_val):
    if val is None:
        return default_val
    return val

def jax_unstack(x, axis=0):
  return [lax.index_in_dim(x, i, axis, keepdims=False) for i in range(x.shape[axis])]

def right_pad_dims_to(x, t):
    padding_dims = len(x.shape) - len(t.shape)
    if padding_dims <= 0:
        return x
    # pytorch version: t.view(*t.shape, *((1,) * padding_dims))
    # jax version:
    return jnp.reshape(t, t.shape + (1,) * padding_dims)