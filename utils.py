from jax import lax
import jax.numpy as jnp
import jax

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

def prob_mask_like(shape, prob, key):
    if prob == 1:
        return jnp.ones(shape, dtype = jnp.bool)
    elif prob == 0:
        return jnp.zeros(shape, dtype = jnp.bool)
    else:
        return jax.random.uniform(key, shape, minval=0, maxval=1) < prob
