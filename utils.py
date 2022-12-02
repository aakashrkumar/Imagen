from jax import lax

def exists(val):
    return val is not None

def default(val, default_val):
    if val is None:
        return default_val
    return val

def jax_unstack(x, axis=0):
  return [lax.index_in_dim(x, i, axis, keepdims=False) for i in range(x.shape[axis])]
