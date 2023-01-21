import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict






def _opt_state_spec_per_leaf(x, spec):
    if isinstance(x, FrozenDict):
        # variables with same structure as params
        return spec
    else:
        # other variables such as count
        return None
