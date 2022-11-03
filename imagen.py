from typing import Tuple
import jax
import flax
# import optax
from flax import linen as nn
import jax.numpy as jnp


class ResNetBlock(nn.Module):
    """ResNet block with a projection shortcut and batch normalization."""
    num_layers: int = 1
    num_channels: int = 1
    strides: Tuple[int, int] = (2, 2)
    dtype: jnp.dtype = jnp.float32
    training: bool = True

    @nn.compact
    def __call__(self, x):
        for i in range(self.num_layers):
            residual = x
            x = nn.GroupNorm(dtype=self.dtype)(x)
            x = nn.swish(x)
            x = nn.Conv(features=self.num_channels, kernel_size=(3, 3), strides=self.strides, dtype=self.dtype)(x)
            x = nn.GroupNorm(dtype=self.dtype)(x)
            x = nn.swish(x)
            x = nn.Conv(features=self.num_channels, kernel_size=(3, 3), dtype=self.dtype)(x)
            residual = nn.Conv(features=self.num_channels, kernel_size=(1, 1), strides=self.strides, dtype=self.dtype)(residual)
            x = x + residual
        return nn.relu(x)

class CombineEmbs(nn.Module):
    d: int = 32
    @nn.compact
    def __call__(self, x, t):
        pe = jnp.zeros((1, self.d))
        position = jnp.arange(0, self.d).reshape(-1, 1)       

class UnetDBlock(nn.Module):
    """UnetD block with a projection shortcut and batch normalization."""
    num_channels: int = 1
    strides: Tuple[int, int] = (2, 2)
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.num_channels, kernel_size=(3, 3), strides=self.strides, dtype=self.dtype)(x)
        x = ResNetBlock(num_layers=3, num_channels=self.num_channels, strides=self.strides, dtype=self.dtype)(x)
        x = nn.SelfAttention(num_heads=8)(x)
        return x

# 3 *  64 x 64 -> 3 * 32 x 32
# 3 *  32 x 32 -> 3 * 16 x 16
# 3 *  16 x 16 -> 3 * 8 x 8
module = UnetDBlock(num_channels=64)
params = module.init(jax.random.PRNGKey(0), jnp.ones((1, 3, 64, 64)))
x = module.apply(params, jnp.ones((1, 3, 64, 64)))
print(x.shape)