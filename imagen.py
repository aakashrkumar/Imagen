from typing import Tuple
import jax
import flax
# import optax
from flax import linen as nn
import jax.numpy as jnp


class ResNetBlock(nn.Module):
    """ResNet block with a projection shortcut and batch normalization."""
    num_layers: int
    num_channels: int
    strides: Tuple[int, int]
    dtype: jnp.dtype = jnp.float32
    training: bool = True

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            residual = x
            x = nn.GroupNorm(dtype=self.dtype)(x)
            x = nn.swish(x)
            x = nn.Conv(self.num_channels, kernel_size=(3, 3), strides=self.strides, dtype=self.dtype, padding="same")(x)
            x = nn.GroupNorm(dtype=self.dtype)(x)
            x = nn.swish(x)
            x = nn.Conv(self.num_channels, kernel_size=(3, 3), strides=self.strides, dtype=self.dtype, padding="same")(x)
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
    num_channels: int
    strides: Tuple[int, int]
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.num_channels, kernel_size=(3, 3), strides=self.strides, dtype=self.dtype, padding=1)(x)
        x = ResNetBlock(num_layers=3, num_channels=self.num_channels, strides=self.strides, dtype=self.dtype)(x)
        #x = nn.SelfAttention(num_heads=8, qkv_features=2 * self.num_channels, out_features=self.num_channels)(x)
        return x

class EfficentUNet(nn.Module):
    strides: Tuple[int, int] = (1, 1)
    dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=128, kernel_size=(3, 3), strides=self.strides, dtype=self.dtype, padding="same")(x)
        x = UnetDBlock(num_channels=128, strides=self.strides, dtype=self.dtype)(x)
        x = UnetDBlock(num_channels=256, strides=self.strides, dtype=self.dtype)(x)
        x = UnetDBlock(num_channels=512, strides=self.strides, dtype=self.dtype)(x)
        return x

def test():
    # 3 *  64 x 64 -> 3 * 32 x 32
    # 3 *  32 x 32 -> 3 * 16 x 16
    # 3 *  16 x 16 -> 3 * 8 x 8
    module = EfficentUNet()
    images = jnp.ones((1, 256, 256, 3))
    params = module.init(jax.random.PRNGKey(0), images)
    x = module.apply(params, images)
    print(x.shape)
    
test()
