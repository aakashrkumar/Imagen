from typing import Tuple
import jax
import flax
from flax import linen as nn
import jax.numpy as jnp

from tqdm import tqdm


class ResNetBlock(nn.Module):
    """ResNet block with a projection shortcut and batch normalization."""
    num_layers: int
    num_channels: int
    strides: Tuple[int, int]
    dtype: jnp.dtype = jnp.float32
    training: bool = True

    @nn.compact
    def __call__(self, x):
        # Iterate over the number of layers.
        for _ in range(self.num_layers):
            # Save the input for the residual connection.
            residual = x

            # Normalization, swish, and convolution.
            x = nn.GroupNorm(dtype=self.dtype)(x)
            x = nn.swish(x)
            x = nn.Conv(self.num_channels, kernel_size=(3, 3),
                        dtype=self.dtype, padding="same")(x)

            # Normalization, swish, and convolution.
            x = nn.GroupNorm(dtype=self.dtype)(x)
            x = nn.swish(x)
            x = nn.Conv(self.num_channels, kernel_size=(3, 3),
                        dtype=self.dtype, padding="same")(x)

            # Projection shortcut.
            residual = nn.Conv(features=self.num_channels,
                               kernel_size=(1, 1), dtype=self.dtype)(residual)

            # Add the residual connection.
            x = x + residual
        return x


class CombineEmbs(nn.Module):
    """Combine positional encoding with text/image encoding."""

    d: int = 32  # should be the dimensions of x
    n: int = 10000  # user defined scalor

    @nn.compact
    def __call__(self, x, t):
        # timestep encoding
        # 1. Create a position encoding matrix with shape (1, d) where d = input dimension.
        #    The matrix will be initialized with zeros.
        d = x.shape[-1]
        pe = jnp.zeros((1, d))
        # 2. Create a position encoding vector with shape (1, 1) where the value is the timestep.
        #    The vector will be initialized with zeros.
        position = jnp.array([t]).reshape(-1, 1)
        # 3. Calculate a denominator term for the sine and cosine functions.
        div_term = jnp.power(self.n, jnp.arange(0, d, 2) / d)
        # 4. Calculate the sine and cosine terms for the position encoding vector.
        #    Add the sine and cosine terms to the position encoding matrix.
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        # 5. Repeat the position encoding matrix for each batch and spatial dimension.
        pe = pe[jnp.newaxis, jnp.newaxis, :]
        pe = jnp.repeat(pe, x.shape[1], axis=1) # repeat for each spatial dimension
        pe = jnp.repeat(pe, x.shape[2], axis=2) # repeat for each batch
        # 6. Add the position encoding to the input.
        x = x + pe  
        # TODO add text/image encoding to x
        return x

class UnetDBlock(nn.Module):
    """UnetD block with a projection shortcut and batch normalization."""
    num_channels: int
    strides: Tuple[int, int]
    dtype: jnp.dtype = jnp.float32
    num_resnet_blocks: int = 3
    text_cross_attention: bool = False
    num_attention_heads: int = 0

    @nn.compact
    def __call__(self, x, time):
        x = nn.Conv(features=self.num_channels, kernel_size=(3, 3),
                    strides=self.strides, dtype=self.dtype, padding=1)(x)
        x = CombineEmbs()(x, time)
        x = ResNetBlock(num_layers=self.num_resnet_blocks,
                        num_channels=self.num_channels, strides=self.strides, dtype=self.dtype)(x)
        if self.num_attention_heads > 0:
            x = nn.SelfAttention(num_heads=self.num_attention_heads, qkv_features=2 *
                                 self.num_channels, out_features=self.num_channels)(x)
        return x


class UnetUBlock(nn.Module):
    """UnetU block with a projection shortcut and batch normalization."""
    num_channels: int
    strides: Tuple[int, int]
    dtype: jnp.dtype = jnp.float32
    num_resnet_blocks: int = 3
    text_cross_attention: bool = False
    num_attention_heads: int = 0

    @nn.compact
    def __call__(self, x, time):
        x = CombineEmbs()(x, time)
        x = ResNetBlock(num_layers=self.num_resnet_blocks,
                        num_channels=self.num_channels, strides=self.strides, dtype=self.dtype)(x)
        if self.num_attention_heads > 0:
            x = nn.SelfAttention(num_heads=self.num_attention_heads, qkv_features=2 *
                                 self.num_channels, out_features=self.num_channels)(x)
        x = jax.image.resize(
            x,
            shape=(x.shape[0], x.shape[1] * 2, x. shape[2] * 2, x.shape[3]),
            method="nearest",
        )
        x = nn.Conv(features=self.num_channels, kernel_size=(
            3, 3), dtype=self.dtype, padding=1)(x)
        return x


class EfficentUNet(nn.Module):
    strides: Tuple[int, int] = (2, 2)
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, time):
        x = nn.Conv(features=128, kernel_size=(3, 3),
            dtype=self.dtype, padding="same")(x)
        uNet256D = UnetDBlock(num_channels=128, strides=self.strides,
                              num_resnet_blocks=2, dtype=self.dtype)(x, time)
        uNet128D = UnetDBlock(num_channels=256, strides=self.strides,
                              num_resnet_blocks=4, dtype=self.dtype)(uNet256D, time)
        uNet64D = UnetDBlock(num_channels=512, strides=self.strides,
                             num_resnet_blocks=8, dtype=self.dtype)(uNet128D, time)
        uNet32D = UnetDBlock(num_channels=1024, strides=self.strides,
                             num_resnet_blocks=8, num_attention_heads=8, dtype=self.dtype)(uNet64D, time)

        uNet32U = UnetUBlock(num_channels=1024, strides=self.strides,
                             num_resnet_blocks=8, num_attention_heads=8, dtype=self.dtype)(uNet32D, time)
        uNet64U = UnetUBlock(num_channels=512, strides=self.strides,
                             num_resnet_blocks=8, dtype=self.dtype)(jnp.concatenate([uNet32U, uNet64D], axis=-1), time)
        uNet128U = UnetUBlock(num_channels=256, strides=self.strides,
                              num_resnet_blocks=4, dtype=self.dtype)(jnp.concatenate([uNet64U, uNet128D], axis=-1), time)
        uNet256U = UnetUBlock(num_channels=128, strides=self.strides,
                              num_resnet_blocks=2, dtype=self.dtype)(jnp.concatenate([uNet128U, uNet256D], axis=-1), time)

        x = nn.Dense(features=3, dtype=self.dtype)(uNet256U)

        return x

def test():
    # 3 *  64 x 64 -> 3 * 32 x 32
    # 3 *  32 x 32 -> 3 * 16 x 16
    # 3 *  16 x 16 -> 3 * 8 x 8
    module = EfficentUNet()
    images = jnp.ones((32, 256, 256, 3))
    params = module.init(jax.random.PRNGKey(0), images, 0)
    for i in tqdm(range(1_000_000)):
        x = jax.jit(module.apply)(params, images, 1)
        # print(x.shape)


test()
