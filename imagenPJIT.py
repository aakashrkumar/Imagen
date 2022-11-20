import time
from typing import Tuple
import jax
import flax
from flax import linen as nn
import jax.numpy as jnp

from tqdm import tqdm
from jax.experimental import pjit, PartitionSpec as P
from jax.experimental import maps
import numpy as np

from flax.linen import partitioning
import partitioning as nnp


mesh_shape = (2, 4)
devices = np.asarray(jax.devices()).reshape(*mesh_shape)
mesh = maps.Mesh(devices, ("X", "Y"))




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
            x = nn.Conv(self.num_channels, kernel_size=(3, 3),
                        dtype=self.dtype, padding="same")(x)
            x = nn.GroupNorm(dtype=self.dtype)(x)
            x = nn.swish(x)
            x = nn.Conv(self.num_channels, kernel_size=(3, 3),
                        dtype=self.dtype, padding="same")(x)
            residual = nn.Conv(features=self.num_channels,
                               kernel_size=(1, 1), dtype=self.dtype)(residual)
            x = x + residual
        return x



class CombineEmbs(nn.Module):
    d: int = 32  # should be the dimensions of x
    n: int = 10000  # user defined scalor
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, t, s):
        # timestep encoding, Note t is a tensor of dimension (batch_size x 1)

        # dimension is nummber of channels
        d = x.shape[-1] 
        # create a tensor of dimensions: batch_size x channels
        pe = jnp.zeros((t.shape[0], d)) 
        # use the formula n ^ (2*i/d) for i∈2Z (even numbers)
        div_term = jnp.power(self.n, jnp.arange(0, d, 2) / d) 
        # set all even indices to sin
        pe = pe.at[:, 0::2].set(jnp.sin(t * div_term)) 
        # set all odd indices to cos
        pe = pe.at[:, 1::2].set(jnp.cos(t * div_term)) 
        # add the height and width channels
        pe = pe[:, jnp.newaxis, jnp.newaxis, :]
        # project accross height and width
        pe = jnp.repeat(pe, x.shape[1], axis=1)
        pe = jnp.repeat(pe, x.shape[2], axis=2)
        # concatinate timestep embeds
        x = x + pe


        # add text/image encoding to x, Note for text, s is a tensor of dimension (batch_size, sequence_length, hidden_latent_size)

        text_embeds = s
        # project to correct number of channels
        text_embeds = nn.Dense(features=self.d, dtype=self.dtype)(text_embeds)
        # mean pooling across sequence
        text_embeds = jnp.mean(text_embeds, axis=2) 
        # add axis for height
        text_embeds = text_embeds[:, jnp.newaxis, :]
        # project across height and width
        text_embeds = jnp.repeat(text_embeds, x.shape[1], axis=1)
        text_embeds = jnp.repeat(text_embeds, x.shape[2], axis=2)
        # concatinate text_embeds
        x = x + text_embeds

        # use layer norm as suggested by the paper
        x = nn.LayerNorm(dtype=self.dtype)(x)
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
    skip_scale_factor: float = 1.0 / jnp.sqrt(2)

    @nn.compact
    def __call__(self, x, time):
        # (batch, height, width, channels)
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
                             num_resnet_blocks=8, dtype=self.dtype)(jnp.concatenate([uNet32U, self.skip_scale_factor * uNet64D], axis=-1), time)
        uNet128U = UnetUBlock(num_channels=256, strides=self.strides,
                              num_resnet_blocks=4, dtype=self.dtype)(jnp.concatenate([uNet64U, self.skip_scale_factor * uNet128D], axis=-1), time)
        uNet256U = UnetUBlock(num_channels=128, strides=self.strides,
                              num_resnet_blocks=2, dtype=self.dtype)(jnp.concatenate([uNet128U, self.skip_scale_factor * uNet256D], axis=-1), time)

        x = nn.Dense(features=3, dtype=self.dtype)(uNet256U)

        return x


def test():
    # 3 *  64 x 64 -> 3 * 32 x 32
    # 3 *  32 x 32 -> 3 * 16 x 16
    # 3 *  16 x 16 -> 3 * 8 x 8
    module = EfficentUNet()
    images = jnp.ones((32, 256, 256, 3))
    st = time.time()
    pinit = pjit.pjit(module.init, in_axis_resources=(None, P("X", "Y"), None), out_axis_resources=(None))
    with mesh, partitioning.axis_rules(nnp.DEFAULT_TPU_RULES):
        params = pinit(jax.random.PRNGKey(0), images, 0)
        print("Params initialized after, ", time.time() - st, " seconds")
        params, params_axes = params.pop("params_axes")
        params_axes = nnp.get_params_axes(params, params_axes, nnp.DEFAULT_TPU_RULES)

    papply = pjit.pjit(module.apply, in_axis_resources=(params_axes, P("X", "Y"), None), out_axis_resources=(None))
    for i in tqdm(range(1_000_000)):
        with mesh, partitioning.axis_rules(nnp.DEFAULT_TPU_RULES):
            x = papply(params, images, 1)
        # print(x.shape)


test()
