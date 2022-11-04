from typing import Tuple
import jax
import flax
# import optax
from flax import linen as nn
import jax.numpy as jnp
from flax.linen import partitioning as nn_partitioning
from jax.experimental import maps, pjit, PartitionSpec as P
from flax.training.train_state import TrainState

import partitioning as nnp

import numpy as np


mesh_shape = (2, 4)

devices = np.asarray(jax.devices()).reshape(*mesh_shape)
mesh = maps.Mesh(devices, ('X', 'Y'))

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
            x = nnp.Conv(self.num_channels, kernel_size=(3, 3),
                        dtype=self.dtype, padding="same")(x)
            x = nn.GroupNorm(dtype=self.dtype)(x)
            x = nn.swish(x)
            x = nnp.Conv(self.num_channels, kernel_size=(3, 3),
                        dtype=self.dtype, padding="same")(x)
            residual = nnp.Conv(features=self.num_channels,
                               kernel_size=(1, 1), dtype=self.dtype)(residual)
            x = x + residual
        return x


class CombineEmbs(nn.Module):
    d: int = 32 # should be the dimensions of x
    n: int = 10000 # user defined scalor

    @nn.compact
    def __call__(self, x, t):
        # timestep encoding
        d = x.shape[-1]
        pe = jnp.zeros((1, d))
        position = jnp.array([t]).reshape(-1, 1)
        div_term = jnp.power(self.n, jnp.arange(0, d, 2) / d)
        pe[:, 0::2] = jnp.sin(position * div_term)
        pe[:, 1::2] = jnp.cos(position * div_term)
        pe = pe[jnp.newaxis,jnp.newaxis,:]
        pe = jnp.repeat(pe, x.shape[1], axis=1)
        pe = jnp.repeat(pe, x.shape[2], axis=2)
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
    def __call__(self, x):
        x = nn.Conv(features=self.num_channels, kernel_size=(3, 3),
                    strides=self.strides, dtype=self.dtype, padding=1)(x)
        # combine embs
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
    def __call__(self, x):
        # combine embs
        x = ResNetBlock(num_layers=self.num_resnet_blocks,
                        num_channels=self.num_channels, strides=self.strides, dtype=self.dtype)(x)
        if self.num_attention_heads > 0:
            x = nnp.SelfAttention(num_heads=self.num_attention_heads, qkv_features=2 *
                                 self.num_channels, out_features=self.num_channels)(x)
        x = jax.image.resize(
            x,
            shape=(x.shape[0], x.shape[1] * 2, x. shape[2] * 2, x.shape[3]),
            method="nearest",
        )
        x = nnp.Conv(features=self.num_channels, kernel_size=(3, 3), dtype=self.dtype, padding=1)(x)
        return x


class EfficentUNet(nn.Module):
    strides: Tuple[int, int] = (2, 2)
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = nnp.Conv(features=128, kernel_size=(3, 3),
                    strides=self.strides, dtype=self.dtype, padding="same")(x)
        uNet256D = UnetDBlock(num_channels=128, strides=self.strides,
                              num_resnet_blocks=2, dtype=self.dtype)(x)
        uNet64D = UnetDBlock(num_channels=256, strides=self.strides,
                             num_resnet_blocks=4, dtype=self.dtype)(uNet256D)
        uNet32D = UnetDBlock(num_channels=512, strides=self.strides,
                             num_resnet_blocks=8, dtype=self.dtype)(uNet64D)
        uNet16D = UnetDBlock(num_channels=1024, strides=self.strides,
                             num_resnet_blocks=8, num_attention_heads=8, dtype=self.dtype)(uNet32D)

        uNet16U = UnetUBlock(num_channels=1024, strides=self.strides,
                             num_resnet_blocks=8, num_attention_heads=8, dtype=self.dtype)(uNet16D)
        uNet32U = UnetUBlock(num_channels=512, strides=self.strides,
                             num_resnet_blocks=8, dtype=self.dtype)(jnp.concatenate([uNet16U, uNet32D], axis=-1))
        uNet64U = UnetUBlock(num_channels=256, strides=self.strides,
                             num_resnet_blocks=4, dtype=self.dtype)(jnp.concatenate([uNet32U, uNet64D], axis=-1))
        uNet256U = UnetUBlock(num_channels=128, strides=self.strides,
                              num_resnet_blocks=2, dtype=self.dtype)(jnp.concatenate([uNet64U, uNet256D], axis=-1))
        
        x = nnp.Dense(features=256 * 256 * 3, dtype=self.dtype)(uNet256U)
        return uNet256U



def test():
    # 3 *  64 x 64 -> 3 * 32 x 32
    # 3 *  32 x 32 -> 3 * 16 x 16
    # 3 *  16 x 16 -> 3 * 8 x 8
    
    module = EfficentUNet()
    images = jnp.ones((1, 256, 256, 3))
    with maps.Mesh(mesh.devices, mesh.axis_names), nn_partitioning.axis_rules(nnp.DEFAULT_TPU_RULES):
        params = jax.jit(module.init)(jax.random.PRNGKey(0), images)
        
        params, params_axes = params["params"], params["params_axes"]
        params_axes = nnp.get_params_axes(params, params_axes, rules=nnp.DEFAULT_TPU_RULES)
        
        preshard_fn = pjit.pjit(
    lambda x: x,  # this function does nothing
    in_axis_resources=(params_axes,),  # but this spec "pre-shards" the params
    out_axis_resources=params_axes,
)
        params_sharded = preshard_fn(params_sharded)
        
        pjitForward = pjit.pjit(module.apply, in_axis_resources=(params_axes, P("X", None)), out_axis_resources=P("X", None, "Y"))
        for i in range(100):
            x = pjitForward(params_sharded, images)
            print(x.shape)


test()
