from typing import Tuple
import jax
import flax
# import optax
from flax import linen as nn
import jax.numpy as jnp
from flax.linen import partitioning as nn_partitioning
from jax.experimental import maps, pjit, PartitionSpec as P
from flax.training.train_state import TrainState
import optax

import partitioning as nnp

import numpy as np
import ray
import alpa

ray.init()
alpa.init(cluster="ray")


method = alpa.PipeshardParallel(num_micro_batches=16,
                                layer_option=alpa.AutoLayerOption(layer_num=2),
                                stage_option="auto")

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
    d: int = 32  # should be the dimensions of x
    n: int = 10000  # user defined scalor

    @nn.compact
    def __call__(self, x, t):
        # timestep encoding
        d = x.shape[-1]
        pe = jnp.zeros((1, d))
        position = jnp.array([t]).reshape(-1, 1)
        div_term = jnp.power(self.n, jnp.arange(0, d, 2) / d)
        pe[:, 0::2] = jnp.sin(position * div_term)
        pe[:, 1::2] = jnp.cos(position * div_term)
        pe = pe[jnp.newaxis, jnp.newaxis, :]
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
        x = nnp.Conv(features=self.num_channels, kernel_size=(
            3, 3), dtype=self.dtype, padding=1)(x)
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
        # return uNet256D
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

@alpa.parallelize(method=method)
def auto_pipeline_train_step(state, batch):

    def loss_func(params):
        out = state.apply_fn(params, batch)
        loss = 1
        return loss

    # Again, we use `alpa.grad` here to seperate the apply gradient stage with
    # the forward/backward stages in the pipeline.
    grads = alpa.grad(loss_func)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state

def test():
    # 3 *  64 x 64 -> 3 * 32 x 32
    # 3 *  32 x 32 -> 3 * 16 x 16
    # 3 *  16 x 16 -> 3 * 8 x 8

    model = EfficentUNet()
    images = jnp.ones((1, 256, 256, 3))
    params = jax.jit(model.init)(jax.random.PRNGKey(0), images)
    tx = optax.adam(learning_rate=1e-3)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    for i in range(100):
        auto_pipeline_actual_state = auto_pipeline_train_step(state, images)
        print("Step")


test()
