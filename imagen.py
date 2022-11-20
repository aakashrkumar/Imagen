from typing import Tuple
import jax
import flax
from flax import linen as nn
import jax.numpy as jnp

from tqdm import tqdm

import optax

from sampler import GaussianDiffusionContinuousTimes, extract
from einops import rearrange, repeat, reduce, pack, unpack


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
        # repeat for each spatial dimension
        pe = jnp.repeat(pe, x.shape[1], axis=1)
        pe = jnp.repeat(pe, x.shape[2], axis=2)  # repeat for each batch
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


class Imagen(nn.Module):

    loss_type: str = "l2"

    def setup(self):
        self.lowres_scheduler = GaussianDiffusionContinuousTimes(
            noise_schedule="cosine", num_timesteps=1000)
        self.unet = EfficentUNet()
        # todo: text encoder

    def p_sample(self, x, t, t_index, rng):
        betas_t = extract(self.lowres_scheduler.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.lowres_scheduler.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(
            self.lowres_scheduler.sqrt_recip_alphas, t, x.shape)
        model_mean = sqrt_recip_alphas_t * \
            (x - betas_t * self.lowres_scheduler.unet(x, t) /
             sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(
                self.lowres_scheduler.posteiror_variance, t, x.shape)
            noise = jax.random.normal(rng, x.shape)  # TODO: use proper key
            return model_mean + noise * jnp.sqrt(posterior_variance_t)

    def p_sample_loop(self, shape, rng):
        b = shape[0]
        rng, key = jax.random.split(rng)
        img = jnp.random.normal(key, shape)
        imgs = []

        for i in tqdm(reversed(range(self.lowres_scheduler.num_timesteps))):
            rng, key = jax.random.split(rng)
            img = self.p_sample(img, jnp.ones(b) * i, i, key)
            imgs.append(img)
        return imgs

    def sample(self, model, image_size=(256, 256, 3), batch_size=16, rng=jax.random.PRNGKey(0)):
        return self.lowres_scheduler.p_sample_loop(model, shape=(batch_size, *image_size), rng=rng)

    def p_losses(self, x_start, t, rng):  # t is an int
        noise = jax.random.normal(rng, x_start.shape)
        x_noisy = self.lowres_scheduler.q_sample(x_start, t, noise)
        predicted = self.unet(x_noisy, t)

        if self.loss_type == "l2":
            loss = jnp.mean((noise - predicted) ** 2)
        else:
            raise NotImplementedError()

        return loss, predicted

    def __call__(self, x, t, rng):
        return self.p_losses(x, t, rng)


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

if __name__ == "__main__":
    test()
