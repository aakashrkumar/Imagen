import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
from tqdm import tqdm
from typing import Tuple


class Test(nn.Module):
    """test conv layer for pjit sharding"""
    num_channels: int
    strides: Tuple[int, int]
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        # residual = x
        x = nn.Conv(self.num_channels, strides=self.strides, kernal_size=(3, 3), dtype=self.dtype, padding="same", )(x)
        x = nn.Conv(self.num_channels, strides=self.strides, kernal_size=(3, 3), dtype=self.dtype, padding="same", )(x)
        x = nn.Conv(self.num_channels, strides=self.strides, kernal_size=(3, 3), dtype=self.dtype, padding="same", )(x)
        # residual = nn.Conv(self.num_channels, strides=self.strides, kernal_size=(1, 1), dtype=self.dtype, padding="same")(residual)
        return x


def main():
    model = Test(num_channels=128, strides=(2, 2))
    images = jnp.ones((1, 256, 256, 3))
    params = model.init(jax.random.PRNGKey(0), images, 0)
    for i in tqdm(range(5)):
        x = jax.jit(model.apply)(params, images, 1)
        print(x.shape)
    

if __name__ == '__main__':
    main()