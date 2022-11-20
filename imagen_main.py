from functools import partial
from modeling_imagen import EfficentUNet
from typing import Any, Dict, Tuple
import jax
import flax
from flax import linen as nn
import jax.numpy as jnp

from tqdm import tqdm

import optax

from sampler import GaussianDiffusionContinuousTimes, extract
from einops import rearrange, repeat, reduce, pack, unpack
from flax.training import train_state


class Imagen:
    def __init__(self, img_size: int = 64, batch_size: int = 16, num_timesteps: int = 1000, loss_type: str = "l2"):
        self.random_state = jax.random.PRNGKey(0)        
        self.lowres_scheduler = GaussianDiffusionContinuousTimes(
            noise_schedule="cosine", num_timesteps=1000
        )
        
        self.unet = EfficentUNet()
        self.params = self.unet.init(self.get_key(), jnp.ones((batch_size, img_size, img_size, 3)), None, jnp.ones(batch_size, dtype=jnp.int16))
        
        self.opt = optax.adafactor(1e-4)
        self.state = train_state.TrainState.create(
            apply_fn=self.unet.apply,
            tx=self.opt,
            params=self.params['params']
        )

    def get_key(self):
        self.random_state, key = jax.random.split(self.random_state)
        return key
    
@jax.jit
def train_step(state, x, texts, timestep, rng):
    noise = jax.random.normal(rng, x.shape)

    def loss_fn(params):
        predicted = state.apply_fn({"params": params}, x, texts, timestep)
        loss = jnp.mean((noise - predicted) ** 2)
        return loss, predicted
    gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = gradient_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def p_sample(state, sampler, x, texts, t, t_index, rng):
    betas_t = extract(sampler.betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sampler.sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(
        sampler.sqrt_recip_alphas, t, x.shape)
    model_mean = sqrt_recip_alphas_t * \
        (x - betas_t * state.unet.apply(x, texts, t) /
            sqrt_one_minus_alphas_cumprod_t)

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(
            sampler.posterior_variance, t, x.shape)
        noise = jax.random.normal(rng, x.shape)  # TODO: use proper key
        return model_mean + noise * jnp.sqrt(posterior_variance_t)

def p_sample_loop(state, sampler, shape, texts, rng):
    b = shape[0]
    rng, key = jax.random.split(rng)
    img = jax.random.normal(key, shape)
    imgs = []

    for i in tqdm(reversed(range(sampler.num_timesteps))):
        rng, key = jax.random.split(rng)
        img = p_sample(state, sampler, img, texts, jnp.ones(b) * i, i, key)
        imgs.append(img)
    return imgs

@jax.jit
def sample(state, sampler, shape, texts, rng):
    return p_sample_loop(state, sampler, shape, texts, rng)


def test():
    imagen = Imagen()
    train_step(imagen.state, jnp.ones((16, 64, 64, 3)), None, jnp.ones(16, dtype=jnp.int16), imagen.get_key())
    
if __name__ == "__main__":
    test()
    