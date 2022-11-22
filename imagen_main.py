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

from jax import tree_util

@jax.jit
def j_sample(state, sampler, x, texts, t, t_index, rng):
    betas_t = extract(sampler.betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sampler.sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(
        sampler.sqrt_recip_alphas, t, x.shape)
    model_mean = sqrt_recip_alphas_t * \
        (x - betas_t * state.apply_fn({"params": state.params}, x, texts, t) /
            sqrt_one_minus_alphas_cumprod_t)
   # s = jnp.percentile(
    #    jnp.abs(model_mean), 0.95,
      #  axis=(1, *tuple(range(1, model_mean.ndim)))
     #   )
    # s = jnp.max(s, 1.0)
    
    model_mean = jnp.clip(model_mean, -1., 1.)
    
    return model_mean
def p_sample(state, sampler, x, texts, t, t_index, rng):
    model_mean = j_sample(state, sampler, x, texts, t, t_index, rng)

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(
            sampler.posterior_variance, t, x.shape)
        noise = jax.random.normal(rng, x.shape)  # TODO: use proper key
        return model_mean + noise * jnp.sqrt(posterior_variance_t)

def p_sample_loop(state, sampler, img, texts, rng):
    # img is x0
    batch_size = img.shape[0]
    rng, key = jax.random.split(rng)
    imgs = []

    for i in reversed(range(sampler.num_timesteps)):
        rng, key = jax.random.split(rng)
        img = p_sample(state, sampler, img, texts, jnp.ones(batch_size, dtype=jnp.int16) * i, i, key)
        imgs.append(img)
    # frames, batch, height, width, channels
    # reshape batch, frames, height, width, channels
    imgs = jnp.stack(imgs, axis=1)
    return imgs

def sample(state, sampler, noise, texts, rng):
    return p_sample_loop(state, sampler, noise, texts, rng)

@jax.jit
def train_step(state, sampler, x, texts, timestep, rng):
    noise = jax.random.normal(rng, x.shape)
    x_noise = sampler.q_sample(x, timestep, noise)
    def loss_fn(params):
        predicted = state.apply_fn({"params": params}, x_noise, texts, timestep)
        loss = jnp.mean((noise - predicted) ** 2)
        return loss, predicted
    gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = gradient_fn(state.params)
    grads = jax.lax.pmean(grads, "batch")
    state = state.apply_gradients(grads=grads)
    return state, compute_metrics(loss, logits)



class Imagen:
    def __init__(self, img_size: int = 64, batch_size: int = 16, num_timesteps: int = 1000, loss_type: str = "l2"):
        self.random_state = jax.random.PRNGKey(0)        
        self.lowres_scheduler = GaussianDiffusionContinuousTimes.create(
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
        
        self.image_size = img_size
        self.ptrain_step = jax.pmap(partial(train_step, sampler=self.lowres_scheduler), axis_name="batch")

    def get_key(self):
        self.random_state, key = jax.random.split(self.random_state)
        return key
    
    def sample(self, texts, batch_size):
        noise = jax.random.normal(self.get_key(), (batch_size, self.image_size, self.image_size, 3))
        return sample(self.state, self.lowres_scheduler, noise, texts, self.get_key())
    
    def train_step(self, image_batch, texts_batchs, timestep):
        self.state, metrics = self.ptrain_step(self.state, image_batch, texts_batchs, timestep, self.get_key())
        return metrics

def compute_metrics(loss, logits):
    return {"loss": loss}

def test():
    import cv2
    import numpy as np
    imagen = Imagen()
    #train_step(imagen.state, imagen.lowres_scheduler, jnp.ones((1, 64, 64, 3)), None, jnp.ones(1, dtype=jnp.int16), imagen.get_key())
    print("Training done")
    for i in tqdm(range(1000)):
        images = imagen.sample(None, 1)
        print(images.shape)
        images = np.asarray(images[0] * 255, dtype=np.uint8)
        for i in range(1000):
            cv2.imshow("image", images[i])
            cv2.waitKey(1)
if __name__ == "__main__":
    test()
    