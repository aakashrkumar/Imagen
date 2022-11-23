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
from flax import struct, jax_utils


class ImagenState(struct.PyTreeNode):
    train_state: train_state.TrainState
    sampler: GaussianDiffusionContinuousTimes
    rng: jax.random.PRNGKey
    
    def get_key(self):
        rng, key = jax.random.split(self.rng)
        return key, self.replace(rng=rng)
    
class GeneratorState(struct.PyTreeNode):
    imagen_state: ImagenState
    image: jnp.ndarray
    text: jnp.ndarray
    rng: jax.random.PRNGKey
    
    

def j_sample(train_state, sampler, x, texts, t, t_index, rng):
    betas_t = extract(sampler.betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sampler.sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(
        sampler.sqrt_recip_alphas, t, x.shape)
    model_mean = sqrt_recip_alphas_t * \
        (x - betas_t * train_state.apply_fn({"params": train_state.params}, x, t, texts) /
            sqrt_one_minus_alphas_cumprod_t)
   # s = jnp.percentile(
    #    jnp.abs(model_mean), 0.95,
      #  axis=(1, *tuple(range(1, model_mean.ndim)))
     #   )
    # s = jnp.max(s, 1.0)
    
    model_mean = jnp.clip(model_mean, -1., 1.)
    
    return model_mean

def p_sample(t_index, generator_state):
    t_index = 999-t_index
    t = jnp.ones(1, dtype=jnp.int16) * t_index
    rng, key = jax.random.split(generator_state.rng)
    model_mean = j_sample(generator_state.imagen_state.train_state, generator_state.imagen_state.sampler, generator_state.image, generator_state.text, t, t_index, key)
    rng, key = jax.random.split(rng)
    posterior_variance_t = extract(
    generator_state.imagen_state.sampler.posterior_variance, t, generator_state.image.shape)
    noise = jax.random.normal(key, generator_state.image.shape)  # TODO: use proper key

    x = jax.lax.cond(t_index > 0, lambda x: model_mean + noise * jnp.sqrt(posterior_variance_t), lambda x: model_mean, None)
   #if t_index == 0:
   #     x = model_mean
   # else:
    #    x = model_mean + noise * jnp.sqrt(posterior_variance_t)
    return GeneratorState.replace(generator_state, image=x, rng=rng)
@jax.jit
def p_sample_loop(imagen_state, img, texts, rng):
    # img is x0
    batch_size = img.shape[0]
    rng, key = jax.random.split(rng)
    # imgs = []
    generator_state = GeneratorState(imagen_state=imagen_state, image=img, text=texts, rng=rng)
    generator_state = jax.lax.fori_loop(0, 1000, p_sample, generator_state)
    img = generator_state.image
    #for i in reversed(range(sampler.num_timesteps)):
     #  rng, key = jax.random.split(rng)
     #  img = p_sample(imagen_state, sampler, img, texts, jnp.ones(batch_size, dtype=jnp.int16) * i, i, key)
       # imgs.append(img)
    # frames, batch, height, width, channels
    # reshape batch, frames, height, width, channels
    return img


def sample(imagen_state, noise, texts, rng):
    return p_sample_loop(imagen_state, noise, texts, rng)

@partial(jax.pmap, axis_name="batch")
def train_step(imagen_state, x, timestep, texts):
    imagen_state,key = imagen_state.get_key()
    noise = jax.random.normal(key, x.shape)
    x_noise = imagen_state.sampler.q_sample(x, timestep, noise)
    def loss_fn(params):
        predicted = imagen_state.train_state.apply_fn({"params": params}, x_noise, timestep, texts)
        loss = jnp.mean((noise - predicted) ** 2)
        return loss, predicted
    gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = gradient_fn(imagen_state.train_state.params)
    loss = jax.lax.pmean(loss, "batch")
    grads = jax.lax.pmean(grads, "batch")
    train_state = imagen_state.train_state.apply_gradients(grads=grads)
    imagen_state = imagen_state.replace(train_state=train_state)
    return imagen_state, compute_metrics(loss, logits)



class Imagen:
    def __init__(self, img_size: int = 64, batch_size: int = 16, num_timesteps: int = 1000, loss_type: str = "l2"):
        self.random_state = jax.random.PRNGKey(0)        
        self.lowres_scheduler = GaussianDiffusionContinuousTimes.create(
            noise_schedule="cosine", num_timesteps=1000
        )
        self.unet = EfficentUNet()
        self.random_state, key = jax.random.split(self.random_state)
        self.params = self.unet.init(key, jnp.ones((batch_size, img_size, img_size, 3)), jnp.ones(batch_size, dtype=jnp.int16), None)
        
        self.opt = optax.adafactor(1e-4)
        self.train_state = train_state.TrainState.create(
            apply_fn=self.unet.apply,
            tx=self.opt,
            params=self.params['params']
        )
        self.imagen_state = ImagenState(train_state=self.train_state, sampler=self.lowres_scheduler, rng=self.random_state)
        self.imagen_state = jax_utils.replicate(self.imagen_state)
        self.image_size = img_size

    def get_key(self):
        self.imagen_state, self.random_state = self.imagen_state.get_key()
        return self.random_state
    
    def sample(self, texts, batch_size):
        noise = jax.random.normal(self.get_key(), (batch_size, self.image_size, self.image_size, 3))
        return sample(self.imagen_state, noise, texts, self.get_key())
    
    def train_step(self, image_batch, timestep, texts_batchs=None):
        self.imagen_state, metrics = train_step(self.imagen_state, image_batch, timestep, texts_batchs)
        return metrics

def compute_metrics(loss, logits):
    return {"loss": loss}

def test():
    import cv2
    import numpy as np
    imagen = Imagen()
    print("Training done")
    batch_size = 8
    for i in tqdm(range(1000)):
        imagen.train_step(jnp.ones((batch_size, 64, 64, 3)), jnp.ones(batch_size, dtype=jnp.int16) * 10)
        #images = imagen.sample(None, 1)
       # print(images.shape)
        #images = np.asarray(images * 127.5 + 127.5, dtype=np.uint8)
        # cv2.imshow("image", images[i])
        #cv2.waitKey(0)
if __name__ == "__main__":
    test()
    