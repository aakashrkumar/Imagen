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
    
    def get_key(self):
        rng, key = jax.random.split(self.rng)
        return self.replace(rng=rng), key
    
class GeneratorState(struct.PyTreeNode):
    imagen_state: ImagenState
    image: jnp.ndarray
    text: jnp.ndarray
    attention: jnp.ndarray
    rng: jax.random.PRNGKey
    

def j_sample(train_state, sampler, imgs, timesteps, texts, attention, t_index, rng):
    betas_t = extract(sampler.betas, timesteps, imgs.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sampler.sqrt_one_minus_alphas_cumprod, timesteps, imgs.shape)
    sqrt_recip_alphas_t = extract(
        sampler.sqrt_recip_alphas, timesteps, imgs.shape)
    model_mean = sqrt_recip_alphas_t * \
        (imgs - betas_t * train_state.apply_fn({"params": train_state.params}, imgs, timesteps, texts, attention) /
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
    model_mean = j_sample(generator_state.imagen_state.train_state, generator_state.imagen_state.sampler, generator_state.image, generator_state.text, generator_state.attention, t, t_index, key)
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
@partial(jax.pmap, axis_name="batch")
def p_sample_loop(imagen_state, img, texts, attention, rng):
    # img is x0
    batch_size = img.shape[0]
    rng, key = jax.random.split(rng)
    # imgs = []
    generator_state = GeneratorState(imagen_state=imagen_state, image=img, text=texts, attention=attention, rng=rng)
    generator_state = jax.lax.fori_loop(0, 1000, p_sample, generator_state)
    img = generator_state.image
    #for i in reversed(range(sampler.num_timesteps)):
     #  rng, key = jax.random.split(rng)
     #  img = p_sample(imagen_state, sampler, img, texts, jnp.ones(batch_size, dtype=jnp.int16) * i, i, key)
       # imgs.append(img)
    # frames, batch, height, width, channels
    # reshape batch, frames, height, width, channels
    return img


def sample(imagen_state, noise, texts, attention, rng):
    return p_sample_loop(imagen_state, noise, texts, attention, rng)

@partial(jax.pmap, axis_name="batch")
def train_step(imagen_state, imgs_start, timestep, texts, attention_masks, rng):
    rng,key = jax.random.split(rng)
    noise = jax.random.normal(key, imgs_start.shape)
    x_noise = imagen_state.sampler.q_sample(imgs_start, timestep, noise)
    def loss_fn(params):
        predicted_noise = imagen_state.train_state.apply_fn({"params": params}, x_noise, timestep, texts, attention_masks)
        loss = jnp.mean((noise - predicted_noise) ** 2)
        return loss, predicted_noise
    gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = gradient_fn(imagen_state.train_state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")
    loss = jax.lax.pmean(loss, axis_name="batch")
    train_state = imagen_state.train_state.apply_gradients(grads=grads)
    imagen_state = imagen_state.replace(train_state=train_state)
    return imagen_state, compute_metrics(loss, logits)



class Imagen:
    def __init__(self, img_size: int = 64, batch_size: int = 16, sequence_length: int = 512, encoder_latent_dims: int = 1024, num_timesteps: int = 1000, loss_type: str = "l2"):
        self.random_state = jax.random.PRNGKey(0)        
        self.lowres_scheduler = GaussianDiffusionContinuousTimes.create(
            noise_schedule="cosine", num_timesteps=1000
        )
        self.unet = EfficentUNet()
        self.random_state, key = jax.random.split(self.random_state)
        self.params = self.unet.init(key, jnp.ones((batch_size, img_size, img_size, 3)), jnp.ones(batch_size, dtype=jnp.int16), jnp.ones((batch_size, sequence_length, encoder_latent_dims)), jnp.ones((batch_size, sequence_length)))
        
        self.opt = optax.adafactor(1e-4)
        self.train_state = train_state.TrainState.create(
            apply_fn=self.unet.apply,
            tx=self.opt,
            params=self.params['params']
        )
        self.imagen_state = ImagenState(train_state=self.train_state, sampler=self.lowres_scheduler)
        self.imagen_state = jax_utils.replicate(self.imagen_state)
        self.image_size = img_size

    def get_key(self):
        self.random_state,key = jax.random.split(self.random_state)
        return key
    
    def sample(self, texts, attention):
        batch_size = texts.shape[0] 
        noise = jax.random.normal(self.get_key(), (batch_size, self.image_size, self.image_size, 3))
        texts = jnp.reshape(texts, (jax.device_count(), -1, texts.shape[1], texts.shape[2]))
        attention = jnp.reshape(attention, (jax.device_count(), -1, attention.shape[1]))
        noise = jnp.reshape(noise, (jax.device_count(), -1, noise.shape[1], noise.shape[2], noise.shape[3]))
        keys = jax.random.split(self.get_key(), jax.device_count())
        return sample(self.imagen_state, noise, texts, attention, keys)
    
    def train_step(self, image_batch, timestep, texts_batches=None, attention_batches=None):
        # shard prng key
        # image_batch_shape = (batch_size, image_size, image_size, 3)
        image_batch = jnp.array(image_batch)
        # reshape images, texts, timestep, attention to (local_devices, device_batch_size, ...)
        image_batch = jnp.reshape(image_batch, (jax.device_count(), -1, self.image_size, self.image_size, 3))
        timestep = jnp.reshape(timestep, (jax.device_count(), -1))
        texts_batches = jnp.reshape(texts_batches, (jax.device_count(), -1, texts_batches.shape[1], texts_batches.shape[2]))
        attention_batches = jnp.reshape(attention_batches, (jax.device_count(), -1, attention_batches.shape[1]))
        keys = jax.random.split(self.get_key(), jax.local_device_count())
        self.imagen_state, metrics = train_step(self.imagen_state, image_batch, timestep, texts_batches, attention_batches, keys)
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
        # imagen.train_step(jnp.ones((batch_size, 64, 64, 3)), jnp.ones(batch_size, dtype=jnp.int16) * 10, jnp.ones((batch_size, 15, 1024)), jnp.ones((batch_size, 15)))
        
        images = imagen.sample(text_encoding, attention_mask)
       # print(images.shape)
        #images = np.asarray(images * 127.5 + 127.5, dtype=np.uint8)
        # cv2.imshow("image", images[i])
        #cv2.waitKey(0)
if __name__ == "__main__":
    test()
    