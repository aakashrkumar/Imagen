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
    conditioning_prob: float

def p_mean_variance(t_index, generator_state):
    t_index = 999-t_index
    t = jnp.ones(1, dtype=jnp.int16) * t_index
    pred = generator_state.imagen_state.train_state.apply_fn({"params": generator_state.imagen_state.train_state.params}, generator_state.image, t, generator_state.text, generator_state.attention, generator_state.conditioning_prob, generator_state.rng)
    x_start = generator_state.imagen_state.sampler.predict_start_from_noise(generator_state.image, t=t, noise=pred)
    
    s = jnp.percentile(
        jnp.abs(rearrange(x_start, 'b ... -> b (...)')),
        0.95,
        axis=-1
    ) # dynamic thresholding percentile
    
    s = jnp.maximum(s, 1.0)
    x_start = jnp.clip(x_start, -s, s) / s
    
    return generator_state.imagen_state.sampler.q_posterior(x_start, x_t=generator_state.image, t=t)

def p_sample(t_index, generator_state):
    model_mean, _, model_log_variance = p_mean_variance(t_index, generator_state)
    rng, key = jax.random.split(generator_state.rng)
    generator_state = generator_state.replace(rng=rng)
    noise = jax.random.normal(key, generator_state.image.shape) 
    x = jax.lax.cond(t_index > 0, lambda x: model_mean + noise * jnp.exp(0.5 * model_log_variance), lambda x: model_mean, None)
    return generator_state.replace(image=x)

@partial(jax.pmap, axis_name="batch")
def p_sample_loop(imagen_state, img, texts, attention, rng):
    # img is x0
    batch_size = img.shape[0]
    rng, key = jax.random.split(rng)
    # imgs = []
    generator_state = GeneratorState(imagen_state=imagen_state, image=img, text=texts, attention=attention, rng=rng, conditioning_prob=0.1)
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
    rng, key2 = jax.random.split(rng)
    noise = jax.random.normal(key, imgs_start.shape)
    x_noise = imagen_state.sampler.q_sample(imgs_start, timestep, noise)
    def loss_fn(params):
        predicted_noise = imagen_state.train_state.apply_fn({"params": params}, x_noise, timestep, texts, attention_masks, 0.1, key2)
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
    def __init__(self, img_size: int = 64, batch_size: int = 16, sequence_length: int = 256, encoder_latent_dims: int = 512, num_timesteps: int = 1000, loss_type: str = "l2"):
        self.random_state = jax.random.PRNGKey(0)        
        self.lowres_scheduler = GaussianDiffusionContinuousTimes.create(
            noise_schedule="cosine", num_timesteps=1000
        )
        self.unet = EfficentUNet(max_token_len=sequence_length)
        self.random_state, key = jax.random.split(self.random_state)
        self.params = self.unet.init(key, jnp.ones((batch_size, img_size, img_size, 3)), jnp.ones(batch_size, dtype=jnp.int16), jnp.ones((batch_size, sequence_length, encoder_latent_dims)), jnp.ones((batch_size, sequence_length)), 0.1, key)
        
        lr = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=1e-4,
            warmup_steps=10000,
            decay_steps=2500000,
            end_value=1e-5)
        # self.opt = optax.adafactor(learning_rate=1e-4)
        self.opt = optax.adafactor(learning_rate=lr)
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
    imagen = Imagen()
    imagen.sample(jnp.ones((8, 256, 512)), jnp.ones((8, 256)))
