from functools import partial
import time
from modeling_imagen import EfficentUNet
from typing import Any, Dict, Tuple
import jax
import flax
from flax import linen as nn
import jax.numpy as jnp

from tqdm import tqdm

import optax

from sampler import GaussianDiffusionContinuousTimes
from einops import rearrange, repeat, reduce, pack, unpack
from flax.training.train_state import TrainState

from jax import tree_util
from flax import struct, jax_utils
from jax.experimental import maps

import numpy as np

from jax.experimental import pjit, PartitionSpec as P

import partitioning as nnp

from flax.linen import partitioning as nn_partitioning
from flax.core.frozen_dict import FrozenDict

from utils import right_pad_dims_to
from config import ImagenConfig

from t5x.partitioning import PjitPartitioner
from t5x.checkpoints import Checkpointer
from t5x.train_state import FlaxOptimTrainState
from t5x.optimizers import adamw
from jax.experimental.maps import Mesh

mesh_shape = (2, 4)

DEFAULT_TPU_RULES = [
    ('batch', 'data'),
    ('mlp', 'model'),
    ('heads', 'model'),
    ('vocab', 'model'),
    ('embed', None),
    ('kv', None),
    ('joined_kv', None),
    ('relpos_buckets', None),
    ('abspos_buckets', None),
    ('length', None),
    ('layers', None),
    ('stack', None),
    ('mlp_activations', None),

    ("width", None),
    ("height", None),

    ("data", "data"),
    ("model", "model"),
    (None, None),
]


class UnetState(struct.PyTreeNode):
    train_state: FlaxOptimTrainState
    apply_fn: Any = struct.field(pytree_node=False)
    lr: Any = struct.field(pytree_node=False)
    step: int
    
    sampler: GaussianDiffusionContinuousTimes
    unet_config: Any
    config: Any


class GeneratorState(struct.PyTreeNode):
    unet_state: UnetState
    image: jnp.ndarray
    lowres_cond_image: jnp.ndarray
    text: jnp.ndarray
    attention: jnp.ndarray
    rng: jax.random.PRNGKey


def conditioning_pred(generator_state, t, cond_scale):
    pred = generator_state.unet_state.apply_fn(
        {"params": generator_state.unet_state.train_state.params},
        generator_state.image,
        t,
        generator_state.text,
        generator_state.attention,
        0.0,
        generator_state.lowres_cond_image,
        jnp.zeros(generator_state.image.shape[0], dtype=jnp.int16) if generator_state.lowres_cond_image is not None else None,
        generator_state.rng
    )
    null_logits = generator_state.unet_state.apply_fn(
        {"params": generator_state.unet_state.train_state.params},
        generator_state.image,
        t,
        generator_state.text,
        generator_state.attention,
        1.0,
        generator_state.lowres_cond_image,
        jnp.ones(generator_state.image.shape[0], dtype=jnp.int16) * 999 if generator_state.lowres_cond_image is not None else None,
        generator_state.rng
    )
    return null_logits + (pred - null_logits) * cond_scale


def p_mean_variance(t_index, generator_state):
    t_index = 999-t_index
    t = jnp.ones(generator_state.image.shape[0]) * t_index/999.0
    pred = conditioning_pred(generator_state, t, 4.0)
    x_start = generator_state.unet_state.sampler.predict_start_from_noise(
        generator_state.image, t=t, noise=pred)

    s = jnp.percentile(
        jnp.abs(rearrange(x_start, 'b ... -> b (...)')),
        0.95,
        axis=-1
    )  # dynamic thresholding percentile

    s = jnp.maximum(s, 1.0)
    s = right_pad_dims_to(x_start, s)
    x_start = jnp.clip(x_start, -s, s) / s

    return generator_state.unet_state.sampler.q_posterior(x_start, x_t=generator_state.image, t=t)


def p_sample(t_index, generator_state):
    model_mean, _, model_log_variance = p_mean_variance(
        t_index, generator_state)
    rng, key = jax.random.split(generator_state.rng)
    generator_state = generator_state.replace(rng=rng)
    noise = jax.random.uniform(key, generator_state.image.shape, minval=-1, maxval=1)
    x = jax.lax.cond(t_index > 0, lambda x: model_mean + noise *
                     jnp.exp(0.5 * model_log_variance), lambda x: model_mean, None)
    return generator_state.replace(image=x)


def p_sample_loop(unet_state, img, texts, attention, lowres_cond_image, rng):
    rng, key = jax.random.split(rng)
    generator_state = GeneratorState(
        unet_state=unet_state, image=img, text=texts, attention=attention, lowres_cond_image=lowres_cond_image, rng=key)
    generator_state = jax.lax.fori_loop(0, 1000, p_sample, generator_state)
    img = generator_state.image
    return img


def sample(unet_state, noise, texts, attention, lowres_cond_image, rng):
    return p_sample_loop(unet_state, noise, texts, attention, lowres_cond_image, rng)


def train_step(unet_state, imgs_start, timestep, texts, attention_masks, lowres_cond_image, lowres_aug_times, rng):
    rng, key = jax.random.split(rng)
    noise = jax.random.uniform(key, imgs_start.shape, minval=-1, maxval=1)
    rng, key = jax.random.split(rng)
    x_noise = unet_state.sampler.q_sample(imgs_start, t=timestep, noise=noise)
    if lowres_cond_image is not None:
        lowres_cond_image_noise = unet_state.sampler.q_sample(lowres_cond_image, t=lowres_aug_times, noise=noise)
    else:
        lowres_cond_image_noise = None

    def loss_fn(params):
        predicted_noise = unet_state.apply_fn(
            {"params": params},
            x_noise,
            timestep,
            texts,
            attention_masks,
            unet_state.config.cond_drop_prob,
            lowres_cond_image_noise,
            lowres_aug_times,
            key
        )
        loss = jnp.mean((noise - predicted_noise) ** 2)
        return loss, predicted_noise
    gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = gradient_fn(unet_state.train_state.params)
    
    train_state = unet_state.train_state.apply_gradient(grads=grads, learning_rate=unet_state.lr(unet_state.step))
    unet_state = unet_state.replace(train_state=train_state, step=unet_state.step + 1)

    return unet_state, compute_metrics(loss, logits, imgs_start.shape[1])




class Imagen:
    def __init__(self, config: ImagenConfig):
        start_time = time.time()
        self.random_state = jax.random.PRNGKey(0)

        self.config = config

        batch_size = config.batch_size

        self.unets = []
        self.train_steps = []
        self.sample_steps = []
        self.schedulers = []
        self.partitioner = PjitPartitioner(num_partitions=8, logical_axis_rules=DEFAULT_TPU_RULES)
        num_total_params = 0
        self.mesh = Mesh(np.asarray(jax.devices(), dtype=object).reshape(2, 4), ('data', 'model'))
        self.devices = np.asarray(jax.devices()).reshape(*mesh_shape)
        with Mesh(self.devices, ("data", "model")):
            for i in range(len(config.unets)):
                unet_config = config.unets[i]
                img_size = self.config.image_sizes[i]
                unet = EfficentUNet(config=unet_config)
                key = self.get_key()
                lr = optax.warmup_cosine_decay_schedule(
                    init_value=0.0,
                    peak_value=1e-4,
                    warmup_steps=10000,
                    decay_steps=2500000,
                    end_value=1e-5
                    )

                opt = adamw(learning_rate=lr, b1=0.9, b2=0.999,
                    eps=1e-8, weight_decay=1e-8)
                
                def init_state():
                    image = jnp.ones((batch_size, img_size, img_size, 3))  # image
                    time_step = jnp.ones(batch_size, dtype=jnp.int16)  # timestep
                    text = jnp.ones((batch_size, unet_config.max_token_len, unet_config.token_embedding_dim))  # text
                    attention_mask = jnp.ones((batch_size, unet_config.max_token_len))  # attention mask

                    lowres_cond_image = jnp.ones((batch_size, img_size, img_size, 3)) if unet_config.lowres_conditioning else None  # lowres_cond_image
                    lowres_aug_times = jnp.ones(batch_size, dtype=jnp.int16) if unet_config.lowres_conditioning else None  # lowres_aug_times

                    params = unet.init(key, image, time_step, text, attention_mask, config.cond_drop_prob, lowres_cond_image, lowres_aug_times, key)
                    # opt = OptaxWrapper(opt)
                    return FlaxOptimTrainState.create(opt, params)
                
                scheduler = GaussianDiffusionContinuousTimes.create(
                    noise_schedule="cosine", num_timesteps=1000
                )

                params_shape = jax.eval_shape(
                    init_state
                )
                
                params_spec = self.partitioner.get_mesh_axes(params_shape)
                p_init = pjit.pjit(init_state, in_axis_resources=(
                    None
                ), out_axis_resources=params_spec)

                params = p_init()
                
                sampler_spec = jax.tree_map(lambda x: None, scheduler)
                config_spec = jax.tree_map(lambda x: None, self.config)
                unet_config_spec = jax.tree_map(lambda x: None, unet_config)
                imagen_spec = UnetState(
                    train_state=params_spec,
                    apply_fn=unet.apply,
                    lr=lr,
                    step=None,
                    sampler=sampler_spec,
                    config=config_spec,
                    unet_config=unet_config_spec
                )

                unet_state = UnetState(
                    train_state=params,
                    apply_fn=unet.apply,
                    lr=lr,
                    step=0,
                    sampler=scheduler,
                    config=self.config,
                    unet_config=unet_config
                )

                self.unets.append(unet_state)
                self.schedulers.append(scheduler)

                p_train_step = pjit.pjit(train_step, in_axis_resources=(
                    imagen_spec,
                    P("data",),  # image
                    P("data",),  # timesteps
                    P("data",),  # text
                    P("data",),  # masks
                    P("data",) if unet_config.lowres_conditioning else None,  # lowres_image
                    P("data",) if unet_config.lowres_conditioning else None,  # lowres_image
                    None
                ), out_axis_resources=(imagen_spec, None))

                p_sample = pjit.pjit(sample, in_axis_resources=(
                    imagen_spec,
                    P("data"),  # image
                    P("data"),  # text
                    P("data"),  # masks
                    P("data") if unet_config.lowres_conditioning else None,  # lowres_image
                    None  # key
                ), out_axis_resources=(P("data"))
                )

                self.train_steps.append(p_train_step)
                self.sample_steps.append(p_sample)
                n_params_flax = sum(
                    jax.tree_leaves(jax.tree_map(lambda x: np.prod(x.shape), params))
                )
                num_total_params += n_params_flax

        print(f"Imagen setup complete, it took {time.time() - start_time: 0.4f} seconds for a total of {num_total_params:,} parameters")

    def get_key(self):
        self.random_state, key = jax.random.split(self.random_state)
        return key

    def sample(self, texts, attention):
        lowres_images = None
        with self.mesh(self.devices, ("data", "model")):
            for i in range(len(self.unets)):
                batch_size = texts.shape[0]
                if self.unets[i].unet_config.lowres_conditioning:
                    lowres_images = jax.image.resize(lowres_images, (texts.shape[0], self.config.image_sizes[i], self.config.image_sizes[i], lowres_images.shape[-1]), method='nearest')
                noise = jax.random.uniform(self.get_key(), (batch_size, self.config.image_sizes[i], self.config.image_sizes[i], 3), minval=-1, maxval=1)
                image = self.sample_steps[i](self.unets[i], noise, texts, attention, lowres_images, self.get_key())
                lowres_images = image
        return image

    def train_step(self, image_batch, texts_batches=None, attention_batches=None):
        image_batch = image_batch.astype(jnp.bfloat16)
        texts_batches = texts_batches.astype(jnp.bfloat16)
        attention_batches = attention_batches.astype(jnp.bfloat16)

        key = self.get_key()
        metrics = {}
        with self.mesh(self.devices, ("data", "model")):
            for i in range(len(self.unets)):
                # resize image batch to the size of the current unet
                lowres_cond_image = None
                lowres_aug_times = None
                timestep = self.schedulers[i].sample_random_timestep(image_batch.shape[0], key)
                if self.config.unets[i].lowres_conditioning:
                    lowres_cond_image = jax.image.resize(image_batch, (image_batch.shape[0], self.config.image_sizes[i], self.config.image_sizes[i], 3), method='nearest')
                    lowres_aug_times = self.schedulers[i].sample_random_timestep(1, key)
                    lowres_aug_times = repeat(lowres_aug_times, '1 -> b', b=image_batch.shape[0])

                image_batch = jax.image.resize(image_batch, (image_batch.shape[0], self.config.image_sizes[i], self.config.image_sizes[i], 3), method='nearest')
                self.unets[i], unet_metrics = self.train_steps[i](
                    self.unets[i],
                    image_batch,
                    timestep,
                    texts_batches,
                    attention_batches,
                    lowres_cond_image,
                    lowres_aug_times,
                    key
                )
                for key in unet_metrics:
                    unet_metrics[key] = np.asarray(unet_metrics[key])
                    unet_metrics[key] = np.mean(unet_metrics[key])
                metrics = {**metrics, **unet_metrics}
        return metrics


def compute_metrics(loss, logits, unet_size):
    return {f"loss_unet_{unet_size}": loss}


def test():
    config = ImagenConfig()
    imagen = Imagen(config=config)
    print("Done with imagen setup. Starting training loop")
    pb = tqdm(range(100000))
    while True:
        imagen.train_step(jnp.ones((config.batch_size, 64, 64, 3)),
                          jnp.ones((config.batch_size, 256, 512)),
                          jnp.ones((config.batch_size, 256)))
        pb.update(1)


if __name__ == "__main__":
    test()
