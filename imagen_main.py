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

from sampler import GaussianDiffusionContinuousTimes, extract
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

mesh_shape = (2, 4)

DEFAULT_TPU_RULES = [
    ('batch', 'X'),
    ('mlp', 'Y'),
    ('heads', 'Y'),
    ('vocab', 'Y'),
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

    ("X", "X"),
    ("Y", "Y"),
    (None, None),
]


class UnetState(struct.PyTreeNode):
    train_state: TrainState
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
    pred = generator_state.unet_state.train_state.apply_fn(
        {"params": generator_state.unet_state.train_state.params},
        generator_state.image,
        t,
        generator_state.text,
        generator_state.attention,
        0.0,
        generator_state.lowres_cond_image,
        jnp.ones(generator_state.image.shape[0], dtype=jnp.int16) * 999 if generator_state.lowres_cond_image is not None else None,
        generator_state.rng
    )
    null_logits = generator_state.unet_state.train_state.apply_fn(
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
    t = jnp.ones(generator_state.image.shape[0], dtype=jnp.int16) * t_index
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
    noise = jax.random.normal(key, generator_state.image.shape)
    x = jax.lax.cond(t_index > 0, lambda x: model_mean + noise *
                     jnp.exp(0.5 * model_log_variance), lambda x: model_mean, None)
    return generator_state.replace(image=x)


def p_sample_loop(unet_state, img, texts, attention, lowres_cond_image, rng):
    # img is x0
    batch_size = img.shape[0]
    rng, key = jax.random.split(rng)
    # imgs = []
    generator_state = GeneratorState(
        unet_state=unet_state, image=img, text=texts, attention=attention, lowres_cond_image=lowres_cond_image, rng=rng)
    generator_state = jax.lax.fori_loop(0, 1000, p_sample, generator_state)
    img = generator_state.image
    # for i in reversed(range(sampler.num_timesteps)):
    #  rng, key = jax.random.split(rng)
    #  img = p_sample(unet_state, sampler, img, texts, jnp.ones(batch_size, dtype=jnp.int16) * i, i, key)
    # imgs.append(img)
    # frames, batch, height, width, channels
    # reshape batch, frames, height, width, channels
    return img


def sample(unet_state, noise, texts, attention, lowres_cond_image, rng):
    return p_sample_loop(unet_state, noise, texts, attention, lowres_cond_image, rng)


def train_step(unet_state, imgs_start, timestep, texts, attention_masks, lowres_cond_image, lowres_aug_times, rng):
    rng, key = jax.random.split(rng)
    rng, key2 = jax.random.split(rng)
    noise = jax.random.normal(key, imgs_start.shape)
    timestep = jnp.array(timestep, dtype=jnp.int16)
    x_noise = unet_state.sampler.q_sample(imgs_start, timestep, noise)
    if lowres_cond_image is not None:
        lowres_cond_image_noise = unet_state.sampler.q_sample(lowres_cond_image, lowres_aug_times, noise)
    else:
        lowres_cond_image_noise = None

    def loss_fn(params):
        predicted_noise = unet_state.train_state.apply_fn(
            {"params": params},
            x_noise,
            timestep,
            texts,
            attention_masks,
            unet_state.config.cond_drop_prob,
            lowres_cond_image_noise,
            lowres_aug_times,
            key2
        )
        loss = jnp.mean((noise - predicted_noise) ** 2)
        return loss, predicted_noise
    gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = gradient_fn(unet_state.train_state.params)
    train_state = unet_state.train_state.apply_gradients(grads=grads)
    unet_state = unet_state.replace(train_state=train_state)

    return unet_state, compute_metrics(loss, logits, imgs_start.shape[1])


def unet_init(unet, *args):
    params, params_axes = unet.init(*args).pop("params_axes")
    return params


def get_vars_pspec(state, rules, params_axes):
    # Apply the mapping rules to all leafs ot `param_axes`. This means we replace
    # the name of the each Flax Module axis (e.g., `mlp_in'`) with the name of a
    # Mesh axis (e.g., `model`), or None (if it should not be partitioned).
    rd = {k: v for k, v in rules}
    vars_pspec = jax.tree_map(lambda x: P(*(rd[k] for k in x)), params_axes)
    # Replace all FrozenDicts in the train state with the new one.
    vars_pspec = jax.tree_map(
        lambda x: vars_pspec if isinstance(x, FrozenDict) else None,
        state,
        is_leaf=lambda x: isinstance(x, FrozenDict))
    return vars_pspec


class Imagen:
    def __init__(self, config: ImagenConfig):
        start_time = time.time()
        self.random_state = jax.random.PRNGKey(0)

        self.config = config

        devices = np.asarray(jax.devices()).reshape(*mesh_shape)
        self.mesh = maps.Mesh(devices, ("X", "Y"))
        batch_size = config.batch_size

        self.unets = []
        self.train_steps = []
        self.sample_steps = []
        self.schedulers = []
        self.partitioner = PjitPartitioner(model_parallel_submesh=(2, 4), logical_axis_rules=DEFAULT_TPU_RULES)
        num_total_params = 0
        # with maps.Mesh(self.mesh.devices, self.mesh.axis_names), nn_partitioning.axis_rules(nnp.DEFAULT_TPU_RULES):
        for i in range(len(config.unets)):
            unet_config = config.unets[i]
            img_size = self.config.image_sizes[i]
            unet = EfficentUNet(config=unet_config)
            def init_state():
                key = self.get_key()
                image = jnp.ones((batch_size, unet_config, unet_config, 3)) # image
                time_step = jnp.ones(batch_size, dtype=jnp.int16) # timestep
                text = jnp.ones((batch_size, unet_config.max_token_len, unet_config.token_embedding_dim)) # text
                attention_mask = jnp.ones((batch_size, unet_config.max_token_len)) # attention mask
                
                lowres_cond_image = jnp.ones((batch_size, unet_config, unet_config, 3)) # lowres_cond_image
                lowres_aug_times = jnp.ones(batch_size, dtype=jnp.int16) # lowres_aug_times
                
                return unet_init(unet, key, image, time_step, text, attention_mask, config.cond_drop_prob, lowres_cond_image, lowres_aug_times)
            
                
            self.random_state, key = jax.random.split(self.random_state)

            scheduler = GaussianDiffusionContinuousTimes.create(
                noise_schedule="cosine", num_timesteps=1000
            )

            params_shape = jax.eval_shape(
                init_state
            )
            params_spec = self.partitioner.get_mesh_axes(params_shape).params
            p_init = self.partitioner.partition(init_state, in_axis_resources=(
                None,
            ), out_axis_resources=params_spec)

            params = p_init()
            # self.paramsB = self.unet.init(key, jnp.ones((batch_size, img_size, img_size, 3)), jnp.ones(batch_size, dtype=jnp.int16), jnp.ones((batch_size, sequence_length, encoder_latent_dims)), jnp.ones((batch_size, sequence_length)), 0.1, key)

            lr = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=1e-4,
                warmup_steps=10000,
                decay_steps=2500000,
                end_value=1e-5)
            # self.opt = optax.adafactor(learning_rate=1e-4)
            opt = optax.chain(
                optax.clip(1.0),
                optax.adamw(learning_rate=lr, b1=0.9, b2=0.999,
                            eps=1e-8, weight_decay=1e-8)
            )  # TODO: this is a hack, fix this later
            # opt = optax.adamw(
            #   learning_rate=lr, b1=0.9, b2=0.999,
            #   eps=1e-8, weight_decay=1e-8
            # )
            train_state = TrainState.create(
                apply_fn=unet.apply,
                tx=opt,
                params=params['params']
            )
            state_spec = self.partitioner.get_mesh_axes(train_state)
            sampler_spec = jax.tree_map(lambda x: None, scheduler)
            config_spec = jax.tree_map(lambda x: None, self.config)
            unet_config_spec = jax.tree_map(lambda x: None, unet_config)

            imagen_spec = UnetState(
                train_state=state_spec,
                sampler=sampler_spec,
                config=config_spec,
                unet_config=unet_config_spec
            )

            unet_state = UnetState(
                train_state=train_state,
                sampler=scheduler,
                config=self.config,
                unet_config=unet_config
            )

            self.unets.append(unet_state)
            self.schedulers.append(scheduler)

            p_train_step = self.partitioner.partition(train_step, in_axis_resources=(
                imagen_spec,
                P("X", None, None, None),  # image
                P("X"),  # timesteps
                P("X", None, "Y"),  # text
                P("X", "Y"),  # masks
                P("X", None, None, None) if unet_config.lowres_conditioning else None,  # lowres_image
                P("X",) if unet_config.lowres_conditioning else None,  # lowres_image
                None
            ), out_axis_resources=(imagen_spec, None))

            p_sample = self.partitioner.partition(sample, in_axis_resources=(
                imagen_spec,
                P("X"),  # image
                P("X", None, "Y"),  # text
                P("X", "Y"),  # masks
                P("X") if unet_config.lowres_conditioning else None,  # lowres_image
                None  # key
            ), out_axis_resources=(P("X"))
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
        for i in range(len(self.unets)):
            batch_size = texts.shape[0]
            if self.unets[i].unet_config.lowres_conditioning:
                lowres_images = jax.image.resize(lowres_images, (texts.shape[0], self.config.image_sizes[i], self.config.image_sizes[i], lowres_images.shape[-1]), method='nearest')
            noise = jax.random.normal(self.get_key(), (batch_size, self.config.image_sizes[i], self.config.image_sizes[i], 3))
            image = self.sample_steps[i](self.unets[i], noise, texts, attention, lowres_images, self.get_key())
            lowres_images = image
        return image
        # return self.p_sample(self.unet_state, noise, texts, attention, self.get_key())

    def train_step(self, image_batch, texts_batches=None, attention_batches=None):
        image_batch = image_batch.astype(jnp.bfloat16)
        texts_batches = texts_batches.astype(jnp.bfloat16)
        attention_batches = attention_batches.astype(jnp.bfloat16)
        # shard prng key
        # image_batch_shape = (batch_size, image_size, image_size, 3)

        key = self.get_key()
        metrics = {}
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
            metrics = {**metrics, **unet_metrics}
        return metrics


def compute_metrics(loss, logits, unet_size):
    return {f"loss_unet_{unet_size}": loss}


def test():
    batch_size = 8
    config = ImagenConfig(batch_size=batch_size)
    imagen = Imagen(config=config)
    print("Done with imagen setup. Starting training loop")
    pb = tqdm(range(100000))
    while True:
        imagen.train_step(jnp.ones((batch_size, 64, 64, 3)),
                          jnp.ones((batch_size, 256, 512)),
                          jnp.ones((batch_size, 256)))
        # imagen.sample(jnp.ones((16, 256, 512)), jnp.ones((16, 256)))
        pb.update(1)
#        print("done")


if __name__ == "__main__":
    test()
