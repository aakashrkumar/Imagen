from functools import partial
import time
from modeling_imagen import EfficentUNet
from typing import Any, Callable, Dict, Tuple
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
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze

from utils import right_pad_dims_to
from config import ImagenConfig
from jax.experimental.maps import Mesh
from jax.experimental import checkify
import model_utils


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

    ("channels", None),

    ("width", None),
    ("height", None),

    ("data", "data"),
    ("model", "model"),
    (None, None),
]


class TrainState(struct.PyTreeNode):
    step: int
    params: FrozenDict[str, Any]
    opt_state: optax.OptState
    apply_fn: Callable = struct.field(pytree_node=False)
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    dropout_rng: jnp.ndarray = None
    epoch: int = 0
    train_time: float = 0.0  # total time the model trained
    train_samples: int = 0  # number of samples seen

    def apply(self, *args, **kwargs):
        return self.apply_fn(self, {'params': self.params}, *args, **kwargs)

    def apply_gradients(self, *, grads, **kwargs):
        update_fn = self.tx.update
        updates, new_opt_state = update_fn(grads, self.opt_state, self.params)
        params = optax.apply_updates(self.params, updates)
        opt_state = new_opt_state
        return self.replace(
            step=self.step + 1,
            params=freeze(params),
            opt_state=freeze(opt_state),
            **kwargs,
        )

    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=freeze(opt_state),
            **kwargs,
        )


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
    pred = generator_state.unet_state.apply_fn(
        {"params": generator_state.unet_state.train_state.params},
        generator_state.image,
        t,
        generator_state.text,
        generator_state.attention,
        0.0,
        generator_state.lowres_cond_image,
        jnp.zeros(generator_state.image.shape[0])*0.1 if generator_state.lowres_cond_image is not None else None,
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
        jnp.zeros(generator_state.image.shape[0])*0.1 if generator_state.lowres_cond_image is not None else None,
        generator_state.rng
    )
    return null_logits + (pred - null_logits) * cond_scale


def p_mean_variance(generator_state, time_steps):
    pred = conditioning_pred(generator_state, time_steps, 5.0)
    x_start = generator_state.unet_state.sampler.predict_start_from_noise(
        generator_state.image, t=time_steps, noise=pred)

    s = jnp.percentile(
        jnp.abs(rearrange(x_start, 'b ... -> b (...)')),
        0.95,
        axis=-1
    )  # dynamic thresholding percentile

    s = jnp.maximum(s, 1.0)
    s = right_pad_dims_to(x_start, s)
    x_start = jnp.clip(x_start, -s, s) / s

    return generator_state.unet_state.sampler.q_posterior(x_start, x_t=generator_state.image, t=time_steps)


def p_sample(generator_state, time_steps):
    model_mean, _, model_log_variance = p_mean_variance(generator_state,
                                                        time_steps)
    rng, key = jax.random.split(generator_state.rng)
    generator_state = generator_state.replace(rng=rng)
    noise = jax.random.uniform(key, generator_state.image.shape, minval=-1, maxval=1)
    x = jax.lax.cond(time_steps[0] > 0, lambda x: model_mean + noise *
                     jnp.exp(0.5 * model_log_variance), lambda x: model_mean, None)
    return generator_state.replace(image=x), x


def p_sample_loop(unet_state, img, texts, attention, lowres_cond_image, rng):
    rng, key = jax.random.split(rng)
    generator_state = GeneratorState(
        unet_state=unet_state, image=img, text=texts, attention=attention, lowres_cond_image=lowres_cond_image, rng=key)
    time_steps = unet_state.sampler.get_sampling_timesteps(img.shape[0])
    generator_state, images = jax.lax.scan(f=p_sample, init=generator_state, xs=time_steps)
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

        mesh_shape = (1,)
        self.devices = np.asarray(jax.devices()).reshape(*mesh_shape)
        mesh = maps.Mesh(self.devices, ("dp", "mp"))

        num_total_params = 0
        for i in range(len(config.unets)):
            unet_config = config.unets[i]
            img_size = self.config.image_sizes[i]
            unet = EfficentUNet(config=unet_config)
            def init_params(key):
                image = jnp.ones((batch_size, img_size, img_size, 3))  # image
                time_step = jnp.ones(batch_size, dtype=jnp.int16)  # timestep
                text = jnp.ones((batch_size, unet_config.max_token_len, unet_config.token_embedding_dim))  # text
                attention_mask = jnp.ones((batch_size, unet_config.max_token_len))  # attention mask

                lowres_cond_image = jnp.ones((batch_size, img_size, img_size, 3)) if unet_config.lowres_conditioning else None  # lowres_cond_image
                lowres_aug_times = jnp.ones(batch_size, dtype=jnp.int16) if unet_config.lowres_conditioning else None  # lowres_aug_times

                params = unet.init(key, image, time_step, text, attention_mask, config.cond_drop_prob, lowres_cond_image, lowres_aug_times, key)
                return params
            params_shape = jax.eval_shape(init_params, self.get_key())
            params_spec = nnp.set_partitions(params_shape)
            params_shape = freeze(params_shape)

            scheduler = GaussianDiffusionContinuousTimes.create(
                noise_schedule="cosine", num_timesteps=1000
            )
            
            params_spec = self.partitioner.get_mesh_axes(params_shape)
            lr = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=1e-5,
                warmup_steps=10000,
                decay_steps=2500000,
                end_value=1e-6
            )
            opt = optax.adamw(learning_rate=1e-4, b1=0.9, b2=0.99,
                              eps=1e-8)
            opt_state_shape = jax.eval_shape(opt.init, params_shape)

            opt_state_spec = jax.tree_util.tree_map(
                partial(model_utils._opt_state_spec_per_leaf, spec=params_spec),
                opt_state_shape,
                # return None spec for empty elements
                is_leaf=lambda x: isinstance(x, (FrozenDict, optax.EmptyState)),
            )
            opt_state_spec = freeze(opt_state_spec)
            opt_state_shape = freeze(opt_state_shape)

            trainStateSpec = TrainState(
                params=params_spec,
                opt_state=opt_state_spec,
                step=None,
                epoch=None,
                train_time=None,
                train_samples=None,
                apply_fn=unet.__call__,
                tx=opt,
            )
            with self.mesh:           
                def init_state(params):
                    # opt = OptaxWrapper(opt)
                    return TrainState.create(
                        apply_fn=unet.__call__,
                        tx=opt,
                        params=params,
                    )
                params = pjit(model_utils.init_params, in_axis_resources=(None,), out_axis_resources=(params_spec,))(self.get_key())
                state = pjit(
                    init_state,
                    in_axis_resources=(params_spec,),
                    out_axis_resources=(trainStateSpec,),
                    donate_argnums=(0,)
                )(params, opt_state_shape)

                sampler_spec = jax.tree_map(lambda x: None, scheduler)
                config_spec = jax.tree_map(lambda x: None, self.config)
                unet_config_spec = jax.tree_map(lambda x: None, unet_config)
                unet_spec = UnetState(
                    train_state=trainStateSpec,
                    sampler=sampler_spec,
                    config=config_spec,
                    unet_config=unet_config_spec
                )

                unet_state = UnetState(
                    train_state=state,
                    apply_fn=unet.apply,
                    lr=lr,
                    step=0,
                    sampler=scheduler,
                    config=self.config,
                    unet_config=unet_config
                )

                self.unets.append(unet_state)
                self.schedulers.append(scheduler)
                p_train_step = pjit(train_step, in_axis_resources=(
                    unet_spec,
                    P("data",),  # image
                    P("data",),  # timesteps
                    P("data",),  # text
                    P("data",),  # masks
                    P("data",) if unet_config.lowres_conditioning else None,  # lowres_image
                    P("data",) if unet_config.lowres_conditioning else None,  # lowres_image
                    None
                ), out_axis_resources=(unet_spec, None))
                p_sample = pjit(sample, in_axis_resources=(
                    unet_spec,
                    P("data"),  # image
                    P("data"),  # text
                    P("data"),  # masks
                    P("data") if unet_config.lowres_conditioning else None,  # lowres_image
                    None  # key
                ), out_axis_resources=(P("data"),)
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
        with maps.mesh(self.devices, ('dp', 'mp')):
            lowres_images = None
            for i in range(len(self.unets)):
                batch_size = texts.shape[0]
                if self.unets[i].unet_config.lowres_conditioning:
                    lowres_images = jax.image.resize(lowres_images, (texts.shape[0], self.config.image_sizes[i], self.config.image_sizes[i], lowres_images.shape[-1]), method='nearest')
                noise = jax.random.uniform(self.get_key(), (batch_size, self.config.image_sizes[i], self.config.image_sizes[i], 3), minval=-1, maxval=1)
                err, image = self.sample_steps[i](self.unets[i], noise, texts, attention, lowres_images, self.get_key())
                err = err.get()
                if err:
                    print("Sample error", err)
                lowres_images = image
        return image

    def train_step(self, image_batch, texts_batches=None, attention_batches=None):
        with maps.mesh(self.devices, ('dp', 'mp')):
            image_batch = image_batch.astype(jnp.bfloat16)
            texts_batches = texts_batches.astype(jnp.bfloat16)
            attention_batches = attention_batches.astype(jnp.bfloat16)

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
                err, (self.unets[i], unet_metrics) = self.train_steps[i](
                    self.unets[i],
                    image_batch,
                    timestep,
                    texts_batches,
                    attention_batches,
                    lowres_cond_image,
                    lowres_aug_times,
                    key
                )
                err.throw()
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
        # imagen.sample(jnp.ones((8, 256, 512)),
        #                 jnp.ones((8, 256)))
        pb.update(1)


if __name__ == "__main__":
    test()
