from jax import tree_util
from typing import Any, Tuple
import jax
import flax
from flax import linen as nn
import jax.numpy as jnp
from torch import unbind

from tqdm import tqdm

from functools import partial
import numpy as np
from flax import struct

from einops import rearrange, repeat
from utils import jax_unstack, default

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return jnp.reshape(t, (*t.shape, *((1,) * padding_dims)))

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> float:
    """Cosine decay schedule."""
    steps = timesteps + 1
    x = jnp.linspace(0, timesteps, steps)
    alphas_cumprod = jnp.cos(((x/timesteps) + s)/(1+s) * jnp.pi/2)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return jnp.clip(betas, a_min=0.0001, a_max=0.9999)


def linear_beta_schedule(timesteps: int):
    beta_start = 0.0001
    beta_end = 0.02

    return jnp.linspace(beta_start, beta_end, timesteps)


def extract(a, t, x_shape):
    # extract the values of a at the positions given by t
    # batch_size = t.shape[0] # get the batch size
    batch_size = t.shape[0]  # get the batch size
    out = jnp.take_along_axis(a, t, -1)  # extract the values
    # reshape the output
    return jnp.reshape(out, (batch_size, *((1,) * (len(x_shape) - 1))))

@jax.jit
def alpha_cosine_log_snr(t, s: float = 0.008):
    x = ((jnp.cos((t + s) / (1 + s) * jnp.pi * 0.5) ** -2) - 1)
    x = jnp.clip(x, a_min=1e-8, a_max=1e8)
    return -jnp.log(x)

def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def log_snr_to_alpha_sigma(log_snr):
    return jnp.sqrt(sigmoid(log_snr)), jnp.sqrt(sigmoid(-log_snr))

class GaussianDiffusionContinuousTimes(struct.PyTreeNode):
    noise_schedule: str = struct.field(pytree_node=False)
    num_timesteps: int = struct.field(pytree_node=False)
    
    log_snr: Any = struct.field(pytree_node=False)
    

    def sample_random_timestep(self, batch_size, rng):
        return jax.random.uniform(key=rng, shape=(batch_size,), minval=0, maxval=1)
    
    def get_sampling_timesteps(self, batch):
        times = jnp.linspace(1., 0., self.num_timesteps + 1)
        times = repeat(times, 't -> b t', b = batch)
        times = jnp.stack((times[:, :-1], times[:, 1:]), dim = 0)
        times = unbind(times, axis =-1)
        return times

    def q_posterior(self, x_start, x_t, t):
        t_next = default(t_next, lambda: jnp.maximum(0, (t - 1. / self.num_timesteps)))
        log_snr = self.log_snr(t)
        log_snr_next = self.log_snr(t_next)
        log_snr, log_snr_next = map(partial(right_pad_dims_to, x_t), (log_snr, log_snr_next))

        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

        # c - as defined near eq 33
        c = -jnp.expm1(log_snr - log_snr_next)
        posterior_mean = alpha_next * (x_t * (1 - c) / alpha + c * x_start)

        # following (eq. 33)
        posterior_variance = (sigma_next ** 2) * c
        # add epsilon to q_posterior_variance to avoid numerical issues
        posterior_variance = jnp.maximum(posterior_variance, 1e-8)
        posterior_log_variance_clipped = jnp.log(posterior_variance)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def q_sample(self, x_start, t, noise):
        dtype = x_start.dtype

        if isinstance(t, float):
            batch = x_start.shape[0]
            t = jnp.full((batch,), t, dtype = dtype)
        log_snr = self.log_snr(t)
        log_snr_padded_dim = right_pad_dims_to(x_start, log_snr)
        alpha, sigma =  log_snr_to_alpha_sigma(log_snr_padded_dim)

        return alpha * x_start + sigma * noise, log_snr, alpha, sigma

    def predict_start_from_noise(self, x_t, t, noise):
        log_snr = self.log_snr(t)
        log_snr = right_pad_dims_to(x_t, log_snr)
        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        return (x_t - sigma * noise) / jnp.maximum(alpha, 1e-8)

    @classmethod
    def create(cls, noise_schedule, num_timesteps):
        if noise_schedule == "cosine":
            log_snr = alpha_cosine_log_snr
        elif noise_schedule == "linear":
            log_snr = linear_beta_schedule
        else:
            ValueError(f"Unknown noise schedule {noise_schedule}")
        return cls(noise_schedule=noise_schedule, num_timesteps=num_timesteps, log_snr=log_snr)

def get_noisy_image(x, t, noise, sampler):
    return sampler.q_sample(x, t, noise)


def test():
    import cv2
    img = jnp.ones((64, 64, 64, 3))
    img = jnp.array(img)
    img /= 255.0
    noise = jax.random.normal(jax.random.PRNGKey(0), img.shape)
    scheduler = GaussianDiffusionContinuousTimes.create(
        noise_schedule="cosine", num_timesteps=1000)
    images = []
    # ts = scheduler.get_sampling_timesteps(64, jax.random.PRNGKey(0))
    ts = 1.
    x_noise, _, _, _ = get_noisy_image(img, ts, noise, scheduler)
    # print(x_noise)
    for i in range(1000):
       
        x_noise = x_noise * 255.0
        x_noise = x_noise.astype(jnp.uint8)
        x_noise = np.asarray(x_noise[0])
        images.append(x_noise)
    videoWriter = cv2.VideoWriter("video.mp4", cv2.VideoWriter_fourcc(
        *'MJPG'), 240, (x_noise.shape[1], x_noise.shape[0]))
    for i in range(1000):
        videoWriter.write(images[i])

    videoWriter.release()


if __name__ == "__main__":
    test()
