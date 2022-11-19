from typing import Tuple
import jax
import flax
from flax import linen as nn
import jax.numpy as jnp

from tqdm import tqdm

from functools import partial

def right_pad_dims_to(x, t):
    padding_dims = t.ndim - x.ndim
    if padding_dims <= 0:
        return t
    return jnp.pad(t, [(0, 0)] * padding_dims + [(0, 0), (0, 0)])

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

def alpha_cosine_log_snr(t, s: float = 0.008):
    return -jnp.log((jnp.cos((t + s) / (1 + s) * jnp.pi * 0.5) ** -2) - 1, eps = 1e-5)

def sigmoid(x):
    return 1. / (1. + jnp.exp(-x))

def log_snr_to_alpha_sigma(log_snr):
    return jnp.sqrt(sigmoid(log_snr)), jnp.sqrt(sigmoid(-log_snr))

class GaussianDiffusionContinuousTimes():
    noise_schedule: str = "linear"
    num_timesteps: int = 1000
    
    def __init__(self):
        if self.noise_schedule == "linear":
            self.beta_schedule = linear_beta_schedule
        if self.noise_schedule == "cosine":
            self.beta_schedule = cosine_beta_schedule
            self.log_snr = alpha_cosine_log_snr
            
    
    def get_times(self):
        return self.beta_schedule(self.num_timesteps)
        
    def q_sample(self, x_start, times, noise):
        log_snr = self.log_snr(times)
        log_snr_padded_dim = right_pad_dims_to(x_start, log_snr)
        alpha, sigma =  log_snr_to_alpha_sigma(log_snr_padded_dim)
        
        return alpha * x_start + sigma * noise, log_snr, alpha, sigma
    
    def get_condition(times):
        return self.log_snr(times)