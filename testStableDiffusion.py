import numpy as np
import jax
import jax.numpy as jnp

from pathlib import Path
from jax import pmap
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from PIL import Image

from huggingface_hub import notebook_login
from diffusers import FlaxStableDiffusionPipeline

pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="bf16",
    dtype=jnp.bfloat16,
)
prompt = "An image of a supernova"
prompt_ids = pipeline.prepare_inputs([prompt] * 8)

p_params = replicate(params)
prompt_ids = shard(prompt_ids)

def create_key(seed=0):
    return jax.random.PRNGKey(seed)

rng = create_key(0)
rng = jax.random.split(rng, jax.device_count())


images = pipeline(prompt_ids, p_params, 0, jit=True)[0]
print(images)

