import jax
from flax.training import checkpoints
from tqdm import tqdm
import optax
import jax.numpy as jnp
from imagen_main import Imagen
import wandb
import numpy as np
from datasets import load_dataset

from functools import partial
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
import urllib

import PIL.Image

from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent

import dataCollector
import ray
ray.init()

# wandb.init(project="imagen", entity="therealaakash")
USER_AGENT = get_datasets_user_agent()


class config:
    batch_size = 64
    seed = 0
    learning_rate = 1e-4
    image_size = 64
    save_every = 1
    eval_every = 10
    steps = 1_000_000


def train(imagen: Imagen, steps):
    collector = dataCollector.DataManager.remote(
        num_workers=5, batch_size=config.batch_size)
    collector.start.remote()

    #dl = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    # dl = iter(dl)
    pbar = tqdm(range(1, steps * 1000 + 1))
    for step in range(1, steps + 1):
        images, texts = ray.get(collector.get_batch.remote())
        images = jnp.array(images)
        # print(images.shape)
        timesteps = list(range(0, 2))
        # shuffle timesteps
        timesteps = np.random.permutation(timesteps)
        for ts in timesteps:
            timestep = jnp.ones(config.batch_size) * ts
            # jax.random.randint(imagen.get_key(), (1,), 0, 999)
            timestep = jnp.array(timestep, dtype=jnp.int16)
            metrics = imagen.train_step(
                images, timestep, texts_batchs=None)  # TODO: Add text(None)
            # wandb.log(metrics)
            pbar.update(1)
        if step % config.eval_every == 0:
            samples = 4
            # TODO: Add text(None)
            imgs = imagen.sample(texts=None, batch_size=samples)
            # print(imgs.shape) # (4, 64, 64, 3)
            # log as 16 gifs
            images = []
            for i in range(samples):
                img = np.asarray(imgs[i])  # (64, 64, 3)
                img = img * 127.5 + 127.5
                # img = wandb.Image(img)
                images.append(img)
            # wandb.log({"samples": images})
        if step % config.save_every == 0:
            checkpoints.save_checkpoint(
                f"checkpoint_{step}", 
                imagen.imagen_state.train_state, 
                imagen.imagen_state.train_state.opt)


def main():
    imagen = Imagen()
    train(imagen, config.steps)


main()
