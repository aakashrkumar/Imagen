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

from torch.utils.data import DataLoader

wandb.init(project="imagen")

USER_AGENT = get_datasets_user_agent()


def fetch_single_image(image_url, timeout=None, retries=0):
    for _ in range(retries + 1):
        try:
            request = urllib.request.Request(
                image_url,
                data=None,
                headers={"user-agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=timeout) as req:
                image = PIL.Image.open(io.BytesIO(req.read())).resize((config.image_size, config.image_size))
                # convert to array H, W, C
                image = np.array(image)[..., :3] / 255.0
                
            break
        except Exception:
            image = None
    return image

class config:
    batch_size = 8
    seed = 0
    learning_rate = 1e-4
    image_size = 64
    save_every = 1000
    eval_every = 1000
    steps = 100_000
    

def fetch_images(batch, num_threads, timeout=None, retries=0):
    fetch_single_image_with_args = partial(fetch_single_image, timeout=timeout, retries=retries)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        batch["image"] = list(executor.map(fetch_single_image_with_args, batch["image_url"]))
    return batch

def train(imagen: Imagen, steps):
    dataset = load_dataset("red_caps", split="validation")
    dataset = dataset.map(fetch_images, batched=True, batch_size=16, fn_kwargs={"num_threads": 20})

    dl = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    for step in tqdm(range(1, steps + 1)):
        images = next(iter(dl))["image"]
        texts = next(iter(dl))["captions"]
        timestep = jnp.ones(config.batch_size) * \
            jax.random.randint(imagen.get_key(), (1,), 0, 999)
        timestep = jnp.array(timestep, dtype=jnp.int16)
        metrics = imagen.train_step(
            images, None, timestep)  # TODO: Add text(None)
        if step % config.eval_every == 0:
            # TODO: Add text(None)
            imgs = imagen.sample(None, 16)
            # log as 16 gifs
            gifs = []
            for i in range(16):
                gifs.append(wandb.Video(np.array(imgs[i] * 255, dtype=np.uint8), fps=60, format="gif"))
            wandb.log({"samples": gifs})
        wandb.log(metrics)


def main():
    imagen = Imagen()
    train(imagen, config.steps)


main()
