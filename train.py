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

class config:
    batch_size = 8
    seed = 0
    learning_rate = 1e-4
    image_size = 64
    save_every = 1000
    eval_every = 1000
    steps = 1_000_000

def fetch_single_image(image_url, timeout=None, retries=0):
    for _ in range(retries + 1):
        try:
            request = urllib.request.Request(
                image_url,
                data=None,
                headers={"user-agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=timeout) as req:
                image = PIL.Image.open(io.BytesIO(req.read())).resize(
                    (config.image_size, config.image_size))
                # convert to array H, W, C
                image = np.array(image)[..., :3] / 255.0

            break
        except Exception:
            image = None
    return image




def fetch_images(batch, num_threads, timeout=None, retries=0):
    fetch_single_image_with_args = partial(
        fetch_single_image, timeout=timeout, retries=retries)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        batch["image"] = list(executor.map(
            fetch_single_image_with_args, batch["image_url"]))
    return batch

def get_image(ds):
    while True:
        item = ds[np.random.randint(len(ds))]
        image = fetch_single_image(item["image_url"])
        if image is None:
            continue
        text = item["caption"]
        return image, text

def get_images(num_images, ds):
    images = []
    text = []
    # parallelized image fetching
    with ThreadPoolExecutor(max_workers=8) as executor:
        for i in tqdm(range(num_images)):
            image, t = executor.submit(get_image, ds).result()
            images.append(image)
            text.append(t)
    return images, text

def train(imagen: Imagen, steps):
    dataset = load_dataset("red_caps", split="train")
    dataset = dataset.remove_columns("created_utc")
    dataset = dataset.remove_columns("crosspost_parents")
    dataset = dataset.remove_columns("author")
    dataset = dataset.remove_columns("subreddit")
    dataset = dataset.remove_columns("score")

    #dl = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    # dl = iter(dl)
    for step in tqdm(range(1, steps + 1)):
        images = []
        texts = []
        while len(images) < config.batch_size:
            item = dataset[np.random.randint(len(dataset))]
            image = fetch_single_image(item["image_url"])
            if image is None:
                continue
            image = jnp.array(image, dtype=jnp.float32)
            if images.shape != (config.image_size, config.image_size, 3):
                continue
            images.append(image)
            texts.append(item["caption"])
        images = jnp.array(images)
        print(images.shape)
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
                gifs.append(wandb.Video(
                    np.array(imgs[i] * 255, dtype=np.uint8), fps=60, format="gif"))
            wandb.log({"samples": gifs})
        wandb.log(metrics)


def main():
    imagen = Imagen()
    train(imagen, config.steps)


main()
