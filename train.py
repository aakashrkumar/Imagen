import cv2
from T5Utils import encode_text, get_tokenizer_and_model
import ray
import dataCollector
from datasets.utils.file_utils import get_datasets_user_agent
import PIL.Image
import urllib
import io
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from datasets import load_dataset
import numpy as np
import wandb
from imagen_main import Imagen
import jax.numpy as jnp
import optax
from tqdm import tqdm
from flax.training import checkpoints
import jax
import tensorflow_datasets as tfds
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# ray.init()

# wandb.init(project="imagen", entity="therealaakash")

class config:
    batch_size = 64
    seed = 0
    learning_rate = 1e-4
    image_size = 64
    save_every = 100
    eval_every = 3
    steps = 1_000_000


def get_datasets():
    """Load MNIST train and test datasets into memory."""
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(
        ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_ds['image'] = jnp.float32(train_ds['image']) / 127.5 - 1.
    test_ds['image'] = jnp.float32(test_ds['image']) / 127.5 - 1.
    return train_ds, test_ds


def train(imagen: Imagen, steps):
    tokenizer, encoder_model = get_tokenizer_and_model()
   # collector = dataCollector.DataManager.remote(
    #   num_workers=5, batch_size=config.batch_size)
    # collector.start.remote()
    train_dataset, _ = get_datasets()
    perms = np.random.permutation(len(train_dataset['image']))
    train_ds_size = len(train_dataset['image'])
    steps_per_epoch = train_ds_size // config.batch_size
    # skip incomplete batch
    perms = perms[:steps_per_epoch * config.batch_size]
    perms = perms.reshape((steps_per_epoch, config.batch_size))
    # resize each image to 64x64 and convert to RGB
    train_dataset['image'] = np.asarray(train_dataset['image'])
    train_dataset['image'] = np.stack([cv2.resize(
        img, (config.image_size, config.image_size)) for img in train_dataset['image']], axis=0)
    train_dataset['image'] = np.stack(
        [cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in train_dataset['image']], axis=0)

    pbar = tqdm(range(1, steps * 1000 + 1))
    step = 0
    for perm in perms:
        step += 1
        batch_images = train_dataset['image'][perm, ...]
        batch_labels = train_dataset['label'][perm, ...]
        # convert labels to text
        batch_labels = [str(label) for label in batch_labels]
        # images, texts = ray.get(collector.get_batch.remote())
        text_sequence, attention_masks = encode_text(
            batch_labels, tokenizer, encoder_model)
        images = jnp.array(batch_images)
        # print(images.shape)
        timesteps = list(range(0, 1000))
        # shuffle timesteps
        timesteps = np.random.permutation(timesteps)

        for ts in timesteps:
            timestep = jnp.ones(config.batch_size) * ts
            # jax.random.randint(imagen.get_key(), (1,), 0, 999)
            timestep = jnp.array(timestep, dtype=jnp.int16)
            metrics = imagen.train_step(
                images, timestep, text_sequence, attention_masks)  # TODO: Add text(None)
            # wandb.log(metrics)
            pbar.update(1)
        if step % config.eval_every == 0:
            # TODO: Add text(None)
            prompts = ["1",
                       "2",
                       "3",
                       "4"]
            text_sequence, attention_masks = encode_text(
                prompts, tokenizer, encoder_model)
            imgs = imagen.sample(
                texts=text_sequence, attention=attention_masks)
            # print(imgs.shape) # (4, 64, 64, 3)
            # log as 16 gifs
            images = []
            for i in range(samples):
                img = np.asarray(imgs[i])  # (64, 64, 3)
                img = img * 127.5 + 127.5
                # img = wandb.Image(img)
                cv2.imwrite(f"samples/sample_{step}_{i}.png", img)
                images.append(img)
            # wandb.log({"samples": images})
        if step % config.save_every == 0:
            checkpoints.save_checkpoint(
                f"ckpt/checkpoint_{step}",
                imagen.imagen_state,
                step=step,

            )


def main():
    imagen = Imagen()
    train(imagen, config.steps)


main()
