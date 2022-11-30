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
from imagen_main import Imagen
import jax.numpy as jnp
import optax
from tqdm import tqdm
from flax.training import checkpoints
import jax
import tensorflow_datasets as tfds
import wandb
import time
import os
from T5Utils import get_tokenizer_and_model, encode_text


def get_datasets():
    """Load MNIST train and test datasets into memory."""
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(
        ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_ds['image'] = np.float32(train_ds['image'])
    test_ds['image'] = np.float32(test_ds['image']) 
    train_ds['image'] = np.asarray(train_ds['image'])
    train_ds['image'] = np.stack([cv2.resize(
        img, (64, 64)) for img in train_ds['image']], axis=0)
    train_ds['image'] = np.stack(
        [cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in train_ds['image']], axis=0)
    train_ds["image"] =  np.array(train_ds["image"], dtype=np.float32)
    return train_ds, test_ds

@ray.remote(resources={"tpu": 1, "host": 1}, num_cpus=30)
class Trainer:
    def __init__(self):
        wandb.init(project="imagen", entity="apcsc")
        wandb.config.batch_size = 64
        wandb.config.num_datacollectors = 5
        wandb.config.seed = 0
        wandb.config.learning_rate = 1e-4
        wandb.config.image_size = 64
        wandb.config.save_every = 100
        wandb.config.eval_every = 3
        
        self.imagen = Imagen()
        # self.T5Encoder = dataCollector.T5Encoder.remote()
        # self.datacollector = dataCollector.DataManager.remote(wandb.config.num_datacollectors, wandb.config.batch_size, self.T5Encoder)
        # self.datacollector.start.remote()
        self.tokenizer, self.model = get_tokenizer_and_model()
        self.train_dataset, _ = get_datasets()
        
    def train(self):
        pbar = tqdm(range(1, 1_000_001))
        step = 0
        while True:
            step += 1
            key = np.random.randint(0, len(self.train_dataset["image"]) - wandb.config.batch_size)
            images = self.train_dataset['image'][key:key+wandb.config.batch_size]
            captions = self.train_dataset['label'][key:key+wandb.config.batch_size]
            captions = [str(caption) for caption in captions]
            captions_encoded, attention_masks = encode_text(captions, self.tokenizer, self.model)
            
            # images, captions, captions_encoded, attention_masks = ray.get(self.datacollector.get_batch.remote())
            images = jnp.array(images)
            captions_encoded = jnp.array(captions_encoded)
            attention_masks = jnp.array(attention_masks)
            
            timesteps = list(range(0, 1000))
            
            timesteps = np.random.permutation(timesteps)[:1000]
            
            for ts in timesteps:
                start_time = time.time()
                timestep = jnp.ones(wandb.config.batch_size) * ts
                timestep = jnp.array(timestep, dtype=jnp.int16)
                metrics = self.imagen.train_step(images, timestep, captions_encoded, attention_masks) # TODO: add guidance free conditioning 
                pbar.update(1)
                metrics["images_per_second"] = wandb.config.batch_size / (time.time() - start_time)
                metrics["loss"] = np.asarray(metrics["loss"])
                metrics["loss"] = np.mean(metrics["loss"])
                # print(metrics)
                wandb.log(metrics)
                
            if step % wandb.config.save_every == 0:
                if not os.path.exists(f"ckpt/{wandb.run.id}/"):
                    os.makedirs(f"ckpt/{wandb.run.id}/")
                checkpoints.save_checkpoint(
                    f"ckpt/{wandb.run.id}/checkpoint_{step}", self.imagen.imagen_state, step=step)
                wandb.save(f"ckpt/{wandb.run.id}/checkpoint_{step}.pkl")
            
            if step % wandb.config.eval_every == 0:
                prompts = [
                    "A black and white photo of a dog",
                    "An image of a supernova",
                    "An image of a sunset",
                    "An image of the earth",
                    "The night sky",
                    "An apple",
                    "An image of an iPhone",
                    "An image of an apple watch",
                ]
                prompts = [
                    str(i) for i in range(1, 9)
                ]
                prompts_encoded, attention_masks = ray.get(self.T5Encoder.encode.remote(prompts))
                prompts_encoded = jnp.array(prompts_encoded)
                attention_masks = jnp.array(attention_masks)
                imgs = self.imagen.sample(texts=prompts_encoded, attention=attention_masks)
                images = []
                for img, prompt in zip(imgs, prompts):
                    img = np.asarray(img)
                    img = img * 127.5 + 127.5
                    img = img.astype(np.uint8)
                    img = wandb.Image(img, caption=prompt)
                    images.append(img)
                wandb.log({"samples": images})
        return 0     
            
