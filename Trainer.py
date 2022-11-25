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


@ray.remote(resources={"tpu": 1, "host": 1})
class Trainer:
    def __init__(self):
        wandb.init(project="imagen", entity="apcsc")
        wandb.config.batch_size = 64
        wandb.config.num_datacollectors = 200
        wandb.config.seed = 0
        wandb.config.learning_rate = 1e-4
        wandb.config.image_size = 64
        wandb.config.save_every = 100
        wandb.config.eval_every = 3
        
        self.imagen = Imagen()
        self.datacollector = dataCollector.DataManager.remote(wandb.config.num_datacollectors, wandb.config.batch_size)
        self.datacollector.start.remote()
        self.T5Encoder = dataCollector.T5Encoder.remote()
        
    def train(self):
        pbar = tqdm(range(1, 1_000_000))
        step = 0
        while True:
            step += 1
            images, captions, captions_encoded, attention_masks = ray.get(self.datacollector.get_batch.remote())
            images = jnp.array(images)
            captions_encoded = jnp.array(captions_encoded)
            attention_masks = jnp.array(attention_masks)
            
            timesteps = list(range(0, 1000))
            
            timesteps = np.random.permutation(timesteps)
            
            for ts in timesteps:
                start_time = time.time()
                timestep = jnp.ones(wandb.config.batch_size) * ts
                timestep = jnp.array(timestep, dtype=jnp.int16)
                metrics = self.imagen.train_step(images, timestep, captions_encoded, attention_masks) # add guidance free conditioning 
                pbar.update(1)
                metrics["images_per_second"] = wandb.config.batch_size / (time.time() - start_time)
                wandb.log(metrics)
                
            if step % wandb.config.save_every == 0:
                if not os.path.exists(f"ckpt/{wandb.run.id}/"):
                    os.makedirs(f"ckpt/{wandb.run.id}/")
                checkpoints.save_checkpoint(
                    f"ckpt/{wandb.run.id}/checkpoint_{step}", self.imagen.imagen_state)
            
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
                prompts_encoded, attention_masks = ray.get(self.T5Encoder.encode_text.remote(prompts))
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
            
