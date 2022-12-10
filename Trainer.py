from T5Utils import encode_text, get_tokenizer_and_model
import numpy as np
from imagen_main import Imagen
import jax.numpy as jnp
from tqdm import tqdm
from flax.training import checkpoints
import tensorflow_datasets as tfds
import wandb
import time
import os
from T5Utils import get_tokenizer_and_model, encode_text

import pickle
from datasets import get_cifar100, get_mnist
from config import ImagenConfig

class Trainer:
    def __init__(self):
        wandb.init(project="imagen", entity="apcsc")
        
        config = ImagenConfig()
        
        wandb.config.batch_size = config.batch_size
        wandb.config.seed = 0
        wandb.config.learning_rate = 1e-4
        wandb.config.image_size = 64
        wandb.config.save_every = 1_000_000
        wandb.config.eval_every = 500
        self.images, self.labels = get_mnist()
        self.tokenizer, self.model = get_tokenizer_and_model()
        # batch encode the text
        SAVE = False
        if os.path.exists("batches.npy"):
            with open("batches.npy", "rb") as f:
                self.batches = pickle.load(f)
                f.close()
            print("Loaded batches from file")
        else:
            self.batches = []
            for i in tqdm(range(len(self.labels)//wandb.config.batch_size)):
                batch_labels = self.labels[i *
                                           wandb.config.batch_size:(i+1)*wandb.config.batch_size]
                batch_images = self.images[i *
                                           wandb.config.batch_size:(i+1)*wandb.config.batch_size]
                batch_labels_encoded, attention_masks = encode_text(
                    batch_labels, self.tokenizer, self.model)
                batch_labels_encoded = np.asarray(batch_labels_encoded)
                self.batches.append(
                    (batch_images, batch_labels_encoded, attention_masks))
            # save the batches to disk as pickle
            if SAVE:
                with open("batches.npy", "wb") as f:
                    pickle.dump(self.batches, f)
                    f.close()
                    print("Saved batches to disk")
        print("Loaded batches, now preparing imagen")
        self.imagen = Imagen(config=config)
        print("Prepared imagen, now begining training")

    def train(self):
        pbar = tqdm(range(1, 1_000_001))
        step = 0
        passes = 0
        timesteps_per_image = 1
        while True:
            step += 1
            key = np.random.randint(0, len(self.batches) - 1)
            images, captions_encoded, attention_masks = self.batches[key]
            # images, captions, captions_encoded, attention_masks = ray.get(self.datacollector.get_batch.remote())
            images = jnp.array(images)
            captions_encoded = jnp.array(captions_encoded)
            attention_masks = jnp.array(attention_masks)

            timesteps = list(range(0, 1000))

            timesteps = np.random.permutation(timesteps)[:timesteps_per_image]

            for ts in timesteps:
                passes += 1
                start_time = time.time()
                # TODO: add guidance free conditioning
                metrics = self.imagen.train_step(
                    images, captions_encoded, attention_masks)
                pbar.update(1)
                metrics["images_per_second"] = wandb.config.batch_size / \
                    (time.time() - start_time)
                metrics["loss"] = np.asarray(metrics["loss"])
                metrics["loss"] = np.mean(metrics["loss"])
                # print(metrics)
                wandb.log(metrics, step=passes)

            if step % wandb.config.save_every == 0:
                if not os.path.exists(f"ckpt/{wandb.run.id}/"):
                    os.makedirs(f"ckpt/{wandb.run.id}/")
                checkpoints.save_checkpoint(
                    f"ckpt/{wandb.run.id}/checkpoint_{step}", self.imagen.imagen_state, step=step)
                wandb.save(f"ckpt/{wandb.run.id}/checkpoint_{step}.pkl")

            if step % wandb.config.eval_every == 0:
                prompts = [
                    "apple",
                    "bridge",
                    "butterfly",
                    "rocket",
                    "dolphin",
                    "dinosaur",
                    "orange",
                    "television",
                ]
                prompts = [f"An image of the number {i}" for i in range(1, 9)]
                
                prompts_encoded, attention_masks = encode_text(
                    prompts, self.tokenizer, self.model)
                prompts_encoded = jnp.array(prompts_encoded)
                attention_masks = jnp.array(attention_masks)
                imgs = self.imagen.sample(
                    texts=prompts_encoded, attention=attention_masks)
                images = []
                for img, prompt in zip(imgs, prompts):
                    img = np.asarray(img)
                    img = img * 127.5 + 127.5
                    img = img.astype(np.uint8)
                    img = wandb.Image(img, caption=prompt)
                    images.append(img)
                wandb.log({"samples": images}, step=passes)
        return 0
