import cv2
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
import sklearn

import pickle


def get_mnist():
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
    # print the max pixel value
    train_ds["image"] = np.array(
        train_ds["image"], dtype=np.float32) / 127.5 - 1

    images, lables = sklearn.utils.shuffle(
        train_ds["image"], train_ds["label"])
    lables = list(map(lambda x: "An image of the number " + str(x), lables))
    # np.save("train_ds.npy", train_ds)
    return images, lables


def get_cifar100():
    """Load MNIST train and test datasets into memory."""
    ds_builder = tfds.builder('cifar100')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(
        ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_ds['image'] = np.float32(train_ds['image'])
    test_ds['image'] = np.float32(test_ds['image'])
    train_ds['image'] = np.asarray(train_ds['image'])
    train_ds['image'] = np.stack([cv2.resize(
        img, (64, 64)) for img in train_ds['image']], axis=0)
    # train_ds['image'] = np.stack(
    #     [cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in train_ds['image']], axis=0)
    # print the max pixel value
    train_ds["image"] = np.array(
        train_ds["image"], dtype=np.float32) / 127.5 - 1
    print(train_ds)
    images, lables = sklearn.utils.shuffle(
        train_ds["image"], train_ds["label"])
    labels_ids = {
        0: "apple",
        1: "aquarium_fish",
        2: "baby",
        3: "bear",
        4: "beaver",
        5: "bed",
        6: "bee",
        7: "beetle",
        8: "bicycle",
        9: "bottle",
        10: "bowl",
        11: "boy",
        12: "bridge",
        13: "bus",
        14: "butterfly",
        15: "camel",
        16: "can",
        17: "castle",
        18: "caterpillar",
        19: "cattle",
        20: "chair",
        21: "chimpanzee",
        22: "clock",
        23: "cloud",
        24: "cockroach",
        25: "couch",
        26: "crab",
        27: "crocodile",
        28: "cup",
        29: "dinosaur",
        30: "dolphin",
        31: "elephant",
        32: "flatfish",
        33: "forest",
        34: "fox",
        35: "girl",
        36: "hamster",
        37: "house",
        38: "kangaroo",
        39: "keyboard",
        40: "lamp",
        41: "lawn_mower",
        42: "leopard",
        43: "lion",
        44: "lizard",
        45: "lobster",
        46: "man",
        47: "maple_tree",
        48: "motorcycle",
        49: "mountain",
        50: "mouse",
        51: "mushroom",
        52: "oak_tree",
        53: "orange",
        54: "orchid",
        55: "otter",
        56: "palm_tree",
        57: "pear",
        58: "pickup_truck",
        59: "pine_tree",
        60: "plain",
        61: "plate",
        62: "poppy",
        63: "porcupine",
        64: "possum",
        65: "rabbit",
        66: "raccoon",
        67: "ray",
        68: "road",
        69: "rocket",
        70: "rose",
        71: "sea",
        72: "seal",
        73: "shark",
        74: "shrew",
        75: "skunk",
        76: "skyscraper",
        77: "snail",
        78: "snake",
        79: "spider",
        80: "squirrel",
        81: "streetcar",
        82: "sunflower",
        83: "sweet_pepper",
        84: "table",
        85: "tank",
        86: "telephone",
        87: "television",
        88: "tiger",
        89: "tractor",
        90: "train",
        91: "trout",
        92: "tulip",
        93: "turtle",
        94: "wardrobe",
        95: "whale",
        96: "willow_tree",
        97: "wolf",
        98: "woman",
        99: "worm"
    }
    lables = list(map(lambda x: labels_ids[x].replace("_", " "), lables))
    # np.save("train_ds.npy", train_ds)
    return images, lables


class Trainer:
    def __init__(self):
        wandb.init(project="imagen", entity="apcsc")
        wandb.config.batch_size = 128
        wandb.config.num_datacollectors = 5
        wandb.config.seed = 0
        wandb.config.learning_rate = 1e-4
        wandb.config.image_size = 64
        wandb.config.save_every = 1_000_000
        wandb.config.eval_every = 500
        self.images, self.labels = get_cifar100()
        self.tokenizer, self.model = get_tokenizer_and_model()
        # batch encode the text
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
            with open("batches.npy", "wb") as f:
                pickle.dump(self.batches, f)
                f.close()
                print("Saved batches to disk")

        self.imagen = Imagen()
        # self.T5Encoder = dataCollector.T5Encoder.remote()
        # self.datacollector = dataCollector.DataManager.remote(wandb.config.num_datacollectors, wandb.config.batch_size, self.T5Encoder)
        # self.datacollector.start.remote()

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
                timestep = jnp.ones(wandb.config.batch_size) * ts
                timestep = jnp.array(timestep, dtype=jnp.int16)
                # TODO: add guidance free conditioning
                metrics = self.imagen.train_step(
                    images, timestep, captions_encoded, attention_masks)
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
