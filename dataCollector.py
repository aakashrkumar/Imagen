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

import ray
from T5Utils import encode_text, get_tokenizer_and_model
import tensorflow_datasets as tfds
import cv2

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
                image = PIL.Image.open(io.BytesIO(req.read())).resize(
                    (64, 64))
                # convert to array H, W, C
                image = np.array(image)[..., :3] / 127.5 - 1.0

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


@ray.remote(resources={"host": 1})
class SharedStorage:
    def __init__(self):
        self.images = []
        self.texts = []
        self.texts_encoded = []
        self.attention_masks = []

        self.images_unencoded = []
        self.texts_unencoded = []

    def add_data(self, images, texts):
        self.images_unencoded.extend(images)
        self.texts_unencoded.extend(texts)
        print(len(self.images_unencoded)))))

    def add_data_encoded(self, images, texts, texts_encoded, attention_masks):
        self.images.extend(images)
        self.texts.extend(texts)
        self.texts_encoded.extend(texts_encoded)
        self.attention_masks.extend(attention_masks)

    def get_batch(self, batch_size):
        if len(self.images) < batch_size:
            return None
        images = []
        texts = []
        texts_encoded = []
        attention_masks = []
        for _ in range(batch_size):
            images.append(self.images.pop(0))
            texts.append(self.texts.pop(0))
            texts_encoded.append(self.texts_encoded.pop(0))
            attention_masks.append(self.attention_masks.pop(0))
        images = np.array(images)
        texts_encoded = np.array(texts_encoded)
        attention_masks = np.array(attention_masks)
        return images, texts, texts_encoded, attention_masks

    def get_batch_unencoded(self, batch_size):
        if len(self.images_unencoded) < batch_size:
            return None
        images = []
        texts = []
        for _ in range(batch_size):
            images.append(self.images_unencoded.pop(0))
            texts.append(self.texts_unencoded.pop(0))
        images = np.array(images)
        return images, texts


@ray.remote(num_cpus=5, resources={"host": 1})
class DatasetFetcher:
    def __init__(self):
        dataset = load_dataset("red_caps", split="train")
        dataset = dataset.remove_columns("created_utc")
        dataset = dataset.remove_columns("crosspost_parents")
        dataset = dataset.remove_columns("author")
        dataset = dataset.remove_columns("subreddit")
        dataset = dataset.remove_columns("score")
        self.dataset = dataset
        self.dataset, _ = get_datasets()

    def get_data(self):
        key = np.random.randint(0, len(self.dataset))
        return self.dataset["image"][key], self.dataset["label"][key]


def get_datasets():
    """Load MNIST train and test datasets into memory."""
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(
        ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_ds['image'] = np.float32(train_ds['image']) / 127.5 - 1.
    test_ds['image'] = np.float32(test_ds['image']) / 127.5 - 1.
    train_ds['image'] = np.asarray(train_ds['image'])
    train_ds['image'] = np.stack([cv2.resize(
        img, (64, 64)) for img in train_ds['image']], axis=0)
    train_ds['image'] = np.stack(
        [cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in train_ds['image']], axis=0)

    return train_ds, test_ds


@ray.remote
class DataCollector:
    def __init__(self, shared_storage, dataset):
        self.shared_storage = shared_storage
        self.dataset = dataset

    def collect(self):
        while True:
            item = self.dataset.get_data.remote()
            image, label = ray.get(item)
            """
            item = ray.get(item)
            image = fetch_single_image(item["image_url"])
            if image is None:
                continue
            image = np.array(image, dtype=np.float32)
            if image.shape != (64, 64, 3):
                continue
            """
            self.shared_storage.add_data.remote([image], [str(label)])


@ray.remote(resources={"tpu": 1})
class T5Encoder:
    def __init__(self):
        self.tokenizer, self.model = get_tokenizer_and_model()

    def encode(self, texts):
        logits, attention_mask = encode_text(texts, self.tokenizer, self.model)
        return np.asarray(logits), np.asarray(attention_mask)


@ray.remote(resources={"host": 1})
class Processor:
    def __init__(self, storage, encoder):
        self.encoder = encoder
        self.shared_storage = storage

    def start_encoding(self):
        while True:
            out = ray.get(self.shared_storage.get_batch_unencoded.remote(32))
            if out:
                images, texts = out
                texts_encoded, attention_masks = ray.get(
                    self.encoder.encode.remote(texts))
                self.shared_storage.add_data_encoded.remote(
                    images, texts, texts_encoded, attention_masks)


@ray.remote(num_cpus=2, resources={"host": 1})
class DataManager:
    def __init__(self, num_workers, batch_size):#, encoder):
        self.shared_storage = SharedStorage.options(max_concurrency=10).remote()
        self.batch_size = batch_size
        self.datasetFetcher = DatasetFetcher.options(max_concurrency=10).remote()
        self.workers = [DataCollector.remote(
            self.shared_storage, self.datasetFetcher) for _ in range(num_workers)]
        # self.processor = Processor.remote(self.shared_storage, encoder)

    def start(self):
        for worker in self.workers:
            worker.collect.remote()
        # self.processor.start_encoding.remote()

    def get_batch(self):
        data = None
        while data is None:
            data = ray.get(
                self.shared_storage.get_batch_unencoded.remote(self.batch_size))
        return data
