from functools import partial
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
import urllib

import PIL.Image

from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent

from torch.utils.data import DataLoader
import cv2
import numpy as np
import jax
import jax.numpy as jnp
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
                image = PIL.Image.open(io.BytesIO(req.read())).resize((64, 64))
                # convert to array H, W, C
                image = np.array(image)[..., :3] / 255.0
                
            break
        except Exception as e:
            print(e)
            image = None
    return image

    

def fetch_images(batch, num_threads, timeout=None, retries=0):
    fetch_single_image_with_args = partial(fetch_single_image, timeout=timeout, retries=retries)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        batch["image"] = list(executor.map(fetch_single_image_with_args, batch["image_url"]))
    return batch

def get_image(ds):
    while True:
        item = ds[np.random.randint(len(ds))]
        image = fetch_single_image(item["image_url"])
        if image is None:
            continue
        text = item["caption"]
        return (image, text)
    
    
def get_images(num_images, ds):
    images = []
    text = []
    # parallelized image fetching
    with ThreadPoolExecutor(max_workers=8) as executor:
        for i in range(num_images):
            batch = list(executor.map(get_image, [ds] * 8))
            images = [item[0] for item in batch]
            text = [item[1] for item in batch]
    return images, text


dataset = load_dataset("red_caps", split="train")
# remove 'created_utc' column
dataset = dataset.remove_columns("created_utc")
dataset = dataset.remove_columns("crosspost_parents")
dataset = dataset.remove_columns("author")
dataset = dataset.remove_columns("subreddit")
dataset = dataset.remove_columns("score")
dataset = dataset.map(fetch_images, batched=True, batch_size=100, fn_kwargs={"num_threads": 10})

images = []
texts = []
import time
st = time.time()
images, texts = get_images(8, dataset)
print("Took", time.time() - st)
images = jnp.array(images)    
print(images.shape)
