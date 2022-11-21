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


dataset = load_dataset("red_caps", split="train")
# remove 'created_utc' column
dataset = dataset.remove_columns("created_utc")
dataset = dataset.remove_columns("crosspost_parents")
dataset = dataset.remove_columns("author")
dataset = dataset.remove_columns("subreddit")
dataset = dataset.remove_columns("score")


dl = DataLoader(dataset, batch_size=64, shuffle=True)
dl = iter(dl)
batch = next(dl)
batch = fetch_images(batch, 8)
for image, caption in zip(batch["image"], batch["caption"]):
    print(caption)
    cv2.imshow("image", image)
    cv2.waitKey(0)
cv2.destroyAllWindows()
print(batch)