import random
import torch
from imagen_pytorch import Unet, Imagen
import numpy as np

from datasets import load_dataset
import cv2
import tensorflow_datasets as tfds
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_datasets():
    """Load MNIST train and test datasets into memory."""
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(
        ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_ds['image'] = np.array(train_ds['image']) / 127.5 - 1.
    test_ds['image'] = np.array(test_ds['image']) / 127.5 - 1.
    return train_ds, test_ds

dataset, _ = get_datasets()
# unet for imagen

unet1 = Unet(
    dim = 32,
    cond_dim = 512,
    dim_mults = (1, 2, 4, 8),
    num_resnet_blocks = 3,
    layer_attns = (False, True, True, True),
    layer_cross_attns = (False, True, True, True)
)
# imagen, which contains the unets above (base unet and super resoluting ones)

imagen = Imagen(
    unets = (unet1),
    image_sizes = (64),
    timesteps = 1000,
    cond_drop_prob = 0.1
).cuda()

# mock images (get a lot of this) and text encodings from large T5



# feed images into imagen, training each unet in the cascade
for i in (1, 20000):
    images = []
    for j in range(64):
        image = dataset["image"][random.randint(0, len(dataset["image"]))]
        image = np.array(image * 127.5 + 127.5, dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = cv2.resize(image, (64, 64)) 
        image = np.array(image, dtype=np.float32) /127.5 - 1.0
        image = np.transpose(image, (2, 0, 1))
        images.append(image)
        # print(image.shape)
        # print(image)
    text_embeds = torch.randn(64, 256, 768).cuda()

    images = torch.tensor(images).cuda()
    loss = imagen(images, text_embeds = text_embeds, unet_number = 1)
    loss.backward()
    print(loss)