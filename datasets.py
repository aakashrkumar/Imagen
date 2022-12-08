import numpy as np
import tensorflow_datasets as tfds
import cv2
import sklearn

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
    DEBUG = True
    if DEBUG:
        partofds = 1024
        return images[:partofds], lables[:partofds]
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
