import time
import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds

from flax import linen as nn

import jax
import wandb
import optax

from flax.training import train_state

import logging
from tqdm import tqdm
from flax.metrics import tensorboard
from jax.experimental import pjit, PartitionSpec as P
from jax.experimental import maps


wandb.init(project="flax-mnist", entity="therealaakash")
config = wandb.config
BATCH_SIZE = 64
EPOCHS = 255
config.batch_size = BATCH_SIZE
config.epochs = EPOCHS

def get_datasets():
  """Load MNIST train and test datasets into memory."""
  ds_builder = tfds.builder('mnist')
  ds_builder.download_and_prepare()
  train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
  test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
  train_ds['image'] = jnp.float32(train_ds['image']) / 255.
  test_ds['image'] = jnp.float32(test_ds['image']) / 255.
  return train_ds, test_ds


class MnistNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


@jax.jit
def apply_model(state, images, labels):
    """Computes gradients, loss and accuracy for a single batch."""
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, images)
        one_hot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(
            logits=logits, labels=one_hot))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grads, loss, accuracy


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def create_train_state(rng):
    """Creates initial `TrainState`."""
    cnn = MnistNet()
    params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))['params']
    tx = optax.sgd(1e-4, 0.999)
    return train_state.TrainState.create(
        apply_fn=cnn.apply, params=params, tx=tx)

def train_step(state, batch):
    """Train for a single step."""
    grads, loss, accuracy = apply_model(state, batch['image'], batch['label'])
    state = update_model(state, grads)
    return state, loss, accuracy

def train_epoch(state, train_ds, batch_size, rng):
    """Train for a single epoch."""
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, len(train_ds['image']))
    perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    epoch_loss = []
    epoch_accuracy = []

    for perm in tqdm(perms):
        batch_images = train_ds['image'][perm, ...]
        batch_labels = train_ds['label'][perm, ...]
        state, loss, accuracy = train_step(state, {'image': batch_images, 'label': batch_labels})
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)
    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    return state, train_loss, train_accuracy

def train_and_evaluate() -> train_state.TrainState:
    """Execute model training and evaluation loop.
    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Directory where the tensorboard summaries are written to.
    Returns:
      The train state (which includes the `.params`).
    """
    train_ds, test_ds = get_datasets()
    rng = jax.random.PRNGKey(0)


    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng)
    for epoch in range(1, EPOCHS + 1):
        st = time.time()
        rng, input_rng = jax.random.split(rng)
        state, train_loss, train_accuracy = train_epoch(state, train_ds,
                                                        BATCH_SIZE,
                                                        input_rng)
        _, test_loss, test_accuracy = apply_model(state, test_ds['image'],
                                                  test_ds['label'])

        print('epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, test_accuracy: %.2f'
            % (epoch, train_loss, train_accuracy * 100, test_loss,
               test_accuracy * 100))
        wandb.log({"train_loss": train_loss, "train_accuracy": train_accuracy, "test_loss": test_loss, "test_accuracy": test_accuracy, "epochTime": time.time() - st})
    
    return state

def main():
    state = train_and_evaluate()
    
if __name__ == '__main__':
    main()