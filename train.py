import jax
from flax.training import checkpoints
from tqdm import tqdm
import optax
import jax.numpy as jnp
from imagen_main import Imagen, sample, train_step
import wandb


wandb.init(project="imagen")


class config:
    batch_size = 8
    seed = 0
    learning_rate = 1e-4
    image_size = 64
    save_every = 1000
    eval_every = 1
    steps = 100_000


def train(imagen, steps):
    for step in tqdm(range(1, steps + 1)):
        images = jax.random.normal(
            imagen.get_key(), (config.batch_size, config.image_size, config.image_size, 3))
        timestep = jnp.ones(config.batch_size) * \
            jax.random.randint(imagen.get_key(), (1,), 0, 999)
        timestep = jnp.array(timestep, dtype=jnp.int16)
        #imagen.state, metrics = train_step(
         #   imagen.state, images, None, timestep, imagen.get_key()) # TODO: Add text(None)
        if step % config.eval_every == 0:
            imgs = sample(imagen.state, imagen.lowres_scheduler,
                          images.shape, None, imagen.get_key()) # TODO: Add text(None)
            # log as 16 gifs
            gifs = []
            for i in range(16):
                gifs.append(wandb.Video(imgs[i], fps=60, format="gif"))
            wandb.log({"samples": gifs})
        # wandb.log(metrics)


def main():
    imagen = Imagen()
    train(imagen, config.steps)


main()
