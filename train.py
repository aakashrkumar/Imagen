import jax
from flax.training import train_state
from tqdm import tqdm
import optax
import jax.numpy as jnp
from imagen import Imagen
import wandb

wandb.init(project="imagen")

class config:
    batch_size = 16
    seed = 0
    learning_rate = 1e-4
    image_size = 256

def init_train_state(
        model, random_key, shape, learning_rate) -> train_state.TrainState:
    # Initialize the Model
    variables = model.init(random_key, jnp.ones(shape), jnp.ones(shape[0]), random_key)
    # Create the optimizer
    optimizer = optax.adam(learning_rate)
    # Create a State
    return train_state.TrainState.create(
        apply_fn=model.apply,
        tx=optimizer,
        params=variables['params']
    )
    
def compute_metrics(loss):
    metrics = {
        'loss': loss,
    }
    return metrics

def train_step(state: train_state.TrainState, images: jnp.ndarray, timestep: int, rng):
    def loss_fn(params):
        loss, logits = state.apply_fn({'params': params}, images, timestep, rng)
        return loss, logits


    gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = gradient_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(loss)
    return state, metrics


def train(state, steps):
    rng = jax.random.PRNGKey(config.seed)
    for i in tqdm(range(1, steps + 1)):
        rng, key = jax.random.split(rng)
        images = jnp.random.normal(key, (config.batch_size, config.image_size, config.image_size, 3))
        timestep = jnp.ones(config.batch_size) * jnp.random.randint(key,0, 999)
        rng, key = jax.random.split(rng)
        state, metrics = train_step(state, images, timestep, rng)
        wandb.log(metrics)

def main():
    model = Imagen()
    
    rng = jax.random.PRNGKey(config.seed)
    state = init_train_state(
        model, 
        rng,
        (config.batch_size, config.image_size, config.image_size, 3),
        config.learning_rate
    )
    train(state, 1000)

main()