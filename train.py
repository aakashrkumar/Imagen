import jax
from flax.training import train_state
from tqdm import tqdm
import optax
import jax.numpy as jnp
from imagen import EfficentUNet

class config:
    batch_size = 16
    seed = 0
    learning_rate = 1e-4
    image_size = 256

def init_train_state(
        model, random_key, shape, learning_rate) -> train_state.TrainState:
    # Initialize the Model
    variables = model.init(random_key, jnp.ones(shape))
    # Create the optimizer
    optimizer = optax.adam(learning_rate)
    # Create a State
    return train_state.TrainState.create(
        apply_fn=model.apply,
        tx=optimizer,
        params=variables['params']
    )
    
def compute_metrics(logits, labels):
    loss = jnp.mean((logits - labels) ** 2)
    metrics = {
        'loss': loss,
    }
    return metrics

def train_step(state: train_state.TrainState, images: jnp.ndarray):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, images)
        loss = cross_entropy_loss(logits=logits, labels=images)
        return loss, logits


    gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = gradient_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=logits, labels=images)
    return state, metrics


def train(state, steps):
    for i in tqdm(range(1, steps + 1)):
        batch = jnp.ones((config.batch_size, config.image_size, config.image_size, 3))
        state, metrics = train_step(state, batch)

def main():
    model = EfficentUNet()
    
    rng = jax.random.PRNGKey(config.seed)
    state = init_train_state(
        model, 
        rng,
        (config.batch_size, config.image_size, config.image_size, 3),
        config.learning_rate
    )

main()