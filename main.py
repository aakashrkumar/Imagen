import wandb
import ray

ray.init(address="auto")
wandb.init(project="imagen", entity="apcsc")

wandb.config.batch_size = 64
wandb.config.num_datacollectors = 300
wandb.config.seed = 0
wandb.config.learning_rate = 1e-4
wandb.config.image_size = 64
wandb.config.save_every = 100
wandb.config.eval_every = 3

