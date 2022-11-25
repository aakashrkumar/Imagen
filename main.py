import time
import wandb
import ray
from Trainer import Trainer
import TPUManager

ray.init(address="auto")
wandb.init(project="imagen", entity="apcsc")

wandb.config.batch_size = 64
wandb.config.num_datacollectors = 300
wandb.config.seed = 0
wandb.config.learning_rate = 1e-4
wandb.config.image_size = 64
wandb.config.save_every = 100
wandb.config.eval_every = 3

def main():
    tpu_manager = TPUManager.TPUManager(8, "globaltpu2.aakashserver.org:6379")
    # tpu_manager.clear()
    tpu_manager.setup()
    trainer = Trainer.remote()
    ray.get(trainer.train.remote())
    
    
if __name__ == "__main__":
    main()