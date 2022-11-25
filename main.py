import time
import wandb
import ray
from Trainer import Trainer
import TPUManager

ray.init(address="auto")


def main():
    #tpu_manager = TPUManager.TPUManager(8, "globaltpu2.aakashserver.org:6379")
    # tpu_manager.clear()
    # tpu_manager.setup()
    trainer = Trainer.remote()
    ray.get(trainer.train.remote())
    
    
if __name__ == "__main__":
    main()