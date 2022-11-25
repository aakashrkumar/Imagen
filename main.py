import ray_tpu
import ray

ray.init(address="auto")


for i in range(10):
    ray_tpu.create_tpu(f"ray-tpu-{i}", "us-central1-f", "v2-8", True)
    