import ray_tpu
import ray

ray.init(address="auto")


for i in range(10):
    # ray_tpu.create_tpu(f"ray-tpu-{i}", "us-central1-f", "v2-8", True)
    ray_tpu.wait_til(f"ray-tpu-{i}", "us-central1-f", {"state": "READY"})
    ray_tpu.start_ray(ray_tpu.get_connection(f"ray-tpu-{i}", "us-central1-f")[0], "globaltpu2.aakashserver.org")