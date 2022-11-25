import ray_tpu
import ray
import multiprocessing
from functools import partial

head_info = ray.init(address="auto")
address = head_info['redis_address']

for i in range(10):
    ray_tpu.create_tpu(f"ray-tpu-{i}", "us-central1-f", "v2-8", True)
for i in range(10):
    ray_tpu.wait_til(f"ray-tpu-{i}", "us-central1-f", {"state": "READY"})
conns = []
for i in range(10):
    conns += ray_tpu.get_connection(f"ray-tpu-{i}", "us-central1-f")

with multiprocessing.Pool(10) as p:
    p.map(partial(ray_tpu.start_ray, address=address), conns)