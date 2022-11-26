import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import ray_tpu
import ray
from functools import partial
import threading

head_info = ray.init(address="auto")
address = "globaltpu2.aakashserver.org:6379"

# for i in range(10):
#ray_tpu.delete_tpu(f"ray-tpu-{i}", "us-central1-f")

for i in range(10):
    ray_tpu.create_tpu(f"ray-tpu-{i}", "us-central1-f", "v2-8", True)
for i in range(10):
    ray_tpu.wait_til(f"ray-tpu-{i}", "us-central1-f", {"state": "READY"})
conns = []
for i in range(10):
    conns += ray_tpu.get_connection(f"ray-tpu-{i}", "us-central1-f")

for i in range(10):
   #  ray_tpu.start_ray(conns[i], address)
    threading.Thread(target=ray_tpu.start_ray,
                     args=(conns[i], address)).start()
