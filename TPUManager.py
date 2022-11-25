import ray_tpu
import ray


@ray.remote(resources={"host": 1}, num_cpus=10)
class TPUManager:
    def __init__(self, num_tpus, address):
        self.num_tpus = num_tpus
        self.tpus = ["ray-tpu-" + str(i) for i in range(1, num_tpus + 1)]
        self.conns = []
        self.address = address

    def setup(self):
        for tpu in self.tpus:
            ray_tpu.create_tpu(tpu, "us-central1-f", "v2-8", True)
        for tpu in self.tpus:
            ray_tpu.wait_til(tpu, 'us-central1-f', {"state": "READY"})
        for tpu in self.tpus:
            conns += ray_tpu.get_connection(tpu)

        for conn in self.conns:
            ray_tpu.start_ray(conn, self.address)
        return True
