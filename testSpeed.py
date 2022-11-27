import time
import ray
import dataCollector
import tqdm
import cv2
import numpy as np
import TPUManager

ray.init()

tpuManager = TPUManager.TPUManager(6, "globaltpu2.aakashserver.org:6379")
# tpuManager.setup()
encoder = dataCollector.T5Encoder.remote()
collector = dataCollector.DataManager.remote(num_workers=200, batch_size=64, encoder=encoder)
collector.start.remote()
pb = tqdm.tqdm(total=1000000)
while True:
    pb.update(1)
    batch = collector.get_batch.remote()
    batch = ray.get(batch)