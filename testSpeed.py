import time
import ray
import dataCollector
import tqdm
import cv2
import numpy as np
ray.init()

collector = dataCollector.DataManager.remote(num_workers=90, batch_size=64)
collector.start.remote()
time.sleep(5)
for i in tqdm.tqdm(range(1000)):
    images, text = ray.get(collector.get_batch.remote())
    images = images * 127.5 + 127.5
    images = np.asarray(images, dtype=np.uint8)
    #cv2.imshow("image", images[0])
    # cv2.waitKey(1)