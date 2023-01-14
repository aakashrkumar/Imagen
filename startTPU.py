import os
import time


while True:
    os.system("gcloud alpha compute tpus tpu-vm create tpu-main --zone europe-west4-a  --accelerator-type='v3-8' --version='tpu-vm-tf-2.9.2' --scopes=https://www.googleapis.com/auth/cloud-platform")
    time.sleep(1)
