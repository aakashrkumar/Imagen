from datasets import load_dataset
import cv2
dataset = load_dataset("laion/laion-art")["train"]
print(dataset)
