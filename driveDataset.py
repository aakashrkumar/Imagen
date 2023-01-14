import pickle
import random
import time
from googleapiclient import errors
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
import io
import cv2
import numpy as np
import ray
import T5Utils
import logging
import os
os.environ["TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD"] = "2000000000000"

# ray.init(logging_level=logging.ERROR, log_to_driver=False)

creds = Credentials.from_authorized_user_file(
            'token.json',
            scopes=["https://www.googleapis.com/auth/drive"]
        )

# 3. Use the `drive_service.files().list()` method to retrieve a list of files in the specified folder


def list_files():
    drive_service = build('drive', 'v3', credentials=creds)
    files = []
    try:
        page_token = None
        while True:
            query = "'1QwRiS6rIfWPrzrVBU9x1Y8S6kHXKZnvN' in parents"
            response = drive_service.files().list(q=f"{query}",
                                                  fields='nextPageToken, '
                                                  'files(id, name, mimeType)',
                                                  includeItemsFromAllDrives=True,
                                                  supportsAllDrives=True, 
                                                  corpora="allDrives",
                                                  pageToken=page_token).execute()
            for file in response.get('files', []):
                # Process the file
                files.append(file)
                # print(F'Found file: {file.get("name")} ({file.get("id")})')
                
            page_token = response.get('nextPageToken', None)
            if page_token is None:
                break
    except errors.HttpError as error:
        print(F'An error occurred: {error}')
        files = None
    return files

def download_pickle(file):
    drive_service = build('drive', 'v3', credentials=creds)
    request = drive_service.files().get_media(fileId=file.get('id')).execute()
    data = pickle.load(io.BytesIO(request))
    return data
@ray.remote
class SharedStorageEncoded:
    def __init__(self):
        self.images = []
        self.texts = []
        self.texts_encoded = []
        self.attention_masks = []

    def get_size(self):
        return len(self.images)

    def add_data(self, images, texts, texts_encoded, attention_masks):
        self.images.extend(images)
        self.texts.extend(texts)
        self.texts_encoded.extend(texts_encoded)
        self.attention_masks.extend(attention_masks)

    def get_batch(self, batch_size):
        if len(self.images) < batch_size:
            return None
        images = self.images[:batch_size]
        self.images = self.images[batch_size:]
        texts = self.texts[:batch_size]
        texts_encoded = self.texts_encoded[:batch_size]
        attention_masks = self.attention_masks[:batch_size]
        images = np.array(images)
        texts_encoded = np.array(texts_encoded)
        attention_masks = np.array(attention_masks)
        return images, texts, texts_encoded, attention_masks

@ray.remote
class DatasetFetcher:
    def __init__(self):
        self.files = list_files()
        self.index = 0
    def get_data(self):
        self.index += 1
        if self.index % 100 == 0:
            self.files = list_files()
        return self.files[self.index % len(self.files)]

def processImage(img):
    # image is a pillow image
    height_diff = img.size[1] - img.size[0]
    width_diff = img.size[0] - img.size[1]
    height_diff = height_diff // 2
    width_diff = width_diff // 2
    width_diff = width_diff if width_diff > 0 else 0
    height_diff = height_diff if height_diff > 0 else 0
    img = img.crop((width_diff, height_diff, min(img.size[0], img.size[1]) + width_diff, min(img.size[0], img.size[1]) + height_diff))
    np_img = np.array(img, dtype=np.uint8)
    if len(np_img.shape) == 2:
        np_img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2RGB)
    if np_img.shape[2] == 4:
        np_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2RGB)
    if np_img.shape[2] == 1:
        np_img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2RGB)
    return np_img

@ray.remote
def collect(dataset:DatasetFetcher):
    file = ray.get(dataset.get_data.remote())
    data = download_pickle(file)
    images = data[0] # list of pil images
    texts = data[1]
    # convert pil images to numpy arrays
    images = [processImage(image) for image in images]
    images = ray.put(images)
    texts = ray.put(texts)
    return images, texts
    

@ray.remote
class Encoder:
    def __init__(self, shared_storage_encoded, dataset:DatasetFetcher):
        self.shared_storage_encoded = shared_storage_encoded
        self.dataset = dataset
        self.tokenizer, self.model = T5Utils.get_tokenizer_and_model()
        self.batch_size = 1000
            
    def process(self, data):
        images, texts = data
        input_ids, attention_mask = T5Utils.tokenize_texts(texts, self.tokenizer)
        input_ids = np.array(input_ids).reshape(8, -1, 512)
        attention_mask = np.array(attention_mask).reshape(8, -1, 512)
        texts_encoded, attention_mask = T5Utils.encode_texts(input_ids, attention_mask, self.model)
        texts_encoded = np.array(texts_encoded)
        texts_encoded = texts_encoded.reshape(-1, 512, 1024)
        attention_mask = np.array(attention_mask)
        attention_mask.reshape(-1, 512)
        texts = [ray.put(text) for text in texts]
        texts_encoded = [ray.put(text_encoded) for text_encoded in texts_encoded]
        attention_mask = [ray.put(mask) for mask in attention_mask]
        self.shared_storage_encoded.add_data.remote(images, texts, texts_encoded, attention_mask)
        
    
    def encode(self):
        while True:
            if ray.get(self.shared_storage_encoded.get_size.remote()) > 10000:
                time.sleep(1)
                continue
            batches = [collect.remote(self.dataset) for _ in range(32)]
            for batch in batches:
                self.process(ray.get(batch))
            

@ray.remote
class DataManager:
    def __init__(self, batch_size):
        self.shared_storage_encoded = SharedStorageEncoded.remote()
        self.batch_size = batch_size
        self.dataset = DatasetFetcher.remote()
        self.processor = Encoder.remote(self.shared_storage_encoded, self.dataset)
        self.processor.encode.remote()
        print("Initialized")
    
    def get_num_images(self):
        return ray.get(self.shared_storage_encoded.get_size.remote())
    
    def get_batch(self):
        return self.shared_storage_encoded.get_batch.remote(self.batch_size)

def test():
    datamanager = DataManager.remote(1024)
    total_processed = 0
    while True:
        time.sleep(1)
        print("Current Storage", ray.get(datamanager.get_num_images.remote()))
        batch = ray.get(ray.get(datamanager.get_batch.remote()))
        if batch is None:
            continue
        else:
            images, texts, texts_encoded, attention_mask = batch
            total_processed += len(images)
            print("Total Processed", total_processed)
    
if __name__ == "__main__":
    test()
