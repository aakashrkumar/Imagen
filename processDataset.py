import gc
import pickle
import random
import time
from googleapiclient import errors
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
import io
import cv2
import numpy as np
import psutil
import ray
import urllib3
import T5Utils
import logging
import os
from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload

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
    with open("test.plk", "rb") as f:
        data = pickle.load(f)
        f.close()
    time.sleep(random.randint(60, 100))
    return data
    drive_service = build('drive', 'v3', credentials=creds)
    request = drive_service.files().get_media(fileId=file.get('id')).execute()
    data = pickle.load(io.BytesIO(request))
    return data

def auto_garbage_collect(pct=30.0):
    """
    auto_garbage_collection - Call the garbage collection if memory used is greater than 80% of total available memory.
                              This is called to deal with an issue in Ray not freeing up used memory.

        pct - Default value of 80%.  Amount of memory in use that triggers the garbage collection call.
    """
    if psutil.virtual_memory().percent >= pct:
        gc.collect()
    return

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


def upload_pickle_to_google_drive(data, pickle_file_name, creds=None, upload_data_without_file=False):
    # Set the credentials object with the required scope for accessing Google Drive
    if creds is None:
        creds = Credentials.from_authorized_user_file(
            'token.json',
            scopes=["https://www.googleapis.com/auth/drive"]
        )

    # Build the service for interacting with Google Drive
    service = build("drive", "v3", credentials=creds)


    # Obtain the ID of the shared drive
    # Replace "SHARED_DRIVE_NAME" with the name of the shared drive
    query = "name='LAION-ART-PROCESSED'"
    results = service.files().list(
        q=query, 
        fields="nextPageToken, files(id, name)", 
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
        corpora="allDrives").execute()
    print(results)

    # results2 = service.files().list().execute()
    # print(results2)
    shared_drive_id = results["files"][0]["id"]


    # Dump the data you want to upload to a pickle file
    if not upload_data_without_file:
        with open(pickle_file_name, "wb") as f:
            pickle.dump(data,f)
            f.close()

    try:
        # Upload the pickle file to Google Drive
        file_metadata = {
            "name": pickle_file_name,
            "parents": [shared_drive_id]
            }
        media = None
        if not upload_data_without_file:
            media = MediaFileUpload(pickle_file_name, resumable=True, mimetype='unknown/pkl')
        else: 
            media = MediaIoBaseUpload(io.BytesIO(data), mimetype='unknown/pkl', resumable=True)
        file = service.files().create(body=file_metadata, media_body=media, fields='id', supportsAllDrives=True).execute()
        print(f"File uploaded: {file.get('id')}")

    except HttpError as error:
        print(f"An error occurred while uploading the file: {error}")
        # Handle the error
@ray.remote
class Uploader:
    def __init__(self):
        self.index = 441
        self.creds = None
        self.save_name = "laion_art"
        with open("uploader.txt", "r") as f:        
            self.run_name = f.read()
            print("Uploading as", self.run_name)
    def save(self, data):
        print(f"Saving {len(data[0])} images")
        upload_pickle_to_google_drive(data, f"{self.run_name}_{self.save_name}_{self.index}.pkl", self.creds)
        print(f"Saved {len(data[0])} images")
        self.index += 1
        return 1

@ray.remote
class DataCollector:
    def __init__(self, dataset, shared_storage):
        self.dataset = dataset        
    def start(self):
        while True:
            file = ray.get(self.dataset.get_data.remote())
            data = download_pickle(file)
            images = data[0] # list of pil images
            texts = data[1]
            # convert pil images to numpy arrays
            images = [processImage(image) for image in images]
            images = np.array(images)
            

@ray.remote
class DataManager:
    def __init__(self, batch_size):
        self.shared_storage_encoded = SharedStorageEncoded.remote()
        self.batch_size = batch_size
        self.dataset = DatasetFetcher.remote()
        print("Initialized")
    
    def get_num_images(self):
        return ray.get(self.shared_storage_encoded.get_size.remote())
    
    def get_batch(self):
        auto_garbage_collect(pct=30)
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
        auto_garbage_collect(pct=30)
if __name__ == "__main__":
    test()
