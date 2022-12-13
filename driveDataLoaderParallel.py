import logging
import time
import ray
import pyarrow.parquet as pq
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
import pickle
import pyarrow.parquet as pq
import urllib.request
from PIL import Image
from helpers import StableDiffusionSafetyChecker, numpy_to_pil
from transformers import CLIPFeatureExtractor, AutoFeatureExtractor
import numpy as np
import os
from datasets.utils.file_utils import get_datasets_user_agent
import PIL.Image
import io
import datetime

USER_AGENT = get_datasets_user_agent()
USE_SAFETY_CHECKER = False


SCOPES = ['https://www.googleapis.com/auth/drive']

def login():
    """Shows basic usage of the Drive v3 API.
    Prints the names and ids of the first 10 files the user has access to.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds


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
    query = "name='APCSC Testing'"
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
    pickle.dump(data, open(pickle_file_name, "wb"))

    try:
        # Upload the pickle file to Google Drive
        file_metadata = {
            "name": pickle_file_name,
            "parents": [shared_drive_id]
            }
        media = MediaFileUpload(pickle_file_name, resumable=True, mimetype='unknown/pkl')
        file = service.files().create(body=file_metadata, media_body=media, fields='id', supportsAllDrives=True).execute()
        print(f"File uploaded: {file.get('id')}")

    except HttpError as error:
        print(f"An error occurred while uploading the file: {error}")
        # Handle the error


def fetch_single_image(image_url, timeout=None, retries=0):
    for _ in range(retries + 1):
        try:
            request = urllib.request.Request(
                image_url,
                data=None,
                headers={"user-agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=timeout) as req:
                image = PIL.Image.open(io.BytesIO(req.read()))
            break
        except Exception:
            image = None
    return image

@ray.remote
class DatasetFetcher:
    def __init__(self, parquet_file):
        self.dataset =  pq.read_table(parquet_file)
        self.index = 0

    def get_data(self):
        url = self.dataset["URL"][self.index]
        text = self.dataset["TEXT"][self.index]
        self.index += 1
        if self.index % 1000 == 0:
            print(f"Index: {self.index}")
        return url, text

@ray.remote
class SharedStorage:
    def __init__(self):
        self.images = []
        self.texts = []


    def get_num_images(self):
        return len(self.images)
    
    def add_data(self, images, texts):
        self.images.extend(images)
        self.texts.extend(texts)


    def get_batch(self, batch_size):
        if len(self.images) < batch_size:
            return None
        images = self.images[:batch_size]
        texts = self.texts[:batch_size]
        self.images = self.images[batch_size:]
        self.texts = self.texts[batch_size:]
        return images, texts

@ray.remote
class DataCollector:
    def __init__(self, dataset, shared_storage):
        self.dataset = dataset
        self.shared_storage = shared_storage
        if USE_SAFETY_CHECKER:
            safety_model_id = "CompVis/stable-diffusion-safety-checker"
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

        
    def start(self):
        while True:
            url, text = ray.get(self.dataset.get_data.remote())
            image = fetch_single_image(url)
            if image is None:
                continue
            w, h= image.size
            MIN_IMAGE_SIZE = 256
            if h < MIN_IMAGE_SIZE or w < MIN_IMAGE_SIZE:
                continue
            if USE_SAFETY_CHECKER:
                safety_cheker_input = self.feature_extractor(image, return_tensors="pt")
                image, has_unsafe_concept = self.safety_checker(images=[image], clip_input=safety_cheker_input.pixel_values)
                if has_unsafe_concept[0]:
                    continue
            self.shared_storage.add_data.remote([image], [text])
            while self.shared_storage.get_num_images.remote() > 10_000:
                time.sleep(5)
            
@ray.remote
class DataManager:
    def __init__(self, num_workers, batch_size):
        self.shared_storage = SharedStorage.remote()
        self.batch_size = batch_size
        self.datasetFetcher = DatasetFetcher.remote("laion/part-00000-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet")
        self.workers = [DataCollector.remote(self.datasetFetcher, self.shared_storage) for _ in range(num_workers)]
        
        
    def start(self):
        for worker in self.workers:
            worker.start.remote()

    def get_batch(self):
        data = None
        while data is None:
            data = ray.get(
                self.shared_storage.get_batch.remote(self.batch_size))
        return data
    def get_shared_storage(self):
        return self.shared_storage

@ray.remote
class Uploader:
    def __init__(self):
        self.index = 0
        self.creds = None
        self.save_name = "laion"
        
    def save(self, data):
        print(f"Saving {len(data)} images")
        upload_pickle_to_google_drive(data, f"pkls/{self.save_name}_{self.index}.pkl", self.creds)
        print(f"Saved {len(data)} images")
        self.index += 1
        return 1

def main():
    dm = DataManager.remote(10, 32)
    dm.start.remote()
    uploader = Uploader.remote()
    shared_storage = ray.get(dm.get_shared_storage.remote())
    BATCH_SIZE = 1000
    start_time = time.time()
    num_uploaded = 0
    while True:
        time.sleep(5)
        batch = ray.get(shared_storage.get_batch.remote(BATCH_SIZE))
        if batch:
            num_uploaded += BATCH_SIZE
            ray.get(uploader.save.remote(batch))
        num_images = ray.get(shared_storage.get_num_images.remote())
        images_per_second = (num_uploaded + num_images) / (time.time() - start_time)
        time_till_next_upload = (BATCH_SIZE - num_images) / (images_per_second + 1e-6)
        print(f"Images per second: {images_per_second:.2f}, uploaded: {num_uploaded}, Images: {num_images}, time till next upload: {datetime.timedelta(seconds=round(time_till_next_upload))}")
            
        
if __name__ == "__main__":
    main()