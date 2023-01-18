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
from googleapiclient.errors import HttpError

os.environ["TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD"] = "2000000000000"

# ray.init(logging_level=logging.ERROR, log_to_driver=False)

creds = Credentials.from_authorized_user_file(
            'token.json',
            scopes=["https://www.googleapis.com/auth/drive"]
        )

# 3. Use the `drive_service.files().list()` method to retrieve a list of files in the specified folder


def list_files(folder_id="1QwRiS6rIfWPrzrVBU9x1Y8S6kHXKZnvN"):
    drive_service = build('drive', 'v3', credentials=creds)
    files = []
    try:
        page_token = None
        while True:
            query = f"'{folder_id}' in parents"
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
class DatasetFetcher:
    def __init__(self):
        self.files = list_files()
        self.index = 0
        self.uploaded_ids = []
        for file in list_files("1fUYXHjDJRhBaDJM3TdxIhGh4NZHi4qM0"):
            if "12" in file.get('name'):
                print(file.get('name'))
            self.uploaded_ids.append(file.get('name'))
        self.sent_ids = []
    def get_data(self):
        self.index += 1
        if self.index % 100 == 0:
            self.files = list_files()
        while self.files[self.index % len(self.files)].get('name') in self.uploaded_ids or self.files[self.index % len(self.files)].get('name') in self.sent_ids:
            self.index += 1
        self.sent_ids.append(self.files[self.index % len(self.files)].get('name'))
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
    img = img.resize((256, 256))
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
        print(f"File uploaded: {file.get('name')} with ID: {file.get('id')}")

    except HttpError as error:
        print(f"An error occurred while uploading the file: {error}")
        # Handle the error

@ray.remote
class DataCollector:
    def __init__(self, dataset):
        self.dataset = dataset
    def start(self):
        while True:
            file = ray.get(self.dataset.get_data.remote())
            data = download_pickle(file)
            images = data[0] # list of pil images
            texts = data[1]
            # convert pil images to numpy arrays
            images = [processImage(image) for image in images]
            data = (images, texts)
            upload_pickle_to_google_drive(data, f"{file.get('name')}")
            
            

@ray.remote
class DataManager:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.dataset = DatasetFetcher.remote()
        collectors = [DataCollector.remote(self.dataset) for _ in range(30)]
        print("Initialized")
        for collector in collectors:
            collector.start.remote()
    
def test():
    datamanager = DataManager.remote(1024)
    total_processed = 0
    while True:
        time.sleep(1)
if __name__ == "__main__":
    test()
