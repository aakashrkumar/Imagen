import os
import pickle
import time
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
os.environ["TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD"] = "2000000000000"

# ray.init(logging_level=logging.ERROR, log_to_driver=False)

creds = Credentials.from_authorized_user_file(
            'token.json',
            scopes=["https://www.googleapis.com/auth/drive"]
        )



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

def reqDownload():
    for file in list_files():
        st = time.time()
        data = download_pickle(file)
        print("Requst Download", time.time() - st)
        image, text = data
def mount():
    for file in os.listdir('drive'):
        if file.endswith('.pkl'):
            st = time.time()
            with open(os.path.join('drive', file), 'rb') as f:
                data = pickle.load(f)
                f.close()
            print("Drive Mounted", time.time() - st)
            image, text = data

if __name__ == "__main__":
    for file in list_files():
        st = time.time()
        data = download_pickle(file)
        print("Requst Download", time.time() - st)
        with open("test.plk", 'wb') as f:
            pickle.dump(data, f)
            f.close()
        quit()