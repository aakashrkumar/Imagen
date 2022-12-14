import time
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request
from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload
from google_auth_oauthlib.flow import InstalledAppFlow
import pickle
import pyarrow.parquet as pq
import urllib.request
from PIL import Image
from helpers import StableDiffusionSafetyChecker, numpy_to_pil
from transformers import CLIPFeatureExtractor, AutoFeatureExtractor
import numpy as np
import os
import io

import multiprocessing as mp


safety_model_id = "CompVis/stable-diffusion-safety-checker"
feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

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
    if not upload_data_without_file:
        pickle.dump(data, open(pickle_file_name, "wb"))

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

def get_data(data):
    url, text = data
    try:
        url = url
        # print(f"url:{url}, ", end='')
        urllib.request.urlretrieve(
            str(url),
            "temp.png"
        )
        img = Image.open("temp.png")
        w, h = img.size
        
        if not (w >= 255 and h >= 255 and w == h):
            # print("img too small, continuing...")
            return None
        safety_cheker_input = feature_extractor(img, return_tensors="pt")
        image, has_unsafe_concept = safety_checker(images=img, clip_input=safety_cheker_input.pixel_values)

        # print(f"img ({w}, {h}), sf: {has_unsafe_concept[0]}, ", end='')
        added = False

        if not has_unsafe_concept[0]:
            sample = {"image":image, "text":text}
            num_img += 1
            added = True
            
            print(f'added: {added}, num: {num_img}')
            return sample
    except:
        # print("failed to open url, continuing...")
        return None

    
def threaded_get_data(urls, texts):
    with mp.Pool(10) as p:
        data = p.map(get_data, zip(urls, texts))
        data = [d for d in data if d is not None]


def process_threaded(parquet_file, save_name, creds, chunk_size = 10000):
    log = open(f"{save_name}-runlog.txt", 'w')
    table = pq.read_table(parquet_file)
    
    data = []
    index = 0
    num_img = 0
    images = []
    texts = []
    
    while index < len(table["URL"]):
        texts.append(table["TEXT"][index])
        images.append(table["URL"][index])
        if len(images) >= 100:
            threaded_get_data(images, texts)
            images = []
            texts = []

def process_parquet(parquet_file, save_name, creds, chunk_size = 10000):
    log = open(f"{save_name}-runlog.txt", 'w')
    table = pq.read_table(parquet_file)
    print(len(table["URL"]))
    # print(table["URL"])
    data = []
    index = 0
    num_img = 0
    for i in range(len(table["URL"])):
        try:
            url = table["URL"][i]
            # print(f"url:{url}, ", end='')
            urllib.request.urlretrieve(
                str(url),
                "temp.png"
            )
            img = Image.open("temp.png")
            w, h = img.size
            
            if not (w >= 255 and h >= 255 and w == h):
                # print("img too small, continuing...")
                continue
            safety_cheker_input = feature_extractor(img, return_tensors="pt")
            print(img)
            image, has_unsafe_concept = safety_checker(images=img, clip_input=safety_cheker_input.pixel_values)

            # print(f"img ({w}, {h}), sf: {has_unsafe_concept[0]}, ", end='')
            added = False

            if not has_unsafe_concept[0]:
                sample = {"image":image, "text":table["TEXT"][i]}
                data.append(sample)
                num_img += 1
                added = True
                
            print(f'added: {added}, num: {num_img}')
            log.write(f"url:{url}, img ({w}, {h}), sf{has_unsafe_concept[0]}, added:{added}")
        
            if num_img >= chunk_size:
                print("saving...")
                upload_pickle_to_google_drive(data, f"{save_name}{index}.pkl", creds)
                index += 1
                data = []
                num_img = 0
        except Exception as e:
            # print(e)
            # print("failed to open url, continuing...")
            pass
    im = Image.fromarray(np.zeros((256, 256, 3)))
    im.save("temp.png")
    log.close()


def main():
    creds = login()
    directory = "./laion" # replace with directory containing parquet files

    # use os.listdir() to get a list of all the files in the directory
    files = os.listdir(directory)
    print(files)

    # iterate through the list of files and print their names
    for parquet_file in files:
        # if os.path.isfile(directory + parquet_file):
        file_name, file_type = os.path.splitext(parquet_file)
        print(file_name)
        process_parquet(directory + '/' + parquet_file, file_name, creds)


if __name__ == "__main__":
    main()