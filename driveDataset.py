import pickle
import time
from googleapiclient import errors
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
import io
import cv2
import numpy as np

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

files = list_files()
for file in files:
    print(file)
    images, labels = download_pickle(file)
    batch = zip(images, labels)
    for img in images:
        try:
            # crop image to center (min between height and width)
            height_diff = img.size[1] - img.size[0]
            width_diff = img.size[0] - img.size[1]
            height_diff = height_diff // 2
            width_diff = width_diff // 2
            width_diff = width_diff if width_diff > 0 else 0
            height_diff = height_diff if height_diff > 0 else 0
            img = img.crop((width_diff, height_diff, min(img.size[0], img.size[1]) + width_diff, min(img.size[0], img.size[1]) + height_diff))

            np_img = np.array(img.getdata()).reshape(img.size[0], img.size[1], len(img.getbands()))
            np_img = np.array(np_img, dtype=np.uint8)
            if np_img.shape[2] == 4:
                np_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2RGB)
            if np_img.shape[2] == 1:
                np_img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2RGB)
            try:
                np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
                cv2.imshow("image", np_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    img.show()
            except Exception as e:
                print(e)
                img.show()
                time.sleep(5)
        except Exception as e:
            print(e)