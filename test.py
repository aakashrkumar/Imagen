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
import asyncio
import concurrent.futures
import functools
import requests
import os


# WARNING:
# Here I'm pointing to a publicly available sample video.
# If you are planning on running this code, make sure the
# video is still available as it might change location or get deleted.
# If necessary, replace it with a URL you know is working.
URL = 'https://doc-00-98-docs.googleusercontent.com/docs/securesc/6359h3lug8caprnv3h927e8jh1kmdpgn/j9o3e8n279g33uq6drcnaac2280leh9p/1673725725000/11602798957414465033/11602798957414465033/15XBBoRKp75WVW_KQU8WWlHxW5LjgHCWT?e=download&ax=ALjR8swq9i2Eg5ED2N3P_gMD_RsmA2YaSjJfs3xThyf6I9oxvJdA39R9W0HsMIygK-MRtBxfhFhQZxWgeiDJN9gz63wlOEF7657RDojql2zFR-Doxd3T77OK5OrqQKVLZIgGaZq7uFqMq66ImA-k-75YmhKa6ud2rXrRIoOH9PG_ZxM10EiN76XHiAgdEM_CJI982XsesMSIPbPym3tW7HbFbgkPOabm28_Ygy__xDwP8S39pV8l7M-gFJWfgnEXgo9inXafzA6mphhth6czUMBR-7g-Kdqh72ugjhKZR7jLTQudO-jcT2_NPz2gUYkLvcPXYw020R121CtWODH4DynM7IjvYoEzSlYQw_3pp0uIVYnTy1-0TUdL9Quauxk820uQJTK84nSjc59L2fAcwE0TklcHb3wVZqu8hxoNs6LAWJ7CRU0cAjqRcWhLVKKNVJpLNNPKrnhuF07xjcJusgT-Qi_Z9pRxUyjVOY1e2VxW5PS9_e3yPtczg9LUi1HOj-7P6nPlbcMcTnjaBHZcRZbifPifMVlZbQltM-YjBwQDolD2fZlBj6crYyPA17a2dP9kKUq7Olr3vtEIlcu8NQFxVUpsyv1rj2HiS3s9jbAkutqisMUrbPML3wKt18UA5zUuNtgtDqwWKgRd-_I-pXZENbifshwVhWtJdN5CaiqcdaKwAaTpLRoh0jWRkwXCZDtWPibSy6AcyHLaNEEl48nz4RqDQ7TidfqdTxdjUsawW0U7fxmQd_jSspXJKqpId0bD59z8rmc7OsqHtLhADtJSTXosvTrN5vR1oJ0rHcILIynir-adbicKuLCJ_qpPhhY0o0qODNo7-CpBbfc_fRicJhx0BNzAWYDz9R_n_lduQaBIAWK9WW31visix3O_m_DNCEpyFEuB21l6Qwokco4cBIg15yXeNPfOSTtbp28XYrg9ZBUxEmmiom8UoZLZ4HWRGGUnKybglhvVBe3uZq1vLxbw2mLpIrylQR5570dfH9ldJw&uuid=ef096343-1ce4-4a54-8db6-d12653996bc8&authuser=0&nonce=k17k0oeirjciu&user=11602798957414465033&hash=g7el9s7rlej5m35kra5f58mrneoludgl'
OUTPUT = 'uploader1_laion_art_526.pkl'


async def get_size(url):
    response = requests.head(url)
    size = int(response.headers['Content-Length'])
    return size


def download_range(url, start, end, output):
    headers = {'Range': f'bytes={start}-{end}'}
    response = requests.get(url, headers=headers)

    with open(output, 'wb') as f:
        for part in response.iter_content(1024):
            f.write(part)


async def download(run, loop, url, output, chunk_size=100000):
    file_size = await get_size(url)
    chunks = range(0, file_size, chunk_size)

    tasks = [
        run(
            download_range,
            url,
            start,
            start + chunk_size - 1,
            f'{output}.part{i}',
        )
        for i, start in enumerate(chunks)
    ]

    await asyncio.wait(tasks)

    with open(output, 'wb') as o:
        for i in range(len(chunks)):
            chunk_path = f'{output}.part{i}'

            with open(chunk_path, 'rb') as s:
                o.write(s.read())

            os.remove(chunk_path)


if __name__ == '__main__':
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
    loop = asyncio.new_event_loop()
    run = functools.partial(loop.run_in_executor, executor)

    asyncio.set_event_loop(loop)

    try:
        st = time.time()
        loop.run_until_complete(
            download(run, loop, URL, OUTPUT)
        )
        print(f'Downloaded in {time.time() - st:.2f} seconds')
    finally:
        loop.close()
