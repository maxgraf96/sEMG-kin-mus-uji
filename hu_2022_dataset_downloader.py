import os
import requests
from tqdm import tqdm

dataset_root = '/import/c4dm-datasets-ext/hu-22-sEMG-myo'

if __name__ == '__main__':
    # Get json response from api url https://api.figshare.com/v2/collections/5670433/articles?page_size=100
    response = requests.get('https://api.figshare.com/v2/collections/5670433/articles?page_size=100')
    # Convert json response to python dictionary
    response_dict = response.json()
    # Get all "id" fields in the response as a list - these correspond to the figshare articles of each individual subject
    ids = [item['id'] for item in response_dict]

    # For each article id, create a folder in "/imports/c4dm-datasets-ext/" get the files associated with it, and download them
    for id in ids:
        response = requests.get(f'https://api.figshare.com/v2/articles/{id}')
        response_dict = response.json()

        # Get subject name from response
        subject_name = response_dict['title']
        files = response_dict['files']

        print("-" * 30)
        print("Processing subject " + subject_name + "...")
        print("-" * 30)

        # Create folder for each subject
        subject_folder = f'{dataset_root}/{subject_name}'
        # If folder already exists, skip
        if os.path.exists(subject_folder):
            print(f'Skipping {subject_name}...')
            continue
        # Otherwise, create the folder
        os.mkdir(subject_folder)

        # For each file in the subject's article, download it to the folder
        for file in tqdm(files):
            file_name = file['name']
            file_url = file['download_url']
            file_path = f'{subject_folder}/{file_name}'

            print(f'Downloading {file_name}...')
            r = requests.get(file_url, allow_redirects=True)
            open(file_path, 'wb').write(r.content)