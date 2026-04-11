import os
import requests
from import_extracted_data import import_clip

dataset_labels_path = "dataset/youtube_boundingboxes_detection_train.csv"
dataset_download_url = "https://research.google.com/youtube-bb/yt_bb_detection_train.csv.gz"

def import_data():
    download_dataset()


def download_dataset():
    if not os.path.exists(dataset_labels_path):
        response = requests.get(dataset_download_url, stream=True)
        if response.status_code == 200:
            pass
            



if __name__ == "__main__":
    import_data()