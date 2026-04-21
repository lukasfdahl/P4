import numpy as np
from numpy.typing import NDArray
from data_classes import Clip, Frame
import os
import requests
import shutil
import gzip
import pandas as pd

MV_STRUCT = np.dtype([
    ('source', np.int32),
    ('motion_x', np.int32),
    ('motion_y', np.int32),
    ('motion_scale', np.int32)
])

dataset_labels_path = "dataset/youtube_boundingboxes_detection_train.csv"
dataset_download_url = "https://research.google.com/youtube-bb/yt_bb_detection_train.csv.gz"
column_names = ["youtube_id",
                "timestamp_ms",
                "class_id",
                "class_name",
                "object_id",
                "object_presence",
                "xmin",
                "xmax",
                "ymin",
                "ymax"]

fps = 30 # google stated in the dataset docs, that they decoded all the videos at 30fps regardless of framerate, so thay assume 30 frames in any video equals a second, weather it is true ot not. therefor this importor does the same to account for it, and keep tracking accurate

labels : pd.DataFrame

def download_dataset():
    print("Checking if dataset is downloaded")
    if not os.path.exists(dataset_labels_path):
        print("Dataset labels not found, downloading dataset, this might take a minute depending on internet speed...")
        response = requests.get(dataset_download_url, stream=True)
        if response.status_code == 200: # if we sucessfully start the download
            
            with open(dataset_labels_path + ".gz", "wb") as compressed_csv: # the wb means write binary, we create a file at the location, and write the binary that google streams us to the file, untill it is fully downloaded
                shutil.copyfileobj(response.raw, compressed_csv) # shutil handles looping and copying data from the stream untill it finishes sending
            print("Finished downloading dataset labels, now unzipping dataset...")
            
            with gzip.open(dataset_labels_path + ".gz", "rb") as zip_file:
                with open(dataset_labels_path, "wb") as csv_file:
                    shutil.copyfileobj(zip_file, csv_file)
            os.remove(dataset_labels_path + ".gz")
            print("finished downloading dataset")
    else:
        print("Dataset labels found")

def import_dataset():
    print("importing dataset...")
    return pd.read_csv(dataset_labels_path, header=None, names=column_names)

# Function to load the .npz file into usable python classes and numpy arrays
def import_clip(clip_path : str) -> Clip:
    # extracted data
    data = np.load(clip_path)
    raw_motion_vectors : np.ndarray = data["motion_vectors"]
    frame_types : NDArray[np.str_] = data["frame_types"]
    residuals : np.ndarray = data["residuals"] # i am pretty sure it uses YUV color format from looking though the code for the cv_reader
    motion_vectors = raw_motion_vectors.view(MV_STRUCT)
    
    # labels
    video_id = os.path.basename(clip_path).replace(".npz", "") # since the files have the video id in the name we just get the name and strip out the .npz to get the id
    video_labels = labels[labels["youtube_id"] == video_id]
    
    # createing the frame and clip objects
    frames = []
    for index in range(len(frame_types)): # loop though each frame
        rounded_ms = (index // fps) * 1000 # get the millisecond time of the current frame at 30fps rounded down to nearest 1000ms
        frame_labels = video_labels[video_labels["timestamp_ms"] == rounded_ms]
        bbox = (-1, -1, -1, -1) # defualt value in case no bounding box/class is found (same goes for variables bellow)
        true_class = -1
        has_object = False
        
        if not frame_labels.empty: # if labels where found at the nearest timestamp to the current frame (then use the first listed label)
            # grabs the first label (in case there are multiple)
            has_object = True
            row = frame_labels.iloc[0]
            bbox = (row["xmin"], row["xmax"], row["ymin"], row["ymax"])
            true_class = int(row["class_id"])
            
        frame = Frame(motion_vectors[index], frame_types[index], residuals[index], bbox, true_class, has_object)
        frames.append(frame)
    return Clip(frames)



download_dataset() # just to ensure the dataset is downloaded no matter what.
labels = import_dataset()


# some test code i made
if __name__ == "__main__":
    test_clip = import_clip("downloaded_videos/ASBfcxcC1hc.npz")
    print(len(test_clip.frames))
    print("frame 0:")
    print(test_clip.frames[0].frame_type)
    print(test_clip.frames[0].motion_vectors.shape)
    print(test_clip.frames[0].residuals.shape)
    print("Frame 0 Bounding Box:", test_clip.frames[0].true_bounding_box)
    print("Frame 0 Class ID:", test_clip.frames[0].true_class)
    print(np.all(test_clip.frames[0].motion_vectors['motion_x'] == 0)) # was all the motion vectors still (which is expected since it is an i frame)
    
    print("frame 100:")
    print(test_clip.frames[100].frame_type)
    print(test_clip.frames[100].motion_vectors.shape)
    print(test_clip.frames[100].residuals.shape)
    print("Frame 100 Bounding Box:", test_clip.frames[100].true_bounding_box)
    print("Frame 100 Class ID:", test_clip.frames[100].true_class)
    print(np.all(test_clip.frames[100].motion_vectors['motion_x'] == 0)) # was all the motion vectors still (which is not expected since it is an B frame)
    
    