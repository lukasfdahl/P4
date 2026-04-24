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

dataset_labels_path = "dataset/master_yt_bb_detection.csv"
dataset_download_url = "https://research.google.com/youtube-bb/yt_bb_detection_train.csv.gz" #also handled now in download.py
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
    residuals : np.ndarray = data["residuals"]
    motion_vectors = raw_motion_vectors.view(MV_STRUCT)
    
    # load labels directly from the baked-in arrays
    boxes = data['boxes']
    true_classes = data['true_class']
    
    # creating the frame and clip objects
    frames = []
    for index in range(len(frame_types)):
        
        # Read directly from our new arrays
        true_class = int(true_classes[index])
        has_object = (true_class != -1)
        
        if has_object:
            bbox = tuple(boxes[index])  # Use the actual box
        else:
            bbox = (-1.0, -1.0, -1.0, -1.0) # Default background box
            
        frame = Frame(
            motion_vectors=motion_vectors[index], 
            frame_type=frame_types[index], 
            residuals=residuals[index], 
            true_bounding_box=bbox, 
            true_class=true_class, 
            has_object=has_object
        )
        frames.append(frame)
        
    return Clip(frames)



download_dataset() # just to ensure the dataset is downloaded no matter what.
labels = import_dataset()


# some test code i made
if __name__ == "__main__":
    test_clip = import_clip("downloaded_videos/test video.npz")
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
    
    