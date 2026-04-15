import pandas as pd
import numpy as np
import yt_dlp
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Variables
Dataset_file = "/app/yt_bb_detection_train.csv"
Samples_size = 1
Output_dir = "/app/test/input_videos"
Threads = 1

# For reproducibility
np.random.seed(42)
os.makedirs(Output_dir, exist_ok=True)

# Dataset Columns
columns = [
    "video_id",
    "timestamp_ms",
    "class_id",
    "class_name",
    "object_id",
    "presence",
    "xmin",
    "xmax",
    "ymin",
    "ymax"
]

# Sample the video ids randomly
df = pd.read_csv(Dataset_file, names=columns)
video_ids = df["video_id"].unique()

# -------------------------------------- CHECK AVAILABLE VIDEOS --------------------------------------

working = []
missing = []
remaining_pool = set(video_ids)

# ------------------- DOWNLOAD AVAILABLE VIDEOS ------------------

def download_video(video_id):
    url = f"https://www.youtube.com/watch?v={video_id}"
    output_path = os.path.join(Output_dir, f"{video_id}.mp4")

    # Skip if already downloaded
    if os.path.exists(output_path):
        return (video_id, "already exists")

    ydl_opts = {
        "outtmpl": os.path.join(Output_dir, "%(id)s.%(ext)s"),
        "format": "mp4[height<=720]/best",
        "quiet": True,
        "ignoreerrors": False,
        "noplaylist": True,
        "merge_output_format": "mp4"
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: # type: ignore (makes the type checker ignore this line (it does not like it))
            ydl.download([url])
        return (video_id, "success")

    except Exception as e:
        return (video_id, f"error: {str(e)}")


# ------------------- MULTI-THREAD EXECUTION ------------------

results = []

print("\nCollecting 1000 downloadable videos...\n")

while len(working) < Samples_size and len(remaining_pool) > 0:

    needed = Samples_size - len(working)
    batch_size = min(Threads * 2, needed, len(remaining_pool))

    batch = np.random.choice(list(remaining_pool), batch_size, replace=False)

    for vid in batch:
        remaining_pool.remove(vid)

    with ThreadPoolExecutor(max_workers=Threads) as executor:
        futures = [executor.submit(download_video, vid) for vid in batch]

        for future in as_completed(futures):
            vid, status = future.result()
            results.append((vid, status))

            if status in ["success", "already exists"]:
                working.append(vid)
            else:
                missing.append(vid)

    print(f"Progress: {len(working)}/{Samples_size} downloaded")

# ------------------- RESULTS ------------------

success = [vid for vid, status in results if status == "success"]
failed = [vid for vid, status in results if status not in ["success", "already exists"]]

print("\nDownload Summary:")
print("Successful downloads:", len(success))
print("Failed downloads:", len(failed))