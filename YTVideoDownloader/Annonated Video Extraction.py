import pandas as pd
import numpy as np
import yt_dlp
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------- VARIABLES -------------------

Dataset_file = "yt_bb_detection_train.csv"
Samples_size = 20
Selected_Classes = 5

Output_dir = "downloaded_clips"
Threads = 8

Clip_seconds = 10
Half_clip = Clip_seconds / 2

np.random.seed(42)
os.makedirs(Output_dir, exist_ok=True)

# ------------------- DATASET COLUMNS -------------------

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

df = pd.read_csv(Dataset_file, names=columns)

# Only keep real annotations
df = df[df["presence"] == "present"].copy()

# ------------------- RANDOM CLASS SELECTION -------------------

all_classes = sorted(df["class_name"].unique())

if Selected_Classes > len(all_classes):
    raise ValueError(
        f"Selected_Classes={Selected_Classes}, but dataset only has {len(all_classes)} classes."
    )

selected_classes = np.random.choice(
    all_classes,
    size=Selected_Classes,
    replace=False
)

print("\nSelected classes:")
for cls in selected_classes:
    print("-", cls)

df = df[df["class_name"].isin(selected_classes)].copy()

# ------------------- BALANCED CLASS SAMPLING -------------------

classes = sorted(df["class_name"].unique())
num_classes = len(classes)

if Samples_size % num_classes != 0:
    raise ValueError(
        f"Samples_size={Samples_size} is not divisible by Selected_Classes={num_classes}. "
        f"Use a sample size like {num_classes}, {num_classes * 2}, {num_classes * 10}, etc."
    )

clips_per_class = Samples_size // num_classes

balanced_samples = []

for class_name in classes:
    class_df = df[df["class_name"] == class_name]

    if len(class_df) < clips_per_class:
        raise ValueError(
            f"Class '{class_name}' only has {len(class_df)} samples, "
            f"but you requested {clips_per_class} clips."
        )

    sampled = class_df.sample(
        n=clips_per_class,
        random_state=42
    )

    balanced_samples.append(sampled)

sample_df = pd.concat(balanced_samples)

# Shuffle final sampled clips
sample_df = sample_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nFinal class distribution:")
print(sample_df["class_name"].value_counts())

tasks = sample_df[[
    "video_id",
    "timestamp_ms",
    "class_id",
    "class_name"
]].to_dict("records")

# ------------------- DOWNLOAD FUNCTION -------------------

def safe_filename(text):
    return (
        str(text)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace("*", "_")
        .replace("?", "_")
        .replace('"', "_")
        .replace("<", "_")
        .replace(">", "_")
        .replace("|", "_")
    )

def download_annotated_clip(task):
    video_id = task["video_id"]
    timestamp_ms = task["timestamp_ms"]
    class_id = task["class_id"]
    class_name = safe_filename(task["class_name"])

    center_time = timestamp_ms / 1000
    start_time = max(0, center_time - Half_clip)
    duration = Clip_seconds

    output_path = os.path.join(
        Output_dir,
        f"{class_name}_{class_id}_{video_id}_{int(start_time)}_{int(start_time + duration)}.mp4"
    )

    if os.path.exists(output_path):
        return video_id, class_name, "already exists"

    url = f"https://www.youtube.com/watch?v={video_id}"

    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "format": "best[ext=mp4][height<=720]/best[height<=720]/best",
        "noplaylist": True,
        "ignoreerrors": True
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            if info is None:
                return video_id, class_name, "missing"

            video_url = info["url"]

        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start_time),
            "-i", video_url,
            "-t", str(duration),
            "-c", "copy",
            output_path
        ]

        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        return video_id, class_name, "success"

    except Exception as e:
        return video_id, class_name, f"error: {str(e)}"

# ------------------- MULTI-THREAD DOWNLOAD -------------------

results = []
working = []
missing = []

print(f"\nCollecting {Samples_size} balanced annotated clips...")
print(f"{clips_per_class} clips per class\n")

with ThreadPoolExecutor(max_workers=Threads) as executor:
    futures = [executor.submit(download_annotated_clip, task) for task in tasks]

    for future in as_completed(futures):
        video_id, class_name, status = future.result()
        results.append((video_id, class_name, status))

        if status in ["success", "already exists"]:
            working.append((video_id, class_name))
        else:
            missing.append((video_id, class_name, status))

        print(f"Progress: {len(working)}/{len(tasks)} clips downloaded", end="\r")

# ------------------- RESULTS -------------------

success = [r for r in results if r[2] == "success"]
already_exists = [r for r in results if r[2] == "already exists"]
failed = [r for r in results if r[2] not in ["success", "already exists"]]

print("\n\nDownload Summary:")
print("Successful clip downloads:", len(success))
print("Already existing clips:", len(already_exists))
print("Failed downloads:", len(failed))

results_df = pd.DataFrame(results, columns=["video_id", "class_name", "status"])
results_df.to_csv(os.path.join(Output_dir, "download_results.csv"), index=False)

print("\nDownloaded class distribution:")
print(
    results_df[
        results_df["status"].isin(["success", "already exists"])
    ]["class_name"].value_counts()
)

print(f"\nResults saved to: {os.path.join(Output_dir, 'download_results.csv')}")