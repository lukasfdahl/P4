"""
downloader.py  –  YouTube-BB annotated video downloader and .npz extractor

Downloads only the annotated time-segments from YouTube-BB (no full videos),
extracts motion vectors + residuals via cv_reader / ffmpeg, and saves each
video as a .npz file.

Usage
-----
    Edit the CONFIG dictionary below, then run:
    python downloader.py
"""

import os
import sys
import subprocess
import tempfile
import shutil
import urllib.request
import gzip
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import random
import time
import numpy as np
import pandas as pd

# CONFIG
CONFIG = {
    "MODE":            "DOWNLOAD",     # "CHECK" or "DOWNLOAD"
    "MAX_VIDEOS":      10,             # Max number of videos to check/download
    "MAX_WORKERS":     1,              # Number of parallel threads for downloading/checking
    "TARGET_CLASSES":  [1],             # List of class IDs to download, e.g., [1, 23]. Leave empty [] for all.
    
    "OUTPUT_DIR":      "downloaded_videos/",
    "DATASET_DIR":     "dataset/",
    "RANDOM_SEED":     42,

    # Auto-download URLs for the dataset
    "CSV_FILES": [
        {
            "url": "https://research.google.com/youtube-bb/yt_bb_detection_train.csv.gz",
            "filename": "yt_bb_detection_train.csv"
        },
        {
            "url": "https://research.google.com/youtube-bb/yt_bb_detection_validation.csv.gz",
            "filename": "yt_bb_detection_validation.csv"
        }
    ],
    # Path to cookies file. If it doesn't exist, the script safely ignores it.
    "COOKIES_FILE":    "my_cookies_.txt",

    "SEGMENT_PADDING": 1.0,            # seconds of extra context around annotation
    "FRAME_H":         64,
    "FRAME_W":         64,
    "FPS":             30,             # must match npz_importer.py
}

# CSV schema
COLUMN_NAMES = [
    "youtube_id", "timestamp_ms", "class_id", "class_name",
    "object_id", "object_presence", "xmin", "xmax", "ymin", "ymax",
]

# Thread lock for clean console printing
print_lock = Lock()

def tprint(*args, **kwargs):
    """Thread-safe print"""
    with print_lock:
        print(*args, **kwargs)


# Dependency andd Dataset checks
def _check_dependencies() -> bool:
    ok = True
    for tool in ("yt-dlp", "ffmpeg"):
        if shutil.which(tool) is None:
            tprint(f"[downloader] ERROR: '{tool}' not found on PATH.")
            ok = False
    try:
        import cv2  # noqa: F401
    except ImportError:
        tprint("[downloader] ERROR: opencv-python not installed (pip install opencv-python)")
        ok = False
    return ok


def _ensure_csvs_exist():
    """Downloads and extracts the train/val CSVs, then combines them into a Master CSV."""
    os.makedirs(CONFIG["DATASET_DIR"], exist_ok=True)
    
    # 1. Download and extract individual files
    for file_info in CONFIG["CSV_FILES"]:
        out_csv = os.path.join(CONFIG["DATASET_DIR"], file_info["filename"])
        gz_path = out_csv + ".gz"
        url = file_info["url"]

        if not os.path.exists(out_csv):
            tprint(f"[downloader] Dataset missing. Downloading {url} ...")
            try:
                urllib.request.urlretrieve(url, gz_path)
                tprint(f"[downloader] Extracting {gz_path} ...")
                with gzip.open(gz_path, 'rb') as f_in:
                    with open(out_csv, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(gz_path)
            except Exception as e:
                tprint(f"[downloader] ERROR fetching dataset: {e}")
                sys.exit(1)

    # 2. Combine into a Master CSV
    master_csv = os.path.join(CONFIG["DATASET_DIR"], "master_yt_bb_detection.csv")
    if not os.path.exists(master_csv):
        tprint(f"[downloader] Combining train/val into {master_csv} ...")
        with open(master_csv, 'wb') as wfd:
            for file_info in CONFIG["CSV_FILES"]:
                fpath = os.path.join(CONFIG["DATASET_DIR"], file_info["filename"])
                with open(fpath, 'rb') as fd:
                    shutil.copyfileobj(fd, wfd)
        tprint(f"[downloader] Master CSV created successfully!")

# CSV helpers
def load_csvs() -> pd.DataFrame:
    master_csv = os.path.join(CONFIG["DATASET_DIR"], "master_yt_bb_detection.csv")
    tprint(f"[downloader] Loading Master CSV from {master_csv} …")
    
    # Load the combined master CSV
    full_df = pd.read_csv(master_csv, header=None, names=COLUMN_NAMES)
    
    # Filter 1: Object must be present
    full_df = full_df[full_df["object_presence"] == "present"].copy()
    
    # Filter 2: Specific classes (if configured)
    if CONFIG["TARGET_CLASSES"]:
        full_df = full_df[full_df["class_id"].isin(CONFIG["TARGET_CLASSES"])]
        tprint(f"[downloader] Filtered down to class IDs: {CONFIG['TARGET_CLASSES']}")

    tprint(f"[downloader] Loaded {len(full_df):,} target rows, "
          f"{full_df['youtube_id'].nunique():,} unique video IDs")
    return full_df

def get_video_segments(df: pd.DataFrame) -> dict[str, list[tuple[float, float]]]:
    pad = CONFIG["SEGMENT_PADDING"]
    segments: dict[str, list[tuple[float, float]]] = {}
    
    for vid_id, group in df.groupby("youtube_id"):
        times = np.sort(group["timestamp_ms"].unique()) / 1000.0
        
        chunks = []
        current_start = times[0]
        current_end = times[0]
        
        # If there is a gap greater than 3 seconds, we break it into a new chunk
        for t in times[1:]:
            if t - current_end > 3.0: 
                chunks.append((max(0.0, current_start - pad), current_end + pad))
                current_start = t
            current_end = t
            
        # Append the final chunk
        chunks.append((max(0.0, current_start - pad), current_end + pad))
        segments[str(vid_id)] = chunks
        
    return segments

# Parallel Availability Check
def _check_single_video(vid_id: str) -> tuple[str, bool]:
    url = f"https://www.youtube.com/watch?v={vid_id}"
    result = subprocess.run(
        ["yt-dlp", "--simulate", "--quiet", "--no-warnings", url],
        capture_output=True,
        text=True,
    )
    return vid_id, result.returncode == 0

def check_availability(video_ids: list[str]) -> list[str]:
    # Slice array to MAX_VIDEOS
    target_vids = video_ids[:CONFIG["MAX_VIDEOS"]]
    total = len(target_vids)
    available: list[str] = []

    tprint(f"[downloader] Checking availability of {total} videos using {CONFIG['MAX_WORKERS']} threads …")
    
    with ThreadPoolExecutor(max_workers=CONFIG["MAX_WORKERS"]) as executor:
        futures = {executor.submit(_check_single_video, vid): vid for vid in target_vids}
        
        checked_count = 0
        for future in as_completed(futures):
            vid_id, is_available = future.result()
            checked_count += 1
            
            if is_available:
                available.append(vid_id)
                status = "✓"
            else:
                status = "✗ (unavail)"

            tprint(f"  [{checked_count}/{total}] {vid_id} {status} (available so far: {len(available)})")

    tprint(f"\n[downloader] Final Available: {len(available)} / {total}")
    return available


# Video download & Extraction core

def _download_segment(vid_id: str, chunks: list[tuple[float, float]], tmp_dir: str) -> tuple[str | None, str]:
    url      = f"https://www.youtube.com/watch?v={vid_id}"
    out_path = os.path.join(tmp_dir, f"{vid_id}.mp4")

    cmd = [
        "yt-dlp", 
        "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
        "--force-keyframes-at-cuts",
        "--output", out_path,
        
        # METHOD 1: The Mobile API Bypass Trick
        "--extractor-args", "youtube:player_client=android,web",
    ]
    
    # METHOD 2: Inject cookies if the file exists in your container
    cookie_path = CONFIG.get("COOKIES_FILE", "")
    if cookie_path and os.path.exists(cookie_path):
        cmd.extend(["--cookies", cookie_path])
    
    # Add a --download-sections flag for EVERY chunk we found
    for start, end in chunks:
        section = f"*{start:.3f}-{end:.3f}"
        cmd.extend(["--download-sections", section])
        
    cmd.append(url)
    time.sleep(random.uniform(1, 3))

    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0 or not os.path.exists(out_path):
        error_lines = [line for line in result.stderr.split('\n') if line.strip()]
        error_msg = error_lines[-1] if error_lines else "Unknown yt-dlp error"
        return None, error_msg
        
    return out_path, ""
MV_STRUCT = np.dtype([
    ("source",       np.int32),
    ("motion_x",     np.int32),
    ("motion_y",     np.int32),
    ("motion_scale", np.int32),
])

def _extract_to_npz(video_path: str, output_npz: str, frame_h: int, frame_w: int) -> bool:
    try:
        import cv2 #debug error fix?
    except ImportError:
        return False

    # 1. ffprobe frame types
    ffprobe_cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "v:0",
        "-show_frames", "-show_entries", "frame=pict_type",
        "-of", "csv=p=0", video_path,
    ]
    result = subprocess.run(ffprobe_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False

    frame_types_raw = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
    frame_types = np.array(
        [ft if ft in ("I", "P", "B") else "?" for ft in frame_types_raw],
        dtype="U1"
    )
    n_frames = len(frame_types)
    if n_frames == 0:
        return False

    # 2. Decode residuals
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    residuals_list = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_resized = cv2.resize(frame, (frame_w, frame_h), interpolation=cv2.INTER_AREA)
        residuals_list.append(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
    cap.release()

    min_frames = min(n_frames, len(residuals_list))
    frame_types = frame_types[:min_frames]
    residuals   = np.stack(residuals_list[:min_frames], axis=0)

    # 3. Motion Vectors
    h_mv, w_mv = frame_h // 16, frame_w // 16
    n_mv = h_mv * w_mv
    mv_array = np.zeros((min_frames, n_mv), dtype=MV_STRUCT)

    try:
        mv_array = _extract_motion_vectors_ffmpeg(video_path, min_frames, h_mv, w_mv)
    except Exception:
        pass # fallback to zeros

    # 4. Save
    np.savez_compressed(
        output_npz,
        frame_types    = frame_types,
        motion_vectors = mv_array.view(np.int32).reshape(min_frames, n_mv, 4),
        residuals      = residuals,
    )
    return True


def _extract_motion_vectors_ffmpeg(video_path: str, n_frames: int, h_mv: int, w_mv: int) -> np.ndarray:
    n_mv   = h_mv * w_mv
    result = np.zeros((n_frames, n_mv), dtype=MV_STRUCT)

    cmd = [
        "ffmpeg", "-flags2", "+export_mvs",
        "-i", video_path, "-vf", "codecview=mv=pf+bf+bb",
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    frame_idx = -1
    for line in proc.stderr.split("\n"):
        if "frame_num=" not in line: continue
        kv = {k: v for token in line.split() if "=" in token for k, _, v in [token.partition("=")]}
        try:
            fn  = int(kv.get("frame_num", frame_idx))
            src = int(kv.get("src", 1))
            mx, my = int(kv.get("dx", 0)), int(kv.get("dy", 0))
            bx, by = int(kv.get("bx", 0)), int(kv.get("by", 0))
        except (ValueError, KeyError):
            continue

        if fn < 0 or fn >= n_frames: continue
        row, col = by // 16, bx // 16
        if 0 <= row < h_mv and 0 <= col < w_mv:
            cell = row * w_mv + col
            result[fn, cell]["source"] = src
            result[fn, cell]["motion_x"], result[fn, cell]["motion_y"] = mx, my
            result[fn, cell]["motion_scale"] = 1

    return result


# Parallel Download Pipeline
def _worker_download_and_extract(vid_id, chunks, tmp_dir, output_dir, frame_h, frame_w):
    out_npz = os.path.join(output_dir, f"{vid_id}.npz")
    
    if os.path.exists(out_npz):
        return vid_id, "skipped"

    mp4_path, error_msg = _download_segment(vid_id, chunks, tmp_dir)
    if not mp4_path:
        return vid_id, f"failed_download ({error_msg})"

    success = _extract_to_npz(mp4_path, out_npz, frame_h, frame_w)
    
    try:
        os.remove(mp4_path)
    except OSError:
        pass

    if success:
        size_kb = os.path.getsize(out_npz) // 1024
        return vid_id, f"success ({size_kb} KB)"
    else:
        return vid_id, "failed_extract"


def run_download(df: pd.DataFrame) -> None:
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
    segments  = get_video_segments(df)

    # 1. Get all IDs and sort them alphabetically to guarantee a stable baseline across machines
    all_video_ids = list(segments.keys())
    all_video_ids.sort()
    
    # 2. Apply a reproducible random shuffle if a seed is set
    if CONFIG.get("RANDOM_SEED") is not None:
        rng = random.Random(CONFIG["RANDOM_SEED"])
        rng.shuffle(all_video_ids)

    target = CONFIG["MAX_VIDEOS"]
    tprint(f"\n[downloader] DOWNLOAD mode")
    tprint(f"  Target : {target} successful downloads → {CONFIG['OUTPUT_DIR']} (Threads: {CONFIG['MAX_WORKERS']})")
    if CONFIG.get("RANDOM_SEED") is not None:
        tprint(f"  Seed   : {CONFIG['RANDOM_SEED']} (Reproducible Mode)")

    results = {"success": 0, "skipped": 0, "failed": 0}
    
    with tempfile.TemporaryDirectory(prefix="yt_bb_tmp_") as tmp_dir:
        index = 0
        
        while index < len(all_video_ids) and (results["success"] + results["skipped"]) < target:
            
            needed = target - (results["success"] + results["skipped"])
            batch_vids = all_video_ids[index : index + needed]
            index += needed
            
            if not batch_vids:
                break 

            with ThreadPoolExecutor(max_workers=CONFIG["MAX_WORKERS"]) as executor:
                futures = {}
                for vid_id in batch_vids:
                    chunks = segments[vid_id]
                    futures[executor.submit(
                        _worker_download_and_extract, vid_id, chunks, 
                        tmp_dir, CONFIG["OUTPUT_DIR"], CONFIG["FRAME_H"], CONFIG["FRAME_W"]
                    )] = vid_id

                for future in as_completed(futures):
                    vid_id = futures[future]
                    try:
                        res_vid, status = future.result()
                        if "success" in status:
                            results["success"] += 1
                            icon = "✓"
                        elif status == "skipped":
                            results["skipped"] += 1
                            icon = "-"
                        else:
                            results["failed"] += 1
                            icon = "✗"
                        
                        current_good = results["success"] + results["skipped"]
                        tprint(f"  [Good: {current_good}/{target}] {res_vid} {icon} {status}")
                        
                    except Exception as e:
                        results["failed"] += 1
                        tprint(f"  [Error] {vid_id} ✗ Exception: {e}")

    tprint(f"\n[downloader] Done.")
    tprint(f"  Successfully Downloaded : {results['success']}")
    tprint(f"  Skipped (already exist) : {results['skipped']}")
    tprint(f"  Failed (unavailable)    : {results['failed']}")
    
    if (results["success"] + results["skipped"]) < target:
        tprint(f"  NOTE: Ran out of videos in CSV before reaching target of {target}.")

# Entry point
def main() -> None:
    if not _check_dependencies():
        tprint("\n[downloader] Missing dependencies — see errors above.")
        sys.exit(1)

    _ensure_csvs_exist()
    df = load_csvs()

    if CONFIG["MODE"].upper() == "CHECK":
        segments = get_video_segments(df)
        video_ids = list(segments.keys())
        check_availability(video_ids)
    elif CONFIG["MODE"].upper() == "DOWNLOAD":
        run_download(df)
    else:
        tprint(f"[downloader] Invalid mode: {CONFIG['MODE']}. Use 'CHECK' or 'DOWNLOAD'.")

if __name__ == "__main__":
    main()