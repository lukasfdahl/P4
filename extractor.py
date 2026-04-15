import cv_reader
import os
import numpy as np
import glob

INPUT_DIR = "/app/test/input_videos" # root/input_videos
OUTPUT_DIR = "/app/test/output_dir" # root/output_dir

#Test to fix problems:
print("\n--- 1. DOWNLOADING ---")
os.system('pip install -q gdown && gdown -q "1A_UYswnCL-jd9UotA8H6wJZ5eajhXwF1" -O downloaded_videos.zip')

print("\n--- 2. CREATING FOLDER ---")
os.makedirs('/app/test/input_videos', exist_ok=True)

print("\n--- 3. UNZIPPING (VISIBLE) ---")
# Removed the 'q' (quiet) flag so it prints exactly what it is extracting and where
os.system('unzip -o -j downloaded_videos.zip -d /app/test/input_videos/')

print("\n--- 4. CHECKING FOLDER CONTENTS ---")
# This will list every single file actually inside that folder
os.system('ls -la /app/test/input_videos/')

working_videos = glob.glob('/app/test/input_videos/*.mp4')
print(f"\n--- SCRIPT FOUND {len(working_videos)} .mp4 VIDEOS ---")





def main():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for filename in os.listdir(INPUT_DIR): # loop though all videos to process
        if filename.endswith(".mp4"): # if it is a video file
            full_file_path = os.path.join(INPUT_DIR, filename)
            video_output_name = filename.removesuffix(".mp4") + ".npz"
            if not os.path.exists(os.path.join(OUTPUT_DIR, video_output_name)): # Only extract data if the file is not already extracted from ealrier run
                type_array, motion_vector_array, residual_array = extract_clip(full_file_path)
                np.savez_compressed(
                    os.path.join(OUTPUT_DIR, video_output_name),
                    frame_types = type_array,
                    motion_vectors = motion_vector_array,
                    residuals = residual_array
                )
                print(f"successfully extracted data from video: {filename} and saved it to {os.path.join(OUTPUT_DIR, video_output_name)}")


# extracts the frame types, motion vectors and reciduals from a given clip, and saves them as 3 seperate numpy arrays, one for each type.
def extract_clip(clip_path : str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        all_frames = cv_reader.read_video(clip_path) #read the video file with the custom ffmpeg

        type_list = []
        motion_vector_list = []
        residual_list = []
        for frame in all_frames: # loops though each frame in the video
            type_list.append(frame["pict_type"])
            motion_vector_list.append(frame["motion_vector"])
            residual_list.append(frame["residual"])

        return np.array(type_list), np.array(motion_vector_list), np.array(residual_list) # converts the lists into np arrays.

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        assert False # to just crash the program if it fails (and prevent type checker form throwing a fit)

if __name__ == "__main__":
    main()