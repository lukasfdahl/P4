import cv_reader
import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_dir, "test_video.mp4")

print("--- DICTIONARY DETECTIVE ---")

try:
    all_frames = cv_reader.read_video(video_path)
    if len(all_frames) > 6:
        f = all_frames[6]
        print(f"Frame 6 detected. Keys in dictionary: {list(f.keys())}")
        
        print("\n" + "🔍" * 15)
        for key, value in f.items():
            # If it's a numpy array, print its shape
            if isinstance(value, np.ndarray):
                print(f"✅ {key}: Array of shape {value.shape}")
            # If it's a small value (like width/height), print it
            elif isinstance(value, (int, str, float)):
                print(f"ℹ️  {key}: {value}")
            else:
                print(f"❓ {key}: Type {type(value)}")
        print("🔍" * 15)
        
    else:
        print(f"Not enough frames. Found {len(all_frames)}")

except Exception as e:
    print(f"CRITICAL ERROR: {e}")