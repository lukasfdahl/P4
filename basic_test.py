import cv_reader
import os

# 1. Dynamically find the folder where THIS script is currently sitting
script_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Join it with the video name so it's an absolute path
video_path = os.path.join(script_dir, "test_video.mp4")

print("--- RECOVERY SMOKE TEST ---")

try:
    if os.path.exists(video_path):
        print(f"Found video at: {video_path}")
        
        # Initialize the reader
        cap = cv_reader.VideoCapture(video_path)
        
        # Try to grab data
        valid, frame, mv, res = cap.read()
        
        if valid:
            print("SUCCESS: Data extracted!")
            print(f"Frame Shape: {frame.shape}")
            # Use getattr or check existence if these are tricky
            print(f"Motion Vectors: {mv.shape if mv is not None else 'None'}")
            print(f"Residuals: {res.shape if res is not None else 'None'}")
        else:
            print("FAIL: Could not read frame (check video codec/format).")
    else:
        print(f"FAIL: File NOT found at {video_path}")
        print(f"I see these files in the folder: {os.listdir(script_dir)}")

except Exception as e:
    print(f"CRITICAL ERROR: {e}")