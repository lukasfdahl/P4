import cv_reader
import os

# Use any small video file you have. 
# If you don't have one in the folder, this will just test the import.
video_path = "test_video.mp4" 

print("--- RECOVERY SMOKE TEST ---")

try:
    # 1. Check if the module actually loads (the NumPy/FFmpeg link check)
    print(f"Checking cv_reader version: {cv_reader.__version__}")
    
    if os.path.exists(video_path):
        # 2. Try to initialize the reader
        cap = cv_reader.VideoCapture(video_path)
        
        # 3. Try to grab one frame's worth of data
        # 'valid' is a boolean, 'mv' are motion vectors, 'res' are residuals
        valid, frame, mv, res = cap.read()
        
        if valid:
            print("SUCCESS: Data extracted!")
            print(f"Frame Shape: {frame.shape}")
            print(f"Motion Vectors: {mv.shape if mv is not None else 'None'}")
            print(f"Residuals: {res.shape if res is not None else 'None'}")
        else:
            print("FAIL: Could not read frame (is the video file corrupt?)")
    else:
        print(f"INFO: No file found at {video_path}, but the module IMPORTED successfully.")
        print("This means the C-Linker is working perfectly!")

except Exception as e:
    print(f"CRITICAL ERROR: {e}")