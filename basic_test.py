import cv_reader
import cv_reader.api # Specifically load the C++ engine
import os

# Path Setup
script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_dir, "test_video.mp4")

print("--- THE FINAL RESIDUAL HUNT ---")

try:
    # 1. Try to find the class in the 'api' sub-module
    # Your log showed 'api' is available. That is where the C++ VideoCapture lives.
    if hasattr(cv_reader.api, 'VideoCapture'):
        print("Success: Found 'VideoCapture' inside the api module.")
        cap = cv_reader.api.VideoCapture(video_path)
    elif hasattr(cv_reader, 'read_video'):
        print("Success: Found 'read_video' function.")
        # Some versions use a function that returns a generator
        # We will try to get the first frame from it
        reader = cv_reader.read_video(video_path)
        valid, frame, mv, res = next(reader)
    else:
        print(f"FAIL: Still can't find the entry point. Attributes: {dir(cv_reader.api)}")
        exit(1)

    # 2. Extract and Print (If we got here, we have a 'cap' or 'reader')
    # If we used the VideoCapture class:
    if 'cap' in locals():
        valid, frame, mv, res = cap.read()

    if valid:
        print("\n" + "💎" * 15)
        print("  THE DATA IS REAL!")
        print("💎" * 15)
        print(f"🖼️  Frame:     {frame.shape}")
        print(f"🏎️  Vectors:   {mv.shape if mv is not None else 'N/A'}")
        print(f"🧹 Residuals: {res.shape if res is not None else 'N/A'}")
        print("💎" * 15)
    else:
        print("FAIL: The reader found the file but couldn't decode the frame.")

except Exception as e:
    print(f"CRITICAL ERROR: {e}")