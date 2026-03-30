import cv_reader.reader  # <-- We import the sub-module directly
import os
import sys

# 1. Path Setup
script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_dir, "test_video.mp4")

print("--- FINAL DATA EXTRACTION ---")

try:
    # 2. Access the class through the sub-module
    # Based on your log, 'reader' is the place where the logic lives.
    print(f"Opening video: {video_path}")
    
    # We use cv_reader.reader.VideoCapture
    cap = cv_reader.reader.VideoCapture(video_path)
    
    # 3. Read the first frame
    valid, frame, mv, res = cap.read()
    
    if valid:
        print("\n" + "⭐" * 30)
        print("   EXTRACTION COMPLETE!")
        print("⭐" * 30)
        print(f"🖼️  Frame Size:     {frame.shape}")
        
        if mv is not None:
            print(f"🏎️  Motion Vectors: {mv.shape}")
        else:
            print("🏎️  Motion Vectors: NOT FOUND (Check P-frames)")
            
        if res is not None:
            print(f"🧹 Residuals:      {res.shape}")
        else:
            print("🧹 Residuals:      NOT FOUND")
        print("⭐" * 30)
    else:
        print("FAIL: The reader opened the file, but couldn't decode data.")

except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    # If VideoCapture isn't in reader, let's see what IS in there
    if 'cv_reader' in sys.modules:
        import cv_reader.reader
        print(f"Contents of cv_reader.reader: {dir(cv_reader.reader)}")