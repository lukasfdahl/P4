import cv_reader
import os
import sys

# 1. Path Setup
script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_dir, "test_video.mp4")

print("--- RESIDUAL EXTRACTION SMOKE TEST ---")

try:
    # 2. Identify the Reader Class
    # The researcher sometimes uses 'VideoCapture' and sometimes 'CVReader'
    # We check both to be safe.
    reader_class = None
    if hasattr(cv_reader, 'VideoCapture'):
        reader_class = cv_reader.VideoCapture
    elif hasattr(cv_reader, 'CVReader'):
        reader_class = cv_reader.CVReader
    
    if reader_class is None:
        print("CRITICAL FAIL: Could not find 'VideoCapture' or 'CVReader' in the module.")
        print(f"Check your import location: {cv_reader.__file__}")
        print(f"Attributes available: {dir(cv_reader)}")
        sys.exit(1)

    # 3. Process the Video
    if os.path.exists(video_path):
        print(f"Opening video: {video_path}")
        cap = reader_class(video_path)
        
        # Read the first frame
        # returns: (is_valid, frame_data, motion_vectors, residuals)
        valid, frame, mv, res = cap.read()
        
        if valid:
            print("\n" + "="*30)
            print("🚀 DATA EXTRACTION SUCCESSFUL")
            print("="*30)
            print(f"🖼️  Frame Size:     {frame.shape}")
            
            if mv is not None:
                print(f"🏎️  Motion Vectors: {mv.shape}")
            else:
                print("🏎️  Motion Vectors: NONE (Check if video has P-frames)")
                
            if res is not None:
                print(f"🧹 Residuals:      {res.shape}")
            else:
                print("🧹 Residuals:      NONE (Check codec compatibility)")
            print("="*30)
        else:
            print("FAIL: The reader opened, but could not decode the first frame.")
    else:
        print(f"FAIL: test_video.mp4 not found at {video_path}")

except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    import traceback
    traceback.print_exc()