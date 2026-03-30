import cv_reader
import os

# Path Setup
script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_dir, "test_video.mp4")

print("--- THE FINAL RESIDUAL HUNT ---")

try:
    if not os.path.exists(video_path):
        print(f"FAIL: Can't find video at {video_path}")
        exit(1)

    # 1. Access the 'read_video' function we confirmed exists
    if hasattr(cv_reader, 'read_video'):
        print("Success: Found 'read_video' function.")
        
        # This returns a LIST of frames
        all_frames = cv_reader.read_video(video_path)
        
        if isinstance(all_frames, list) and len(all_frames) > 0:
            print(f"Successfully decoded {len(all_frames)} frames!")
            
            # 2. Extract data from the first frame
            # The researcher's tuple is usually (frame, motion_vectors, residuals)
            first_frame_data = all_frames[0]
            
            # Handle different possible tuple lengths (3 or 4 items)
            if len(first_frame_data) == 3:
                frame, mv, res = first_frame_data
            else:
                _, frame, mv, res = first_frame_data # Skip the 'valid' bit if it's there
                
            print("\n" + "💎" * 15)
            print("  THE DATA IS REAL!")
            print("💎" * 15)
            print(f"🖼️  Frame Shape:     {frame.shape}")
            print(f"🏎️  Motion Vectors:  {mv.shape if mv is not None else 'N/A'}")
            print(f"🧹 Residuals:       {res.shape if res is not None else 'N/A'}")
            print("💎" * 15)
        else:
            print("FAIL: read_video returned an empty list or invalid data.")
    else:
        print(f"FAIL: Could not find read_video. Attributes: {dir(cv_reader)}")

except Exception as e:
    print(f"CRITICAL ERROR: {e}")