import cv_reader
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_dir, "test_video.mp4")

print("--- FRAME 6 RESIDUAL EXTRACTION ---")

try:
    # 1. Decode the video
    all_frames = cv_reader.read_video(video_path)
    print(f"Total frames decoded: {len(all_frames)}")

    if len(all_frames) > 6:
        # 2. Grab Frame 6 (The 7th frame)
        f = all_frames[6]
        
        print("\n" + "🔍" * 15)
        print("  FRAME 6 DATA FOUND")
        print("🔍" * 15)
        
        # In this library, 'bgr' contains the Residual/Difference image
        res_data = f.get('bgr')
        # 'mv' contains the Motion Vectors
        mv_data = f.get('mv')
        
        print(f"📄 Frame Type:    {f.get('type', 'P-Frame (assumed)')}")
        
        if res_data is not None:
            print(f"🧹 Residual Shape: {res_data.shape}")
            print(f"   (This is the pixel-level difference data)")
        else:
            print("🧹 Residuals:      NOT FOUND in 'bgr' key.")
            
        if mv_data is not None:
            print(f"🏎️  Vector Shape:   {mv_data.shape}")
            print(f"   (This is the macroblock movement data)")
        else:
            print("🏎️  Vectors:        NOT FOUND in 'mv' key.")
            
        print("🔍" * 15)
        
        # If keys are still weird, this will tell us why:
        if res_data is None or mv_data is None:
            print(f"\nAvailable keys in this frame: {list(f.keys())}")
            
    else:
        print(f"FAIL: Video only has {len(all_frames)} frames. Need at least 7.")

except Exception as e:
    print(f"CRITICAL ERROR: {e}")