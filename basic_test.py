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

    print("Decoding video...")
    all_frames = cv_reader.read_video(video_path)
    
    if all_frames and len(all_frames) > 0:
        print(f"Successfully decoded {len(all_frames)} frames!")
        
        # 1. Grab the first dictionary
        f = all_frames[0]
        
        print("\n" + "💎" * 15)
        print("  THE DATA IS REAL!")
        print("💎" * 15)
        
        # 2. Pull data using the keys we found in the diagnostic
        # We'll use .get() to be safe
        print(f"📄 Frame Type:    {f.get('type', 'N/A')}")
        print(f"📐 Dimensions:    {f.get('width')}x{f.get('height')}")
        
        # The actual image/residual data is usually under 'data' or 'residual'
        data = f.get('data')
        if data is not None:
            print(f"🖼️  Data Shape:    {data.shape} (Residual/Frame)")
        
        # Motion vectors are usually 'mv'
        mv = f.get('mv')
        if mv is not None:
            print(f"🏎️  Motion Vectors: {mv.shape}")
        else:
            print("🏎️  Motion Vectors: None")
            
        print("💎" * 15)
        
    else:
        print("FAIL: read_video returned no data.")

except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    if 'all_frames' in locals() and len(all_frames) > 0:
        print(f"Dictionary Keys available: {list(all_frames[0].keys())}")