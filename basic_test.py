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

    # 1. Use the read_video function
    print("Decoding video...")
    all_frames = cv_reader.read_video(video_path)
    
    if all_frames and len(all_frames) > 0:
        print(f"Successfully decoded {len(all_frames)} frames!")
        
        # 2. THE FLEXIBLE UNPACK
        # This takes the first 3 items and puts everything else into 'extra'
        # Researcher format is usually: (Type, Data, MotionVectors, ...rest)
        first_frame = all_frames[0]
        
        # We grab the first 3 elements and ignore the rest
        f_type, data, mv, *extra = first_frame
        
        print("\n" + "💎" * 15)
        print("  THE DATA IS REAL!")
        print("💎" * 15)
        print(f"📄 Frame Type:    {f_type}")
        print(f"🖼️  Data Shape:    {data.shape} (Residual/Frame)")
        
        if mv is not None:
            print(f"🏎️  Motion Vectors: {mv.shape}")
        else:
            print("🏎️  Motion Vectors: None (I-Frame)")
            
        print(f"🎁 Extra Info:    {len(extra)} other hidden fields found")
        print("💎" * 15)
        
    else:
        print("FAIL: read_video returned no data.")

except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    # Final fallback: just tell us what the first frame actually IS
    if 'all_frames' in locals() and len(all_frames) > 0:
        print(f"Raw data structure of frame 0: {type(all_frames[0])} with length {len(all_frames[0])}")