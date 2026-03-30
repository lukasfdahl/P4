import cv_reader
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_dir, "test_video.mp4")

print("--- THE VICTORY LAP ---")

try:
    all_frames = cv_reader.read_video(video_path)
    print(f"Decoded {len(all_frames)} frames.")

    # We look at Frame 1 (usually the first P-frame with data)
    # If the video is long, we search for the first frame with motion vectors
    target_frame = None
    for f in all_frames:
        if f.get('mv') is not None:
            target_frame = f
            break
    
    # Fallback to frame 0 if no MVs found
    if target_frame is None:
        target_frame = all_frames[0]

    print("\n" + "⭐" * 20)
    print("      FINAL DATA")
    print("⭐" * 20)
    
    # Dimensions
    print(f"📐 Size:      {target_frame.get('width')}x{target_frame.get('height')}")
    
    # Residuals (In this library, the 'bgr' key holds the residual data)
    res_data = target_frame.get('bgr')
    if res_data is not None:
        print(f"🧹 Residuals: {res_data.shape} (Key: 'bgr')")
    
    # Motion Vectors
    mv_data = target_frame.get('mv')
    if mv_data is not None:
        print(f"🏎️  Vectors:   {mv_data.shape} (Key: 'mv')")
    else:
        print("🏎️  Vectors:   None (This is an I-Frame)")
        
    print("⭐" * 20)

except Exception as e:
    print(f"Error: {e}")
    if all_frames:
        print(f"Available keys in frame: {list(all_frames[0].keys())}")