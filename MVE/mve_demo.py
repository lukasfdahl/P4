from mvextractor.videocap import VideoCap

cap = VideoCap()
cap.open("MVE/test video.mp4")

# (optional) skip decoding frames
cap.set_decode_frames(False)

while True:
    ret, frame, motion_vectors, frame_type = cap.read()
    if not ret:
        break
    print(f"Num. motion vectors: {len(motion_vectors)}")
    print(f"Frame type: {frame_type}")
    if frame is not None:
        print(f"Frame size: {frame.shape}")

cap.release()