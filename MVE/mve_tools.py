from mvextractor.videocap import VideoCap
from numpy import ndarray
import warnings
from typing import NamedTuple

#a class to just make video stats return more readable
class FrameStats(NamedTuple):
    status: bool
    frame_count: int
    keyframe_count: int
    past_frame_count: int
    future_frame_count: int
    unknown_frame_count: int
    total_motion_vector_count: int

    #a class to just make video stats return more readable
class Frame(NamedTuple):
    frame_type: str # Frametype (I = Keyframe, P = frame that only referes to past frames, B = frame that referes to both past and future frames. ? = unkown frame type)
    motion_vector_array: ndarray



cap = VideoCap() #VideoCap is the class that has all the functions for extracting relevant information from videos

# (optional) skip decoding frames
cap.set_decode_frames(False)

#url can be either a path to a file or the ip of a camera streaming with RTSP (VIDEO MUST BE H.264 encoded)
def extract_vectors(url : str) -> list[Frame]:
    status = cap.open(url)
    if status == False:
        warnings.warn("Chould not open video file, or connect to cammera", UserWarning)
        cap.release()
        return [] #returning an empty list means it failed
    
    frames : list[Frame] = [] #Each element in the list is a frame, each frame is a tuple with first the type of frame as a string, follow by the motion vectors as an ndarray

    while True: #Loop through all frames in the video. (IMORTANT. if we need to work on streams, then impliment some system to make it not run forever)
        success, _, motion_vectors, frame_type = cap.read()
        if success == False: #IF the video has ended or mve fails to extract motion vectors
            break
        frames.append(Frame(frame_type, motion_vectors))

    cap.release()
    return frames
    

#A function that returnes some basic stats about a video like frame count, and the number of motion vectors and the different types of frames
def get_video_stats(url: str) -> FrameStats:
    frames = extract_vectors(url)
    if frames == []:
        return FrameStats(False, 0, 0, 0, 0, 0, 0)
    
    frame_count = 0
    keyframe_count = 0
    past_frame_count = 0
    future_frame_count = 0
    unknown_frame_count = 0
    total_motion_vector_count = 0

    for frame in frames:
        total_motion_vector_count += len(frame[1])
        frame_count += 1
        if frame.frame_type == "I":
            keyframe_count += 1
        elif frame.frame_type == "P":
            past_frame_count += 1
        elif frame.frame_type == "B":
            future_frame_count += 1
        elif frame.frame_type == "?":
            unknown_frame_count += 1

    return FrameStats(True, frame_count, keyframe_count, past_frame_count, future_frame_count, unknown_frame_count, total_motion_vector_count)
    
# Just printes the stats gotten by get_video_stats in a little overview
def print_video_stats(url: str) -> FrameStats:
    stats = get_video_stats(url)
    print("----------------------------------------------------------")
    print(f"Stats for video ( {url} )")
    print(f"Success: {stats.status}")
    print(f"Number of Key frames: {stats.keyframe_count}")
    print(f"Number of Past frames: {stats.past_frame_count}")
    print(f"Number of Future & Past frames: {stats.future_frame_count}")
    print(f"Number of Unkown frames: {stats.unknown_frame_count}")
    print(f"Total number of Motion Vectors: {stats.total_motion_vector_count}")
    print("----------------------------------------------------------")

    return stats