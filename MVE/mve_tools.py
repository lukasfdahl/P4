from mvextractor.videocap import VideoCap
from numpy import ndarray
import warnings

cap = VideoCap() #VideoCap is the class that has all the functions for extracting relevant information from videos

# (optional) skip decoding frames
cap.set_decode_frames(False)

#url can be either a path to a file or the ip of a camera streaming with RTSP (VIDEO MUST BE H.264 encoded)
def extract_vectors(url : str) -> list[tuple[str, ndarray]] | bool:
    status = cap.open(url)
    if status == False:
        warnings.warn("Chould not open video file, or connect to cammera", UserWarning)
        cap.release()
        return False #Returns false to show that the function failed
    
    frames : list[tuple[str, ndarray]] = [] #Each element in the list is a frame, each frame is a tuple with first the type of frame as a string, follow by the motion vectors as an ndarray

    while True: #Loop through all frames in the video. (IMORTANT. if we need to work on streams, then impliment some system to make it not run forever)
        success, _, motion_vectors, frame_type = cap.read()
        if success == False: #IF the video has ended or mve fails to extract motion vectors
            break

        frames.append((frame_type, motion_vectors)) # Frametype (I = Keyframe, P = frame that only referes to past frames, B = frame that referes to both past and future frames. ? = unkown frame type)

    cap.release()
    return frames

