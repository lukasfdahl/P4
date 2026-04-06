import numpy as np
from numpy.typing import NDArray
from data_classes import Clip, Frame

MV_STRUCT = np.dtype([
    ('source', np.int32),
    ('motion_x', np.int32),
    ('motion_y', np.int32),
    ('motion_scale', np.int32)
])


# Function to load the .npz file into usable python classes and numpy arrays
def import_clip(clip_path : str) -> Clip:
    data = np.load(clip_path)
    raw_motion_vectors : np.ndarray = data["motion_vectors"]
    frame_types : NDArray[np.str_] = data["frame_types"]
    residuals : np.ndarray = data["residuals"] # i am pretty sure it uses YUV color format from looking though the code for the cv_reader
    motion_vectors = raw_motion_vectors.view(MV_STRUCT)

    frames = []
    for index in range(len(frame_types)): # loop though each frame
        frame = Frame(motion_vectors[index], frame_types[index], residuals[index])
        frames.append(frame)
    return Clip(frames)


