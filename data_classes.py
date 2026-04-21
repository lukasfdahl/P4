from dataclasses import dataclass # Dataclass adds a few functions to the class autiomatically, like an __init__() dfunction that automatically has all the variables defined as arguments
from numpy import ndarray # Just numpy's version of a list. it supports matricies as well
import numpy as np

# The 4 MV fields the model actually consumes.
# (Raw H.264 also gives block width/height and source/dest pixel coords, but those
#  are not fed to the network — only direction + scale matter for motion understanding.)
MV_STRUCT = np.dtype([
    ('source',       np.int32),   # reference direction: -1 = backward, +1 = forward
    ('motion_x',     np.int32),   # horizontal displacement (in units of 1/motion_scale pixels)
    ('motion_y',     np.int32),   # vertical displacement
    ('motion_scale', np.int32),   # sub-pixel precision: 1, 2, or 4
])

@dataclass
class Frame:
    motion_vectors : ndarray # look at MV_Struct for structure. use the labels shown in MV_struct like motion_vectors[source]. the vectors come in one large grid
    frame_type : str # Frametype (I = Keyframe, P = frame that only referes to past frames, B = frame that referes to both past and future frames. ? = unkown frame type)
    residuals         : ndarray      # H x W x 3  uint8 (likely YUV color format)
    true_bounding_box : tuple[float, float, float, float]  # [xmin, xmax, ymin, ymax]  normalised to [0, 1]
    true_class          : int          # single class label per frame
    has_object          : bool # true if there is an object tracked in the frame. false if not (not all frames have stuff on them)


@dataclass
class Clip:
    frames : list[Frame] # a list of all the frames in the clip in order.