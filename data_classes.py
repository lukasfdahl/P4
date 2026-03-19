from dataclasses import dataclass # Dataclass adds a few functions to the class autiomatically, like an __init__() dfunction that automatically has all the variables defined as arguments
from numpy import ndarray # Just numpy's version of a list. it supports matricies as well

@dataclass
class MotionVector: # (read here for information about what element in the array means (it is documented) https://github.com/LukasBommes/mv-extractor)
    source      : int   # reference frame offset (e.g. -1 = previous frame)
    width       : int   # macroblock width  (pixels)
    height      : int   # macroblock height (pixels)
    source_x    : int   # source macroblock top-left x
    source_y    : int   # source macroblock top-left y
    destination_x : int # destination macroblock top-left x
    destination_y : int # destination macroblock top-left y
    motion_x    : int   # raw MV x component (in 1/motion_scale px)
    motion_y    : int   # raw MV y component (in 1/motion_scale px)
    motion_scale: int   # sub-pixel precision (e.g. 2 = half-pixel)

@dataclass
class Frame:
    motion_vectors : list[MotionVector]
    frame_type : str # Frametype (I = Keyframe, P = frame that only referes to past frames, B = frame that referes to both past and future frames. ? = unkown frame type)
    residuals         : ndarray      # H x W x 3  uint8
    true_bounding_boxes : list[float]  # [xmin, xmax]  normalised to [0, 1]
    true_class          : int          # single class label per frame