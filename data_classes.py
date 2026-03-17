from dataclasses import dataclass # Dataclass adds a few functions to the class autiomatically, like an __init__() dfunction that automatically has all the variables defined as arguments
from numpy import ndarray # Just numpy's version of a list. it supports matricies as well

@dataclass
class MotionVector: # (read here for information about what element in the array means (it is documented) https://github.com/LukasBommes/mv-extractor)
    source : int
    width : int
    height : int
    source_x : int
    source_y : int
    destination_x : int
    destination_y : int
    motion_x : int
    motion_y : int
    motion_scale : int
    
@dataclass
class Frame:
    motion_vectors : list[MotionVector]
    frame_type : str # Frametype (I = Keyframe, P = frame that only referes to past frames, B = frame that referes to both past and future frames. ? = unkown frame type)
    residuals : ndarray
    true_bounding_boxes : list[list[float]]
    true_classes : list[int]