
import numpy as np
from enum import Enum



class LandmarkRepresentation:
    class Representation(Enum):
        GLOBAL_3D = 0
        GLOBAL_FULL_INVERSE_DEPTH = 1
        ANCHORED_3D = 2
        ANCHORED_FULL_INVERSE_DEPTH = 3
        ANCHORED_MSCKF_INVERSE_DEPTH = 4
        ANCHORED_INVERSE_DEPTH_SINGLE = 5
        UNKNOWN = 6

    @staticmethod
    def as_string(feat_representation):
        if feat_representation == LandmarkRepresentation.Representation.GLOBAL_3D:
            return "GLOBAL_3D"
        if feat_representation == LandmarkRepresentation.Representation.GLOBAL_FULL_INVERSE_DEPTH:
            return "GLOBAL_FULL_INVERSE_DEPTH"
        if feat_representation == LandmarkRepresentation.Representation.ANCHORED_3D:
            return "ANCHORED_3D"
        if feat_representation == LandmarkRepresentation.Representation.ANCHORED_FULL_INVERSE_DEPTH:
            return "ANCHORED_FULL_INVERSE_DEPTH"
        if feat_representation == LandmarkRepresentation.Representation.ANCHORED_MSCKF_INVERSE_DEPTH:
            return "ANCHORED_MSCKF_INVERSE_DEPTH"
        if feat_representation == LandmarkRepresentation.Representation.ANCHORED_INVERSE_DEPTH_SINGLE:
            return "ANCHORED_INVERSE_DEPTH_SINGLE"
        return "UNKNOWN"

    @staticmethod
    def from_string(feat_representation):
        if feat_representation == "GLOBAL_3D":
            return LandmarkRepresentation.Representation.GLOBAL_3D
        if feat_representation == "GLOBAL_FULL_INVERSE_DEPTH":
            return LandmarkRepresentation.Representation.GLOBAL_FULL_INVERSE_DEPTH
        if feat_representation == "ANCHORED_3D":
            return LandmarkRepresentation.Representation.ANCHORED_3D
        if feat_representation == "ANCHORED_FULL_INVERSE_DEPTH":
            return LandmarkRepresentation.Representation.ANCHORED_FULL_INVERSE_DEPTH
        if feat_representation == "ANCHORED_MSCKF_INVERSE_DEPTH":
            return LandmarkRepresentation.Representation.ANCHORED_MSCKF_INVERSE_DEPTH
        if feat_representation == "ANCHORED_INVERSE_DEPTH_SINGLE":
            return LandmarkRepresentation.Representation.ANCHORED_INVERSE_DEPTH_SINGLE
        return LandmarkRepresentation.Representation.UNKNOWN

    @staticmethod
    def is_relative_representation(feat_representation):
        return feat_representation in [
            LandmarkRepresentation.Representation.ANCHORED_3D,
            LandmarkRepresentation.Representation.ANCHORED_FULL_INVERSE_DEPTH,
            LandmarkRepresentation.Representation.ANCHORED_MSCKF_INVERSE_DEPTH,
            LandmarkRepresentation.Representation.ANCHORED_INVERSE_DEPTH_SINGLE,
        ]